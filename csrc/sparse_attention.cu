/**
 * Sparse Mask Attention - CUDA Implementation
 *
 * WMMA FP16 kernel with:
 * - Double-buffered K/V with cp.async prefetch
 * - Tensor-core QK^T and P@V
 * - In-fragment O accumulation with runtime row mapping
 * Scalar BF16 fallback.
 */

#include "sparse_attention.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <float.h>
#include <math.h>

using namespace nvcuda;

// ============================================================
// Pack Mask Kernel
// ============================================================

__global__ void pack_mask_kernel(
    const bool* __restrict__ mask,
    uint32_t* __restrict__ mask_packed,
    int total_rows, int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    int n_words = (N + 31) / 32;
    for (int w = 0; w < n_words; w++) {
        uint32_t packed = 0u;
        int cs = w * 32;
        #pragma unroll
        for (int bit = 0; bit < 32; bit++) {
            int col = cs + bit;
            if (col < N && mask[row * N + col])
                packed |= (1u << bit);
        }
        mask_packed[row * n_words + w] = packed;
    }
}

void pack_mask_bits(
    const bool* mask, uint32_t* mask_packed,
    int B, int H, int N, cudaStream_t stream
) {
    int total = B * H * N;
    pack_mask_kernel<<<CEIL_DIV(total, 256), 256, 0, stream>>>(
        mask, mask_packed, total, N);
}

// ============================================================
// WMMA FP16 Forward — double-buffered K/V with cp.async
// ============================================================

#define WM 16
#define WK 16
#define WMMA_N 16
#define BN 32
#define NWARPS 8
#define BM (WM * NWARPS)
#define D_CHUNKS 4
#define S_STRIDE (BN + 4)   // padded stride for float S_s to avoid bank conflicts (36 floats=144B, 144%128=16)
#define P_STRIDE (BN + 16)  // padded stride for half P_s, must be multiple of 16 for WMMA (48 halfs=96B)

// Helper: issue cp.async for one K/V tile into smem buffer
__device__ __forceinline__ void prefetch_kv_tile(
    __half* K_dst, __half* V_dst,
    const __half* K_src_base, const __half* V_src_base,
    int col0, int N, int D, int bhND
) {
    const int n_vec = (BN * D) / 8;
    float4* K4 = (float4*)K_dst;
    float4* V4 = (float4*)V_dst;
    const float4* Ks = (const float4*)(K_src_base + bhND + col0 * D);
    const float4* Vs = (const float4*)(V_src_base + bhND + col0 * D);
    const float4 z4 = {0.f, 0.f, 0.f, 0.f};
    for (int i = threadIdx.x; i < n_vec; i += blockDim.x) {
        int gc = col0 + (i * 8) / D;
        if (gc < N) {
            __pipeline_memcpy_async(&K4[i], &Ks[i], 16);
            __pipeline_memcpy_async(&V4[i], &Vs[i], 16);
        } else {
            K4[i] = z4;
            V4[i] = z4;
        }
    }
}

__global__ void sparse_attn_wmma_fp16(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const uint32_t* __restrict__ mask_packed,
    __half* __restrict__ Out,
    float* __restrict__ lse,
    int B, int H, int N, int D,
    float scale, bool is_training
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row_start = blockIdx.x * BM + warp_id * WM;
    const int bh = blockIdx.y;
    const int b = bh / H, h = bh % H;

    if (b >= B) return;

    const int bhND = ((b * H + h) * N) * D;
    const int n_words = (N + 31) / 32;

    /* -------- shared memory: double-buffered K/V -------- */
    extern __shared__ char smem[];
    __half* Q_s   = (__half*)smem;                                  // [BM, D]
    __half* K_buf = Q_s + BM * D;                                  // [2, BN, D]
    __half* V_buf = K_buf + 2 * BN * D;                            // [2, BN, D]
    float*  S_s   = (float*)(V_buf + 2 * BN * D);                  // [NWARPS, WM, S_STRIDE] padded
    __half* P_s   = (__half*)(S_s + NWARPS * WM * S_STRIDE);       // [NWARPS, WM, P_STRIDE] padded

    /* -------- fragment row mapping -------- */
    __half* tmpA = Q_s;
    __half* tmpB = Q_s + 256;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        int r = i / 16, c = i % 16;
        tmpA[i] = __float2half((r == c) ? 1.0f : 0.0f);
        tmpB[i] = __float2half((float)r);
    }
    __syncthreads();

    int elem_rows[8];
    {
        wmma::fragment<wmma::matrix_a, WM, WMMA_N, WK, __half, wmma::row_major> mA;
        wmma::fragment<wmma::matrix_b, WM, WMMA_N, WK, __half, wmma::row_major> mB;
        wmma::fragment<wmma::accumulator, WM, WMMA_N, WK, float> mC;
        wmma::load_matrix_sync(mA, tmpA, 16);
        wmma::load_matrix_sync(mB, tmpB, 16);
        wmma::fill_fragment(mC, 0.0f);
        wmma::mma_sync(mC, mA, mB, mC);
        #pragma unroll
        for (int i = 0; i < 8; i++) elem_rows[i] = __float2int_rn(mC.x[i]);
    }
    // Identify unique rows for this thread (for alpha shuffle)
    int my_r0 = elem_rows[0], my_r1 = my_r0;
    for (int i = 1; i < 8; i++) {
        if (elem_rows[i] != my_r0) { my_r1 = elem_rows[i]; break; }
    }
    __syncthreads();

    /* -------- load Q -------- */
    {
        const int n_vec = (BM * D) / 8;
        const float4 z4 = {0.f, 0.f, 0.f, 0.f};
        float4* Q4 = (float4*)Q_s;
        const float4* Qs = (const float4*)(Q + bhND + blockIdx.x * BM * D);
        for (int i = threadIdx.x; i < n_vec; i += blockDim.x) {
            int gr = blockIdx.x * BM + (i * 8) / D;
            Q4[i] = (gr < N) ? Qs[i] : z4;
        }
    }

    /* -------- prefetch tile 0 into buffer 0 -------- */
    prefetch_kv_tile(K_buf, V_buf, K, V, 0, N, D, bhND);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    /* -------- per-row running state -------- */
    const int srow  = lane_id / 2;
    const int shalf = lane_id % 2;
    const int grow  = row_start + srow;
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    wmma::fragment<wmma::accumulator, WM, WMMA_N, WK, float> O_acc[D_CHUNKS];
    for (int d = 0; d < D_CHUNKS; d++)
        wmma::fill_fragment(O_acc[d], 0.0f);

    const int mask_base = (grow < N) ? ((b * H + h) * N + grow) * n_words : 0;

    /* -------- tile loop with double buffering -------- */
    for (int tile = 0; tile < n_words; tile++) {
        const int buf = tile & 1;
        __half* K_s = K_buf + buf * BN * D;
        __half* V_s = V_buf + buf * BN * D;
        const int col0 = tile * 32;
        const int tlen = min(32, N - col0);

        // Async prefetch NEXT tile into other buffer
        if (tile + 1 < n_words) {
            int next_buf_off = (1 - buf) * BN * D;
            prefetch_kv_tile(K_buf + next_buf_off, V_buf + next_buf_off,
                             K, V, (tile + 1) * 32, N, D, bhND);
            __pipeline_commit();
        }

        /* -- WMMA: S = Q × K^T -- */
        wmma::fragment<wmma::accumulator, WM, WMMA_N, WK, float> S_left, S_right;
        wmma::fill_fragment(S_left, 0.0f);
        wmma::fill_fragment(S_right, 0.0f);

        #pragma unroll
        for (int k = 0; k < D_CHUNKS; k++) {
            wmma::fragment<wmma::matrix_a, WM, WMMA_N, WK, __half, wmma::row_major> A;
            wmma::load_matrix_sync(A, &Q_s[warp_id * WM * D + k * WK], D);
            wmma::fragment<wmma::matrix_b, WM, WMMA_N, WK, __half, wmma::col_major> Bl, Br;
            wmma::load_matrix_sync(Bl, &K_s[k * WK], D);
            wmma::load_matrix_sync(Br, &K_s[16 * D + k * WK], D);
            wmma::mma_sync(S_left, A, Bl, S_left);
            wmma::mma_sync(S_right, A, Br, S_right);
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) { S_left.x[i] *= scale; S_right.x[i] *= scale; }

        // Store S → smem (padded stride)
        float* my_S = &S_s[warp_id * WM * S_STRIDE];
        wmma::store_matrix_sync(&my_S[0], S_left, S_STRIDE, wmma::mem_row_major);
        wmma::store_matrix_sync(&my_S[16], S_right, S_STRIDE, wmma::mem_row_major);
        __syncwarp();

        /* -- softmax (process in 2 sub-passes of 8 to reduce stack) -- */
        float* S_row = &my_S[srow * S_STRIDE];
        __half* P_row = &P_s[warp_id * WM * P_STRIDE + srow * P_STRIDE];
        uint32_t mword = 0u;
        if (grow < N) mword = mask_packed[mask_base + tile];

        int c_off = shalf * 16;
        float lmax = -FLT_MAX;

        // Sub-pass 1: cols [c_off, c_off+8) — find max, store vals, compute P
        float vals_a[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int j = c_off + i;
            bool ok = (grow < N) && (j < tlen) && ((mword >> j) & 1u);
            vals_a[i] = ok ? S_row[j] : -FLT_MAX;
            lmax = fmaxf(lmax, vals_a[i]);
        }
        // Sub-pass 2: cols [c_off+8, c_off+16) — find max
        float vals_b[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int j = c_off + 8 + i;
            bool ok = (grow < N) && (j < tlen) && ((mword >> j) & 1u);
            vals_b[i] = ok ? S_row[j] : -FLT_MAX;
            lmax = fmaxf(lmax, vals_b[i]);
        }

        float pmax = __shfl_xor_sync(0xffffffff, lmax, 1);
        float tile_max = fmaxf(lmax, pmax);
        float new_max = fmaxf(running_max, tile_max);
        float alpha = (running_max > -FLT_MAX) ? expf(running_max - new_max) : 0.0f;

        float lsum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float p = (vals_a[i] > -FLT_MAX) ? expf(vals_a[i] - new_max) : 0.0f;
            lsum += p;
            P_row[c_off + i] = __float2half(p);
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float p = (vals_b[i] > -FLT_MAX) ? expf(vals_b[i] - new_max) : 0.0f;
            lsum += p;
            P_row[c_off + 8 + i] = __float2half(p);
        }
        float psum = __shfl_xor_sync(0xffffffff, lsum, 1);
        running_sum = running_sum * alpha + lsum + psum;
        running_max = new_max;

        /* -- rescale O_acc via shuffle (no smem needed) -- */
        // srow = lane_id/2 owns alpha for its row. Get alpha for fragment rows via shfl.
        float alpha_r0 = __shfl_sync(0xffffffff, alpha, 2 * my_r0);
        float alpha_r1 = __shfl_sync(0xffffffff, alpha, 2 * my_r1);
        #pragma unroll
        for (int d = 0; d < D_CHUNKS; d++) {
            #pragma unroll
            for (int i = 0; i < 8; i++)
                O_acc[d].x[i] *= (elem_rows[i] == my_r0) ? alpha_r0 : alpha_r1;
        }

        /* -- WMMA: O_acc += P × V -- */
        // Load P fragments once, reuse across all D chunks
        {
            wmma::fragment<wmma::matrix_a, WM, WMMA_N, WK, __half, wmma::row_major> Pl, Pr;
            wmma::load_matrix_sync(Pl, &P_s[warp_id * WM * P_STRIDE], P_STRIDE);
            wmma::load_matrix_sync(Pr, &P_s[warp_id * WM * P_STRIDE + 16], P_STRIDE);

            #pragma unroll
            for (int dc = 0; dc < D_CHUNKS; dc++) {
                wmma::fragment<wmma::matrix_b, WM, WMMA_N, WK, __half, wmma::row_major> Vt, Vb;
                wmma::load_matrix_sync(Vt, &V_s[dc * WK], D);
                wmma::load_matrix_sync(Vb, &V_s[16 * D + dc * WK], D);
                wmma::mma_sync(O_acc[dc], Pl, Vt, O_acc[dc]);
                wmma::mma_sync(O_acc[dc], Pr, Vb, O_acc[dc]);
            }
        }

        // Wait for next tile's async prefetch, then sync all threads
        if (tile + 1 < n_words) {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
    }

    /* -------- write output -------- */
    for (int w = 0; w < NWARPS; w++) {
        if (warp_id == w) {
            float* O_buf = S_s;
            #pragma unroll
            for (int d = 0; d < D_CHUNKS; d++)
                wmma::store_matrix_sync(&O_buf[d * WMMA_N], O_acc[d], D,
                                        wmma::mem_row_major);
            __syncwarp();
            if (grow < N) {
                float inv = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
                int d_start = shalf * 32;
                for (int d = 0; d < 32; d++) {
                    float v = O_buf[srow * D + d_start + d] * inv;
                    Out[bhND + grow * D + d_start + d] = __float2half(v);
                }
                if (is_training && shalf == 0) {
                    int idx = (b * H + h) * N + grow;
                    lse[idx] = (running_sum > 0.0f)
                             ? (running_max + logf(running_sum)) : -FLT_MAX;
                }
            }
        }
        __syncthreads();
    }
}

// ============================================================
// Scalar BF16 fallback
// ============================================================

#define SCALAR_BM 16
#define SCALAR_BN 32

template <typename T>
__global__ void sparse_attn_scalar_kernel(
    const T* __restrict__ Q, const T* __restrict__ K,
    const T* __restrict__ V, const uint32_t* __restrict__ mask_packed,
    T* __restrict__ Out, float* __restrict__ lse,
    int B, int H, int N, int D, float scale, bool is_training
) {
    const int wid = threadIdx.x >> 5, lid = threadIdx.x & 31;
    const int row = blockIdx.x * SCALAR_BM + wid;
    const int bh = blockIdx.y, b_ = bh / H, h_ = bh % H;
    const bool ok = (row < N && b_ < B);
    const int base = ((b_ * H + h_) * N) * D;
    const int nw = (N + 31) / 32;
    const int mb = ok ? ((b_ * H + h_) * N + row) * nw : 0;
    float q_r[8], acc[8]; int nd = 0;
    for (int d = lid; d < D; d += 32) {
        q_r[nd] = ok ? to_float(Q[base + row * D + d]) : 0.f;
        acc[nd] = 0.f; nd++;
    }
    float rm = -FLT_MAX, rs = 0.f;
    extern __shared__ char sm[];
    T* Ks = (T*)sm; T* Vs = (T*)(sm + SCALAR_BN * D * sizeof(T));
    for (int t = 0; t < nw; t++) {
        int cs = t * 32, tl = min(32, N - cs);
        for (int i = threadIdx.x; i < SCALAR_BN * D; i += blockDim.x) {
            int kn = i / D, kd = i % D, c = cs + kn;
            if (c < N) { Ks[i] = K[base + c * D + kd]; Vs[i] = V[base + c * D + kd]; }
        }
        __syncthreads();
        if (ok) {
            uint32_t mw = mask_packed[mb + t];
            if (mw) for (int j = 0; j < tl; j++) {
                if (!((mw >> j) & 1u)) continue;
                float dot = 0.f;
                for (int di = 0, d = lid; d < D; d += 32, di++)
                    dot += q_r[di] * to_float(Ks[j * D + d]);
                float sc = warp_reduce_sum(dot) * scale;
                float nm = fmaxf(rm, sc), al = expf(rm - nm), p = expf(sc - nm);
                rs = rs * al + p;
                for (int di = 0, d = lid; d < D; d += 32, di++)
                    acc[di] = acc[di] * al + p * to_float(Vs[j * D + d]);
                rm = nm;
            }
        }
        __syncthreads();
    }
    if (ok) {
        float iv = (rs > 0.f) ? (1.f / rs) : 0.f;
        for (int di = 0, d = lid; d < D; d += 32, di++)
            Out[base + row * D + d] = from_float<T>(acc[di] * iv);
        if (is_training && lid == 0) {
            int idx = (b_ * H + h_) * N + row;
            lse[idx] = (rs > 0.f) ? (rm + logf(rs)) : -FLT_MAX;
        }
    }
}

// ============================================================
// Host Launchers
// ============================================================

void sparse_attention_fwd_bf16(
    const SparseAttentionParams& params, cudaStream_t stream
) {
    dim3 grid(CEIL_DIV(params.seq_len, SCALAR_BM),
              params.batch_size * params.num_heads);
    dim3 block(SCALAR_BM * 32);
    int smem = 2 * SCALAR_BN * params.head_dim * sizeof(__nv_bfloat16);
    sparse_attn_scalar_kernel<__nv_bfloat16><<<grid, block, smem, stream>>>(
        (const __nv_bfloat16*)params.q, (const __nv_bfloat16*)params.k,
        (const __nv_bfloat16*)params.v, params.mask_packed,
        (__nv_bfloat16*)params.out, params.lse,
        params.batch_size, params.num_heads, params.seq_len, params.head_dim,
        params.scale, params.is_training);
}

void sparse_attention_fwd_fp16(
    const SparseAttentionParams& params, cudaStream_t stream
) {
    int D = params.head_dim;
    int N = params.seq_len;
    dim3 grid(CEIL_DIV(N, BM), params.batch_size * params.num_heads);
    dim3 block(NWARPS * 32);
    // Double-buffered K/V: 2× K_s + 2× V_s, padded S_s and P_s
    int smem_bytes = BM * D * 2             // Q_s
                   + 2 * BN * D * 2         // K_buf[2]
                   + 2 * BN * D * 2         // V_buf[2]
                   + NWARPS * WM * S_STRIDE * 4   // S_s (padded)
                   + NWARPS * WM * P_STRIDE * 2;  // P_s (padded)

    // Request extended shared memory if needed (>48KB)
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(sparse_attn_wmma_fp16,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes);
    }

    sparse_attn_wmma_fp16<<<grid, block, smem_bytes, stream>>>(
        (const __half*)params.q, (const __half*)params.k,
        (const __half*)params.v, params.mask_packed,
        (__half*)params.out, params.lse,
        params.batch_size, params.num_heads, params.seq_len, D,
        params.scale, params.is_training);
}

void sparse_attention_bwd_bf16(
    const SparseAttentionParams& params,
    const void* dout, void* dq, void* dk, void* dv,
    cudaStream_t stream
) { /* stub */ }
