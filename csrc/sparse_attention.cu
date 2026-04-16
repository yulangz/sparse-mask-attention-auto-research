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
#define BN_TILE 64          // outer tile width for K/V prefetch (2x BN sub-tiles)
#define P_STRIDE (BN + 16)  // padded stride for half P_s, must be multiple of 16 for WMMA (48 halfs=96B)

// Helper: issue cp.async for one K/V tile (BN_TILE=64 cols) into smem buffer
__device__ __forceinline__ void prefetch_kv_tile(
    __half* K_dst, __half* V_dst,
    const __half* K_src_base, const __half* V_src_base,
    int col0, int N, int D, int bhND
) {
    const int n_vec = (BN_TILE * D) / 8;
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

    /* -------- shared memory: no S_s needed -------- */
    extern __shared__ char smem[];
    __half* Q_s   = (__half*)smem;                                  // [BM, D]
    __half* K_buf = Q_s + BM * D;                                  // [2, BN_TILE, D]
    __half* V_buf = K_buf + 2 * BN_TILE * D;                       // [2, BN_TILE, D]
    __half* P_s   = (__half*)(V_buf + 2 * BN_TILE * D);            // [NWARPS, WM, P_STRIDE] padded

    /* -------- fragment row AND column mapping -------- */
    __half* tmpA = Q_s;
    __half* tmpB = Q_s + 256;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        int r = i / 16, c = i % 16;
        tmpA[i] = __float2half((r == c) ? 1.0f : 0.0f);
        tmpB[i] = __float2half((float)(r * 16 + c));  // encode row*16+col
    }
    __syncthreads();

    int elem_rows[8], elem_cols[8];
    {
        wmma::fragment<wmma::matrix_a, WM, WMMA_N, WK, __half, wmma::row_major> mA;
        wmma::fragment<wmma::matrix_b, WM, WMMA_N, WK, __half, wmma::row_major> mB;
        wmma::fragment<wmma::accumulator, WM, WMMA_N, WK, float> mC;
        wmma::load_matrix_sync(mA, tmpA, 16);
        wmma::load_matrix_sync(mB, tmpB, 16);
        wmma::fill_fragment(mC, 0.0f);
        wmma::mma_sync(mC, mA, mB, mC);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int val = __float2int_rn(mC.x[i]);
            elem_rows[i] = val / 16;
            elem_cols[i] = val % 16;
        }
    }
    // Identify unique rows for this thread (for alpha shuffle + running state)
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

    /* -------- per-row running state (2 rows per thread) -------- */
    float running_max_r0 = -FLT_MAX, running_max_r1 = -FLT_MAX;
    float running_sum_r0 = 0.0f, running_sum_r1 = 0.0f;

    wmma::fragment<wmma::accumulator, WM, WMMA_N, WK, float> O_acc[D_CHUNKS];
    for (int d = 0; d < D_CHUNKS; d++)
        wmma::fill_fragment(O_acc[d], 0.0f);

    /* -------- mask base addresses for our two rows -------- */
    const int grow_r0 = row_start + my_r0;
    const int grow_r1 = row_start + my_r1;
    const int mask_base_r0 = (grow_r0 < N) ? ((b * H + h) * N + grow_r0) * n_words : 0;
    const int mask_base_r1 = (grow_r1 < N) ? ((b * H + h) * N + grow_r1) * n_words : 0;

    /* -------- tile loop: 64-col outer tiles, 2x 32-col sub-tiles -------- */
    const int n_tiles_64 = (N + BN_TILE - 1) / BN_TILE;
    for (int tile64 = 0; tile64 < n_tiles_64; tile64++) {
        const int buf = tile64 & 1;

        // Async prefetch NEXT 64-col tile into other buffer
        if (tile64 + 1 < n_tiles_64) {
            int next_buf_off = (1 - buf) * BN_TILE * D;
            prefetch_kv_tile(K_buf + next_buf_off, V_buf + next_buf_off,
                             K, V, (tile64 + 1) * BN_TILE, N, D, bhND);
            __pipeline_commit();
        }

        // Process 2 sub-tiles of 32 cols each
        for (int sub = 0; sub < 2; sub++) {
            const int sub_col0 = tile64 * BN_TILE + sub * BN;
            if (sub_col0 >= N) break;
            const int tlen = min(BN, N - sub_col0);
            const int mask_word = tile64 * 2 + sub;

            __half* K_s = K_buf + buf * BN_TILE * D + sub * BN * D;
            __half* V_s = V_buf + buf * BN_TILE * D + sub * BN * D;

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

        /* -- In-register softmax: no S_s needed -- */
        // Load mask words for our two rows
        uint32_t mword_r0 = (grow_r0 < N && mask_word < n_words) ? mask_packed[mask_base_r0 + mask_word] : 0u;
        uint32_t mword_r1 = (grow_r1 < N && mask_word < n_words) ? mask_packed[mask_base_r1 + mask_word] : 0u;

        // Apply mask to S values in registers and find per-row max
        float max_r0 = -FLT_MAX, max_r1 = -FLT_MAX;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            // S_left: actual column = elem_cols[i] (0-15)
            int col_l = elem_cols[i];
            uint32_t mw = (elem_rows[i] == my_r0) ? mword_r0 : mword_r1;
            int gr = (elem_rows[i] == my_r0) ? grow_r0 : grow_r1;
            bool ok = (gr < N) && (col_l < tlen) && ((mw >> col_l) & 1u);
            S_left.x[i] = ok ? S_left.x[i] : -FLT_MAX;
            if (elem_rows[i] == my_r0) max_r0 = fmaxf(max_r0, S_left.x[i]);
            else                        max_r1 = fmaxf(max_r1, S_left.x[i]);

            // S_right: actual column = elem_cols[i] + 16
            int col_r = col_l + 16;
            ok = (gr < N) && (col_r < tlen) && ((mw >> col_r) & 1u);
            S_right.x[i] = ok ? S_right.x[i] : -FLT_MAX;
            if (elem_rows[i] == my_r0) max_r0 = fmaxf(max_r0, S_right.x[i]);
            else                        max_r1 = fmaxf(max_r1, S_right.x[i]);
        }

        // Reduce max across 4 threads in octet (XOR 1 + XOR 2)
        float t;
        t = __shfl_xor_sync(0xffffffff, max_r0, 1); max_r0 = fmaxf(max_r0, t);
        t = __shfl_xor_sync(0xffffffff, max_r0, 2); max_r0 = fmaxf(max_r0, t);
        t = __shfl_xor_sync(0xffffffff, max_r1, 1); max_r1 = fmaxf(max_r1, t);
        t = __shfl_xor_sync(0xffffffff, max_r1, 2); max_r1 = fmaxf(max_r1, t);

        // Online softmax update
        float new_max_r0 = fmaxf(running_max_r0, max_r0);
        float new_max_r1 = fmaxf(running_max_r1, max_r1);
        float alpha_r0 = (running_max_r0 > -FLT_MAX) ? expf(running_max_r0 - new_max_r0) : 0.0f;
        float alpha_r1 = (running_max_r1 > -FLT_MAX) ? expf(running_max_r1 - new_max_r1) : 0.0f;

        /* -- rescale O_acc -- */
        #pragma unroll
        for (int d = 0; d < D_CHUNKS; d++) {
            #pragma unroll
            for (int i = 0; i < 8; i++)
                O_acc[d].x[i] *= (elem_rows[i] == my_r0) ? alpha_r0 : alpha_r1;
        }

        // Compute exp, accumulate sum, write P to smem
        float sum_r0 = 0.0f, sum_r1 = 0.0f;
        __half* my_P = &P_s[warp_id * WM * P_STRIDE];

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float nm, s_val;
            int row_local;

            // S_left
            row_local = elem_rows[i];
            nm = (row_local == my_r0) ? new_max_r0 : new_max_r1;
            s_val = S_left.x[i];
            float p_l = (s_val > -FLT_MAX) ? expf(s_val - nm) : 0.0f;
            if (row_local == my_r0) sum_r0 += p_l; else sum_r1 += p_l;
            my_P[row_local * P_STRIDE + elem_cols[i]] = __float2half(p_l);

            // S_right
            nm = (row_local == my_r0) ? new_max_r0 : new_max_r1;
            s_val = S_right.x[i];
            float p_r = (s_val > -FLT_MAX) ? expf(s_val - nm) : 0.0f;
            if (row_local == my_r0) sum_r0 += p_r; else sum_r1 += p_r;
            my_P[row_local * P_STRIDE + elem_cols[i] + 16] = __float2half(p_r);
        }

        // Reduce sum across 4 threads in octet
        t = __shfl_xor_sync(0xffffffff, sum_r0, 1); sum_r0 += t;
        t = __shfl_xor_sync(0xffffffff, sum_r0, 2); sum_r0 += t;
        t = __shfl_xor_sync(0xffffffff, sum_r1, 1); sum_r1 += t;
        t = __shfl_xor_sync(0xffffffff, sum_r1, 2); sum_r1 += t;

        running_sum_r0 = running_sum_r0 * alpha_r0 + sum_r0;
        running_sum_r1 = running_sum_r1 * alpha_r1 + sum_r1;
        running_max_r0 = new_max_r0;
        running_max_r1 = new_max_r1;

        /* -- WMMA: O_acc += P × V -- */
        __syncwarp();
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

        } // end sub-tile loop

        // Wait for next 64-col tile's async prefetch, then sync all threads
        if (tile64 + 1 < n_tiles_64) {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
    }

    /* -------- write output directly from registers -------- */
    float my_inv_r0 = (running_sum_r0 > 0.0f) ? (1.0f / running_sum_r0) : 0.0f;
    float my_inv_r1 = (running_sum_r1 > 0.0f) ? (1.0f / running_sum_r1) : 0.0f;

    #pragma unroll
    for (int d = 0; d < D_CHUNKS; d++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int r = row_start + elem_rows[i];
            int c = d * WMMA_N + elem_cols[i];
            if (r < N) {
                float inv = (elem_rows[i] == my_r0) ? my_inv_r0 : my_inv_r1;
                Out[bhND + r * D + c] = __float2half(O_acc[d].x[i] * inv);
            }
        }
    }

    // Write LSE (one thread per row in octet)
    if (is_training && (lane_id % 4) == 0) {
        if (grow_r0 < N) {
            int idx = (b * H + h) * N + grow_r0;
            lse[idx] = (running_sum_r0 > 0.0f) ? (running_max_r0 + logf(running_sum_r0)) : -FLT_MAX;
        }
        if (grow_r1 < N) {
            int idx = (b * H + h) * N + grow_r1;
            lse[idx] = (running_sum_r1 > 0.0f) ? (running_max_r1 + logf(running_sum_r1)) : -FLT_MAX;
        }
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
    // No S_s needed (in-register softmax). BN_TILE=64 for K/V prefetch.
    int smem_bytes = BM * D * 2             // Q_s
                   + 2 * BN_TILE * D * 2    // K_buf[2] (64-col tiles)
                   + 2 * BN_TILE * D * 2    // V_buf[2] (64-col tiles)
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
