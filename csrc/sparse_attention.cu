/**
 * Sparse Mask Attention - CUDA Implementation
 *
 * WMMA FP16 kernel: tensor-core QK^T and P@V, online softmax,
 * O accumulated in shared memory (no fragment layout assumptions).
 * Scalar BF16 fallback.
 */

#include "sparse_attention.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
// WMMA FP16 Forward Kernel
//
// Block: 4 warps (128 threads), each warp owns 16 Q rows.
// KV tile = 16 columns (one WMMA tile).
// QK^T via 4 WMMA m16n16k16 ops.
// O accumulated in smem to avoid fragment layout assumptions.
// ============================================================

#define WM 16
#define WN 16
#define WK 16
#define NWARPS 4
#define BM (WM * NWARPS)  // 64

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
    const int D_CHUNKS = D / WK;  // 4 for D=64

    /* -------- shared memory layout -------- */
    extern __shared__ char smem[];
    __half* Q_s = (__half*)smem;                                // [BM, D]
    __half* K_s = Q_s + BM * D;                                // [WN, D]
    __half* V_s = K_s + WN * D;                                // [WN, D]
    float*  S_s = (float*)(V_s + WN * D);                      // [NWARPS, WM, WN] (temp per tile)
    __half* P_s = (__half*)(S_s + NWARPS * WM * WN);           // [NWARPS, WM, WN]
    float*  O_s = (float*)(P_s + NWARPS * WM * WN);            // [NWARPS, WM, D]

    /* -------- initialise O and running state in smem -------- */
    // O_s: NWARPS * WM * D = 4*16*64 = 4096 floats
    for (int i = threadIdx.x; i < NWARPS * WM * D; i += blockDim.x)
        O_s[i] = 0.0f;

    /* -------- load Q once -------- */
    for (int i = threadIdx.x; i < BM * D; i += blockDim.x) {
        int r = i / D, d = i % D;
        int gr = blockIdx.x * BM + r;
        Q_s[i] = (gr < N) ? Q[bhND + gr * D + d] : __float2half(0.0f);
    }
    __syncthreads();

    /* -------- per-row running state (registers) -------- */
    // softmax thread mapping: lane/2 → local row (0..15), lane%2 → half
    const int srow = lane_id / 2;
    const int shalf = lane_id % 2;
    const int grow = row_start + srow;
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    /* -------- tile over KV columns -------- */
    const int n_tiles = CEIL_DIV(N, WN);
    for (int tile = 0; tile < n_tiles; tile++) {
        const int col0 = tile * WN;
        const int tlen = min(WN, N - col0);

        // cooperative load K, V
        for (int i = threadIdx.x; i < WN * D; i += blockDim.x) {
            int kn = i / D, kd = i % D, gc = col0 + kn;
            K_s[i] = (gc < N) ? K[bhND + gc * D + kd] : __float2half(0.0f);
            V_s[i] = (gc < N) ? V[bhND + gc * D + kd] : __float2half(0.0f);
        }
        __syncthreads();

        /* -- WMMA: S[16,16] = Q × K^T -- */
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> S_frag;
        wmma::fill_fragment(S_frag, 0.0f);
        #pragma unroll
        for (int k = 0; k < D_CHUNKS; k++) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> A;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> Bf;
            wmma::load_matrix_sync(A, &Q_s[warp_id * WM * D + k * WK], D);
            wmma::load_matrix_sync(Bf, &K_s[k * WK], D);
            wmma::mma_sync(S_frag, A, Bf, S_frag);
        }
        // scale
        for (int i = 0; i < S_frag.num_elements; i++)
            S_frag.x[i] *= scale;

        // store S → smem
        float* my_S = &S_s[warp_id * WM * WN];
        wmma::store_matrix_sync(my_S, S_frag, WN, wmma::mem_row_major);
        __syncwarp();

        /* -- mask + online softmax (2 threads per row) -- */
        float* S_row = &my_S[srow * WN];
        __half* P_row = &P_s[warp_id * WM * WN + srow * WN];

        // read mask word
        uint32_t mword = 0u;
        if (grow < N) {
            int mbase = ((b * H + h) * N + grow) * n_words;
            int widx = col0 / 32;
            int boff = col0 % 32;
            if (widx < n_words) mword = mask_packed[mbase + widx] >> boff;
        }

        int c_off = shalf * 8;
        float vals[8];
        float lmax = -FLT_MAX;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int j = c_off + i;
            bool ok = (grow < N) && (j < tlen) && ((mword >> j) & 1u);
            vals[i] = ok ? S_row[j] : -FLT_MAX;
            lmax = fmaxf(lmax, vals[i]);
        }
        float pmax = __shfl_xor_sync(0xffffffff, lmax, 1);
        float tile_max = fmaxf(lmax, pmax);

        float new_max = fmaxf(running_max, tile_max);
        float alpha = (running_max > -FLT_MAX) ? expf(running_max - new_max) : 0.0f;

        // P values + tile sum
        float lsum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float p = (vals[i] > -FLT_MAX) ? expf(vals[i] - new_max) : 0.0f;
            lsum += p;
            P_row[c_off + i] = __float2half(p);
        }
        float psum = __shfl_xor_sync(0xffffffff, lsum, 1);
        float tile_sum = lsum + psum;

        running_sum = running_sum * alpha + tile_sum;
        running_max = new_max;

        /* -- rescale O_s by alpha (smem, per warp) -- */
        // Each pair handles one row; thread writes 32 of 64 D values
        float* my_O = &O_s[warp_id * WM * D];
        if (grow < N) {
            int d_start = shalf * 32;
            for (int d = 0; d < 32; d++)
                my_O[srow * D + d_start + d] *= alpha;
        }
        __syncwarp();

        /* -- WMMA: O_partial = P × V, add to O_s -- */
        // Process one D-chunk at a time: compute fragment, store to temp, add to O_s
        float* temp_buf = my_S;  // reuse S_s area for this warp (256 floats)

        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS; dc++) {
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> Ofrag;
            wmma::fill_fragment(Ofrag, 0.0f);
            wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> Pf;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> Vf;
            wmma::load_matrix_sync(Pf, &P_s[warp_id * WM * WN], WN);
            wmma::load_matrix_sync(Vf, &V_s[dc * WK], D);
            wmma::mma_sync(Ofrag, Pf, Vf, Ofrag);

            // store to temp
            wmma::store_matrix_sync(temp_buf, Ofrag, WN, wmma::mem_row_major);
            __syncwarp();

            // add to O_s: each pair of threads handles one row
            if (grow < N) {
                int d_base = shalf * 8;
                for (int i = 0; i < 8; i++) {
                    my_O[srow * D + dc * WK + d_base + i] += temp_buf[srow * WN + d_base + i];
                }
            }
            __syncwarp();
        }

        __syncthreads();
    }

    /* -------- write output -------- */
    float* my_O = &O_s[warp_id * WM * D];
    if (grow < N) {
        float inv = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        int d_start = shalf * 32;
        for (int d = 0; d < 32; d++) {
            float v = my_O[srow * D + d_start + d] * inv;
            Out[bhND + grow * D + d_start + d] = __float2half(v);
        }
        if (is_training && shalf == 0) {
            int idx = (b * H + h) * N + grow;
            lse[idx] = (running_sum > 0.0f)
                     ? (running_max + logf(running_sum)) : -FLT_MAX;
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
    int smem_bytes = BM * D * 2                // Q_s
                   + WN * D * 2                // K_s
                   + WN * D * 2                // V_s
                   + NWARPS * WM * WN * 4      // S_s
                   + NWARPS * WM * WN * 2      // P_s
                   + NWARPS * WM * D * 4;      // O_s

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
