/**
 * Sparse Mask Attention - CUDA Implementation
 *
 * WMMA FP16 kernel: tensor-core QK^T and P@V, online softmax,
 * O accumulated in shared memory. BN=32 columns per KV tile.
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
// WMMA FP16 Forward Kernel  (BN=32)
//
// Block: 4 warps (128 threads), each warp owns 16 Q rows.
// KV tile = 32 columns (two WMMA-N tiles, aligned with mask word).
// QK^T: 2×4 = 8 WMMA ops per tile.
// P@V : 2×4 = 8 WMMA ops per tile (P split left/right, V top/bottom).
// Online softmax, O accumulated in smem.
// ============================================================

#define WM 16
#define WK 16
#define WMMA_N 16       // WMMA N dimension (fixed)
#define BN 32           // KV tile size (= 2 WMMA tiles = 1 mask word)
#define NWARPS 4
#define BM (WM * NWARPS) // 64

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

    /* -------- shared memory layout (no O_s needed) -------- */
    extern __shared__ char smem[];
    __half* Q_s = (__half*)smem;                                    // [BM, D]
    __half* K_s = Q_s + BM * D;                                    // [BN, D]
    __half* V_s = K_s + BN * D;                                    // [BN, D]
    float*  S_s = (float*)(V_s + BN * D);                          // [NWARPS, WM, BN]
    __half* P_s = (__half*)(S_s + NWARPS * WM * BN);               // [NWARPS, WM, BN]

    /* -------- determine fragment row mapping (one-time, arch-independent) -------- */
    // Use WMMA to compute C = Identity × RowIndex → C[i,j] = i (row of each element)
    __half* tmpA = Q_s;         // reuse Q_s before Q is loaded (need 256 halfs)
    __half* tmpB = Q_s + 256;   // next 256 halfs
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        int r = i / 16, c = i % 16;
        tmpA[i] = __float2half((r == c) ? 1.0f : 0.0f);  // identity
        tmpB[i] = __float2half((float)r);                  // B[k,j] = k
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, WM, WMMA_N, WK, float> map_frag;
    {
        wmma::fragment<wmma::matrix_a, WM, WMMA_N, WK, __half, wmma::row_major> mA;
        wmma::fragment<wmma::matrix_b, WM, WMMA_N, WK, __half, wmma::row_major> mB;
        wmma::load_matrix_sync(mA, tmpA, 16);
        wmma::load_matrix_sync(mB, tmpB, 16);
        wmma::fill_fragment(map_frag, 0.0f);
        wmma::mma_sync(map_frag, mA, mB, map_frag);
    }
    int elem_rows[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        elem_rows[i] = __float2int_rn(map_frag.x[i]);
    __syncthreads();

    /* -------- load Q -------- */
    for (int i = threadIdx.x; i < BM * D; i += blockDim.x) {
        int r = i / D, d = i % D;
        int gr = blockIdx.x * BM + r;
        Q_s[i] = (gr < N) ? Q[bhND + gr * D + d] : __float2half(0.0f);
    }
    __syncthreads();

    /* -------- per-row running state -------- */
    const int srow  = lane_id / 2;          // 0..15
    const int shalf = lane_id % 2;          // 0 or 1
    const int grow  = row_start + srow;
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    /* -------- O accumulator in WMMA fragments -------- */
    wmma::fragment<wmma::accumulator, WM, WMMA_N, WK, float> O_acc[4]; // D/16 chunks
    for (int d = 0; d < D_CHUNKS; d++)
        wmma::fill_fragment(O_acc[d], 0.0f);

    /* -------- tile over KV columns (BN=32, one mask word) -------- */
    for (int tile = 0; tile < n_words; tile++) {
        const int col0 = tile * 32;
        const int tlen = min(32, N - col0);

        // cooperative load K, V  [BN, D]
        for (int i = threadIdx.x; i < BN * D; i += blockDim.x) {
            int kn = i / D, kd = i % D, gc = col0 + kn;
            K_s[i] = (gc < N) ? K[bhND + gc * D + kd] : __float2half(0.0f);
            V_s[i] = (gc < N) ? V[bhND + gc * D + kd] : __float2half(0.0f);
        }
        __syncthreads();

        /* -- WMMA: S[WM, BN] = Q × K^T  (two [16,16] halves) -- */
        wmma::fragment<wmma::accumulator, WM, WMMA_N, WK, float> S_left, S_right;
        wmma::fill_fragment(S_left, 0.0f);
        wmma::fill_fragment(S_right, 0.0f);

        #pragma unroll
        for (int k = 0; k < D_CHUNKS; k++) {
            wmma::fragment<wmma::matrix_a, WM, WMMA_N, WK, __half, wmma::row_major> A;
            wmma::load_matrix_sync(A, &Q_s[warp_id * WM * D + k * WK], D);

            wmma::fragment<wmma::matrix_b, WM, WMMA_N, WK, __half, wmma::col_major> Bl, Br;
            // K_left = K_s[0:16, :], K_right = K_s[16:32, :]
            wmma::load_matrix_sync(Bl, &K_s[k * WK], D);               // rows 0-15
            wmma::load_matrix_sync(Br, &K_s[16 * D + k * WK], D);      // rows 16-31

            wmma::mma_sync(S_left,  A, Bl, S_left);
            wmma::mma_sync(S_right, A, Br, S_right);
        }

        // scale
        for (int i = 0; i < S_left.num_elements; i++) {
            S_left.x[i] *= scale;
            S_right.x[i] *= scale;
        }

        // store S → smem [WM, BN] row-major, stride BN=32
        float* my_S = &S_s[warp_id * WM * BN];
        wmma::store_matrix_sync(&my_S[0],  S_left,  BN, wmma::mem_row_major);  // cols 0-15
        wmma::store_matrix_sync(&my_S[16], S_right, BN, wmma::mem_row_major);  // cols 16-31
        __syncwarp();

        /* -- mask + online softmax (2 threads per row, 16 cols each) -- */
        float* S_row = &my_S[srow * BN];
        __half* P_row = &P_s[warp_id * WM * BN + srow * BN];

        // BN=32 aligned to mask word boundary → clean read
        uint32_t mword = 0u;
        if (grow < N && tile < n_words)
            mword = mask_packed[((b * H + h) * N + grow) * n_words + tile];

        int c_off = shalf * 16;
        float vals[16];
        float lmax = -FLT_MAX;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int j = c_off + i;
            bool ok = (grow < N) && (j < tlen) && ((mword >> j) & 1u);
            vals[i] = ok ? S_row[j] : -FLT_MAX;
            lmax = fmaxf(lmax, vals[i]);
        }
        float pmax = __shfl_xor_sync(0xffffffff, lmax, 1);
        float tile_max = fmaxf(lmax, pmax);

        float new_max = fmaxf(running_max, tile_max);
        float alpha = (running_max > -FLT_MAX) ? expf(running_max - new_max) : 0.0f;

        float lsum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float p = (vals[i] > -FLT_MAX) ? expf(vals[i] - new_max) : 0.0f;
            lsum += p;
            P_row[c_off + i] = __float2half(p);
        }
        float psum = __shfl_xor_sync(0xffffffff, lsum, 1);
        float tile_sum = lsum + psum;

        running_sum = running_sum * alpha + tile_sum;
        running_max = new_max;

        /* -- rescale O_acc fragments by alpha -- */
        // Share alpha via smem (reuse S_s area for this warp)
        float* alpha_buf = my_S;  // only need 16 floats, S_s has 512 per warp
        if (shalf == 0) alpha_buf[srow] = alpha;
        __syncwarp();

        // Fragment layout (m16n16k16 acc): thread group g=lane/4 handles rows 2g, 2g+1
        // Use runtime-discovered mapping instead of hardcoded assumption
        #pragma unroll
        for (int d = 0; d < D_CHUNKS; d++) {
            #pragma unroll
            for (int i = 0; i < 8; i++)
                O_acc[d].x[i] *= alpha_buf[elem_rows[i]];
        }
        __syncwarp();

        /* -- WMMA: O_acc += P × V  (accumulate directly) -- */
        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS; dc++) {
            wmma::fragment<wmma::matrix_a, WM, WMMA_N, WK, __half, wmma::row_major> Pl, Pr;
            wmma::fragment<wmma::matrix_b, WM, WMMA_N, WK, __half, wmma::row_major> Vt, Vb;

            wmma::load_matrix_sync(Pl, &P_s[warp_id * WM * BN],      BN);
            wmma::load_matrix_sync(Vt, &V_s[dc * WK],                 D);
            wmma::load_matrix_sync(Pr, &P_s[warp_id * WM * BN + 16], BN);
            wmma::load_matrix_sync(Vb, &V_s[16 * D + dc * WK],        D);

            wmma::mma_sync(O_acc[dc], Pl, Vt, O_acc[dc]);
            wmma::mma_sync(O_acc[dc], Pr, Vb, O_acc[dc]);
        }

        __syncthreads();
    }

    /* -------- write output (sequential per warp using S_s as temp) -------- */
    for (int w = 0; w < NWARPS; w++) {
        if (warp_id == w) {
            float* O_buf = S_s;  // reuse S_s (2048 floats, need 1024)
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
    int smem_bytes = BM * D * 2                 // Q_s
                   + BN * D * 2                 // K_s
                   + BN * D * 2                 // V_s
                   + NWARPS * WM * BN * 4       // S_s
                   + NWARPS * WM * BN * 2;      // P_s

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
