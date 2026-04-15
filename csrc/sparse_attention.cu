/**
 * Sparse Mask Attention - CUDA Implementation
 *
 * Tiled kernel: BLOCK_M query rows per block, shared memory K/V tiles.
 * Each warp handles one query row. Online softmax with FP32 accumulation.
 */

#include "sparse_attention.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

// ============================================================
// Pack Mask Kernel
// ============================================================

__global__ void pack_mask_kernel(
    const bool* __restrict__ mask,
    uint32_t* __restrict__ mask_packed,
    int total_rows,
    int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;

    int n_words = (N + 31) / 32;
    for (int w = 0; w < n_words; w++) {
        uint32_t packed = 0u;
        int col_start = w * 32;
        #pragma unroll
        for (int bit = 0; bit < 32; bit++) {
            int col = col_start + bit;
            if (col < N && mask[row * N + col]) {
                packed |= (1u << bit);
            }
        }
        mask_packed[row * n_words + w] = packed;
    }
}

void pack_mask_bits(
    const bool* mask,
    uint32_t* mask_packed,
    int B, int H, int N,
    cudaStream_t stream
) {
    int total_rows = B * H * N;
    int threads = 256;
    int blocks = CEIL_DIV(total_rows, threads);
    pack_mask_kernel<<<blocks, threads, 0, stream>>>(mask, mask_packed, total_rows, N);
}

// ============================================================
// Forward Kernel: Tiled with shared memory K/V caching
//
// BLOCK_M query rows per block (one warp per row).
// K/V tiles of BLOCK_N=32 columns loaded cooperatively into
// shared memory — all BLOCK_M warps share the same K/V data.
// Online softmax in FP32 for numerical stability.
// ============================================================

#define BLOCK_M 16
#define BLOCK_N 32

template <typename T>
__global__ void sparse_attn_fwd_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    const uint32_t* __restrict__ mask_packed,
    T* __restrict__ Out,
    float* __restrict__ lse,
    int B, int H, int N, int D,
    float scale, bool is_training
) {
    const int warp_id = threadIdx.x >> 5;     // 0 .. BLOCK_M-1
    const int lane_id = threadIdx.x & 31;     // 0 .. 31

    const int row = blockIdx.x * BLOCK_M + warp_id;
    const int bh  = blockIdx.y;
    const int b   = bh / H;
    const int h   = bh % H;

    const bool valid_row = (row < N && b < B);

    // Contiguous [B, H, N, D] layout
    const int bhN_D = ((b * H + h) * N) * D;

    // Mask layout: [B, H, N, n_words]  where n_words = ceil(N/32)
    const int n_words = (N + 31) / 32;
    const int mask_row_base = valid_row ? ((b * H + h) * N + row) * n_words : 0;

    // Load Q[row] into registers — each thread handles dims lane, lane+32, ...
    const int MAX_D_PER_THREAD = 8;
    float q_reg[MAX_D_PER_THREAD];
    float acc[MAX_D_PER_THREAD];
    int ndims = 0;
    for (int d = lane_id; d < D; d += 32) {
        q_reg[ndims] = valid_row ? to_float(Q[bhN_D + row * D + d]) : 0.0f;
        acc[ndims] = 0.0f;
        ndims++;
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    // Shared memory: K tile [BLOCK_N, D] followed by V tile [BLOCK_N, D]
    extern __shared__ char smem[];
    T* K_smem = (T*)smem;
    T* V_smem = (T*)(smem + BLOCK_N * D * sizeof(T));

    // Tile over KV columns — each tile = one mask word (32 columns)
    for (int tile = 0; tile < n_words; tile++) {
        const int col_start = tile * 32;
        const int tile_len  = min(32, N - col_start);

        // Cooperative load: all threads load K and V tile into smem
        const int total_elems = BLOCK_N * D;
        for (int i = threadIdx.x; i < total_elems; i += blockDim.x) {
            int kn = i / D;   // column index within tile  (0..BLOCK_N-1)
            int kd = i % D;   // dimension index           (0..D-1)
            int col = col_start + kn;
            if (col < N) {
                K_smem[i] = K[bhN_D + col * D + kd];
                V_smem[i] = V[bhN_D + col * D + kd];
            }
        }
        __syncthreads();

        // Per-warp: process masked columns in this tile
        if (valid_row) {
            uint32_t mword = mask_packed[mask_row_base + tile];

            if (mword != 0u) {
                for (int j = 0; j < tile_len; j++) {
                    if (!((mword >> j) & 1u)) continue;

                    // dot(Q_reg, K_smem[j])
                    float dot = 0.0f;
                    for (int di = 0, d = lane_id; d < D; d += 32, di++) {
                        dot += q_reg[di] * to_float(K_smem[j * D + d]);
                    }
                    float score = warp_reduce_sum(dot) * scale;

                    // Online softmax update
                    float new_max = fmaxf(row_max, score);
                    float alpha   = expf(row_max - new_max);
                    float p       = expf(score - new_max);

                    row_sum = row_sum * alpha + p;
                    for (int di = 0, d = lane_id; d < D; d += 32, di++) {
                        acc[di] = acc[di] * alpha + p * to_float(V_smem[j * D + d]);
                    }
                    row_max = new_max;
                }
            }
        }
        __syncthreads();
    }

    // Write output
    if (valid_row) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (int di = 0, d = lane_id; d < D; d += 32, di++) {
            Out[bhN_D + row * D + d] = from_float<T>(acc[di] * inv_sum);
        }
        if (is_training && lane_id == 0) {
            int lse_idx = (b * H + h) * N + row;
            lse[lse_idx] = (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -FLT_MAX;
        }
    }
}

// ============================================================
// Host Launchers
// ============================================================

void sparse_attention_fwd_bf16(
    const SparseAttentionParams& params,
    cudaStream_t stream
) {
    dim3 grid(CEIL_DIV(params.seq_len, BLOCK_M), params.batch_size * params.num_heads);
    dim3 block(BLOCK_M * 32);
    int smem_bytes = 2 * BLOCK_N * params.head_dim * sizeof(__nv_bfloat16);

    sparse_attn_fwd_kernel<__nv_bfloat16><<<grid, block, smem_bytes, stream>>>(
        (const __nv_bfloat16*)params.q,
        (const __nv_bfloat16*)params.k,
        (const __nv_bfloat16*)params.v,
        params.mask_packed,
        (__nv_bfloat16*)params.out,
        params.lse,
        params.batch_size, params.num_heads, params.seq_len, params.head_dim,
        params.scale, params.is_training
    );
}

void sparse_attention_fwd_fp16(
    const SparseAttentionParams& params,
    cudaStream_t stream
) {
    dim3 grid(CEIL_DIV(params.seq_len, BLOCK_M), params.batch_size * params.num_heads);
    dim3 block(BLOCK_M * 32);
    int smem_bytes = 2 * BLOCK_N * params.head_dim * sizeof(__half);

    sparse_attn_fwd_kernel<__half><<<grid, block, smem_bytes, stream>>>(
        (const __half*)params.q,
        (const __half*)params.k,
        (const __half*)params.v,
        params.mask_packed,
        (__half*)params.out,
        params.lse,
        params.batch_size, params.num_heads, params.seq_len, params.head_dim,
        params.scale, params.is_training
    );
}

void sparse_attention_bwd_bf16(
    const SparseAttentionParams& params,
    const void* dout,
    void* dq, void* dk, void* dv,
    cudaStream_t stream
) {
    // Stub: backward not bound in pybind11; Python fallback is used instead.
}
