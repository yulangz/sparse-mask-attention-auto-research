/**
 * Sparse Mask Attention - CUDA Implementation
 *
 * Simple scalar kernel: one warp (32 threads) per query row,
 * online softmax with FP32 accumulation.
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
// Forward Kernel: one warp (32 threads) per query row
//
// Each thread handles output dimensions tid, tid+32, tid+64, ...
// Uses warp_reduce_sum for dot product (result broadcast to all lanes).
// Online softmax in FP32 for numerical stability.
// ============================================================

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
    int row = blockIdx.x;
    int bh  = blockIdx.y;
    int b   = bh / H;
    int h   = bh % H;
    int tid = threadIdx.x;  // 0..31

    if (row >= N || b >= B) return;

    // Contiguous [B, H, N, D] layout
    int bhN_D = ((b * H + h) * N) * D;

    // Mask layout: [B, H, N, n_words] where n_words = ceil(N/32)
    int n_words = (N + 31) / 32;
    int mask_row_base = ((b * H + h) * N + row) * n_words;

    // Load Q[row] for this thread's dimensions
    // Each thread handles dims: tid, tid+32, tid+64, ...
    // For D up to 256, at most 8 dims per thread
    const int MAX_D_PER_THREAD = 8;
    float q_reg[MAX_D_PER_THREAD];
    float acc[MAX_D_PER_THREAD];
    int ndims = 0;
    for (int d = tid; d < D; d += 32) {
        q_reg[ndims] = to_float(Q[bhN_D + row * D + d]);
        acc[ndims] = 0.0f;
        ndims++;
    }

    // Online softmax state
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    // Iterate over all K/V columns
    for (int col = 0; col < N; col++) {
        // Check bit-packed mask
        uint32_t mword = mask_packed[mask_row_base + (col >> 5)];
        if (!((mword >> (col & 31)) & 1u)) continue;

        // Compute dot(Q[row], K[col])
        float dot = 0.0f;
        for (int di = 0, d = tid; d < D; d += 32, di++) {
            dot += q_reg[di] * to_float(K[bhN_D + col * D + d]);
        }
        float score = warp_reduce_sum(dot) * scale;

        // Online softmax update
        float new_max = fmaxf(row_max, score);
        float alpha = expf(row_max - new_max);   // rescale old accumulators
        float p     = expf(score - new_max);      // new weight

        row_sum = row_sum * alpha + p;
        for (int di = 0, d = tid; d < D; d += 32, di++) {
            acc[di] = acc[di] * alpha + p * to_float(V[bhN_D + col * D + d]);
        }
        row_max = new_max;
    }

    // Normalize and write output
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int di = 0, d = tid; d < D; d += 32, di++) {
        Out[bhN_D + row * D + d] = from_float<T>(acc[di] * inv_sum);
    }

    // Optionally save LSE for backward pass
    if (is_training && tid == 0) {
        int lse_idx = (b * H + h) * N + row;
        lse[lse_idx] = (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -FLT_MAX;
    }
}

// ============================================================
// Host Launchers
// ============================================================

void sparse_attention_fwd_bf16(
    const SparseAttentionParams& params,
    cudaStream_t stream
) {
    dim3 grid(params.seq_len, params.batch_size * params.num_heads);
    dim3 block(32);  // one warp per row

    sparse_attn_fwd_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
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
    dim3 grid(params.seq_len, params.batch_size * params.num_heads);
    dim3 block(32);  // one warp per row

    sparse_attn_fwd_kernel<__half><<<grid, block, 0, stream>>>(
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
