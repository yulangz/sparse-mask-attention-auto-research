# Performance Optimization Log

## Baseline (Round 0)
- **Kernel**: Scalar, 1 warp (32 threads) per query row
- **GPU**: NVIDIA L20Y (132 SMs, compute 9.0)
- **Config**: B=1, H=12, N=16384, D=64, FP16, mask density ~61.7%
- **Latency**: 444.392 ms
- **TFLOPS**: 1.86 (1.6% MFU)
- **Reference points**: PyTorch Ref 32.3ms, Triton 20.4ms, cuDNN 11.0ms, flash-attn 2.7ms

Analysis: The scalar kernel is ~22x slower than our Triton baseline. Key bottlenecks:
1. Only 32 threads per row — extremely low occupancy
2. No tensor core usage
3. No shared memory — all K/V loads go to global memory
4. Column-by-column iteration — no tiling or reuse
5. No vectorized loads

## Round 1: Tiled smem K/V caching (BLOCK_M=4, BLOCK_N=32)
- **Change**: Restructured kernel to use 4 warps per block (one per query row), with
  cooperative K/V tile loading into shared memory. Each tile = 32 KV columns (one mask word).
  4 query rows share the same K/V data → 4x reduction in global memory reads.
- **Latency**: 312.561 ms (was 444.392 ms)
- **Speedup**: 1.42x
- **Status**: ACCEPTED

Analysis: Moderate improvement from better memory reuse and doubled warp occupancy (32→64
warps/SM). Still memory-latency bound — the per-column inner loop with warp_reduce_sum +
online softmax is heavy on instruction overhead relative to the small D=64 compute.

