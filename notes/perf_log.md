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

## Summary after 13 Rounds

| Round | Optimization | Latency (ms) | Speedup | Status |
|-------|-------------|-------------|---------|--------|
| 0 | Baseline (scalar 1-warp) | 444.4 | 1.00x | - |
| 1 | Tiled smem K/V (BM=4) | 312.6 | 1.42x | ACCEPT |
| 2 | BLOCK_M=16 (16 warps) | 236.6 | 1.88x | ACCEPT |
| 3 | BLOCK_N=128 | 252.1 | - | REJECT |
| 4 | Column-split 16 warps | 236.9 | - | REJECT |
| 5 | WMMA TC BN=16 (smem O) | 231.2 | 1.92x | ACCEPT |
| 6 | WMMA BN=32 (halved tiles) | 140.3 | 3.17x | ACCEPT |
| 7 | Block-level mask skip | 141.5 | - | REJECT |
| 8 | Fragment O_acc + runtime row mapping | 53.1 | 8.36x | ACCEPT |
| 9 | NWARPS=8 (BM=128) | **49.9** | **8.91x** | **ACCEPT** |
| 10 | Q from global/L2 | 52.1 | - | REJECT |
| 11 | Warp-level mask skip | 50.4 | - | REJECT |
| 12 | BN=64 | 60.4 | - | REJECT |
| 13 | Two-pass softmax | 58.8 | - | REJECT |

**Current best**: 47.4 ms (17.40 TFLOPS, 15.1% MFU) — R16 vectorized loads
**Key breakthroughs**: WMMA tensor cores (R5-R6), fragment O accumulation with runtime
mapping (R8), larger BLOCK_M for K/V sharing (R9), vectorized float4 loads (R16).

| 14-17 | Various (NWARPS, in-reg softmax, etc.) | - | - | ALL REJECT |
| 16 | Vectorized K/V float4 loads | **47.4** | **9.38x** | **ACCEPT** |

Remaining gap vs Triton (20.4ms): ~2.3x. Bottleneck: per-tile softmax overhead, 67 regs
limiting to 3 blocks/SM (37.5% occupancy), and smem round-trips for S and P.

