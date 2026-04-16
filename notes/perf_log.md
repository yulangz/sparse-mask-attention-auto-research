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

| 16 | Vectorized K/V float4 loads | **47.4** | **9.38x** | **ACCEPT** |
| 18 | cp.async double-buffered K/V | **44.9** | **9.90x** | **ACCEPT** |
| 20 | Alpha via shuffle (no smem) | **44.6** | **9.96x** | **ACCEPT** |
| 14-15,17,19,21-23 | Various attempts | - | - | ALL REJECT |

**Final best**: 44.6 ms (18.48 TFLOPS, 16.1% MFU) — R20
From 444.4ms to 44.6ms = **9.96x speedup** over 23 optimization rounds.

## Round 24: Smem Padding for S_s/P_s (Bank Conflict Reduction)
- **Change**: Added stride padding to S_s (float, stride 32→36) and P_s (half, stride 32→48)
  to break 128-byte bank alignment. S_STRIDE*4=144B (144%128=16, offset by 4 banks),
  P_STRIDE*2=96B. Eliminates worst-case 16-way bank conflicts in softmax smem accesses.
- **Latency**: 41.506 ms (was 44.6 ms)
- **TFLOPS**: 19.87 (was 18.48)
- **Speedup**: 1.08x over R20, **10.71x** vs baseline
- **Status**: ACCEPTED

Analysis: Clean 8% gain from reducing bank conflicts in the S→softmax→P shared memory
round-trip, which is the hottest path (executed 512× per block). Smem increased from 57KB
to 63KB but still fits 3 blocks/SM (registers remain the limiter at 67/thread).

## Round 25: In-Register Softmax + Direct Fragment Output
- **Change**: Eliminated S_s shared memory entirely. Softmax now operates directly on WMMA
  accumulator fragment values in registers. Each thread handles 2 rows (my_r0, my_r1) via
  octet-level reductions (__shfl_xor with masks 1,2). Also replaced serial 8-warp output
  loop with direct fragment-to-global writes using elem_rows/elem_cols mapping.
  Computed elem_cols via combined row*16+col identity matrix trick.
- **Latency**: 39.576 ms (was 41.5 ms)
- **TFLOPS**: 20.84 (was 19.87)
- **Speedup**: 1.05x over R24, **11.23x** vs baseline
- **Status**: ACCEPTED

Analysis: 5% gain from eliminating S_s store/load (17KB per tile × 512 tiles) and replacing
the serial 8-warp output with parallel direct writes. Smem dropped from 63KB to 42KB.
Register pressure increased (elem_cols[8] + 2 running states) but stays within limits.

## Round 26: BN_TILE=64 (2×32 Sub-Tiles per Prefetch)
- **Change**: Doubled the outer tile width to 64 columns for K/V prefetch while keeping
  BN=32 for sub-tile WMMA processing. Each 64-col tile processes 2 sub-tiles of 32 cols.
  K/V buffers doubled (8KB→16KB each). Halves the number of __syncthreads, cp.async
  prefetches, and pipeline commits.
- **Latency**: 37.710 ms (was 39.6 ms)
- **TFLOPS**: 21.87 (was 20.84)
- **Speedup**: 1.05x over R25, **11.78x** vs baseline
- **Status**: ACCEPTED

Analysis: 5% gain from reducing synchronization and prefetch overhead (256 syncs instead of
512). Smem increased from 42KB to 58KB (larger K/V buffers) but still 3 blocks/SM.

## Round 27: BN_TILE=128 (4×32 Sub-Tiles per Prefetch)
- **Change**: Further doubled outer tile to 128 columns (4 sub-tiles of 32). K/V buffers
  now 32KB each. Smem 92KB → 2 blocks/SM (was 3). Reduces syncs from 256 to 128.
- **Latency**: 36.755 ms total (23.304 ms kernel-only)
- **TFLOPS**: 22.44 total / 35.39 kernel-only
- **Speedup**: 1.03x over R26
- **Status**: ACCEPTED

**Key discovery**: pack_mask takes **13.5ms** — 37% of total time! All prior measurements
included this overhead. Kernel-only: 23.3ms vs Triton 20.4ms = only 1.14× gap.

## Round 28: Vectorized pack_mask (__ballot_sync)
- **Change**: Replaced scalar per-bit pack_mask with warp-level __ballot_sync. One warp
  packs 32 bools → 1 uint32 in a single intrinsic. Coalesced 32-byte reads.
- **pack_mask latency**: 1.983 ms (was 13.452 ms) — **6.8× speedup**
- **Total latency**: 25.290 ms (was 36.755 ms)
- **TFLOPS**: 32.61 total (was 22.44)
- **Speedup**: 1.45x over R27 total, **17.6x** vs baseline
- **Status**: ACCEPTED

## Round 29: __launch_bounds__ + Remove Sub-Tile Unroll
- **Change**: Added `__launch_bounds__(256, 2)` to force 2 blocks/SM. Removed `#pragma
  unroll` from the sub-tile loop (was causing register explosion: 193→133 regs).
  With launch_bounds, compiler targets 128 regs (20B spill, negligible).
- **Kernel latency**: 20.355 ms (was 23.3 ms) — **12.6% improvement**
- **Total latency**: 22.339 ms (was 25.3 ms)
- **TFLOPS**: 40.51 kernel / 36.91 total
- **Speedup**: **19.88×** vs baseline, kernel matches Triton (20.4ms)!
- **Status**: ACCEPTED

**Key insight**: Register pressure was the hidden bottleneck. With 193 regs, only 1 block/SM
(8 warps). At 128 regs, 2 blocks/SM (16 warps) — 2× occupancy = 12% faster.

**Updated comparison at B=1, N=16384, H=12, D=64 (FP16):**
```
flash-attn:      2.65ms  (311 TFLOPS)  dense, no mask
cuDNN SDPA:     11.02ms  ( 75 TFLOPS)  dense
Triton:         20.40ms  ( 40 TFLOPS)  sparse     ← 1.00× (kernel parity!)
Ours (R29):     22.34ms  ( 37 TFLOPS)  sparse     ← HERE (total w/ pack_mask)
  kernel-only:  20.36ms  ( 41 TFLOPS)                  (matches Triton!)
PyTorch Ref:    32.26ms  ( 26 TFLOPS)  sparse     ← 1.44× slower
FlashInfer:     71.20ms  ( 12 TFLOPS)  sparse     ← 3.19× slower
Baseline:      444.39ms  (  2 TFLOPS)  scalar     ← 19.9× slower
```

## Rounds 30-32: Rejected Attempts
- **R30**: BN_TILE=64 → 21.0ms kernel (worse, reverted)
- **R30b**: BN_TILE=256 → 22.8ms kernel (worse, smem pressure)
- **R31**: launch_bounds(256,3) → 80 regs, 340B spills → 23.2ms (worse)
- **R32**: NWARPS=16 (BM=256) → 20.8ms kernel (slightly worse, less scheduling flexibility)
BN_TILE=128, NWARPS=8, 2 blocks/SM remains optimal.

## Summary: Rounds 24-32

| Round | Optimization | Kernel (ms) | Total (ms) | Status |
|-------|-------------|-------------|------------|--------|
| R24 | S_s/P_s smem padding | 28.0 | 41.5 | ACCEPT |
| R25 | In-register softmax + direct output | 26.1 | 39.6 | ACCEPT |
| R26 | BN_TILE=64 (2x sub-tiles) | 24.2 | 37.7 | ACCEPT |
| R27 | BN_TILE=128 (4x sub-tiles) | 23.3 | 36.8 | ACCEPT |
| R28 | Vectorized pack_mask (__ballot_sync) | 23.3 | 25.3 | ACCEPT |
| R29 | launch_bounds + occupancy fix | **20.4** | **22.3** | ACCEPT |
| R30-32 | BN_TILE/NWARPS/occupancy sweeps | — | — | REJECT |

**Current best**: 22.3ms total / 20.4ms kernel — **19.9× vs baseline, parity with Triton**
From 444.4ms → 22.3ms over 32 optimization rounds.

**Key breakthroughs** (in order of impact):
1. **R8**: Fragment O accumulation w/ runtime row mapping (140→53ms, 2.6x single-round gain)
2. **R6**: WMMA BN=32 (231→140ms, 1.65x) 
3. **R2**: BLOCK_M=16 for K/V sharing (312→237ms, 1.32x)
4. **R18**: cp.async double-buffered K/V prefetch (47→45ms)
5. **R20**: Alpha via shuffle (no smem round-trip)

**Comparison at B=1, N=16384, H=12, D=64 (FP16):**
```
flash-attn:     2.65ms  (311 TFLOPS)  dense, no mask
cuDNN SDPA:    11.02ms  ( 75 TFLOPS)  dense
Triton:        20.40ms  ( 40 TFLOPS)  sparse     ← 2.2x faster
PyTorch Ref:   32.26ms  ( 26 TFLOPS)  sparse     ← 1.4x faster
Ours (R20):    44.63ms  ( 18 TFLOPS)  sparse     ← HERE
FlashInfer:    71.20ms  ( 12 TFLOPS)  sparse     ← 1.6x slower
Baseline:     444.39ms  (  2 TFLOPS)  scalar     ← 10x slower
```

