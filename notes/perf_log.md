# Sparse Mask Attention — Performance Optimization Log

## Environment
- **GPU**: NVIDIA L20Y (132 SMs, compute 9.0, Ada/Hopper architecture)
- **Peak FP16 Tensor Core**: ~114.9 TFLOPS (estimated)
- **Benchmark config**: B=1, H=12, N=16384, D=64, FP16, mask density ~61.7%
- **Sparsity pattern**: upper-triangular + 2048 random previous tokens per query

---

## Final Result (Round 35)

| Metric | Value |
|--------|-------|
| **Kernel latency** | **18.84 ms** |
| **Total latency (kernel + pack_mask)** | **20.82 ms** |
| **pack_mask latency** | 1.98 ms |
| **TFLOPS (total)** | 39.61 |
| **TFLOPS (kernel only)** | 43.78 |
| **MFU** | 34.5% |
| **Speedup vs baseline** | **21.4×** |
| **Registers / thread** | 118 (zero spills) |
| **Blocks / SM** | 2 |
| **Shared memory / block** | ~81 KB |

---

## Comparison with Other Implementations

All measured at B=1, H=12, N=16384, D=64, FP16 on NVIDIA L20Y.

| Implementation | Latency (ms) | TFLOPS | Mask Support | vs Ours (total) |
|:---------------|:-------------|:-------|:-------------|:----------------|
| flash-attn 2 | 2.66 | 310.0 | dense only | 0.13× (dense) |
| cuDNN SDPA | 11.02 | 74.8 | dense only | 0.53× (dense) |
| **Ours (kernel)** | **18.84** | **43.8** | **sparse** | — |
| **Ours (total)** | **20.82** | **39.6** | **sparse** | **baseline** |
| Triton | 20.41 | 40.4 | sparse | 0.98× |
| PyTorch Ref | 32.25 | 25.6 | sparse | 1.55× slower |
| FlashInfer | 71.32 | 11.6 | sparse | 3.43× slower |
| Baseline (R0) | 444.39 | 1.9 | sparse | 21.4× slower |

**Key takeaway**: Our CUDA kernel (18.84 ms) is **7.6% faster than Triton** (20.41 ms) on
the attention computation itself. The total (20.82 ms) includes a 1.98 ms pack_mask
preprocessing step that Triton doesn't need (Triton reads bool masks directly).

---

## Full Optimization History (35 Rounds)

### Phase 1: Foundations (R0–R9) — 444 ms → 50 ms

| Round | Optimization | Latency (ms) | Speedup | Status |
|:------|:-------------|:-------------|:--------|:-------|
| 0 | Baseline: scalar, 1 warp/row | 444.4 | 1.00× | — |
| 1 | Tiled smem K/V (BM=4, BN=32, 4 warps) | 312.6 | 1.42× | ACCEPT |
| 2 | BLOCK_M=16 (16 warps, 512 threads) | 236.6 | 1.88× | ACCEPT |
| 3 | BLOCK_N=128 larger tiles | 252.1 | — | REJECT |
| 4 | Column-split 16 warps/row | 236.9 | — | REJECT |
| 5 | WMMA tensor cores (BN=16, smem O) | 231.2 | 1.92× | ACCEPT |
| 6 | WMMA BN=32 (halved tile count) | 140.3 | 3.17× | ACCEPT |
| 7 | Block-level mask skip | 141.5 | — | REJECT |
| 8 | **Fragment O_acc + runtime row mapping** | **53.1** | **8.36×** | **ACCEPT** |
| 9 | NWARPS=8 (BM=128, doubled K/V sharing) | 49.9 | 8.91× | ACCEPT |

### Phase 2: Memory Optimization (R10–R23) — 50 ms → 44.6 ms

| Round | Optimization | Latency (ms) | Speedup | Status |
|:------|:-------------|:-------------|:--------|:-------|
| 10–15 | Various (Q global, mask skip, BN=64, …) | — | — | ALL REJECT |
| 16 | Vectorized K/V loading (float4) | 47.4 | 9.38× | ACCEPT |
| 17 | NWARPS=4 + vectorized | 49.2 | — | REJECT |
| 18 | cp.async double-buffered K/V prefetch | 44.9 | 9.90× | ACCEPT |
| 19 | S/P smem merge | 46.1 | — | REJECT |
| 20 | Alpha via shuffle (no smem round-trip) | 44.6 | 9.96× | ACCEPT |
| 21–23 | Direct output / reg reduction / maxreg | — | — | ALL REJECT |

### Phase 3: Smem Elimination (R24–R29) — 44.6 ms → 20.8 ms (total)

| Round | Optimization | Total (ms) | Kernel (ms) | Status |
|:------|:-------------|:-----------|:------------|:-------|
| 24 | S_s/P_s bank-conflict padding | 41.5 | 28.0 | ACCEPT |
| 25 | **In-register softmax + direct output** | 39.6 | 26.1 | **ACCEPT** |
| 26 | BN_TILE=64 (2×32 sub-tiles) | 37.7 | 24.2 | ACCEPT |
| 27 | BN_TILE=128 (4×32 sub-tiles) | 36.8 | 23.3 | ACCEPT |
| 28 | **Vectorized pack_mask (__ballot_sync)** | **25.3** | 23.3 | **ACCEPT** |
| 29 | **launch_bounds occupancy fix (128 regs)** | **22.3** | **20.4** | **ACCEPT** |

### Phase 4: Register-Only P@V (R30–R35) — 20.8 ms (total) / 18.8 ms (kernel)

| Round | Optimization | Total (ms) | Kernel (ms) | Status |
|:------|:-------------|:-----------|:------------|:-------|
| 30–32 | BN_TILE sweep / 3-block / NWARPS=16 | — | — | ALL REJECT |
| 33 | **Eliminate P_s (acc↔mat_a layout match)** | 21.2 | **19.2** | **ACCEPT** |
| 34 | BN_TILE=192 (max 2-block smem budget) | 21.0 | 19.1 | ACCEPT |
| 35 | **Hardcode WMMA layout (0 spills, 118 regs)** | **20.8** | **18.8** | **ACCEPT** |

---

## Key Breakthroughs (ranked by single-round impact)

1. **R8 — Fragment O accumulation** (140→53 ms, **2.64×**):
   Eliminated shared memory round-trip for output by accumulating directly in WMMA
   accumulator fragments. Used runtime identity-matrix probe to discover fragment layout.

2. **R6 — WMMA BN=32** (231→140 ms, **1.65×**):
   Halved the number of KV tiles by doubling the WMMA tile width. Split P@V into two
   16×16 sub-tiles to keep within WMMA dimensions.

3. **R28 — Vectorized pack_mask** (36.8→25.3 ms total, **1.45×**):
   Replaced scalar bit-by-bit mask packing with warp-level `__ballot_sync()`. One intrinsic
   packs 32 bools → 1 uint32, with coalesced global memory reads.

4. **R29 — Occupancy fix via launch_bounds** (25.3→22.3 ms, **1.13×**):
   Discovered register explosion (193 regs → 1 block/SM) from sub-tile loop unrolling.
   `__launch_bounds__(256, 2)` forced 128 regs → 2 blocks/SM = 2× occupancy.

5. **R25 — In-register softmax** (41.5→39.6 ms, **1.05×**):
   Computed softmax directly on WMMA accumulator values using octet-level warp shuffles
   (XOR masks 1, 2). Eliminated S_s shared memory buffer entirely.

6. **R33 — Eliminate P_s via layout identity** (22.3→21.2 ms, **1.05×**):
   Probed WMMA matrix_a fragment layout and discovered it is **identical** to the accumulator
   layout on sm_90. P values from softmax directly populate matrix_a fragments in registers —
   no shared memory store/load needed. Zero-cost P@V operand construction.

---

## Architecture of Final Kernel (R35)

```
Grid:  [ceil(N/BM), B*H]  = [128, 12] = 1536 blocks
Block: NWARPS × 32        = 256 threads
BM=128 (Q rows per block), BN=32 (sub-tile KV cols), BN_TILE=192 (prefetch cols)

Shared memory (81 KB):
  Q_s   [128, 64]  __half  — 16 KB   (loaded once, reused every sub-tile)
  K_buf [2×192, 64] __half — 48 KB   (double-buffered, cp.async prefetch)
  V_buf [2×192, 64] __half — 48 KB   (double-buffered, cp.async prefetch)
  (no S_s, no P_s — everything in registers)

Per sub-tile (32 KV columns):
  1. WMMA QK^T:   8× mma_sync (Q from smem, K from smem → S in registers)
  2. Softmax:     in-register mask + max + exp + sum (octet shuffles)
  3. P fragment:  direct register construction (acc layout == mat_a layout)
  4. WMMA P@V:    8× mma_sync (P from registers, V from smem → O_acc in registers)

Outer tile loop: 192-col tiles, 6× sub-tiles each, double-buffered K/V prefetch.
Online softmax across tiles: per-thread running state for 2 rows (my_r0, my_r1).
Output: direct register-to-global writes using hardcoded WMMA layout.
```

---

## Sequence Length Scaling (Final Kernel)

| N | B | Latency (ms) | TFLOPS | MFU% | Density% |
|---:|---:|---:|---:|---:|---:|
| 64 | 16 | 0.025 | 8.13 | 7.1 | 100.0 |
| 128 | 16 | 0.041 | 19.66 | 17.1 | 100.0 |
| 256 | 16 | 0.123 | 26.26 | 22.9 | 100.0 |
| 512 | 16 | 0.363 | 35.47 | 30.9 | 100.0 |
| 1024 | 16 | 1.357 | 37.99 | 33.1 | 100.0 |
| 2048 | 4 | 1.335 | 38.62 | 33.6 | 100.0 |
| 4096 | 4 | 5.248 | 39.28 | 34.2 | 87.5 |
| 8192 | 2 | 10.432 | 39.53 | 34.4 | 71.9 |
| 16384 | 1 | 20.817 | 39.61 | 34.5 | 61.7 |
