# Sparse Mask Attention — CUDA Kernel Optimization Report

## Overview

This project implements and optimizes a custom CUDA kernel for **sparse masked attention**, achieving **21.4× speedup** over the scalar baseline and **matching Triton performance** on NVIDIA L20Y (sm_90).

The attention kernel supports arbitrary boolean masks where each query attends to a random subset of up to 2048 preceding KV positions (plus all subsequent positions), yielding ~62% mask density at N=16384.

| Metric | Baseline (R0) | Final (R35) | Improvement |
|:-------|:-------------|:------------|:------------|
| Total latency | 444.4 ms | **20.8 ms** | **21.4×** |
| Kernel latency | 444.4 ms | **18.8 ms** | **23.6×** |
| TFLOPS | 1.86 | **43.78** | **23.5×** |
| MFU (vs FP16 TC peak) | 1.6% | **34.5%** | — |

---

## Hardware & Software

| Item | Spec |
|:-----|:-----|
| GPU | NVIDIA L20Y (132 SMs, compute 9.0) |
| FP16 Tensor Core peak | ~114.9 TFLOPS (estimated) |
| CUDA | 12.x |
| PyTorch | 2.x |
| Benchmark config | B=1, H=12, N=16384, D=64, FP16, mask density ~61.7% |

---

## Backend Comparison

All measured at B=1, H=12, N=16384, D=64, FP16 on NVIDIA L20Y:

| Implementation | Latency (ms) | TFLOPS | Mask Type | vs Ours |
|:---------------|:-------------|:-------|:----------|:--------|
| flash-attn 2 | 2.66 | 310 | dense only | 0.13× (no custom mask) |
| cuDNN SDPA | 11.02 | 75 | dense only | 0.53× (no custom mask) |
| **Ours kernel** | **18.84** | **43.8** | **sparse** | **fastest sparse kernel** |
| **Ours total** | **20.82** | **39.6** | **sparse** | **includes 2ms pack_mask** |
| Triton | 20.41 | 40.4 | sparse | 0.98× (kernel: 1.08× slower) |
| PyTorch Ref | 32.25 | 25.6 | sparse | 1.55× slower |
| FlashInfer | 71.32 | 11.6 | sparse | 3.43× slower |
| Scalar baseline | 444.39 | 1.9 | sparse | 21.4× slower |

Among all sparse-mask implementations, our CUDA kernel is the **fastest**, surpassing Triton by 7.6% on the attention kernel itself. The 2ms gap in total latency comes from `pack_mask` (bool→uint32 bit-packing), which Triton doesn't need.

---

## Optimization Journey (35 Rounds)

### Phase 1: Foundations (R0–R9) — 444 ms → 50 ms
- Tiled shared memory K/V caching with cooperative warp loading
- Increased parallelism from 1 to 16 warps per block
- **Key breakthrough R8**: Fragment O accumulation with runtime row mapping eliminated the smem output round-trip (140→53 ms, 2.64× single-round gain)

### Phase 2: Memory Optimization (R10–R23) — 50 ms → 44.6 ms
- WMMA tensor cores for QK^T and P@V (R5–R6)
- Vectorized float4 K/V loads (R16)
- cp.async double-buffered K/V prefetch (R18)
- Alpha rescale via warp shuffle (R20)

### Phase 3: Smem Elimination (R24–R29) — 44.6 ms → 22.3 ms (total)
- **R24**: Bank-conflict padding for S_s/P_s (8% gain)
- **R25**: In-register softmax — eliminated S_s shared memory entirely using octet-level warp shuffles
- **R26–R27**: Multi-sub-tile prefetch (BN_TILE=128) — halved sync/prefetch overhead
- **R28**: Vectorized pack_mask with `__ballot_sync` (13.5→2.0 ms, 6.8× on pack_mask alone)
- **R29**: `__launch_bounds__` occupancy fix — discovered register explosion from loop unrolling (193→128 regs), doubling occupancy to 2 blocks/SM

### Phase 4: Register-Only P@V (R30–R35) — 22.3 ms → 20.8 ms (total)
- **R33**: Discovered WMMA accumulator and matrix_a fragment layouts are **identical** on sm_90. P values from softmax directly populate matrix_a fragments — eliminated P_s shared memory (zero smem for softmax + P)
- **R35**: Hardcoded WMMA layout, eliminated runtime probe. 118 registers, zero spills.

---

## Key Technical Insights

### 1. Register Pressure is the Hidden Bottleneck
Removing `#pragma unroll` from a 4-iteration loop reduced registers from 193→133. `__launch_bounds__(256, 2)` then brought them to 128, doubling occupancy (1→2 blocks/SM) for a 12.6% speedup. **Occupancy matters more than loop overhead.**

### 2. WMMA Fragment Layout Identity (sm_90)
The accumulator (C/D) and matrix_a (A) fragment layouts share the same thread↔element mapping on sm_90:
```
Thread t: group g = t/4, lane l = t%4
  x[i] → (row, col) where row = g + (i&2 ? 8 : 0), col = l*2 + (i&1) + (i&4 ? 8 : 0)
  matrix_a x[8..15] = x[0..7] (duplicated for m16n16k16)
```
This allows **zero-cost P fragment construction** — softmax output in accumulator layout directly becomes the matrix_a operand for P@V.

### 3. Bank Conflict Padding
Stride padding for shared memory (e.g., 32→48 halfs for P_s) breaks 128-byte bank alignment, reducing 16-way conflicts. With in-register softmax, this primarily affects WMMA loads.

### 4. Multi-Sub-Tile Prefetch
Decoupling K/V prefetch granularity (BN_TILE=192) from WMMA tile size (BN=32) reduces `__syncthreads` and `cp.async` overhead by 6× while maintaining the same compute structure.

### 5. __ballot_sync for Mask Packing
Replacing scalar bit-by-bit packing (13.5ms) with warp-level `__ballot_sync()` (2.0ms) — one intrinsic packs 32 bools into one uint32 with coalesced memory access.

---

## Final Kernel Architecture

```
Grid:  [ceil(N/128), B×H] = [128, 12] = 1536 blocks
Block: 8 warps × 32 threads = 256 threads
Registers: 118/thread, 0 spills, 2 blocks/SM

Shared memory (81 KB):
  Q_s   [128, 64]   __half — 16 KB  (loaded once)
  K_buf [2×192, 64] __half — 48 KB  (double-buffered, cp.async)
  V_buf [2×192, 64] __half — 48 KB  (double-buffered, cp.async)
  (no S_s, no P_s — softmax + P fully in registers)

Per sub-tile (32 KV columns):
  1. WMMA QK^T:   8 mma_sync → S accumulators in registers
  2. Softmax:     in-register (octet shuffle: XOR 1,2 for max/sum)
  3. P fragment:  direct construction (acc layout == mat_a layout)
  4. WMMA P@V:    8 mma_sync → O_acc accumulators
  
Output: direct register-to-global writes (no smem round-trip)
```

---

## Performance Charts

See [notes/perf_chart.png](notes/perf_chart.png) for:
1. Latency per round (bar chart with accept/reject coloring)
2. Optimization trajectory vs Triton/PyTorch baselines
3. TFLOPS throughput progression
4. Backend comparison (log-scale horizontal bars)

---

## How to Run

```bash
pip install -r requirements.txt
python run.py                       # correctness + performance
python run.py --mode correctness    # correctness only
python run.py --mode perf           # performance only
python run.py --mode compare        # full cross-backend comparison
python notes/gen_chart.py           # regenerate performance charts
```
