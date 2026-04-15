# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
pip install -r requirements.txt
pip install -e .
```

Compiles `csrc/sparse_attention.cu` into the `sparse_attn_cuda` Python extension. Auto-detects GPU arch via `run.py`.

## Tests

All testing is done via the unified `run.py` entry point:

```bash
python run.py --mode correctness   # correctness vs FP32 PyTorch reference
python run.py --mode perf           # latency/MFU scaling + baseline comparison
python run.py --mode compare        # full横向对比 ours/triton/cudnn/flashinfer/flash-attn
python run.py                       # runs correctness + perf (default --mode all)
```

## Architecture

CUDA kernel for sparse masked attention.

The sparsity pattern is based on standard decode attention (causal mask over preceding KV), where each query only attends to a random subset of up to 2048 preceding KV positions. If a query has fewer than 2048 preceding positions, it attends to all of them.

The sparse attention implementation assumes N varies (tested N in [64, 128, 256, 512] for correctness; up to 16384 for perf). Mask shape is always `[B,H,N,N]` and must match the sequence length N of Q/K/V.


### File roles

- `csrc/sparse_attention.cu` — WMMA FP16 kernel (`sparse_attn_wmma_full_fp16`), scalar BF16 fallback, mask packing kernel, host launchers
- `csrc/sparse_attention.h` — `SparseAttentionParams` struct, function declarations
- `csrc/binding.cpp` — pybind11 bindings
- `csrc/utils.cuh` — warp reductions, bit-mask helpers, type conversions
- `python/sparse_attention.py` — PyTorch autograd `Function` wrapper, `sparse_attention()` user API, `sparse_attention_ref()` reference impl
- `run.py` — unified test/benchmark entry (JIT compiles, includes correctness/perf/compare modes, Triton baseline)
- `notes/perf_log.md` — detailed per-round optimization log
- `notes/optimization_rules.md` — optimization methodology rules
- `notes/gen_chart.py` — generates performance chart

### Data flow

```
Q, K, V [B,H,N,D] + bool mask [B,H,N,N]
  → pack_mask_kernel → uint32 mask [B,H,N,N/32]
  → sparse_attn_wmma_full_fp16 (register softmax, skips all-masked tiles)
  → output [B,H,N,D]
```

Backward saves LSE (log-sum-exp) from forward for gradient computation.
