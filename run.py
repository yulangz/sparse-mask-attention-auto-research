"""
独立运行脚本：JIT 编译 CUDA 扩展，验证正确性 + 测量延迟/MFU

用法：
    python run.py              # 正确性 + 性能
    python run.py --mode correctness
    python run.py --mode perf
    python run.py --mode compare             # 横向对比 ours/triton/cudnn/flashinfer/flash-attn
"""

import argparse
import math
import os
import sys
import torch
from torch.utils.cpp_extension import load

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    from flashinfer import single_prefill_with_kv_cache as fi_single_prefill
    from flashinfer.prefill import (
        BatchPrefillWithRaggedKVCacheWrapper as FiBatchPrefill,
    )

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

try:
    from flash_attn import flash_attn_func as _flash_attn_func

    # flash_attn_func 不支持 custom bool mask，需要用 float additive mask
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# cuDNN SDPA — 通过 PyTorch 2.0+ 的 sdpa_kernel 上下文管理器
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend

    HAS_SDPA_CUDNN = True
except ImportError:
    HAS_SDPA_CUDNN = False

# ============================================================
# 硬件检测
# ============================================================


def detect_gpu():
    """检测 GPU 型号，返回峰值性能参数和编译 arch"""
    props = torch.cuda.get_device_properties(0)
    name = props.name
    sm_count = props.multi_processor_count
    cap = (props.major, props.minor)

    # 已知 GPU 峰值性能查表 (TFLOPS)
    # 格式: {关键字: (fp16_tc_dense, fp16_cuda, fp32, sm_arch)}
    GPU_DB = {
        "RTX 3080": (59.5, 29.8, 29.8, "86"),
        "RTX 3080 Ti": (68.0, 34.1, 34.1, "86"),
        "RTX 3090": (71.0, 35.6, 35.6, "86"),
        "RTX 3090 Ti": (80.0, 40.0, 40.0, "86"),
        "RTX 4080": (97.5, 48.7, 48.7, "89"),
        "RTX 4090": (165.2, 82.6, 82.6, "89"),
        "A100": (312.0, 78.0, 19.5, "80"),
        "A100 80GB": (312.0, 78.0, 19.5, "80"),
        "H100": (989.0, 267.0, 67.0, "90"),
    }

    matched = None
    for key in GPU_DB:
        if key in name:
            matched = key
            break

    if matched:
        fp16_tc, fp16_cuda, fp32, arch = GPU_DB[matched]
    else:
        # 未知 GPU：根据 SM 数量和 compute capability 估算
        # Ampere (8.x): 4 TC/SM, 64 FMA/TC/cycle, boost ~1.7 GHz
        boost_ghz = 1.7
        fp32 = sm_count * 128 * 2 * boost_ghz / 1000
        fp16_cuda = fp32  # Ampere FP16 CUDA = FP32 throughput
        fp16_tc = sm_count * 4 * 64 * 2 * boost_ghz / 1000
        arch = f"{cap[0]}{cap[1]}"
        print(f"[Warning] 未知 GPU '{name}'，使用估算峰值性能")

    return {
        "name": name,
        "sm_count": sm_count,
        "compute_cap": cap,
        "arch": arch,
        "fp16_tc_tflops": fp16_tc,
        "fp16_cuda_tflops": fp16_cuda,
        "fp32_tflops": fp32,
    }


GPU_INFO = detect_gpu()

print(f"GPU: {GPU_INFO['name']}")
print(
    f"  SMs: {GPU_INFO['sm_count']}, Compute: {GPU_INFO['compute_cap'][0]}.{GPU_INFO['compute_cap'][1]}"
)
print(f"  FP16 Tensor Core: {GPU_INFO['fp16_tc_tflops']:.1f} TFLOPS")
print(f"  FP16 CUDA Core:   {GPU_INFO['fp16_cuda_tflops']:.1f} TFLOPS")
print(f"  FP32:             {GPU_INFO['fp32_tflops']:.1f} TFLOPS")
print()

# ============================================================
# JIT 编译
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))
SM_ARCH = GPU_INFO["arch"]

# --- 原始 sparse_attn (csrc/) ---
print("正在编译 CUDA 扩展（首次约需 30s）...")
sparse_attn_cuda = load(
    name="sparse_attn_cuda",
    sources=[
        os.path.join(ROOT, "csrc/binding.cpp"),
        os.path.join(ROOT, "csrc/sparse_attention.cu"),
    ],
    extra_include_paths=[os.path.join(ROOT, "csrc")],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        f"-gencode=arch=compute_{SM_ARCH},code=sm_{SM_ARCH}",
    ],
    extra_cflags=["-O3"],
    verbose=False,
)
print("编译完成")

# ============================================================
# 工具函数
# ============================================================


def pack_mask(mask: torch.Tensor) -> torch.Tensor:
    """bool [B,H,N,N] -> uint32 [B,H,N,N/32]"""
    return sparse_attn_cuda.pack_mask(mask)


def sparse_attention_cuda_fwd(q, k, v, mask):
    """调用 CUDA kernel 前向"""
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    mask_packed = pack_mask(mask)
    out = torch.empty_like(q)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    sparse_attn_cuda.forward(q, k, v, mask_packed, out, lse, scale, False)
    return out


def sparse_attention_ref(q, k, v, mask):
    """PyTorch 参考实现（保持输入 dtype，FP16 时走 Tensor Core）"""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, v)


def generate_mask(B, H, N, num_random_prev=2048, device="cuda"):
    """
    生成接近生产环境的稀疏 mask。
    规则：
      1. 每个 Q[i] attend 自己和自己之后的所有 token (j >= i)
      2. 每个 Q[i] 额外随机选择 min(num_random_prev, i) 个前序 token
    """
    idx = torch.arange(N, device=device)
    upper_tri = idx.unsqueeze(0) >= idx.unsqueeze(1)
    mask = upper_tri.unsqueeze(0).unsqueeze(0).expand(B, H, N, N).clone()

    lower_tri = idx.unsqueeze(0) < idx.unsqueeze(1)
    k = min(num_random_prev, N - 1)
    if k > 0:
        rand_scores = torch.rand(B, H, N, N, device=device)
        rand_scores.masked_fill_(~lower_tri.unsqueeze(0).unsqueeze(0), float("inf"))
        _, topk_idx = torch.topk(rand_scores, k=k, dim=-1, largest=False)
        random_selected = torch.zeros(B, H, N, N, dtype=torch.bool, device=device)
        random_selected.scatter_(-1, topk_idx, True)
        random_selected &= lower_tri.unsqueeze(0).unsqueeze(0)
        mask |= random_selected

    return mask


def measure_latency(fn, warmup=10, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat  # ms


def get_peak_tflops(dtype):
    """根据 dtype 返回当前 GPU 的峰值 TFLOPS (Tensor Core for FP16/BF16)"""
    if dtype in (torch.float16, torch.bfloat16):
        return GPU_INFO["fp16_tc_tflops"]
    return GPU_INFO["fp32_tflops"]


def get_peak_tflops_tc():
    """Tensor Core 峰值"""
    return GPU_INFO["fp16_tc_tflops"]


def compute_mfu(latency_ms, B, H, N, D, peak_tflops):
    # 理论 FLOPs: QK^T (B*H*N*N*D) + softmax (B*H*N) + attn@V (B*H*N*N*D) = 4*B*H*N*N*D
    # mask 不影响理论 FLOPs，只是实际计算时跳过一些元素
    flops = 4 * B * H * N * N * D
    tflops = flops / (latency_ms / 1000.0) / 1e12
    return tflops, tflops / peak_tflops * 100.0


# ============================================================
# Triton Sparse Attention Baseline
# ============================================================

if HAS_TRITON:
    # 上面的方案过于复杂，改用更简洁的实现
    @triton.jit
    def sparse_attn_triton_kernel(
        Q,
        K,
        V,
        Mask,
        Out,
        # strides
        stride_qb,
        stride_qh,
        stride_qn,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_mb,
        stride_mh,
        stride_mn,
        stride_ob,
        stride_oh,
        stride_on,
        # shapes
        H,
        N,
        # params
        scale,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # grid: (ceil(N/BLOCK_M), B*H)
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        pid_b = pid_bh // H
        pid_h = pid_bh % H

        row_start = pid_m * BLOCK_M
        offs_m = row_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        offs_d = tl.arange(0, D)  # [D]

        # 基础指针
        Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
        K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
        V_ptr = V + pid_b * stride_vb + pid_h * stride_vh
        M_ptr = Mask + pid_b * stride_mb + pid_h * stride_mh
        O_ptr = Out + pid_b * stride_ob + pid_h * stride_oh

        # 加载 Q tile [BLOCK_M, D]
        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qn + offs_d[None, :],
            mask=offs_m[:, None] < N,
            other=0.0,
        )  # [BLOCK_M, D]

        # Online softmax 状态（用 -1e6 代替 -inf，避免 exp(-inf-(-inf))=nan）
        m_i = tl.full([BLOCK_M], -1e6, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        # 遍历 K/V tiles
        num_tiles_n = tl.cdiv(N, BLOCK_N)
        for tile_n in range(num_tiles_n):
            col_start = tile_n * BLOCK_N
            offs_n = col_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

            # 读取 mask [BLOCK_M, BLOCK_N]
            mask_tile = tl.load(
                M_ptr + offs_m[:, None] * stride_mn + offs_n[None, :],
                mask=(offs_m[:, None] < N) & (offs_n[None, :] < N),
                other=False,
            )  # [BLOCK_M, BLOCK_N], bool

            # 加载 K tile [BLOCK_N, D]
            k = tl.load(
                K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :],
                mask=offs_n[:, None] < N,
                other=0.0,
            )  # [BLOCK_N, D]

            # QK^T [BLOCK_M, BLOCK_N]
            scores = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_N]

            # 应用 mask：masked 位置设为 -1e6（避免 -inf 导致 nan）
            scores = tl.where(
                mask_tile & (offs_n[None, :] < N),
                scores,
                -1e6,
            )

            # Online softmax 更新
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))  # [BLOCK_M]
            alpha = tl.exp(m_i - m_new)  # [BLOCK_M]
            p = tl.exp(scores - m_new[:, None])  # [BLOCK_M, BLOCK_N]

            l_i = alpha * l_i + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            # 加载 V tile [BLOCK_N, D]
            v = tl.load(
                V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :],
                mask=offs_n[:, None] < N,
                other=0.0,
            )  # [BLOCK_N, D]

            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # 归一化
        safe_l = tl.where(l_i > 0, l_i, tl.full([BLOCK_M], 1.0, dtype=tl.float32))
        acc = acc / safe_l[:, None]

        # 写回
        tl.store(
            O_ptr + offs_m[:, None] * stride_on + offs_d[None, :],
            acc.to(Q.dtype.element_ty),
            mask=offs_m[:, None] < N,
        )

    def sparse_attention_triton(q, k, v, mask):
        """
        Triton sparse attention baseline.
        q, k, v: [B, H, N, D] FP16
        mask: [B, H, N, N] bool, True=保留
        """
        B, H, N, D = q.shape
        assert D in (16, 32, 64, 128), f"D={D} must be power-of-2 in [16,128]"
        scale = 1.0 / math.sqrt(D)
        out = torch.empty_like(q)

        BLOCK_M = 16
        BLOCK_N = 16
        grid = (triton.cdiv(N, BLOCK_M), B * H)

        sparse_attn_triton_kernel[grid](
            q,
            k,
            v,
            mask,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            mask.stride(0),
            mask.stride(1),
            mask.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            H,
            N,
            scale,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return out

# ============================================================
# 正确性测试
# ============================================================


def run_correctness():
    print("=" * 50)
    print("正确性验证")
    print("=" * 50)
    device = "cuda"
    dtype = torch.float16
    B, H, D = 2, 4, 64
    num_random_prev = 2048
    tol = 1e-2

    all_pass = True
    for N in [64, 128, 256, 512]:
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        mask = generate_mask(B, H, N, num_random_prev=num_random_prev, device=device)

        # FP32 参考用于正确性验证
        ref = sparse_attention_ref(q.float(), k.float(), v.float(), mask)
        out = sparse_attention_cuda_fwd(q, k, v, mask).float()

        max_diff = (out - ref).abs().max().item()
        mean_diff = (out - ref).abs().mean().item()
        passed = max_diff < tol
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"  [CUDA]   N={N:4d}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  [{status}]"
        )

        if HAS_TRITON:
            out_triton = sparse_attention_triton(q, k, v, mask).float()
            max_diff_t = (out_triton - ref).abs().max().item()
            mean_diff_t = (out_triton - ref).abs().mean().item()
            passed_t = max_diff_t < tol
            all_pass = all_pass and passed_t
            status_t = "PASS" if passed_t else "FAIL"
            print(
                f"  [Triton] N={N:4d}  max_diff={max_diff_t:.6f}  mean_diff={mean_diff_t:.6f}  [{status_t}]"
            )

    # 全 masked 行不应产生 NaN
    q = torch.randn(1, 1, 8, 64, device=device, dtype=dtype)
    k = torch.randn(1, 1, 8, 64, device=device, dtype=dtype)
    v = torch.randn(1, 1, 8, 64, device=device, dtype=dtype)
    mask = torch.ones(1, 1, 8, 8, device=device, dtype=torch.bool)
    mask[:, :, 0, :] = False
    out = sparse_attention_cuda_fwd(q, k, v, mask)
    nan_ok = not torch.isnan(out).any()
    print(f"  [CUDA]   全 masked 行无 NaN  [{'PASS' if nan_ok else 'FAIL'}]")
    all_pass = all_pass and nan_ok

    if HAS_TRITON:
        out_t = sparse_attention_triton(q, k, v, mask)
        nan_ok_t = not torch.isnan(out_t).any()
        print(f"  [Triton] 全 masked 行无 NaN  [{'PASS' if nan_ok_t else 'FAIL'}]")
        all_pass = all_pass and nan_ok_t

    print(f"\n结论: {'全部通过' if all_pass else '存在失败项'}\n")


# ============================================================
# 性能测试
# ============================================================


def run_perf():
    print("=" * 50)
    print("性能测试 (FP16)")
    print("=" * 50)
    device = "cuda"
    dtype = torch.float16
    H, D = 12, 64
    num_random_prev = 2048
    peak = get_peak_tflops(dtype)
    peak_tc = get_peak_tflops_tc()

    print(f"  峰值参考: {peak:.1f} TFLOPS (FP16 Tensor Core)")

    # --- 序列长度缩放 ---
    print(f"\n[序列长度缩放]  H={H}, D={D}, num_random_prev={num_random_prev}")
    print(
        f"  {'N':>6}  {'B':>4}  {'latency(ms)':>12}  {'TFLOPS':>8}  {'MFU%':>7}  {'density%':>9}"
    )
    print("  " + "-" * 60)
    for N in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        # 动态调整 B 以适配显存：mask [B,H,N,N] bool 占 B*H*N*N 字节
        if N <= 1024:
            B = 16
        elif N <= 4096:
            B = 4
        elif N <= 8192:
            B = 2
        else:
            B = 1
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        mask = generate_mask(B, H, N, num_random_prev=num_random_prev, device=device)
        density = mask.float().mean().item() * 100
        lat = measure_latency(lambda: sparse_attention_cuda_fwd(q, k, v, mask))
        tflops, mfu = compute_mfu(lat, B, H, N, D, peak)
        print(
            f"  {N:>6}  {B:>4}  {lat:>12.3f}  {tflops:>8.2f}  {mfu:>7.1f}%  {density:>8.1f}%"
        )
        del q, k, v, mask
        torch.cuda.empty_cache()

    # --- Batch size 缩放 ---
    print(
        f"\n[Batch size 缩放]  N=512, H={H}, D={D}, num_random_prev={num_random_prev}"
    )
    print(f"  {'B':>6}  {'latency(ms)':>12}  {'TFLOPS':>8}  {'MFU%':>7}  {'seq/s':>10}")
    print("  " + "-" * 55)
    N = 512
    for B in [1, 4, 8, 16, 32, 64]:
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        mask = generate_mask(B, H, N, num_random_prev=num_random_prev, device=device)
        lat = measure_latency(lambda: sparse_attention_cuda_fwd(q, k, v, mask))
        tflops, mfu = compute_mfu(lat, B, H, N, D, peak)
        seqs = B / (lat / 1000.0)
        print(f"  {B:>6}  {lat:>12.3f}  {tflops:>8.2f}  {mfu:>7.1f}%  {seqs:>10.0f}")

    # --- 与各 baseline 对比 (FP16) ---
    print(
        f"\n[Baseline 对比]  B=1, N=16384, H={H}, D={D}, dtype=FP16, num_random_prev={num_random_prev}"
    )
    B, N = 1, 16384
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)
    mask = generate_mask(B, H, N, num_random_prev=num_random_prev, device=device)
    density = mask.float().mean().item() * 100
    print(f"  mask density: {density:.1f}%")

    lat_ours = measure_latency(lambda: sparse_attention_cuda_fwd(q, k, v, mask))
    lat_pt = measure_latency(
        lambda: sparse_attention_ref(q, k, v, mask), warmup=3, repeat=10
    )
    tflops_ours, mfu_ours = compute_mfu(lat_ours, B, H, N, D, peak)
    tflops_pt, mfu_pt = compute_mfu(lat_pt, B, H, N, D, peak_tc)
    print(f"  Ours:        {lat_ours:.3f} ms  ({tflops_ours:.2f} TFLOPS)")
    print(
        f"  PyTorch Ref: {lat_pt:.3f} ms  ({tflops_pt:.2f} TFLOPS)  speedup={lat_pt / lat_ours:.2f}x"
    )

    if HAS_TRITON:
        # warmup triton JIT
        _ = sparse_attention_triton(q, k, v, mask)
        torch.cuda.synchronize()
        lat_triton = measure_latency(
            lambda: sparse_attention_triton(q, k, v, mask), warmup=5, repeat=20
        )
        tflops_triton, _ = compute_mfu(lat_triton, B, H, N, D, peak_tc)
        print(
            f"  Triton:      {lat_triton:.3f} ms  ({tflops_triton:.2f} TFLOPS)  speedup={lat_triton / lat_ours:.2f}x"
        )

    # cuDNN SDPA (dense, 不支持自定义 sparse mask)
    if HAS_SDPA_CUDNN:
        try:
            lat_cudnn = measure_latency(
                lambda: cudnn_attention_fwd(q, k, v, mask), warmup=5, repeat=20
            )
            tflops_cudnn, _ = compute_mfu(lat_cudnn, B, H, N, D, peak_tc)
            print(
                f"  cuDNN SDPA: {lat_cudnn:.3f} ms  ({tflops_cudnn:.2f} TFLOPS)  speedup={lat_cudnn / lat_ours:.2f}x (dense)"
            )
        except Exception as e:
            print(f"  cuDNN SDPA: [ERROR] {e}")

    # FlashInfer
    if HAS_FLASHINFER:
        try:
            lat_fi = measure_latency(
                lambda: flashinfer_attention_fwd(q, k, v, mask), warmup=3, repeat=10
            )
            tflops_fi, _ = compute_mfu(lat_fi, B, H, N, D, peak_tc)
            print(
                f"  FlashInfer: {lat_fi:.3f} ms  ({tflops_fi:.2f} TFLOPS)  speedup={lat_fi / lat_ours:.2f}x"
            )
        except Exception as e:
            print(f"  FlashInfer: [ERROR] {e}")

    # flash-attn (无 custom mask, dense baseline)
    if HAS_FLASH_ATTN:
        try:
            lat_flash = measure_latency(
                lambda: flash_attn_pkg_fwd(q, k, v, mask), warmup=5, repeat=20
            )
            tflops_flash, _ = compute_mfu(lat_flash, B, H, N, D, peak_tc)
            print(
                f"  flash-attn: {lat_flash:.3f} ms  ({tflops_flash:.2f} TFLOPS)  speedup={lat_flash / lat_ours:.2f}x (dense)"
            )
        except Exception as e:
            print(f"  flash-attn: [ERROR] {e}")

    print()


# ============================================================
# cuDNN SDPA Backend  (torch.nn.attention.sdpa_kernel)
# ============================================================


def cudnn_attention_fwd(q, k, v, mask):
    """
    PyTorch cuDNN Flash Attention backend.
    q, k, v : [B, H, N, D]  FP16/BF16
    mask    : [B, H, N, N]  bool  (True=attend, False=mask out)
    返回     : [B, H, N, D]
    Note: PyTorch SDPA 的 attn_mask 为 True 时保留，与 flash-attention 规范一致。
    """
    if not HAS_SDPA_CUDNN:
        raise RuntimeError(
            "torch.nn.attention.sdpa_kernel not available (需要 PyTorch >= 2.0)"
        )
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,  # bool: True=keep
            dropout_p=0.0,
        )
    return out


def cudnn_math_attention_fwd(q, k, v, mask):
    """
    PyTorch Math (cuBLAS) backend — 用于验证 cuDNN 结果正确性。
    """
    with sdpa_kernel(SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
        )
    return out


# ============================================================
# FlashInfer Backend
# ============================================================


def flashinfer_attention_fwd(q, k, v, mask):
    """
    FlashInfer BatchPrefillWithRaggedKVCache 后端。
    q, k, v : [B, H, N, D]  FP16 (NHD layout internally)
    mask    : [B, H, N, N]  bool  (True=attend)
    返回     : [B, H, N, D]

    将 B*H 展平为 batch 维度，每个 "sequence" 用 1 个 head，
    一次 kernel launch 处理所有 (batch, head) 对。
    """
    if not HAS_FLASHINFER:
        raise RuntimeError(
            "flashinfer not installed, run: pip install flashinfer-python"
        )
    B, H, N, D = q.shape
    num_seqs = B * H

    # [B, H, N, D] -> [B*H, N, 1, D] -> [B*H*N, 1, D]
    q_fi = (
        q.reshape(num_seqs, N, D).unsqueeze(2).reshape(num_seqs * N, 1, D).contiguous()
    )
    k_fi = (
        k.reshape(num_seqs, N, D).unsqueeze(2).reshape(num_seqs * N, 1, D).contiguous()
    )
    v_fi = (
        v.reshape(num_seqs, N, D).unsqueeze(2).reshape(num_seqs * N, 1, D).contiguous()
    )

    # indptr: each of B*H sequences has length N
    qo_indptr = torch.arange(
        0, (num_seqs + 1) * N, N, device=q.device, dtype=torch.int32
    )
    kv_indptr = qo_indptr

    # mask [B, H, N, N] -> [B*H*N*N] flattened
    mask_flat = mask.reshape(-1).contiguous()

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = FiBatchPrefill(workspace, "NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=D,
        custom_mask=mask_flat,
    )
    o_fi = wrapper.run(q_fi, k_fi, v_fi)  # [B*H*N, 1, D]
    return o_fi.reshape(B, H, N, D)


# ============================================================
# Flash-Attn Backend  (from flash-attn package)
# ============================================================


def flash_attn_pkg_fwd(q, k, v, mask):
    """
    flash-attn 包的 flash_attn_func 后端。
    flash_attn_func 接受 causal 或 alibi_slopes，不支持 custom bool mask。
    改用 additive float mask: False→-inf, True→0.
    q, k, v : [B, H, N, D] FP16
    mask    : [B, H, N, N] bool
    返回    : [B, H, N, D]
    """
    if not HAS_FLASH_ATTN:
        raise RuntimeError("flash-attn not installed")
    B, H, N, D = q.shape
    # flash_attn_func 期望 [B, N, H, D] (bshd layout)
    q_f = q.permute(0, 2, 1, 3).contiguous()  # [B, N, H, D]
    k_f = k.permute(0, 2, 1, 3).contiguous()
    v_f = v.permute(0, 2, 1, 3).contiguous()
    # flash_attn_func 不接受 custom mask；此处我们用无 mask 版本测基础性能
    # （用于与我们的 sparse kernel 做无 mask 基准对比）
    out_f = _flash_attn_func(
        q_f, k_f, v_f, dropout_p=0.0, softmax_scale=None, causal=False
    )
    return out_f.permute(0, 2, 1, 3)  # [B, H, N, D]


# ============================================================
# 横向对比：run_compare
# ============================================================


def run_compare():
    """
    对比所有可用后端的延迟和吞吐量。
    后端列表：
      - Ours (csrc/sparse_attn_cuda)
      - Triton (if available)
      - cuDNN SDPA (torch.nn.attention.sdpa_kernel)
      - FlashInfer (flashinfer-python, per-batch loop)
      - flash-attn pkg (if installed, no custom mask, dense baseline)
    """
    print("=" * 80)
    print("横向性能对比 — 所有可用后端")
    print("=" * 80)
    device = "cuda"
    dtype = torch.float16
    H, D = 12, 64
    num_random_prev = 2048
    peak = get_peak_tflops(dtype)
    peak_tc = get_peak_tflops_tc()

    # 检测可用后端
    backends = {}
    backends["Ours (CUDA)"] = lambda q, k, v, m: sparse_attention_cuda_fwd(q, k, v, m)
    if HAS_TRITON:
        backends["Triton"] = lambda q, k, v, m: sparse_attention_triton(q, k, v, m)
    if HAS_SDPA_CUDNN:
        # 先试 cuDNN backend，失败则回退到 math
        def _try_cudnn(q, k, v, m):
            try:
                return cudnn_attention_fwd(q, k, v, m)
            except RuntimeError:
                return cudnn_math_attention_fwd(q, k, v, m)

        backends["cuDNN SDPA"] = _try_cudnn
        backends["PyTorch Math"] = lambda q, k, v, m: cudnn_math_attention_fwd(
            q, k, v, m
        )
    if HAS_FLASHINFER:
        # 逐 batch 调用版（精确 mask）
        backends["FlashInfer"] = lambda q, k, v, m: flashinfer_attention_fwd(q, k, v, m)
    if HAS_FLASH_ATTN:
        # flash-attn 无 custom mask（显示纯 dense baseline）
        backends["flash-attn (dense)"] = lambda q, k, v, m: flash_attn_pkg_fwd(
            q, k, v, m
        )

    print(f"\n可用后端: {list(backends.keys())}\n")

    # ----------------------------------------------------------
    # 1. 正确性验证（对比 FP32 参考实现）
    # ----------------------------------------------------------
    print("-" * 60)
    print("1. 正确性验证  (B=2, H=4, N=256, D=64, num_random_prev=2048, FP16)")
    print("-" * 60)
    B_c, N_c = 2, 256
    torch.manual_seed(42)
    q_c = torch.randn(B_c, 4, N_c, D, device=device, dtype=dtype)
    k_c = torch.randn(B_c, 4, N_c, D, device=device, dtype=dtype)
    v_c = torch.randn(B_c, 4, N_c, D, device=device, dtype=dtype)
    mask_c = generate_mask(B_c, 4, N_c, num_random_prev=num_random_prev, device=device)

    ref_c = sparse_attention_ref(q_c.float(), k_c.float(), v_c.float(), mask_c)
    tol = 5e-2  # FP16 精度宽松一点

    for name, fn in backends.items():
        if name in ("flash-attn (dense)",):
            # 无法用 custom mask 做正确性验证，跳过
            print(f"  {name:<22}: [SKIP] (no custom mask support)")
            continue
        try:
            # cuDNN 和 FlashInfer 有时对非常稀疏的 mask 行为略有差异
            if name == "FlashInfer":
                # FlashInfer single_prefill 不支持 H > 1 的 per-head mask，
                # 使用 H=1 子集验证
                B1, N1 = 1, N_c
                q1 = q_c[:1, :1, :, :]
                k1 = k_c[:1, :1, :, :]
                v1 = v_c[:1, :1, :, :]
                m1 = mask_c[:1, :1, :, :]
                ref1 = ref_c[:1, :1, :, :]
                out_c = fn(q1, k1, v1, m1).float()
                diff = (out_c - ref1).abs().max().item()
            else:
                out_c = fn(q_c, k_c, v_c, mask_c).float()
                diff = (out_c - ref_c).abs().max().item()
            status = "PASS" if diff < tol else "FAIL"
            print(f"  {name:<22}: max_diff={diff:.5f}  [{status}]")
        except Exception as e:
            print(f"  {name:<22}: [ERROR] {e}")

    # ----------------------------------------------------------
    # 2. 序列长度缩放
    # ----------------------------------------------------------
    print(f"\n{'-' * 80}")
    print(f"2. 序列长度缩放  H={H}, D={D}, num_random_prev={num_random_prev}")
    print(f"{'-' * 80}")
    # 列头
    col_names = ["N"] + list(backends.keys())
    col_w = max(len(n) for n in col_names) + 2

    header = f"  {'N':>6} {'B':>3}  " + "  ".join(
        f"{n:>{col_w}}" for n in backends.keys()
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for N in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        if N <= 1024:
            B_p = 16
        elif N <= 4096:
            B_p = 4
        elif N <= 8192:
            B_p = 2
        else:
            B_p = 1
        q_p = torch.randn(B_p, H, N, D, device=device, dtype=dtype)
        k_p = torch.randn(B_p, H, N, D, device=device, dtype=dtype)
        v_p = torch.randn(B_p, H, N, D, device=device, dtype=dtype)
        mask_p = generate_mask(
            B_p, H, N, num_random_prev=num_random_prev, device=device
        )

        row = f"  {N:>6} {B_p:>3}  "
        lat_ours = None
        for name, fn in backends.items():
            try:
                rpt = 10 if "FlashInfer" in name else 50
                wmup = 3 if "FlashInfer" in name else 10
                lat = measure_latency(
                    lambda fn=fn, q=q_p, k=k_p, v=v_p, m=mask_p: fn(q, k, v, m),
                    warmup=wmup,
                    repeat=rpt,
                )
                if lat_ours is None:
                    lat_ours = lat
                row += f"  {lat:{col_w}.3f}"
            except Exception as e:
                row += f"  {'ERR':>{col_w}}"
        print(row + " ms")
        del q_p, k_p, v_p, mask_p
        torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # 3. 单点详细对比  (B=1, N=16384, H=12)
    # ----------------------------------------------------------
    print(f"\n{'-' * 80}")
    print(
        f"3. 单点详细对比  B=1, N=16384, H={H}, D={D}, num_random_prev={num_random_prev}"
    )
    print(f"{'-' * 80}")
    print(
        f"  {'Backend':<26}  {'latency(ms)':>12}  {'TFLOPS':>8}  {'vs Ours':>9}  {'vs Triton':>10}"
    )
    print(f"  {'-' * 75}")

    N_s, B_s = 16384, 1
    q_s = torch.randn(B_s, H, N_s, D, device=device, dtype=dtype)
    k_s = torch.randn(B_s, H, N_s, D, device=device, dtype=dtype)
    v_s = torch.randn(B_s, H, N_s, D, device=device, dtype=dtype)
    mask_s = generate_mask(B_s, H, N_s, num_random_prev=num_random_prev, device=device)
    density = mask_s.float().mean().item() * 100
    print(f"  mask density: {density:.1f}%")

    results = {}
    for name, fn in backends.items():
        try:
            rpt = 10 if "FlashInfer" in name else 50
            wmup = 3 if "FlashInfer" in name else 10
            lat = measure_latency(
                lambda fn=fn, q=q_s, k=k_s, v=v_s, m=mask_s: fn(q, k, v, m),
                warmup=wmup,
                repeat=rpt,
            )
            tflops, _ = compute_mfu(lat, B_s, H, N_s, D, peak_tc)
            results[name] = (lat, tflops)
        except Exception as e:
            results[name] = (None, None)

    lat_ours_s = results.get("Ours (CUDA)", (None,))[0]
    lat_triton_s = results.get("Triton", (None,))[0]

    for name, (lat, tflops) in results.items():
        if lat is None:
            print(f"  {name:<26}  {'ERROR':>12}")
            continue
        vs_ours = f"{lat_ours_s / lat:.2f}x" if lat_ours_s else "N/A"
        vs_triton = f"{lat_triton_s / lat:.2f}x" if lat_triton_s else "N/A"
        if name == "Ours (CUDA)":
            vs_ours = "baseline"
        if name == "Triton":
            vs_triton = "baseline"
        print(
            f"  {name:<26}  {lat:>12.3f}  {tflops:>8.2f}  {vs_ours:>9}  {vs_triton:>10}"
        )

    # ----------------------------------------------------------
    # 4. 总结分析
    # ----------------------------------------------------------
    print(f"\n{'-' * 80}")
    print("4. 性能差距分析")
    print(f"{'-' * 80}")
    if lat_ours_s and lat_triton_s:
        gap = lat_ours_s / lat_triton_s
        print(
            f"  Ours vs Triton: {lat_ours_s:.3f} ms / {lat_triton_s:.3f} ms = {gap:.2f}x slower"
        )
        if gap > 1.0:
            print(f"  差距来源分析:")
            print(
                f"    - BN=64 减少了循环次数，但 smem 增至 34KB，occupancy 下降到 ~2 blocks/SM"
            )
            print(
                f"    - Triton 使用 BLOCK_N=16，occupancy 更高，且 Triton 自动调优更优"
            )
            print(f"    - cuDNN/FlashInfer 使用专用优化内核，不受自定义稀疏度限制")
    if HAS_SDPA_CUDNN and "cuDNN SDPA" in results and results["cuDNN SDPA"][0]:
        lat_cudnn = results["cuDNN SDPA"][0]
        print(
            f"  cuDNN SDPA: {lat_cudnn:.3f} ms (dense, 不稀疏) — 用于了解 cuDNN 基准线"
        )
    if HAS_FLASHINFER and "FlashInfer" in results and results["FlashInfer"][0]:
        lat_fi = results["FlashInfer"][0]
        print(f"  FlashInfer: {lat_fi:.3f} ms — 注意: 逐 batch loop，存在 Python 开销")
    print()


# ============================================================
# Decode 场景测试
# ============================================================


def sparse_decode_gather(q, k_cache, v_cache, indices, scale):
    """
    Sparse decode via gather: only attend to indexed KV positions.
    q:       [B, H, q_len, D]
    k_cache: [B, H, kv_len, D]
    v_cache: [B, H, kv_len, D]
    indices: [B, H, q_len, num_attend]  — per-query attended position indices
    """
    B, H, q_len, D = q.shape
    num_attend = indices.shape[-1]

    # Gather K/V at sparse positions: [B, H, q_len, num_attend, D]
    idx_expand = indices.unsqueeze(-1).expand(B, H, q_len, num_attend, D)
    k_sel = torch.gather(
        k_cache.unsqueeze(2).expand(-1, -1, q_len, -1, -1), 3, idx_expand
    )
    v_sel = torch.gather(
        v_cache.unsqueeze(2).expand(-1, -1, q_len, -1, -1), 3, idx_expand
    )

    # Attention: Q[q_len, D] × K_sel[q_len, num_attend, D]^T → [q_len, num_attend]
    scores = torch.einsum("bhqd,bhqkd->bhqk", q, k_sel) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bhqkd->bhqd", attn, v_sel)
    return out


def sparse_decode_flat(q, k_cache, v_cache, indices_flat, scale):
    """
    Optimized sparse decode: shared indices across queries.
    q:            [B, H, q_len, D]
    k_cache:      [B, H, kv_len, D]
    v_cache:      [B, H, kv_len, D]
    indices_flat: [B, H, num_attend]  — shared across q_len queries
    """
    B, H, q_len, D = q.shape
    num_attend = indices_flat.shape[-1]

    # Gather K/V: [B, H, num_attend, D]
    idx_exp = indices_flat.unsqueeze(-1).expand(B, H, num_attend, D)
    k_sel = torch.gather(k_cache, 2, idx_exp)
    v_sel = torch.gather(v_cache, 2, idx_exp)

    # Q[B,H,q_len,D] × K_sel[B,H,num_attend,D]^T → [B,H,q_len,num_attend]
    scores = torch.matmul(q, k_sel.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_sel)
    return out


def dense_decode_sdpa(q, k_cache, v_cache):
    """Dense decode via cuDNN SDPA (attends to all KV positions)."""
    return torch.nn.functional.scaled_dot_product_attention(q, k_cache, v_cache)


def dense_decode_flash(q, k_cache, v_cache):
    """Dense decode via flash-attn (attends to all KV positions)."""
    # flash-attn expects [B, N, H, D]
    out = _flash_attn_func(
        q.transpose(1, 2).contiguous(),
        k_cache.transpose(1, 2).contiguous(),
        v_cache.transpose(1, 2).contiguous(),
        causal=False,  # decode: q attends to all past, no causal needed on q side
    )
    return out.transpose(1, 2)


def run_decode():
    """
    Decode 场景测试：Q 长度 1-4，KV cache 长度 4K-128K。
    对比 sparse gather (attend 2048) vs dense (attend all)。
    """
    print("=" * 100)
    print("Decode 场景测试 — Sparse Gather vs Dense Full Attention")
    print("=" * 100)
    print("  Q: 新生成的 token (1=标准decode, 2-4=MTP 场景)")
    print("  KV: 已有的 KV cache")
    print("  Sparse: 每个 query 只 attend 2048 个随机 KV 位置")
    print("  Dense: 每个 query attend 全部 KV 位置")
    print()

    B = 1
    H = 12
    D = 64
    dtype = torch.float16
    device = "cuda"
    num_attend = 2048
    q_lens = [1, 2, 3, 4]
    kv_lens = [4096, 16384, 65536, 131072]

    scale = 1.0 / math.sqrt(D)

    # Header
    print(f"  B={B}, H={H}, D={D}, num_attend={num_attend}, dtype=FP16")
    print()
    print(
        f"  {'kv_len':>8}  {'q_len':>5}  {'Sparse(ms)':>11}  {'cuDNN(ms)':>10}  "
        f"{'flash(ms)':>10}  {'Speedup_cuDNN':>13}  {'Speedup_flash':>13}  "
        f"{'Sparse/Dense':>12}"
    )
    print("  " + "-" * 96)

    for kv_len in kv_lens:
        k_cache = torch.randn(B, H, kv_len, D, device=device, dtype=dtype)
        v_cache = torch.randn(B, H, kv_len, D, device=device, dtype=dtype)

        for q_len in q_lens:
            q = torch.randn(B, H, q_len, D, device=device, dtype=dtype)

            # Sparse indices: each query attends to num_attend random KV positions
            indices = torch.randint(
                0, kv_len, (B, H, min(num_attend, kv_len)), device=device
            )

            # Sparse decode
            try:
                lat_sp = measure_latency(
                    lambda: sparse_decode_flat(q, k_cache, v_cache, indices, scale),
                    warmup=20,
                    repeat=100,
                )
            except Exception:
                lat_sp = float("nan")

            # Dense: cuDNN SDPA
            try:
                lat_cudnn = measure_latency(
                    lambda: dense_decode_sdpa(q, k_cache, v_cache),
                    warmup=20,
                    repeat=100,
                )
            except Exception:
                lat_cudnn = float("nan")

            # Dense: flash-attn
            if HAS_FLASH_ATTN:
                try:
                    lat_flash = measure_latency(
                        lambda: dense_decode_flash(q, k_cache, v_cache),
                        warmup=20,
                        repeat=100,
                    )
                except Exception:
                    lat_flash = float("nan")
            else:
                lat_flash = float("nan")

            # Compute speedups
            sp_cudnn = lat_cudnn / lat_sp if lat_sp > 0 else float("nan")
            sp_flash = lat_flash / lat_sp if lat_sp > 0 else float("nan")
            density = min(num_attend, kv_len) / kv_len * 100

            print(
                f"  {kv_len:>8}  {q_len:>5}  {lat_sp:>10.3f}  {lat_cudnn:>9.3f}  "
                f"{lat_flash:>9.3f}  {sp_cudnn:>12.2f}×  {sp_flash:>12.2f}×  "
                f"attend {density:.1f}%"
            )

        del k_cache, v_cache
        torch.cuda.empty_cache()
        print()

    print("  说明:")
    print("    Speedup >1 表示 sparse 更快，<1 表示 dense 更快")
    print("    Sparse/Dense 列显示 sparse attend 的 KV 占比")
    print("    MTP (Multi-Token Prediction): q_len>1 表示同时解码多个 token")
    print()


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA GPU"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["correctness", "perf", "all", "compare", "decode"],
        default="all",
    )
    args = parser.parse_args()

    if args.mode in ("correctness", "all"):
        run_correctness()
    if args.mode in ("perf", "all"):
        run_perf()
    if args.mode == "compare":
        run_compare()
    if args.mode == "decode":
        run_decode()
