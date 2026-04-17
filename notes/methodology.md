# 优化方法论复盘

## 使用的分析手段

整个 35 轮优化过程中**没有使用任何 GPU profiling 工具**（nsight compute、nsight systems、ncu、nvprof 均未使用），完全依靠以下手段：

### 1. CUDA Event 端到端计时

唯一的定量度量。每轮修改后跑一次，看数字变了没有。

```python
start.record()
for _ in range(50): kernel()
end.record()
latency = start.elapsed_time(end) / 50
```

### 2. ptxas -v 寄存器/溢出报告

```
ptxas info: Used 193 registers, used 1 barriers          # R25后，1 block/SM
ptxas info: Used 128 registers, 20B spill stores/loads    # R29 launch_bounds后
ptxas info: Used 118 registers, 0 spill                   # R35 最终版
```

R29 的关键发现——寄存器从 67 跳到 193 导致 occupancy 腰斩，直到主动查看 `--ptxas-options=-v` 才发现。

### 3. 运行时 Layout Probe（自制分析手段）

用 identity matrix × encoded matrix 探测 WMMA fragment 布局：

```cpp
// mC = I × encoded(r*16+c)  →  mC.x[i] 暴露 fragment element i 的 (row, col) 映射
wmma::mma_sync(mC, identity, encoded, mC);
```

R8（fragment O 累加）和 R33（发现 accumulator == matrix_a layout）的关键突破都来自此技巧。

### 4. 纯脑内理论分析

所有架构推理均无工具辅助，包括：
- Bank conflict 分析：`stride=64 halfs=128 bytes=32 banks → 16-way conflict`
- Occupancy 计算：`128 regs × 256 threads = 32768 → 65536/32768 = 2 blocks/SM`
- Smem traffic 估算：`S store 16KB + P store 8KB → ×512 tiles = 12.3 GB total`

---

## 如果用了 ncu 会怎样

以下信息本可以直接从 nsight compute 获取，而不需要猜测：

| 我猜测/推算的 | ncu 能直接给的指标 |
|:-------------|:-----------------|
| "bank conflict 可能是瓶颈" (R24) | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` |
| "S_s smem traffic 很大" (R25) | `l1tex__t_bytes_pipe_lsu_mem_shared_op_ld/st` |
| "寄存器涨到了多少" (R29) | `launch__registers_per_thread` |
| "compute vs memory 谁是瓶颈" | Roofline chart / `sm__throughput` vs `dram__throughput` |
| "expf 调用占多少时间" | SASS instruction mix |
| "tile loop 各阶段时间分布" | Source-correlated SASS profiling |

估计使用 ncu 可以节省 5-8 轮无效尝试。

---

## 优化思路框架

整个过程遵循 `notes/optimization_rules.md` 的循环：

```
while True:
    1. 提出假设（理论分析 / 上轮教训 / 参数扫描）
    2. 实现修改
    3. 验证正确性 (python run.py --mode correctness)
    4. 测量延迟 (CUDA events)
    5. 接受 (latency 下降) / 拒绝 (回退)
    6. 记录到 experiments.csv + perf_log.md
```

### 假设来源分类（按成功率排序）

**A. 理论驱动（成功率最高）**

从架构原理出发推导优化方向，命中率高：

| 轮次 | 假设 | 理论依据 | 结果 |
|:-----|:-----|:--------|:-----|
| R24 | S_s/P_s padding 减少 bank conflict | stride%128==0 导致 16-way conflict | **+8%** |
| R25 | 消除 S_s smem（寄存器内 softmax） | smem round-trip 占热路径主要开销 | **+5%** |
| R33 | 消除 P_s smem（acc==mat_a layout） | 运行时 probe 证实 layout 相同 | **+5.5%** |
| R35 | 硬编码 WMMA layout 省寄存器 | 消除 16 int 寄存器 + probe 开销 | **+1.2%** |

**B. 偶然发现/教训驱动（最大单步收益）**

不是主动分析发现的，而是"突然意识到遗漏了什么"：

| 轮次 | 发现 | 怎么发现的 | 结果 |
|:-----|:-----|:---------|:-----|
| R28 | pack_mask 占 37% 总延迟 | 分开测量 kernel-only vs total | **+45%** |
| R29 | 寄存器从 67 爆炸到 193 | 查看 `ptxas -v` 输出 | **+12.6%** |

**C. 参数扫描（低风险低收益）**

系统性遍历参数空间，稳定但边际递减：

| 轮次 | 扫描内容 | 结果 |
|:-----|:--------|:-----|
| R26-27 | BN_TILE=64→128 | 每轮 +3-5% |
| R34 | BN_TILE=96/128/160/192/256 | 最优 192，+0.9% |
| R30-32 | blocks/SM、NWARPS 扫描 | 全部拒绝 |

---

## 最有价值的方法论经验

### 1. 分开测量各组件

R27 之前一直测的是 `pack_mask + kernel` 总延迟，以为"kernel 是 36.8ms"。
分开测量后才发现 **kernel 只有 23.3ms，pack_mask 占了 13.5ms（37%）**。

**教训：永远分开测量 preprocessing 和 kernel 本身。**

### 2. 每次修改后检查寄存器数

R25 引入 in-register softmax 后寄存器从 ~67 跳到 193，但直到 R29 才发现（中间浪费了 R26-R28 三轮——它们的优化效果全被低 occupancy 削弱了）。

**教训：每次修改后必跑 `--ptxas-options=-v`。**

### 3. 运行时 Probe 比文档靠谱

WMMA fragment layout 在不同架构上不同，NVIDIA 文档也不总是完整。
R33 发现 accumulator 和 matrix_a layout 完全相同——这在任何文档中都没有记载。

**教训：对于硬件 layout，写一个 probe kernel 比读文档更快更准。**

### 4. 反向思维比正向优化更有效

R28 和 R29 都是"突然发现一直忽略的问题"，而非"对已知瓶颈做更深优化"。35 轮中最大的两个 single-round gain 都来自发现**隐藏的瓶颈**，而不是优化已知的瓶颈。

**教训：定期问自己"什么东西我一直没测量？"**

---

## 如果重来一遍的检查清单

```bash
# 每次修改后必做的 4 步：
python run.py --mode correctness              # 1. 正确性
# 分开测 pack_mask / kernel                    # 2. 延迟（分组件）
nvcc --ptxas-options=-v ...                    # 3. 寄存器/溢出

# 如果有 ncu（可选但强烈推荐）：
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained,\   # 计算利用率
  dram__throughput.avg.pct_of_peak_sustained,\ # 显存带宽
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,\ # bank conflict
  launch__occupancy \                          # 实际 occupancy
  python -c "..."                              # 4. 瓶颈定位
```
