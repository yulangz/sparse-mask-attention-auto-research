#!/usr/bin/env python3
"""
Visualization script for sparse attention optimization experiments.
Generates 4 charts:
  1. Latency per round (bar chart, green=accept, red=reject)
  2. Best-so-far latency trajectory (log scale)
  3. TFLOPS progression for accepted rounds
  4. Backend comparison (bar chart)

Usage:
    python notes/gen_chart.py
"""

import csv
import os
import sys

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    print("matplotlib/numpy not installed. Install via: pip install matplotlib numpy")
    sys.exit(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "experiments.csv")
CHART_PATH = os.path.join(SCRIPT_DIR, "perf_chart.png")

# ── Reference baselines (B=1, H=12, N=16384, D=64, FP16, L20Y) ─────────────
BASELINES = {
    "flash-attn\n(dense)": 2.66,
    "cuDNN SDPA\n(dense)": 11.02,
    "Ours R35\n(total)": 20.82,
    "Ours R35\n(kernel)": 18.84,
    "Triton\n(sparse)": 20.41,
    "PyTorch Ref\n(sparse)": 32.25,
    "FlashInfer\n(sparse)": 71.32,
    "Baseline R0\n(scalar)": 444.39,
}


def load_csv():
    rows = []
    seen = set()
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # skip duplicate round 16 entry
            key = (row["round"], row["optimization"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def main():
    rows = load_csv()
    if not rows:
        print("No data in CSV.")
        return

    # ── Parse data ───────────────────────────────────────────────────────────
    rounds, latencies, accepted_flags, tflops_vals, labels = [], [], [], [], []
    for r in rows:
        try:
            rnd = int(r["round"])
        except ValueError:
            continue
        try:
            lat = float(r["latency_ms"])
        except (ValueError, KeyError):
            lat = None
        acc = r.get("accepted", "").strip().lower() == "yes"
        try:
            tf = float(r["tflops"]) if r.get("tflops") else None
        except ValueError:
            tf = None

        rounds.append(rnd)
        latencies.append(lat)
        accepted_flags.append(acc)
        tflops_vals.append(tf)
        labels.append(r["optimization"][:35])

    # ── Compute best-so-far ──────────────────────────────────────────────────
    best_lat = []
    cur_best = float("inf")
    for lat, acc in zip(latencies, accepted_flags):
        if acc and lat is not None:
            cur_best = min(cur_best, lat)
        best_lat.append(cur_best if cur_best != float("inf") else lat)

    # ── Accepted-only data ───────────────────────────────────────────────────
    acc_rounds = [r for r, a, l in zip(rounds, accepted_flags, latencies) if a and l]
    acc_latencies = [l for a, l in zip(accepted_flags, latencies) if a and l]
    acc_tflops = [t for a, t in zip(accepted_flags, tflops_vals) if a and t]
    acc_tflops_rounds = [
        r for r, a, t in zip(rounds, accepted_flags, tflops_vals) if a and t
    ]

    # ── Create figure with 4 subplots ────────────────────────────────────────
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(
        "Sparse Mask Attention — Optimization Journey\n"
        "GPU: NVIDIA L20Y  |  B=1, H=12, N=16384, D=64, FP16  |  35 rounds",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # ┌─────────────────────────────────────────────────────────────────────────
    # │ Chart 1: Latency per round (bar chart)
    # └─────────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    valid_rounds = [r for r, l in zip(rounds, latencies) if l is not None]
    valid_lats = [l for l in latencies if l is not None]
    valid_colors = [
        "#2ecc71" if a else "#e74c3c"
        for a, l in zip(accepted_flags, latencies)
        if l is not None
    ]
    ax1.bar(
        valid_rounds,
        valid_lats,
        color=valid_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.3,
        width=0.8,
    )

    # Best-so-far line
    best_valid_r = [r for r, b in zip(rounds, best_lat) if b and b != float("inf")]
    best_valid_l = [b for b in best_lat if b and b != float("inf")]
    ax1.plot(
        best_valid_r,
        best_valid_l,
        "b-",
        linewidth=2,
        alpha=0.7,
        label="Best accepted",
        marker=".",
        markersize=3,
    )

    ax1.set_xlabel("Round", fontsize=11)
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.set_title("Latency per Round", fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=15, top=500)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))

    # Add phase annotations
    ax1.axvspan(-0.5, 9.5, alpha=0.05, color="blue", label="Phase 1")
    ax1.axvspan(9.5, 23.5, alpha=0.05, color="orange")
    ax1.axvspan(23.5, 29.5, alpha=0.05, color="green")
    ax1.axvspan(29.5, 35.5, alpha=0.05, color="purple")
    ax1.text(4, 420, "Phase 1\nFoundations", fontsize=7, ha="center", color="blue")
    ax1.text(16, 420, "Phase 2\nMemory", fontsize=7, ha="center", color="orange")
    ax1.text(26, 420, "Phase 3\nSmem Elim", fontsize=7, ha="center", color="green")
    ax1.text(33, 420, "Phase 4\nReg P@V", fontsize=7, ha="center", color="purple")

    # ┌─────────────────────────────────────────────────────────────────────────
    # │ Chart 2: Best-so-far trajectory + baselines
    # └─────────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(
        acc_rounds,
        acc_latencies,
        "o-",
        color="#2c3e50",
        linewidth=2,
        markersize=5,
        label="Ours (total, accepted)",
    )

    # Reference lines
    ax2.axhline(
        y=20.41,
        color="#e67e22",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Triton (20.4 ms)",
    )
    ax2.axhline(
        y=32.25,
        color="#95a5a6",
        linestyle=":",
        linewidth=1,
        alpha=0.6,
        label="PyTorch Ref (32.3 ms)",
    )

    ax2.set_xlabel("Round (accepted only)", fontsize=11)
    ax2.set_ylabel("Latency (ms)", fontsize=11)
    ax2.set_title(
        "Optimization Trajectory vs Baselines", fontsize=12, fontweight="bold"
    )
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=15, top=max(acc_latencies) * 1.1 if acc_latencies else 500)

    # Annotate key milestones
    milestones = {
        8: "R8: Fragment O_acc\n(2.6× gain)",
        28: "R28: Fast\npack_mask",
        29: "R29: Occupancy\nfix",
        33: "R33: Elim P_s\n(beats Triton)",
    }
    for r_ms, txt in milestones.items():
        lat_ms = next(
            (l for rr, l in zip(acc_rounds, acc_latencies) if rr == r_ms), None
        )
        if lat_ms:
            ax2.annotate(
                txt,
                xy=(r_ms, lat_ms),
                fontsize=7,
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8),
            )

    # ┌─────────────────────────────────────────────────────────────────────────
    # │ Chart 3: TFLOPS progression
    # └─────────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.fill_between(acc_tflops_rounds, acc_tflops, alpha=0.3, color="#3498db")
    ax3.plot(
        acc_tflops_rounds,
        acc_tflops,
        "s-",
        color="#2980b9",
        linewidth=2,
        markersize=5,
        label="Ours (total TFLOPS)",
    )
    ax3.axhline(
        y=40.4,
        color="#e67e22",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Triton (40.4 TFLOPS)",
    )
    ax3.axhline(
        y=114.9,
        color="#c0392b",
        linestyle=":",
        linewidth=1,
        alpha=0.4,
        label="Peak FP16 TC (114.9)",
    )

    ax3.set_xlabel("Round (accepted only)", fontsize=11)
    ax3.set_ylabel("TFLOPS", fontsize=11)
    ax3.set_title("Throughput Progression", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_ylim(bottom=0, top=50)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(5))

    # ┌─────────────────────────────────────────────────────────────────────────
    # │ Chart 4: Backend comparison (horizontal bar)
    # └─────────────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)

    names = list(BASELINES.keys())
    vals = list(BASELINES.values())
    colors = []
    for n in names:
        if "Ours" in n and "kernel" in n:
            colors.append("#27ae60")
        elif "Ours" in n:
            colors.append("#2ecc71")
        elif "dense" in n:
            colors.append("#bdc3c7")
        elif "Triton" in n:
            colors.append("#e67e22")
        elif "Baseline" in n:
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    y_pos = range(len(names))
    bars = ax4.barh(
        y_pos,
        vals,
        color=colors,
        edgecolor="black",
        linewidth=0.3,
        height=0.6,
        alpha=0.85,
    )

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(names, fontsize=9)
    ax4.set_xlabel("Latency (ms) — lower is better", fontsize=11)
    ax4.set_title("Backend Comparison (B=1, N=16384)", fontsize=12, fontweight="bold")
    ax4.set_xscale("log")
    ax4.set_xlim(1, 600)
    ax4.grid(axis="x", alpha=0.3)
    ax4.invert_yaxis()

    # Value labels on bars
    for bar, val in zip(bars, vals):
        ax4.text(
            val * 1.15,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} ms",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()
