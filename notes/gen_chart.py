#!/usr/bin/env python3
"""
Visualization script for sparse attention optimization experiments.
Reads notes/experiments.csv and generates performance charts.
"""

import csv
import os
import sys

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("matplotlib not installed, skipping chart generation")
    sys.exit(0)

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.csv")
CHART_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_chart.png")


def load_csv():
    rows = []
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    rows = load_csv()
    if not rows:
        print("No data in CSV yet.")
        return

    rounds = []
    latencies = []
    labels = []
    colors = []

    for r in rows:
        rnd = int(r["round"])
        lat = float(r["latency_ms"])
        accepted = r["accepted"].strip().lower() == "yes"
        rounds.append(rnd)
        latencies.append(lat)
        labels.append(r["optimization"][:30])
        colors.append("green" if accepted else "red")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]}
    )

    # --- Top: Latency over rounds ---
    ax1.bar(
        rounds, latencies, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5
    )

    # Best-so-far line
    best = []
    cur_best = float("inf")
    for lat, acc, col in zip(
        latencies, [r["accepted"].strip().lower() for r in rows], colors
    ):
        if acc == "yes":
            cur_best = min(cur_best, lat)
        best.append(cur_best if cur_best != float("inf") else lat)
    ax1.plot(
        rounds,
        best,
        "b--",
        linewidth=2,
        label="Best accepted",
        marker="o",
        markersize=4,
    )

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Sparse Attention Optimization — Latency per Round")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # --- Bottom: Speedup relative to baseline ---
    if latencies:
        baseline = latencies[0]
        speedups = [baseline / lat if lat > 0 else 0 for lat in best]
        ax2.plot(rounds, speedups, "g-", linewidth=2, marker="s", markersize=4)
        ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Speedup vs Baseline")
        ax2.set_title("Cumulative Speedup")
        ax2.grid(axis="y", alpha=0.3)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    print(f"Chart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()
