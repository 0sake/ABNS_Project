"""
Progressive pruning analysis for Block Influence pipeline.

Two removal strategies:
  1. bi_acc ascending  — remove least accuracy-impactful blocks first
  2. delta=(bi_rep - bi_acc) descending — remove silent-failure blocks first

Plots cumulative acc_lost and rep_lost curves side by side.
Standalone: requires only numpy, matplotlib, json.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_JSON = os.path.join(os.path.dirname(__file__), "results", "phase2_results.json")
OUT_PATH     = os.path.join(os.path.dirname(__file__), "figures", "progressive_pruning.png")


def load_metrics(path: str):
    with open(path) as f:
        data = json.load(f)

    blocks  = list(data["bi_acc"].keys())
    bi_acc  = np.array([data["bi_acc"][b]  for b in blocks])
    bi_rep  = np.array([data["bi_rep"][b]  for b in blocks])
    return blocks, bi_acc, bi_rep


def cumulative_losses(order: np.ndarray, bi_acc: np.ndarray, bi_rep: np.ndarray):
    """Return (acc_cum, rep_cum) of shape (n_blocks,) for the given removal order."""
    acc_cum = np.cumsum(bi_acc[order])
    rep_cum = np.cumsum(bi_rep[order])
    return acc_cum, rep_cum


def plot_strategy(ax, steps, acc_cum, rep_cum, labels, title, color_acc="#e05c5c", color_rep="#5c8ee0"):
    ax.plot(steps, acc_cum, marker="o", markersize=5, linewidth=1.8,
            color=color_acc, label="Cumulative BIacc")
    ax.plot(steps, rep_cum, marker="s", markersize=5, linewidth=1.8,
            color=color_rep, label="Cumulative BIrep", linestyle="--")

    ax.set_xticks(steps)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xlabel("Block removed (left → right = pruning order)", fontsize=9)
    ax.set_ylabel("Cumulative loss", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
    ax.set_ylim(bottom=0)


def main():
    blocks, bi_acc, bi_rep = load_metrics(RESULTS_JSON)
    n = len(blocks)
    steps = np.arange(1, n + 1)

    # Strategy 1: ascending bi_acc (least damaging first)
    order1  = np.argsort(bi_acc)
    labels1 = [blocks[i] for i in order1]
    acc1, rep1 = cumulative_losses(order1, bi_acc, bi_rep)

    # Strategy 2: descending delta = bi_rep - bi_acc (silent failures first)
    delta   = bi_rep - bi_acc
    order2  = np.argsort(-delta)        # descending
    labels2 = [blocks[i] for i in order2]
    acc2, rep2 = cumulative_losses(order2, bi_acc, bi_rep)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Progressive Block Pruning — Cumulative Impact", fontsize=12, fontweight="bold")

    plot_strategy(axes[0], steps, acc1, rep1, labels1,
                  "Strategy 1: BIacc ascending\n(least accuracy-impactful first)")
    plot_strategy(axes[1], steps, acc2, rep2, labels2,
                  "Strategy 2: Δ=(BIrep−BIacc) descending\n(silent-failure blocks first)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
