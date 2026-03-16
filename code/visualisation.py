"""
visualisation.py — Generate all 7 figures specified in Phase 3.

Figures:
    Fig 1  Kendall's tau 3×3 heatmap with p-values
    Fig 2  Grouped bar chart of three BI vectors (sorted by BIacc)
    Fig 3  Jaccard heatmaps for k=3 and k=5
    Fig 4  BIacc vs BIrep scatter (silent failure candidates annotated)
    Fig 5  CKA damage propagation profile for primary candidate
    Fig 6  Per-class CKA disruption bar chart
    Fig 7  Paired entropy violin plot (Hintact vs Hablated on C_l)
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import FIGURES_DIR, PHASE2_RESULTS, PHASE3_CORRELATIONS, PHASE3_JACCARD, RESULTS_DIR, TARGET_BLOCKS

logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

METRIC_LABELS = {"bi_geo": "BIgeo", "bi_acc": "BIacc", "bi_rep": "BIrep"}
COLORS = {"bi_geo": "#4C72B0", "bi_acc": "#DD8452", "bi_rep": "#55A868"}


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# Fig 1 — Kendall's tau heatmap
# ─────────────────────────────────────────────────────────────

def fig1_tau_heatmap(corr: dict) -> None:
    names = ["bi_geo", "bi_acc", "bi_rep"]
    labels = [METRIC_LABELS[n] for n in names]
    n = len(names)
    matrix = np.eye(n)
    p_matrix = np.zeros((n, n))

    pair_map = {
        ("bi_geo", "bi_acc"): "bi_geo_vs_bi_acc",
        ("bi_geo", "bi_rep"): "bi_geo_vs_bi_rep",
        ("bi_acc", "bi_rep"): "bi_acc_vs_bi_rep",
    }

    for i, m1 in enumerate(names):
        for j, m2 in enumerate(names):
            if i == j:
                continue
            key = pair_map.get((m1, m2)) or pair_map.get((m2, m1))
            if key:
                matrix[i, j] = corr[key]["tau"]
                p_matrix[i, j] = corr[key]["p_value"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=11)
    fig.colorbar(im, ax=ax, label="Kendall's τ")
    ax.set_title("Fig 1 — Kendall's τ Rank Correlation Matrix", fontsize=12, pad=10)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            p   = p_matrix[i, j]
            sig = "*" if p < 0.05 else ""
            txt = f"{val:.2f}{sig}" if i != j else "1.00"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    color="black" if abs(val) < 0.7 else "white")

    fig.tight_layout()
    _save(fig, "fig1_tau_heatmap.png")


# ─────────────────────────────────────────────────────────────
# Fig 2 — Grouped bar chart
# ─────────────────────────────────────────────────────────────

def fig2_grouped_bar(bi_geo: dict, bi_acc: dict, bi_rep: dict) -> None:
    # Sort blocks by BIacc
    order = sorted(TARGET_BLOCKS, key=lambda b: bi_acc[b])
    x = np.arange(len(order))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, [bi_geo[b] for b in order], width, label="BIgeo", color=COLORS["bi_geo"], alpha=0.85)
    ax.bar(x,         [bi_acc[b] for b in order], width, label="BIacc", color=COLORS["bi_acc"], alpha=0.85)
    ax.bar(x + width, [bi_rep[b] for b in order], width, label="BIrep", color=COLORS["bi_rep"], alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Block Influence Score")
    ax.set_title("Fig 2 — Block Influence Scores (sorted by BIacc)", fontsize=12)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    fig.tight_layout()
    _save(fig, "fig2_grouped_bar.png")


# ─────────────────────────────────────────────────────────────
# Fig 3 — Jaccard heatmaps
# ─────────────────────────────────────────────────────────────

def fig3_jaccard_heatmaps(jaccard: dict) -> None:
    k_vals = [int(k) for k in jaccard.keys()]
    fig, axes = plt.subplots(1, len(k_vals), figsize=(5 * len(k_vals), 4))
    if len(k_vals) == 1:
        axes = [axes]

    metrics = ["bi_geo", "bi_acc", "bi_rep"]
    labels  = [METRIC_LABELS[m] for m in metrics]
    pair_map = {
        ("bi_geo", "bi_acc"): "bi_geo_vs_bi_acc",
        ("bi_geo", "bi_rep"): "bi_geo_vs_bi_rep",
        ("bi_acc", "bi_rep"): "bi_acc_vs_bi_rep",
    }

    for ax, k in zip(axes, k_vals):
        mat = np.eye(3)
        jd = jaccard[str(k)]
        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                if i == j: continue
                key = pair_map.get((m1, m2)) or pair_map.get((m2, m1))
                if key:
                    mat[i, j] = jd[key]["jaccard"]

        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels)
        ax.set_yticks(range(3)); ax.set_yticklabels(labels)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=11)
        ax.set_title(f"k = {k}")
        fig.colorbar(im, ax=ax, label="Jaccard")

    fig.suptitle("Fig 3 — Top-k Jaccard Similarity", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_jaccard_heatmaps.png")


# ─────────────────────────────────────────────────────────────
# Fig 4 — BIacc vs BIrep scatter
# ─────────────────────────────────────────────────────────────

def fig4_scatter(bi_acc: dict, bi_rep: dict, candidates: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for block in TARGET_BLOCKS:
        color = "crimson" if block in candidates else "#4C72B0"
        size  = 120 if block in candidates else 50
        ax.scatter(bi_acc[block], bi_rep[block], color=color, s=size, zorder=3)
        if block in candidates:
            ax.annotate(block, (bi_acc[block], bi_rep[block]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8, color="crimson")

    ax.set_xlabel("BIacc (accuracy drop)", fontsize=11)
    ax.set_ylabel("BIrep (representational disruption)", fontsize=11)
    ax.set_title("Fig 4 — BIacc vs BIrep (silent failure candidates in red)", fontsize=12)
    ax.axhline(np.percentile(list(bi_rep.values()), 66.7), color="grey", linestyle="--", linewidth=0.8, label="θ_rep")
    ax.axvline(np.percentile(list(bi_acc.values()), 33.3), color="grey", linestyle=":",  linewidth=0.8, label="θ_acc")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "fig4_scatter.png")


# ─────────────────────────────────────────────────────────────
# Fig 5 — CKA propagation profile
# ─────────────────────────────────────────────────────────────

def fig5_propagation_profile(block_name: str, profile: dict) -> None:
    if not profile:
        logger.warning(f"No propagation profile data for {block_name}; skipping Fig 5.")
        return
    stages = list(profile.keys())
    cka_vals = [profile[s] for s in stages]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(stages, cka_vals, marker="o", color="#4C72B0", linewidth=2, markersize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Network Stage", fontsize=11)
    ax.set_ylabel("CKA (intact vs ablated)", fontsize=11)
    ax.set_title(f"Fig 5 — CKA Propagation Profile: {block_name}", fontsize=12)
    ax.axhline(1.0, color="green", linestyle="--", linewidth=0.8, label="No disruption")
    ax.legend()
    fig.tight_layout()
    _save(fig, f"fig5_propagation_{block_name.replace('.','_')}.png")


# ─────────────────────────────────────────────────────────────
# Fig 6 — Per-class CKA bar chart
# ─────────────────────────────────────────────────────────────

def fig6_per_class_cka(block_name: str, per_class_cka: dict) -> None:
    classes = [CIFAR10_CLASSES[int(c)] for c in per_class_cka.keys()]
    cka_vals = list(per_class_cka.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(classes, cka_vals, color="#55A868", alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("CIFAR-10 Class")
    ax.set_ylabel("CKA (intact vs ablated)")
    ax.set_title(f"Fig 6 — Per-Class CKA Disruption: {block_name}", fontsize=12)
    ax.axhline(np.mean(cka_vals), color="crimson", linestyle="--", linewidth=1.2, label=f"Mean={np.mean(cka_vals):.3f}")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, f"fig6_per_class_cka_{block_name.replace('.','_')}.png")


# ─────────────────────────────────────────────────────────────
# Fig 7 — Paired entropy violin plot
# ─────────────────────────────────────────────────────────────

def fig7_entropy_violin(block_name: str, sf_json_path: Path) -> None:
    """
    Reads the saved silent-failure JSON to reconstruct entropy data,
    or if raw values aren't available, produces a summary bar chart.
    """
    with open(sf_json_path) as f:
        sf = json.load(f)

    ent = sf["confidence"]["entropy"]
    # If we don't have raw paired values (not stored), plot summary bars
    labels = ["H_intact", "H_ablated"]
    means  = [ent["mean_H_intact"], ent["mean_H_ablated"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, means, color=["#4C72B0", "#DD8452"], alpha=0.85, width=0.4)
    ax.set_ylabel("Mean Shannon Entropy")
    ax.set_title(f"Fig 7 — Entropy Comparison on C_l: {block_name}\n"
                 f"ΔH_mean={ent['mean_delta_H']:.4f}  "
                 f"Wilcoxon p={ent['wilcoxon_p']:.4f}"
                 f"{'*' if ent['significant'] else ''}", fontsize=11)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    _save(fig, f"fig7_entropy_{block_name.replace('.','_')}.png")


# ─────────────────────────────────────────────────────────────
# Main: generate all figures from saved result files
# ─────────────────────────────────────────────────────────────

def generate_all_figures(phase3_results: dict) -> None:
    """Generate Figs 1–7 from cached result dicts."""
    logger.info("=" * 60)
    logger.info("Generating figures …")

    with open(PHASE2_RESULTS) as f:
        p2 = json.load(f)
    with open(PHASE3_CORRELATIONS) as f:
        corr = json.load(f)
    with open(PHASE3_JACCARD) as f:
        jac = json.load(f)

    bi_geo = p2["bi_geo"]
    bi_acc = p2["bi_acc"]
    bi_rep = p2["bi_rep"]
    bi_rep_ml = p2["bi_rep_multilayer"]

    candidates = (
        phase3_results.get("primary_candidate", []) +
        phase3_results.get("secondary_candidates", [])
    )

    fig1_tau_heatmap(corr)
    fig2_grouped_bar(bi_geo, bi_acc, bi_rep)
    fig3_jaccard_heatmaps(jac)
    fig4_scatter(bi_acc, bi_rep, candidates)

    for block_name in candidates:
        sf_path = RESULTS_DIR / f"phase3_silent_failure_{block_name.replace('.', '_')}.json"
        if sf_path.exists():
            with open(sf_path) as f:
                sf = json.load(f)
            fig5_propagation_profile(
                block_name,
                sf.get("geometry", {}).get("propagation_profile", {})
            )
            fig6_per_class_cka(
                block_name,
                sf.get("geometry", {}).get("per_class_cka", {})
            )
            fig7_entropy_violin(block_name, sf_path)

    logger.info("All figures generated ✓")
