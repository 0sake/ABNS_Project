"""
visualisation.py — Generate all figures for Phase 3 and extended BIrep analysis.

Phase 3 figures (original):
    Fig 1  Kendall's tau 3x3 heatmap with p-values
    Fig 2  Grouped bar chart of three BI vectors (sorted by BIacc)
    Fig 3  Jaccard heatmaps for k=3 and k=5
    Fig 4  BIacc vs BIrep scatter (silent failure candidates annotated)
    Fig 5  CKA damage propagation profile for primary candidate
    Fig 6  Per-class CKA disruption bar chart
    Fig 7  Paired entropy violin plot (Hintact vs Hablated on C_l)

Extended BIrep figures (bi_rep_extended.py):
    Fig 8  BIrep vs BIrep_gram scatter - mathematical equivalence validation
    Fig 9  BIrep_class bar chart for all 16 blocks
    Fig 10 10x10 class cosine heatmaps: intact | ablated | difference
    Fig 11 Top-5 class-pair cosine similarity changes (before/after bar chart)

Progressive pruning (phase3_analysis.py):
    Fig 12 Progressive pruning — 3-panel cumulative BIacc/BIrep curves
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import FIGURES_DIR, PHASE2_RESULTS, PHASE3_CORRELATIONS, PHASE3_JACCARD, PHASE3_PER_CLASS_CKA, RESULTS_DIR, TARGET_BLOCKS

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
    ax.set_title(f"Fig 7 — Entropy Comparison on C_l (correctly classified samples): {block_name}\n"
                 f"ΔH_mean={ent['mean_delta_H']:.4f}  "
                 f"Wilcoxon p={ent['wilcoxon_p']:.4f}"
                 f"{'*' if ent['significant'] else ''}", fontsize=11)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    _save(fig, f"fig7_entropy_{block_name.replace('.','_')}.png")


# ─────────────────────────────────────────────────────────────
# Fig 8 — BIrep vs BIrep_gram scatter (validation)
# ─────────────────────────────────────────────────────────────

def fig8_birep_vs_gram(bi_rep: dict, bi_rep_gram: dict) -> None:
    """
    Scatter plot of BIrep (Linear CKA on NxD) vs BIrep_gram (Gram CKA on NxN).

    For linear kernels the two formulations are mathematically equivalent,
    so points should fall on the y=x diagonal. Deviations reveal numerical
    differences between implementations. This serves as a methodological
    validation figure.
    """
    vals_rep  = [bi_rep[b]      for b in TARGET_BLOCKS]
    vals_gram = [bi_rep_gram[b] for b in TARGET_BLOCKS]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Reference diagonal
    lim = max(max(vals_rep), max(vals_gram)) * 1.05
    ax.plot([0, lim], [0, lim], color="#bbbbbb", linewidth=1.2,
            linestyle="--", label="y = x (perfect equivalence)", zorder=1)

    scatter = ax.scatter(vals_rep, vals_gram, c=vals_rep,
                         cmap="viridis", s=90, zorder=3, edgecolors="white", linewidths=0.5)

    # Annotate outliers (largest deviation from diagonal)
    for b in TARGET_BLOCKS:
        dev = abs(bi_rep[b] - bi_rep_gram[b])
        if dev > 0.001 or bi_rep[b] > 0.1:
            ax.annotate(b, (bi_rep[b], bi_rep_gram[b]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color="#333333")

    fig.colorbar(scatter, ax=ax, label="BIrep value")
    ax.set_xlabel("BIrep  (Linear CKA on N×D matrix)", fontsize=11)
    ax.set_ylabel("BIrep_gram  (Gram CKA on N×N matrix)", fontsize=11)
    ax.set_title(
        "Fig 8 — BIrep vs BIrep_gram\n"
        "Mathematical equivalence for linear kernels", fontsize=11
    )
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "fig8_birep_vs_gram.png")


# ─────────────────────────────────────────────────────────────
# Fig 9 — BIrep_class bar chart for all 16 blocks
# ─────────────────────────────────────────────────────────────

def fig9_birep_class_bar(bi_acc: dict, bi_rep: dict, bi_rep_class: dict) -> None:
    """
    Bar chart of BIrep_class (class-level cosine similarity disruption)
    alongside BIrep, sorted by BIacc.

    The scale difference between BIrep (~0.35 for transition blocks) and
    BIrep_class (~0.03) is the key visual finding: the class centroid
    structure is much more robust to ablation than instance-level geometry.
    """
    order = sorted(TARGET_BLOCKS, key=lambda b: bi_acc[b])
    x = np.arange(len(order))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    # BIrep on primary axis
    bars1 = ax1.bar(x - width/2, [bi_rep[b] for b in order],
                    width, label="BIrep (instance)", color="#55A868", alpha=0.85)
    # BIrep_class on secondary axis (different scale)
    bars2 = ax2.bar(x + width/2, [bi_rep_class[b] for b in order],
                    width, label="BIrep_class (class)", color="#C44E52", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("BIrep — instance-level disruption", color="#55A868", fontsize=10)
    ax2.set_ylabel("BIrep_class — class-level disruption", color="#C44E52", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#55A868")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    ax1.set_title(
        "Fig 9 — BIrep vs BIrep_class (sorted by BIacc)\n"
        "Note: dual y-axis — class-level disruption is ~10× smaller", fontsize=11
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig9_birep_class_bar.png")


# ─────────────────────────────────────────────────────────────
# Fig 10 — 10×10 class cosine heatmaps (intact | ablated | diff)
# ─────────────────────────────────────────────────────────────

def fig10_class_heatmaps(
    s_intact: list,
    bi_rep_class: dict,
    block_name: str,
    model,
    registry: dict,
    calib_loader,
    device,
    labels_intact,
) -> None:
    """
    Three-panel figure:
        Left  — S_intact (10×10 cosine similarity matrix of intact model)
        Center — S_ablated (same, after ablating block_name)
        Right  — S_ablated - S_intact (difference, diverging colormap)

    This is the most interpretable figure of the extended analysis:
    the difference panel shows exactly which class pairs became more/less
    similar after ablation.
    """
    from bi_rep_extended import class_cosine_matrix
    from utils import ablated_block
    from config import DOWNSAMPLING_BLOCKS

    S_int = np.array(s_intact)

    # Re-extract ablated class matrix for block_name
    block = registry[block_name]
    is_ds = block_name in DOWNSAMPLING_BLOCKS
    import torch
    with torch.no_grad():
        from utils import ActivationCapture, gap
        feats = []
        with ablated_block(block, is_downsampling=is_ds):
            with ActivationCapture(model.avgpool) as cap:
                for images, _ in calib_loader:
                    images = images.to(device)
                    _ = model(images)
                    feats.append(gap(cap.output).cpu())
        F_abl = torch.cat(feats, dim=0)

    S_abl_t = class_cosine_matrix(F_abl.to(device), labels_intact.to(device))
    S_abl   = S_abl_t.cpu().numpy()
    S_diff  = S_abl - S_int

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = CIFAR10_CLASSES

    panels = [
        (S_int,  "S_intact\n(Intact model)",   "Blues",   0.0, 1.0),
        (S_abl,  f"S_ablated\n({block_name} removed)", "Blues", 0.0, 1.0),
        (S_diff, "Difference\n(Ablated − Intact)", "RdBu_r", -0.3, 0.3),
    ]

    for ax, (mat, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(10)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(10)); ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10, pad=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate each cell with value
        for i in range(10):
            for j in range(10):
                val = mat[i, j]
                color = "white" if abs(val) > (0.6 if cmap != "RdBu_r" else 0.2) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    fig.suptitle(
        f"Fig 10 — Class×Class Cosine Similarity: Intact vs {block_name} Ablated",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    _save(fig, f"fig10_class_heatmaps_{block_name.replace('.','_')}.png")


# ─────────────────────────────────────────────────────────────
# Fig 11 — Top-5 class-pair changes (before/after bar chart)
# ─────────────────────────────────────────────────────────────

def fig11_class_pair_changes(
    s_intact: list,
    model,
    registry: dict,
    calib_loader,
    device,
    labels_intact,
    block_name: str,
    top_k: int = 5,
) -> None:
    """
    Horizontal grouped bar chart showing the top-k class pairs whose
    cosine similarity changed most after ablating block_name.

    For each pair: two bars side by side (before in blue, after in orange).
    The delta is annotated on each bar pair.

    This is the most directly citable figure for the silent failure argument:
    "cat and dog became 25% more similar after removing layer4.0."
    """
    from bi_rep_extended import class_cosine_matrix
    from utils import ablated_block, ActivationCapture, gap
    from config import DOWNSAMPLING_BLOCKS
    import torch

    S_int = np.array(s_intact)

    # Re-extract ablated representations
    block = registry[block_name]
    is_ds = block_name in DOWNSAMPLING_BLOCKS
    with torch.no_grad():
        feats = []
        with ablated_block(block, is_downsampling=is_ds):
            with ActivationCapture(model.avgpool) as cap:
                for images, _ in calib_loader:
                    images = images.to(device)
                    _ = model(images)
                    feats.append(gap(cap.output).cpu())
        F_abl = torch.cat(feats, dim=0)

    S_abl = class_cosine_matrix(F_abl.to(device), labels_intact.to(device)).cpu().numpy()

    # Collect all unique off-diagonal pairs and their deltas
    pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            before = float(S_int[i, j])
            after  = float(S_abl[i, j])
            delta  = after - before
            pairs.append((abs(delta), delta, before, after,
                          CIFAR10_CLASSES[i], CIFAR10_CLASSES[j]))
    pairs.sort(reverse=True)
    top = pairs[:top_k]

    # Build chart
    pair_labels = [f"{p[4]} ↔ {p[5]}" for p in top]
    befores = [p[2] for p in top]
    afters  = [p[3] for p in top]
    deltas  = [p[1] for p in top]

    y = np.arange(top_k)
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_b = ax.barh(y + height/2, befores, height,
                     label="Before (intact)",  color="#4C72B0", alpha=0.85)
    bars_a = ax.barh(y - height/2, afters,  height,
                     label="After (ablated)",  color="#DD8452", alpha=0.85)

    # Delta annotations
    for i, (bef, aft, d) in enumerate(zip(befores, afters, deltas)):
        sign = "▲" if d > 0 else "▼"
        color = "#c0392b" if d > 0 else "#1a6b3a"
        ax.text(max(bef, aft) + 0.01, i, f"{sign} {abs(d):.3f}",
                va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(pair_labels, fontsize=10)
    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_xlim(0, 1.12)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(
        f"Fig 11 — Top-{top_k} Class-Pair Similarity Changes: {block_name} ablated\n"
        "All pairs become MORE similar (▲) — class separation degrades silently",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.invert_yaxis()   # largest delta at top
    fig.tight_layout()
    _save(fig, f"fig11_class_pair_changes_{block_name.replace('.','_')}.png")


# ─────────────────────────────────────────────────────────────
# Fig 12 — Progressive pruning (3-panel cumulative curves)
# ─────────────────────────────────────────────────────────────

def fig12_progressive_pruning(pruning: dict) -> None:
    """
    Three-panel figure showing cumulative BIacc and BIrep as blocks are
    progressively removed under each strategy.

    Each panel plots two lines (BIacc and BIrep) against the pruning step,
    with block names on the x-axis in removal order.
    """
    strategies = [
        ("strategy1", "Strategy 1: BIacc ascending\n(least accuracy-impactful first)"),
        ("strategy2", "Strategy 2: Δ=(BIrep−BIacc) descending\n(silent-failure blocks first)"),
        ("strategy3", "Strategy 3: BIrep ascending\n(least representation-impactful first)"),
    ]

    steps = list(range(1, len(TARGET_BLOCKS) + 1))
    color_acc = "#e05c5c"
    color_rep = "#5c8ee0"

    fig, axes = plt.subplots(1, 3, figsize=(21, 5), constrained_layout=True)
    fig.suptitle("Fig 12 — Progressive Block Pruning — Cumulative Impact",
                 fontsize=12, fontweight="bold")

    for ax, (key, title) in zip(axes, strategies):
        order   = pruning[f"{key}_order"]
        acc_cum = pruning[f"{key}_cumulative_biacc"]
        rep_cum = pruning[f"{key}_cumulative_birep"]

        ax.plot(steps, acc_cum, marker="o", markersize=5, linewidth=1.8,
                color=color_acc, label="Cumulative BIacc")
        ax.plot(steps, rep_cum, marker="s", markersize=5, linewidth=1.8,
                color=color_rep, label="Cumulative BIrep", linestyle="--")

        ax.set_xticks(steps)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Block removed (left → right = pruning order)", fontsize=9)
        ax.set_ylabel("Cumulative loss", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
        ax.set_ylim(bottom=0)

    _save(fig, "fig12_progressive_pruning.png")


# ─────────────────────────────────────────────────────────────
# Fig 13 — Per-class CKA heatmap (all blocks × all classes)
# ─────────────────────────────────────────────────────────────

def fig13_per_class_cka_heatmap(per_class_cka: dict) -> None:
    """16×10 heatmap of per-class linear CKA (intact vs ablated).

    Rows = all 16 TARGET_BLOCKS (top to bottom).
    Columns = 10 CIFAR-10 classes.
    Blocks are visually grouped by ResNet stage with horizontal separators.
    """
    n_blocks = len(TARGET_BLOCKS)
    n_classes = len(CIFAR10_CLASSES)

    data = np.zeros((n_blocks, n_classes))
    for i, block_name in enumerate(TARGET_BLOCKS):
        for j in range(n_classes):
            data[i, j] = per_class_cka.get(block_name, {}).get(j, 0.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Linear CKA")

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels(TARGET_BLOCKS, fontsize=8)

    # Stage separators
    for sep_y in (2.5, 6.5, 12.5):
        ax.axhline(sep_y, color="white", linewidth=2)

    # Stage labels on the left margin
    stage_centres = [(1, "Stage 1"), (4.5, "Stage 2"), (9.5, "Stage 3"), (14, "Stage 4")]
    for y_centre, label in stage_centres:
        ax.text(-1.6, y_centre, label, va="center", ha="right",
                fontsize=8, fontweight="bold", color="#333333")

    ax.set_title("Fig 13 — Per-Class Linear CKA (Intact vs Ablated) — All Blocks",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, "fig13_per_class_cka_heatmap.png")


# ─────────────────────────────────────────────────────────────
# Fig 14 — Per-class CKA bar charts for top-5 blocks (multi-panel)
# ─────────────────────────────────────────────────────────────

def fig14_top5_per_class_bar(per_class_cka: dict) -> None:
    """Multi-panel bar chart: per-class CKA for the top-5 blocks by BIrep.

    Fixed block order (top-5 by BIrep influence):
        layer1.0, layer3.0, layer2.0, layer4.0, layer1.1
    Each panel replicates the Fig 6 bar-chart style for one block.
    """
    top5 = ["layer1.0", "layer3.0", "layer2.0", "layer4.0", "layer1.1"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True,
                             constrained_layout=True)
    fig.suptitle("Fig 14 — Per-Class CKA — Top-5 Blocks by BIrep",
                 fontsize=13, fontweight="bold")

    for ax, block_name in zip(axes, top5):
        scores = [per_class_cka.get(block_name, {}).get(j, 0.0)
                  for j in range(len(CIFAR10_CLASSES))]
        variance = float(np.var(scores))
        ax.bar(range(len(CIFAR10_CLASSES)), scores, color="#4C72B0")
        ax.set_xticks(range(len(CIFAR10_CLASSES)))
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{block_name}\nvar={variance:.4f}", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Linear CKA", fontsize=10)
    _save(fig, "fig14_top5_per_class_bar.png")


# ─────────────────────────────────────────────────────────────
# Fig 15 — Real vs Simulated Progressive Pruning (2×2 grid)
# ─────────────────────────────────────────────────────────────

def plot_progressive_pruning_real(pruning_real_results: dict) -> None:
    """
    2×2 figure comparing real and simulated progressive pruning under two strategies.

    Rows  : strategy1 (BIacc ascending) | strategy3 (BIrep ascending)
    Col 0 : Accuracy & Representation
              left  y-axis — BIrep_k (real, blue dashed)
                           — simulated cumulative BIacc normalized (red dotted)
                           — simulated cumulative BIrep normalized (blue dotted)
              right y-axis — acc_k (real, red solid)
    Col 1 : Confidence & Entropy
              conf_mean (orange solid)
              H_Cl      (purple solid — entropy on correctly predicted samples)
              H_mean    (gray dashed  — entropy on all samples)

    x-ticks: block name added at each pruning step.
    """
    strategy_labels = {
        "strategy1": "Strategy 1: BIacc ascending\n(least accuracy-impactful first)",
        "strategy3": "Strategy 3: BIrep ascending\n(least representation-impactful first)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("Fig 15 — Real vs Simulated Progressive Pruning",
                 fontsize=13, fontweight="bold")

    for row_idx, strategy_key in enumerate(("strategy1", "strategy3")):
        if strategy_key not in pruning_real_results:
            continue

        strat = pruning_real_results[strategy_key]
        order     = strat["order"]
        real      = strat["real"]
        simulated = strat["simulated"]

        n = len(order)
        ks = list(range(1, n + 1))

        acc_vals   = [real[f"k{k}"]["acc"]       for k in ks]
        birep_vals = [real[f"k{k}"]["BIrep"]      for k in ks]
        conf_vals  = [real[f"k{k}"]["conf_mean"]  for k in ks]
        H_vals     = [real[f"k{k}"]["H_mean"]     for k in ks]
        HCl_vals   = [real[f"k{k}"]["H_Cl"]       for k in ks]

        sim_bacc = [simulated[f"k{k}"]["cumulative_biacc"] for k in ks]
        sim_brep = [simulated[f"k{k}"]["cumulative_birep"] for k in ks]

        # Normalize simulated curves to their final value for comparability
        norm_bacc = sim_bacc[-1] if sim_bacc[-1] != 0 else 1.0
        norm_brep = sim_brep[-1] if sim_brep[-1] != 0 else 1.0
        sim_bacc_norm = [1.0 - v / norm_bacc for v in sim_bacc]  # reversed: cumulative loss 0→1 becomes 1→0
        sim_brep_norm = [v / norm_brep for v in sim_brep]

        # ── Panel [row, 0]: Accuracy & Representation ──────────
        ax0 = axes[row_idx, 0]
        ax0_r = ax0.twinx()   # right y-axis for accuracy

        ln1, = ax0.plot(ks, birep_vals,     color="#5c8ee0", linestyle="--",
                        linewidth=1.8, marker="s", markersize=4, label="BIrep (real)")
        ln2, = ax0.plot(ks, sim_bacc_norm,  color="#e05c5c", linestyle=":",
                        linewidth=1.5, label="Sim BIacc (norm)")
        ln3, = ax0.plot(ks, sim_brep_norm,  color="#5c8ee0", linestyle=":",
                        linewidth=1.5, label="Sim BIrep (norm)")
        ln4, = ax0_r.plot(ks, acc_vals,     color="#e05c5c", linestyle="-",
                          linewidth=2.0, marker="o", markersize=4, label="Acc (real)")

        ax0.set_ylabel("BIrep / Normalized Simulated", fontsize=9)
        ax0_r.set_ylabel("Accuracy", fontsize=9, color="#e05c5c")
        ax0_r.tick_params(axis="y", labelcolor="#e05c5c")
        ax0_r.set_ylim(0, 1.05)
        ax0.set_ylim(bottom=0)

        ax0.set_xticks(ks)
        ax0.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax0.set_xlabel("Block removed (left → right)", fontsize=8)
        ax0.set_title(
            f"{strategy_labels[strategy_key]}\nAccuracy & Representation",
            fontsize=9, fontweight="bold"
        )
        ax0.grid(True, linestyle=":", alpha=0.5)
        lines = [ln1, ln2, ln3, ln4]
        ax0.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="upper left")

        # ── Panel [row, 1]: Confidence & Entropy ───────────────
        ax1 = axes[row_idx, 1]

        ax1.plot(ks, conf_vals, color="orange",  linestyle="-",  linewidth=1.8,
                 marker="o", markersize=4, label="Confidence (mean top-1)")
        ax1.plot(ks, HCl_vals,  color="purple",  linestyle="-",  linewidth=1.8,
                 marker="^", markersize=4, label="H_Cl (entropy, correct samples)")
        ax1.plot(ks, H_vals,    color="gray",    linestyle="--", linewidth=1.5,
                 label="H_mean (entropy, all samples)")

        ax1.set_xticks(ks)
        ax1.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax1.set_xlabel("Block removed (left → right)", fontsize=8)
        ax1.set_ylabel("Value (nats / probability)", fontsize=9)
        ax1.set_title(
            f"{strategy_labels[strategy_key]}\nConfidence & Entropy",
            fontsize=9, fontweight="bold"
        )
        ax1.legend(fontsize=7)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.set_ylim(bottom=0)

    _save(fig, "fig_progressive_pruning_real.png")


# ─────────────────────────────────────────────────────────────
# Main: generate all figures from saved result files
# ─────────────────────────────────────────────────────────────

def generate_all_figures(phase3_results: dict, model=None, registry=None,
                         calib_loader=None, device=None,
                         pruning_real_results=None) -> None:
    """Generate Figs 1–15 from cached result dicts."""
    logger.info("=" * 60)
    logger.info("Generating figures …")

    with open(PHASE2_RESULTS) as f:
        p2 = json.load(f)
    with open(PHASE3_CORRELATIONS) as f:
        corr = json.load(f)
    with open(PHASE3_JACCARD) as f:
        jac = json.load(f)

    per_class_cka: dict = {}
    if PHASE3_PER_CLASS_CKA.exists():
        with open(PHASE3_PER_CLASS_CKA) as f:
            raw = json.load(f)
        per_class_cka = {
            block: {int(k): v for k, v in cls_dict.items()}
            for block, cls_dict in raw.items()
        }

    bi_geo       = p2["bi_geo"]
    bi_acc       = p2["bi_acc"]
    bi_rep       = p2["bi_rep"]
    bi_rep_ml    = p2["bi_rep_multilayer"]
    bi_rep_gram  = p2.get("bi_rep_gram", {})
    bi_rep_class = p2.get("bi_rep_class", {})
    s_intact     = p2.get("s_intact_10x10", None)

    candidates = (
        phase3_results.get("primary_candidate", []) +
        phase3_results.get("secondary_candidates", [])
    )

    # ── Figs 1–7 (original) ──────────────────────────────────
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

    # ── Figs 8–11 (extended BIrep) ────────────────────────────
    if bi_rep_gram and bi_rep_class and s_intact is not None:
        logger.info("Generating extended BIrep figures (8–11) …")
        fig8_birep_vs_gram(bi_rep, bi_rep_gram)
        fig9_birep_class_bar(bi_acc, bi_rep, bi_rep_class)

        # Figs 10–11 need the model to re-extract ablated representations
        if model is not None and registry is not None and calib_loader is not None:
            import torch
            # Collect calibration labels once
            all_labels = []
            for _, lbl in calib_loader:
                all_labels.append(lbl)
            labels_intact = torch.cat(all_labels, dim=0).to(device)

            # Use primary silent failure candidate
            primary = candidates[0] if candidates else max(
                TARGET_BLOCKS,
                key=lambda b: bi_rep.get(b, 0) - bi_acc.get(b, 0)
            )
            fig10_class_heatmaps(
                s_intact, bi_rep_class, primary,
                model, registry, calib_loader, device, labels_intact
            )
            fig11_class_pair_changes(
                s_intact, model, registry, calib_loader,
                device, labels_intact, primary
            )
        else:
            logger.warning(
                "Figs 10–11 skipped: model/registry/calib_loader not provided. "
                "Call generate_all_figures(..., model=model, registry=registry, "
                "calib_loader=calib_loader, device=device) to generate them."
            )
    else:
        logger.info(
            "Extended BIrep figures (8–11) skipped: "
            "bi_rep_gram / bi_rep_class / s_intact_10x10 not found in phase2_results.json. "
            "Run bi_rep_extended first."
        )

    # ── Fig 12 (progressive pruning) ──────────────────────────
    pruning = phase3_results.get("pruning")
    if pruning:
        fig12_progressive_pruning(pruning)
    else:
        logger.info("Fig 12 skipped: no pruning data in phase3_results.")

    # ── Figs 13–14 (per-class CKA) ───────────────────────────
    if per_class_cka:
        fig13_per_class_cka_heatmap(per_class_cka)
        fig14_top5_per_class_bar(per_class_cka)
    else:
        logger.info("Figs 13–14 skipped: per-class CKA data not found.")

    # ── Fig 15 (real progressive pruning) ────────────────────
    if pruning_real_results:
        plot_progressive_pruning_real(pruning_real_results)
    else:
        logger.info("Fig 15 skipped: no real progressive pruning data provided.")

    logger.info("All figures generated ✓")
