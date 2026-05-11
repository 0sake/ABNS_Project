import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PHASE4_DIR  = Path(__file__).parent.parent / "phase4_results"
FIGURES_DIR = PHASE4_DIR / "figures"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CONDITIONS = ["bi_acc", "bi_rep", "uniform", "vanilla"]

COND_COLORS = {
    "bi_acc":  "#DD8452",
    "bi_rep":  "#55A868",
    "uniform": "#4C72B0",
    "vanilla": "#C44E52",
}
TEACHER_COLOR = "#8172B2"

COND_LABELS = {
    "bi_acc":  "BI-Acc",
    "bi_rep":  "BI-Rep",
    "uniform": "Uniform",
    "vanilla": "Vanilla",
}

# per_student key prefix → canonical condition name
COND_NORM = {
    "biacc":   "bi_acc",
    "birep":   "bi_rep",
    "uniform": "uniform",
    "vanilla": "vanilla",
}

FONT = {"title": 14, "label": 12, "tick": 10, "annot": 9}
DPI  = 200

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    if not path.exists():
        logging.warning(f"Missing JSON: {path} — skipping related figures")
        return None
    with open(path) as f:
        return json.load(f)


def _save_fig(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {out}")


def _parse_per_student(per_student: dict) -> dict:
    """Group per_student entries by condition name (bi_acc / bi_rep / uniform / vanilla)."""
    grouped = {c: [] for c in CONDITIONS}
    for key, entry in per_student.items():
        prefix = key.rsplit("_seed_", 1)[0]
        cond = COND_NORM.get(prefix)
        if cond:
            grouped[cond].append(entry)
    return grouped


def _cond_mean_std(grouped: dict, metric_fn) -> dict:
    """Return {cond: (mean, std)} by applying metric_fn to each per-student entry."""
    result = {}
    for cond, entries in grouped.items():
        vals = [metric_fn(e) for e in entries] if entries else [0.0]
        result[cond] = (float(np.mean(vals)), float(np.std(vals)))
    return result


# ---------------------------------------------------------------------------
# Figures 1–6  (phase4_fine_analysis.json)
# ---------------------------------------------------------------------------

def fig1_aggregate_bar(fine: dict):
    """Dual-axis grouped bar: cosine_frob (left) and class_cka_var (right) per condition."""
    grouped  = _parse_per_student(fine["per_student"])
    frob_ms  = _cond_mean_std(grouped, lambda e: e["cosine_diff_frob"])
    var_ms   = _cond_mean_std(grouped, lambda e: e["class_cka_variance"])

    x     = np.arange(len(CONDITIONS))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.bar(
        x - width / 2,
        [frob_ms[c][0] for c in CONDITIONS],
        width,
        yerr=[frob_ms[c][1] for c in CONDITIONS],
        color=[COND_COLORS[c] for c in CONDITIONS],
        capsize=4, alpha=0.85, label="Cosine Frob (left axis)",
    )
    ax2.bar(
        x + width / 2,
        [var_ms[c][0] for c in CONDITIONS],
        width,
        yerr=[var_ms[c][1] for c in CONDITIONS],
        color=[COND_COLORS[c] for c in CONDITIONS],
        capsize=4, alpha=0.45, hatch="//", label="Class CKA Var (right axis)",
    )

    ax1.set_xlabel("Condition", fontsize=FONT["label"])
    ax1.set_ylabel("Cosine Frobenius Norm", fontsize=FONT["label"])
    ax2.set_ylabel("Class CKA Variance", fontsize=FONT["label"])
    ax1.set_xticks(x)
    ax1.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=FONT["tick"])
    ax1.set_title("Aggregate Representation Metrics by Condition", fontsize=FONT["title"])

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=FONT["tick"])

    plt.tight_layout()
    _save_fig(fig, "fig1_aggregate_bar.png")


def fig2_per_class_cka(fine: dict):
    """Grouped bar: per-class CKA for each of 10 classes × 4 conditions + teacher line."""
    grouped   = _parse_per_student(fine["per_student"])
    n_classes = len(CIFAR10_CLASSES)
    n_conds   = len(CONDITIONS)
    width     = 0.18
    x         = np.arange(n_classes)

    stats = {}
    for cond, entries in grouped.items():
        stats[cond] = []
        for ci in range(n_classes):
            vals = [e["per_class_cka"][str(ci)] for e in entries]
            stats[cond].append((np.mean(vals), np.std(vals)))

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, cond in enumerate(CONDITIONS):
        offsets = x + (i - n_conds / 2 + 0.5) * width
        means   = [stats[cond][ci][0] for ci in range(n_classes)]
        stds    = [stats[cond][ci][1] for ci in range(n_classes)]
        ax.bar(offsets, means, width, yerr=stds, capsize=3,
               color=COND_COLORS[cond], label=COND_LABELS[cond], alpha=0.85)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.2, label="Teacher (CKA=1.0)")
    ax.set_xlabel("Class", fontsize=FONT["label"])
    ax.set_ylabel("Per-Class CKA", fontsize=FONT["label"])
    ax.set_title("Per-Class CKA by Condition vs. Teacher", fontsize=FONT["title"])
    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=30, ha="right", fontsize=FONT["tick"])
    ax.legend(fontsize=FONT["tick"])
    plt.tight_layout()
    _save_fig(fig, "fig2_per_class_cka.png")


def fig3_critical_pairs(fine: dict):
    """Grouped bar: cosine diff (student − teacher) for 3 confusable pairs × 4 conditions."""
    pairs        = ["cat_dog", "cat_deer", "dog_deer"]
    pair_labels  = ["Cat–Dog", "Cat–Deer", "Dog–Deer"]
    n_pairs      = len(pairs)
    n_conds      = len(CONDITIONS)
    width        = 0.18
    x            = np.arange(n_pairs)

    grouped = _parse_per_student(fine["per_student"])

    stats = {}
    for cond, entries in grouped.items():
        stats[cond] = []
        for p in pairs:
            vals = [e["critical_pairs"][p] for e in entries]
            stats[cond].append((np.mean(vals), np.std(vals)))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, cond in enumerate(CONDITIONS):
        offsets = x + (i - n_conds / 2 + 0.5) * width
        means   = [stats[cond][pi][0] for pi in range(n_pairs)]
        stds    = [stats[cond][pi][1] for pi in range(n_pairs)]
        ax.bar(offsets, means, width, yerr=stds, capsize=4,
               color=COND_COLORS[cond], label=COND_LABELS[cond], alpha=0.85)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Confusable Pair", fontsize=FONT["label"])
    ax.set_ylabel("Cosine Diff (Student − Teacher)", fontsize=FONT["label"])
    ax.set_title("Critical Pair Cosine Similarity Differences", fontsize=FONT["title"])
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=FONT["tick"])
    ax.legend(fontsize=FONT["tick"])
    plt.tight_layout()
    _save_fig(fig, "fig3_critical_pairs.png")


def fig4_cosine_diff_heatmaps(fine: dict):
    """4-panel heatmap: mean cosine_diff matrix (student − teacher) per condition."""
    grouped = _parse_per_student(fine["per_student"])

    matrices = {}
    for cond, entries in grouped.items():
        arrs = [np.array(e["cosine_diff"]) for e in entries]
        matrices[cond] = np.mean(arrs, axis=0) if arrs else np.zeros((10, 10))

    all_vals = np.concatenate([m.ravel() for m in matrices.values()])
    vmax = float(np.max(np.abs(all_vals)))
    vmin = -vmax

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    im = None
    for ax, cond in zip(axes, CONDITIONS):
        mat = matrices[cond]
        im  = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"Cosine Diff: {COND_LABELS[cond]}", fontsize=FONT["title"])
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=FONT["tick"])
        ax.set_yticklabels(CIFAR10_CLASSES, fontsize=FONT["tick"])
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=FONT["annot"] - 1)

    fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.04)
    _save_fig(fig, "fig4_cosine_diff_heatmaps.png")


def fig5_confusion_delta_heatmaps(fine: dict):
    """4-panel heatmap: mean (CM_student − CM_teacher) per condition, diagonal zeroed."""
    grouped = _parse_per_student(fine["per_student"])

    matrices = {}
    for cond, entries in grouped.items():
        deltas = []
        for e in entries:
            cm_t  = np.array(e["confusion_matrix_teacher"], dtype=float)
            cm_s  = np.array(e["confusion_matrix_student"], dtype=float)
            delta = cm_s - cm_t
            np.fill_diagonal(delta, 0.0)
            deltas.append(delta)
        matrices[cond] = np.mean(deltas, axis=0) if deltas else np.zeros((10, 10))

    all_vals = np.concatenate([m.ravel() for m in matrices.values()])
    vmax = float(np.max(np.abs(all_vals)))
    vmin = -vmax

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    im = None
    for ax, cond in zip(axes, CONDITIONS):
        mat = matrices[cond]
        im  = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"ΔConfusion: {COND_LABELS[cond]}", fontsize=FONT["title"])
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=FONT["tick"])
        ax.set_yticklabels(CIFAR10_CLASSES, fontsize=FONT["tick"])
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                        fontsize=FONT["annot"] - 1)

    fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.04)
    _save_fig(fig, "fig5_confusion_delta_heatmaps.png")


def fig6_per_class_cka_violin(fine: dict):
    """Violin plot: 30 points per condition (10 classes × 3 seeds) of per-class CKA."""
    grouped = _parse_per_student(fine["per_student"])

    data = []
    for cond in CONDITIONS:
        pts = []
        for entry in grouped[cond]:
            pts.extend(entry["per_class_cka"][str(ci)] for ci in range(10))
        data.append(pts if pts else [0.0])

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data, positions=range(len(CONDITIONS)),
                          showmedians=True, showextrema=True)

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COND_COLORS[CONDITIONS[i]])
        body.set_alpha(0.7)

    for part_name in ("cmedians", "cmins", "cmaxes", "cbars"):
        if part_name in parts:
            parts[part_name].set_color("black")
            parts[part_name].set_linewidth(1.2)

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=FONT["tick"])
    ax.set_xlabel("Condition", fontsize=FONT["label"])
    ax.set_ylabel("Per-Class CKA", fontsize=FONT["label"])
    ax.set_title("Distribution of Per-Class CKA (10 classes × 3 seeds)", fontsize=FONT["title"])
    plt.tight_layout()
    _save_fig(fig, "fig6_per_class_cka_violin.png")


# ---------------------------------------------------------------------------
# Figures 7–9  (phase4_extended_analysis.json)
# ---------------------------------------------------------------------------

def fig7_intra_class_variance(ext: dict):
    """Bar chart: mean intra-class variance for teacher + 4 conditions."""
    teacher_val = ext["teacher"]["intra_class_variance"]["mean"]
    grouped     = _parse_per_student(ext["per_student"])
    stats       = _cond_mean_std(grouped, lambda e: e["intra_class_variance"]["mean"])

    labels = ["Teacher"] + [COND_LABELS[c] for c in CONDITIONS]
    means  = [teacher_val] + [stats[c][0] for c in CONDITIONS]
    errs   = [0.0]         + [stats[c][1] for c in CONDITIONS]
    colors = [TEACHER_COLOR] + [COND_COLORS[c] for c in CONDITIONS]

    fig, ax = plt.subplots(figsize=(8, 5))
    x    = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=errs, capsize=5, color=colors, alpha=0.85)
    bars[0].set_hatch("//")

    y_pad = max(means) * 0.01
    for xi, (val, err) in enumerate(zip(means, errs)):
        ax.text(xi, val + err + y_pad, f"{val:.4f}",
                ha="center", va="bottom", fontsize=FONT["annot"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT["tick"])
    ax.set_ylabel("Mean Intra-Class Variance", fontsize=FONT["label"])
    ax.set_title("Intra-Class Variance by Condition", fontsize=FONT["title"])
    plt.tight_layout()
    _save_fig(fig, "fig7_intra_class_variance.png")


def fig8_fdr_critical_pairs(ext: dict):
    """Grouped bar: FDR for 3 critical pairs + all-pairs mean, 5 bars each (teacher + 4 conds)."""
    pairs       = ["cat_dog", "cat_deer", "dog_deer"]
    pair_labels = ["Cat–Dog", "Cat–Deer", "Dog–Deer", "All Pairs Mean"]
    n_groups    = 4
    n_bars      = 5
    width       = 0.14
    x           = np.arange(n_groups)

    teacher = ext["teacher"]
    grouped = _parse_per_student(ext["per_student"])

    teacher_vals = (
        [teacher["fdr_critical_pairs"].get(p, 0.0) for p in pairs]
        + [teacher["fdr_mean_all_pairs"]]
    )

    def cond_stats(cond):
        result = []
        for p in pairs:
            vals = [e["fdr_critical_pairs"].get(p, 0.0) for e in grouped[cond]]
            result.append((np.mean(vals), np.std(vals)))
        all_v = [e["fdr_mean_all_pairs"] for e in grouped[cond]]
        result.append((np.mean(all_v), np.std(all_v)))
        return result

    all_bars = (
        [(teacher_vals, [0.0] * n_groups, TEACHER_COLOR, "Teacher", "//")]
        + [([cond_stats(c)[g][0] for g in range(n_groups)],
            [cond_stats(c)[g][1] for g in range(n_groups)],
            COND_COLORS[c], COND_LABELS[c], None)
           for c in CONDITIONS]
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for bi, (means, stds, color, label, hatch) in enumerate(all_bars):
        offsets = x + (bi - n_bars / 2 + 0.5) * width
        b = ax.bar(offsets, means, width, yerr=stds, capsize=3,
                   color=color, alpha=0.85, label=label)
        if hatch:
            for bar in b:
                bar.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=FONT["tick"])
    ax.set_ylabel("FDR Value", fontsize=FONT["label"])
    ax.set_title("Fisher Discriminant Ratio for Critical Pairs", fontsize=FONT["title"])
    ax.legend(fontsize=FONT["tick"])
    plt.tight_layout()
    _save_fig(fig, "fig8_fdr_critical_pairs.png")


def fig9_participation_ratio(ext: dict):
    """Two panels: global PR bar chart (left) + per-class PR grouped bar (right)."""
    teacher_global = ext["teacher"]["participation_ratio"]["global"]
    teacher_pc     = ext["teacher"]["participation_ratio"]["per_class"]

    grouped      = _parse_per_student(ext["per_student"])
    global_stats = _cond_mean_std(grouped, lambda e: e["participation_ratio"]["global"])

    labels_all = ["Teacher"] + [COND_LABELS[c] for c in CONDITIONS]
    means_all  = [teacher_global] + [global_stats[c][0] for c in CONDITIONS]
    errs_all   = [0.0]            + [global_stats[c][1] for c in CONDITIONS]
    colors_all = [TEACHER_COLOR]  + [COND_COLORS[c] for c in CONDITIONS]

    n_classes = len(CIFAR10_CLASSES)
    n_bars_pc = 5
    width_pc  = 0.14
    x_pc      = np.arange(n_classes)

    per_class_stats = {}
    for cond, entries in grouped.items():
        per_class_stats[cond] = []
        for ci in range(n_classes):
            vals = [e["participation_ratio"]["per_class"][str(ci)] for e in entries]
            per_class_stats[cond].append((np.mean(vals), np.std(vals)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Left panel — global PR
    ax = axes[0]
    x_g  = np.arange(len(labels_all))
    bars = ax.bar(x_g, means_all, yerr=errs_all, capsize=5, color=colors_all, alpha=0.85)
    bars[0].set_hatch("//")
    y_pad = max(means_all) * 0.01
    for xi, (val, err) in enumerate(zip(means_all, errs_all)):
        ax.text(xi, val + err + y_pad, f"{val:.1f}",
                ha="center", va="bottom", fontsize=FONT["annot"])
    ax.set_xticks(x_g)
    ax.set_xticklabels(labels_all, fontsize=FONT["tick"])
    ax.set_ylabel("Participation Ratio (global)", fontsize=FONT["label"])
    ax.set_title("Global Participation Ratio by Condition", fontsize=FONT["title"])

    # Right panel — per-class PR
    ax2 = axes[1]
    pc_bars = (
        [([float(teacher_pc[str(ci)]) for ci in range(n_classes)],
          [0.0] * n_classes, TEACHER_COLOR, "Teacher", "//")]
        + [([per_class_stats[c][ci][0] for ci in range(n_classes)],
            [per_class_stats[c][ci][1] for ci in range(n_classes)],
            COND_COLORS[c], COND_LABELS[c], None)
           for c in CONDITIONS]
    )
    for bi, (means_pc, stds_pc, color, label, hatch) in enumerate(pc_bars):
        offsets = x_pc + (bi - n_bars_pc / 2 + 0.5) * width_pc
        b = ax2.bar(offsets, means_pc, width_pc, yerr=stds_pc, capsize=2,
                    color=color, alpha=0.85, label=label)
        if hatch:
            for bar in b:
                bar.set_hatch(hatch)
    ax2.set_xticks(x_pc)
    ax2.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=FONT["tick"])
    ax2.set_ylabel("Participation Ratio (per class)", fontsize=FONT["label"])
    ax2.set_title("Per-Class Participation Ratio", fontsize=FONT["title"])
    ax2.legend(fontsize=FONT["tick"])

    plt.tight_layout()
    _save_fig(fig, "fig9_participation_ratio.png")


# ---------------------------------------------------------------------------
# Figures 10–11  (phase4_confidence_analysis.json)
# ---------------------------------------------------------------------------

def fig10_confidence_entropy(conf: dict):
    """3-panel bar: confidence on correct, entropy overall, calibration gap."""
    teacher = conf["teacher"]
    grouped = _parse_per_student(conf["per_student"])

    panels = [
        ("mean_confidence_correct", "Mean Confidence (Correct)"),
        ("mean_entropy_overall",    "Mean Entropy (Overall)"),
        ("calibration_gap",         "Calibration Gap"),
    ]

    labels_all = ["Teacher"] + [COND_LABELS[c] for c in CONDITIONS]
    colors_all = [TEACHER_COLOR] + [COND_COLORS[c] for c in CONDITIONS]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (key, title) in zip(axes, panels):
        teacher_val = teacher[key]
        stats = _cond_mean_std(grouped, lambda e, k=key: e[k])
        means = [teacher_val] + [stats[c][0] for c in CONDITIONS]
        errs  = [0.0]         + [stats[c][1] for c in CONDITIONS]

        x    = np.arange(len(labels_all))
        bars = ax.bar(x, means, yerr=errs, capsize=5, color=colors_all, alpha=0.85)
        bars[0].set_hatch("//")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_all, fontsize=FONT["tick"])
        ax.set_ylabel(title, fontsize=FONT["label"])
        ax.set_title(title, fontsize=FONT["title"])

    plt.tight_layout()
    _save_fig(fig, "fig10_confidence_entropy.png")


def fig11_logit_margin(conf: dict):
    """Bar chart: mean logit margin, annotated with ratio vs teacher."""
    teacher_val = conf["teacher"]["mean_logit_margin"]
    grouped     = _parse_per_student(conf["per_student"])
    stats       = _cond_mean_std(grouped, lambda e: e["mean_logit_margin"])

    labels_all = ["Teacher"] + [COND_LABELS[c] for c in CONDITIONS]
    means_all  = [teacher_val] + [stats[c][0] for c in CONDITIONS]
    errs_all   = [0.0]         + [stats[c][1] for c in CONDITIONS]
    colors_all = [TEACHER_COLOR] + [COND_COLORS[c] for c in CONDITIONS]

    fig, ax = plt.subplots(figsize=(8, 5))
    x    = np.arange(len(labels_all))
    bars = ax.bar(x, means_all, yerr=errs_all, capsize=5, color=colors_all, alpha=0.85)
    bars[0].set_hatch("//")

    y_pad = max(means_all) * 0.01
    for xi, (val, err) in enumerate(zip(means_all, errs_all)):
        annot = "1.00×" if xi == 0 else f"{val / teacher_val:.2f}×" if teacher_val else "—"
        ax.text(xi, val + err + y_pad, annot,
                ha="center", va="bottom", fontsize=FONT["annot"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels_all, fontsize=FONT["tick"])
    ax.set_ylabel("Mean Logit Margin", fontsize=FONT["label"])
    ax.set_title("Logit Margin by Condition (ratio vs. teacher annotated)", fontsize=FONT["title"])
    plt.tight_layout()
    _save_fig(fig, "fig11_logit_margin.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_figures():
    fine = _load_json(PHASE4_DIR / "phase4_fine_analysis.json")
    ext  = _load_json(PHASE4_DIR / "phase4_extended_analysis.json")
    conf = _load_json(PHASE4_DIR / "phase4_confidence_analysis.json")

    if fine:
        fig1_aggregate_bar(fine)
        fig2_per_class_cka(fine)
        fig3_critical_pairs(fine)
        fig4_cosine_diff_heatmaps(fine)
        fig5_confusion_delta_heatmaps(fine)
        fig6_per_class_cka_violin(fine)
    else:
        logging.warning("Skipping Figs 1–6 (phase4_fine_analysis.json not found)")

    if ext:
        fig7_intra_class_variance(ext)
        fig8_fdr_critical_pairs(ext)
        fig9_participation_ratio(ext)
    else:
        logging.warning("Skipping Figs 7–9 (phase4_extended_analysis.json not found)")

    if conf:
        fig10_confidence_entropy(conf)
        fig11_logit_margin(conf)
    else:
        logging.warning("Skipping Figs 10–11 (phase4_confidence_analysis.json not found)")


if __name__ == "__main__":
    generate_all_figures()
