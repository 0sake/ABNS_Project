"""
phase4_cifar10c_visualisation.py — 9-figure visualisation suite for CIFAR-10-C/P results.

Reads phase4_results/phase4_cifar10c_results.json produced by phase4_cifar10c_eval.py
and saves 9 PNG figures to phase4_results/figures/:

  figc1_mce_bar.png                — mCE mean ± std per condition
  figc2_rmce_bar.png               — Relative mCE mean ± std per condition
  figc3_severity_curves.png        — Accuracy vs severity, averaged over all corruptions
  figc4_category_severity_curves.png — 4-panel severity curves per corruption category
  figc5_corruption_heatmap.png     — 19 corruptions × 5 conditions accuracy heatmap
  figc6_ece_comparison.png         — ECE: clean vs corrupted per condition
  figc7_mfp_bar.png                — mFP (mean flip probability) per condition
  figc8_fpr_per_perturbation.png   — FPR grouped by perturbation type
  figc9_category_fp.png            — Flip probability per perturbation category

Usage:
    python phase4_cifar10c_visualisation.py
    python phase4_cifar10c_visualisation.py --data path/to/results.json
    python phase4_cifar10c_visualisation.py --figures-dir path/to/output/
"""

# ── Phase 1: Imports ──────────────────────────────────────────────────────────

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import config

# ── Phase 1: Paths & constants ────────────────────────────────────────────────

PHASE4_DIR  = config.PHASE4_DIR
FIGURES_DIR = PHASE4_DIR / "figures"
DATA_PATH   = PHASE4_DIR / "phase4_cifar10c_results.json"

DPI  = 200
FONT = {"title": 14, "label": 12, "tick": 10, "annot": 9}

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

# Map raw checkpoint prefix → canonical condition (mirrors CONDITION_MAP in eval script)
_COND_NORM = {
    "biacc":   "bi_acc",
    "birep":   "bi_rep",
    "uniform": "uniform",
    "vanilla": "vanilla",
}

CORRUPTION_TYPES = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter", "saturate",
]
CORRUPTION_CATEGORIES = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
    "weather": ["snow", "frost", "fog", "brightness", "spatter", "saturate"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
SEVERITY_LEVELS = [1, 2, 3, 4, 5]

PERTURBATION_TYPES = [
    "gaussian_noise", "shot_noise", "motion_blur", "zoom_blur",
    "snow", "brightness", "translate", "rotate", "tilt", "scale",
]
PERTURBATION_CATEGORIES = {
    "noise":     ["gaussian_noise", "shot_noise"],
    "geometric": ["translate", "rotate", "tilt", "scale"],
    "other":     ["motion_blur", "zoom_blur", "snow", "brightness"],
}

CAT_LABELS = {
    "noise": "Noise", "blur": "Blur", "weather": "Weather", "digital": "Digital",
}
PERT_CAT_LABELS = {
    "noise": "Noise", "geometric": "Geometric", "other": "Other",
}

# ── Phase 1: Shared helpers ───────────────────────────────────────────────────

def _load_json(path: Path):
    if not path.exists():
        logging.warning(f"Missing JSON: {path} — skipping related figures")
        return None
    with open(path) as f:
        return json.load(f)


def _save_fig(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {out}")


def _group_student_models(models: dict) -> dict:
    """
    Group per-model result dicts by canonical condition name.
    Handles keys like "biacc_seed_42", "birep_seed_123", etc.
    Returns {canon_cond: [result_dict, ...]}. Excludes "teacher".
    """
    groups: dict = {c: [] for c in CONDITIONS}
    for name, result in models.items():
        if name == "teacher":
            continue
        parts = name.split("_seed_")
        if len(parts) != 2:
            continue
        canon = _COND_NORM.get(parts[0])
        if canon and canon in groups:
            groups[canon].append(result)
    return groups


def _weighted_severity_curve(per_category_accuracy: dict) -> list:
    """
    Compute overall mean accuracy per severity level by taking a weighted
    average across corruption categories (weights = number of corruptions per category).
    """
    weights = {cat: len(members) for cat, members in CORRUPTION_CATEGORIES.items()}
    total = sum(weights.values())  # 19
    return [
        sum(weights[cat] * per_category_accuracy[cat][s] for cat in weights) / total
        for s in range(5)
    ]


# ── Phase 2: CIFAR-10-C figures ───────────────────────────────────────────────

def figc1_mce_bar(agg: dict) -> None:
    """
    Bar chart: mCE mean ± std per condition.
    A dashed line at y=1.0 marks the teacher reference (mCE=1 by definition).
    Lower bar = more robust student.
    """
    means = [agg[c]["mCE_mean"] for c in CONDITIONS]
    stds  = [agg[c]["mCE_std"]  for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.55,
                  color=[COND_COLORS[c] for c in CONDITIONS], alpha=0.85,
                  error_kw={"elinewidth": 1.5})
    ax.axhline(1.0, color=TEACHER_COLOR, linestyle="--", linewidth=1.5,
               label="Teacher (mCE = 1.0)")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=FONT["annot"])
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=FONT["tick"])
    ax.set_ylabel("Mean Corruption Error (mCE)", fontsize=FONT["label"])
    ax.set_title("CIFAR-10-C: mCE per Distillation Condition", fontsize=FONT["title"])
    ax.legend(fontsize=FONT["tick"])
    plt.tight_layout()
    _save_fig(fig, "figc1_mce_bar.png")


def figc2_rmce_bar(agg: dict) -> None:
    """
    Bar chart: Relative mCE mean ± std per condition.
    RmCE normalises by clean-accuracy degradation, isolating robustness from
    absolute accuracy gaps between teacher and student.
    """
    means = [agg[c]["rmCE_mean"] for c in CONDITIONS]
    stds  = [agg[c]["rmCE_std"]  for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.55,
                  color=[COND_COLORS[c] for c in CONDITIONS], alpha=0.85,
                  error_kw={"elinewidth": 1.5})
    ax.axhline(1.0, color=TEACHER_COLOR, linestyle="--", linewidth=1.5,
               label="Teacher reference (RmCE = 1.0)")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=FONT["annot"])
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=FONT["tick"])
    ax.set_ylabel("Relative Mean Corruption Error (RmCE)", fontsize=FONT["label"])
    ax.set_title("CIFAR-10-C: Relative mCE per Distillation Condition",
                 fontsize=FONT["title"])
    ax.legend(fontsize=FONT["tick"])
    plt.tight_layout()
    _save_fig(fig, "figc2_rmce_bar.png")


def figc3_severity_curves(agg: dict, teacher_result: dict) -> None:
    """
    Line plot: mean accuracy vs severity level (1→5), averaged across all
    19 corruptions (weighted by category size). One line per condition + teacher.
    Shows how quickly each condition degrades as corruption strengthens.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = SEVERITY_LEVELS

    # Teacher curve from its per_category_accuracy
    t_curve = _weighted_severity_curve(teacher_result["per_category_accuracy"])
    ax.plot(x, t_curve, color=TEACHER_COLOR, marker="o", linestyle="--",
            linewidth=2, label="Teacher", zorder=5)

    for cond in CONDITIONS:
        curve = _weighted_severity_curve(agg[cond]["per_category_accuracy"])
        ax.plot(x, curve, color=COND_COLORS[cond], marker="o",
                linewidth=2, label=COND_LABELS[cond])

    ax.set_xlabel("Severity Level", fontsize=FONT["label"])
    ax.set_ylabel("Mean Accuracy (all 19 corruptions)", fontsize=FONT["label"])
    ax.set_title("CIFAR-10-C: Accuracy Degradation vs Severity", fontsize=FONT["title"])
    ax.set_xticks(SEVERITY_LEVELS)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=FONT["tick"])
    plt.tight_layout()
    _save_fig(fig, "figc3_severity_curves.png")


def figc4_category_severity_curves(agg: dict, teacher_result: dict) -> None:
    """
    1×4 panel: one accuracy-vs-severity line plot per corruption category
    (noise / blur / weather / digital). Reveals whether robustness advantages
    are category-specific. Uses constrained_layout to prevent legend overlap.
    """
    categories = list(CORRUPTION_CATEGORIES.keys())
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    x = SEVERITY_LEVELS

    for ax, cat in zip(axes, categories):
        t_curve = teacher_result["per_category_accuracy"][cat]
        ax.plot(x, t_curve, color=TEACHER_COLOR, marker="o", linestyle="--",
                linewidth=2, label="Teacher")
        for cond in CONDITIONS:
            curve = agg[cond]["per_category_accuracy"][cat]
            ax.plot(x, curve, color=COND_COLORS[cond], marker="o",
                    linewidth=2, label=COND_LABELS[cond])
        ax.set_title(CAT_LABELS[cat], fontsize=FONT["title"])
        ax.set_xlabel("Severity", fontsize=FONT["label"])
        if ax is axes[0]:
            ax.set_ylabel("Accuracy", fontsize=FONT["label"])
        ax.set_xticks(SEVERITY_LEVELS)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=FONT["tick"],
               bbox_to_anchor=(1.02, 1.0))
    _save_fig(fig, "figc4_category_severity_curves.png")


# ── Phase 3: CIFAR-10-C heatmap + ECE ────────────────────────────────────────

def figc5_corruption_heatmap(c_models: dict, agg: dict) -> None:
    """
    Heatmap: 19 corruption types (rows) × 5 conditions (teacher + 4 students, cols).
    Cell value = mean accuracy across 5 severity levels.
    Condition columns show the mean across seeds.
    Uses constrained_layout=True to accommodate the colorbar without overlap.
    """
    groups = _group_student_models(c_models)

    # Teacher column
    teacher = c_models.get("teacher", {})
    teacher_col = [
        teacher.get("per_corruption", {}).get(c, {}).get("mean_accuracy", 0.0)
        for c in CORRUPTION_TYPES
    ]

    # Condition columns (mean across seeds)
    cond_cols = []
    for cond in CONDITIONS:
        results = groups[cond]
        if not results:
            cond_cols.append([0.0] * len(CORRUPTION_TYPES))
        else:
            col = [
                float(np.mean([
                    r.get("per_corruption", {}).get(c, {}).get("mean_accuracy", 0.0)
                    for r in results
                ]))
                for c in CORRUPTION_TYPES
            ]
            cond_cols.append(col)

    mat = np.column_stack([teacher_col] + cond_cols)  # (19, 5)
    col_labels = ["Teacher"] + [COND_LABELS[c] for c in CONDITIONS]

    fig, ax = plt.subplots(figsize=(10, 12), constrained_layout=True)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(len(CORRUPTION_TYPES)):
        for j in range(len(col_labels)):
            val = mat[i, j]
            text_color = "black" if 0.25 < val < 0.85 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=FONT["annot"] - 1, color=text_color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=FONT["tick"])
    ax.set_yticks(range(len(CORRUPTION_TYPES)))
    ax.set_yticklabels(
        [c.replace("_", " ").title() for c in CORRUPTION_TYPES],
        fontsize=FONT["tick"] - 1,
    )
    ax.set_title("CIFAR-10-C: Mean Accuracy per Corruption × Condition",
                 fontsize=FONT["title"])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Mean Accuracy")
    _save_fig(fig, "figc5_corruption_heatmap.png")


def figc6_ece_comparison(agg: dict, teacher_result: dict) -> None:
    """
    Dual bar chart: clean ECE (solid) vs mean corrupted ECE (hatched) per condition.
    Teacher reference lines show its clean and corrupted ECE.
    Calibration quality tends to degrade under distribution shift; this figure
    shows whether that degradation is condition-dependent.
    """
    x = np.arange(len(CONDITIONS))
    width = 0.35

    clean_means = [agg[c]["mean_ece_clean"]        for c in CONDITIONS]
    clean_stds  = [agg[c]["mean_ece_clean_std"]    for c in CONDITIONS]
    corr_means  = [agg[c]["mean_ece_corrupted"]    for c in CONDITIONS]
    corr_stds   = [agg[c]["mean_ece_corrupted_std"] for c in CONDITIONS]

    t_clean_ece = teacher_result.get("clean_ece", 0.0)
    t_corr_ece = float(np.mean([
        teacher_result.get("per_corruption", {}).get(c, {}).get("mean_ece", 0.0)
        for c in CORRUPTION_TYPES
    ]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, clean_means, width, yerr=clean_stds, capsize=4,
           color=[COND_COLORS[c] for c in CONDITIONS], alpha=0.85,
           label="Clean ECE", error_kw={"elinewidth": 1.5})
    ax.bar(x + width / 2, corr_means, width, yerr=corr_stds, capsize=4,
           color=[COND_COLORS[c] for c in CONDITIONS], alpha=0.45, hatch="//",
           label="Corrupted ECE (mean)", error_kw={"elinewidth": 1.5})

    ax.axhline(t_clean_ece, color=TEACHER_COLOR, linestyle="-", linewidth=1.5,
               label=f"Teacher clean ECE ({t_clean_ece:.3f})")
    ax.axhline(t_corr_ece, color=TEACHER_COLOR, linestyle="--", linewidth=1.5,
               label=f"Teacher corrupted ECE ({t_corr_ece:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=FONT["tick"])
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=FONT["label"])
    ax.set_title("CIFAR-10-C: Calibration (ECE) — Clean vs Corrupted",
                 fontsize=FONT["title"])
    ax.legend(fontsize=FONT["tick"])
    plt.tight_layout()
    _save_fig(fig, "figc6_ece_comparison.png")


# ── Phase 4: CIFAR-10-P figures ───────────────────────────────────────────────

def figc7_mfp_bar(p_agg: dict, p_teacher: dict) -> None:
    """
    Bar chart: mean flip probability (mFP) for teacher + each condition (mean ± std).
    Lower bar = more temporally stable predictions under perturbation sequences.
    """
    all_labels = ["Teacher"] + [COND_LABELS[c] for c in CONDITIONS]
    all_colors = [TEACHER_COLOR] + [COND_COLORS[c] for c in CONDITIONS]
    means = [p_teacher.get("mFP", 0.0)] + [p_agg[c]["mFP_mean"] for c in CONDITIONS]
    stds  = [0.0]                        + [p_agg[c]["mFP_std"]  for c in CONDITIONS]

    x = np.arange(len(all_labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.55,
                  color=all_colors, alpha=0.85, error_kw={"elinewidth": 1.5})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.002,
                f"{m:.3f}", ha="center", va="bottom", fontsize=FONT["annot"])
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=FONT["tick"])
    ax.set_ylabel("Mean Flip Probability (mFP)", fontsize=FONT["label"])
    ax.set_title("CIFAR-10-P: Prediction Stability per Condition",
                 fontsize=FONT["title"])
    plt.tight_layout()
    _save_fig(fig, "figc7_mfp_bar.png")


def figc8_fpr_per_perturbation(p_models: dict) -> None:
    """
    Grouped bar chart: Flip Probability Ratio (FPR) per perturbation type per condition.
    FPR = FP_student / FP_teacher. A dashed line at y=1.0 is the teacher baseline.
    FPR > 1 means the student is more jittery than the teacher on that perturbation.
    Condition means are averaged across seeds.
    """
    groups = _group_student_models(p_models)

    # Mean FPR per condition per perturbation type (across seeds)
    cond_fpr: dict = {}
    for cond in CONDITIONS:
        results = groups[cond]
        if not results:
            cond_fpr[cond] = {p: 0.0 for p in PERTURBATION_TYPES}
            continue
        cond_fpr[cond] = {}
        for p in PERTURBATION_TYPES:
            seed_vals = [
                r["FPR_per_perturbation"][p]
                for r in results
                if "FPR_per_perturbation" in r and p in r["FPR_per_perturbation"]
            ]
            cond_fpr[cond][p] = float(np.mean(seed_vals)) if seed_vals else 0.0

    n_pert = len(PERTURBATION_TYPES)
    n_cond = len(CONDITIONS)
    width = 0.18
    x = np.arange(n_pert)

    fig, ax = plt.subplots(figsize=(16, 5))
    for i, cond in enumerate(CONDITIONS):
        offsets = x + (i - n_cond / 2 + 0.5) * width
        vals = [cond_fpr[cond].get(p, 0.0) for p in PERTURBATION_TYPES]
        ax.bar(offsets, vals, width=width, color=COND_COLORS[cond], alpha=0.85,
               label=COND_LABELS[cond])

    ax.axhline(1.0, color=TEACHER_COLOR, linestyle="--", linewidth=1.5,
               label="Teacher (FPR = 1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [p.replace("_", "\n") for p in PERTURBATION_TYPES],
        fontsize=FONT["tick"],
    )
    ax.set_ylabel("Flip Probability Ratio (FPR = student / teacher)",
                  fontsize=FONT["label"])
    ax.set_title("CIFAR-10-P: FPR per Perturbation Type", fontsize=FONT["title"])
    ax.legend(fontsize=FONT["tick"])
    plt.tight_layout()
    _save_fig(fig, "figc8_fpr_per_perturbation.png")


def figc9_category_fp(p_agg: dict, p_teacher: dict) -> None:
    """
    Grouped bar chart: mean flip probability per perturbation category
    (noise / geometric / other) for teacher + each condition.
    Concise summary of which motion type destabilises each condition most.
    """
    categories = list(PERTURBATION_CATEGORIES.keys())
    n_cat = len(categories)
    n_entities = 1 + len(CONDITIONS)  # teacher + 4 conditions
    width = 0.15
    x = np.arange(n_cat)

    # Compute teacher's per-category FP from its per_perturbation data
    t_per_pert = p_teacher.get("per_perturbation", {})
    t_cat_fp = {
        cat: float(np.mean([
            t_per_pert[p]["flip_probability"]
            for p in members
            if p in t_per_pert
        ])) if any(p in t_per_pert for p in members) else 0.0
        for cat, members in PERTURBATION_CATEGORIES.items()
    }

    # Build ordered list: (label, color, per-category FP dict)
    entities = [("Teacher", TEACHER_COLOR, t_cat_fp)] + [
        (COND_LABELS[c], COND_COLORS[c], p_agg[c]["per_category_FP"])
        for c in CONDITIONS
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, color, cat_fp) in enumerate(entities):
        offsets = x + (i - n_entities / 2 + 0.5) * width
        vals = [cat_fp.get(cat, 0.0) for cat in categories]
        ax.bar(offsets, vals, width=width, color=color, alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([PERT_CAT_LABELS[c] for c in categories], fontsize=FONT["tick"])
    ax.set_ylabel("Mean Flip Probability", fontsize=FONT["label"])
    ax.set_title("CIFAR-10-P: Flip Probability per Perturbation Category",
                 fontsize=FONT["title"])
    ax.legend(fontsize=FONT["tick"])
    plt.tight_layout()
    _save_fig(fig, "figc9_category_fp.png")


# ── Phase 5: Main function ─────────────────────────────────────────────────────

def run_visualisation(
    data_path: Path = DATA_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """
    Load phase4_cifar10c_results.json and generate all 9 figures.
    Skips figure groups gracefully if the corresponding data section is absent,
    so partial results (e.g. only CIFAR-10-C done) still produce valid output.
    """
    global FIGURES_DIR
    FIGURES_DIR = figures_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    data = _load_json(data_path)
    if data is None:
        logging.error(f"No data at {data_path} — run phase4_cifar10c_eval.py first")
        return

    cifar10c = data.get("cifar10c", {})
    cifar10p = data.get("cifar10p", {})
    c_agg    = cifar10c.get("aggregate", {})
    c_models = cifar10c.get("models", {})
    p_agg    = cifar10p.get("aggregate", {})
    p_models = cifar10p.get("models", {})

    teacher_c = c_models.get("teacher", {})
    teacher_p = p_models.get("teacher", {})

    # ── CIFAR-10-C figures ────────────────────────────────────────────────────
    _can_c_agg = bool(c_agg and all(c in c_agg for c in CONDITIONS) and teacher_c)

    if _can_c_agg:
        logging.info("--- CIFAR-10-C aggregate figures ---")
        figc1_mce_bar(c_agg)
        figc2_rmce_bar(c_agg)
        figc3_severity_curves(c_agg, teacher_c)
        figc4_category_severity_curves(c_agg, teacher_c)
        figc6_ece_comparison(c_agg, teacher_c)
    else:
        logging.warning(
            "CIFAR-10-C aggregate data incomplete — skipping figc1, figc2, figc3, figc4, figc6"
        )

    if c_models:
        logging.info("--- CIFAR-10-C heatmap ---")
        figc5_corruption_heatmap(c_models, c_agg)
    else:
        logging.warning("CIFAR-10-C model data missing — skipping figc5")

    # ── CIFAR-10-P figures ────────────────────────────────────────────────────
    _can_p_agg = bool(p_agg and all(c in p_agg for c in CONDITIONS) and teacher_p)

    if _can_p_agg:
        logging.info("--- CIFAR-10-P aggregate figures ---")
        figc7_mfp_bar(p_agg, teacher_p)
        figc9_category_fp(p_agg, teacher_p)
    else:
        logging.warning("CIFAR-10-P aggregate data incomplete — skipping figc7, figc9")

    if p_models:
        logging.info("--- CIFAR-10-P per-perturbation figure ---")
        figc8_fpr_per_perturbation(p_models)
    else:
        logging.warning("CIFAR-10-P model data missing — skipping figc8")

    logging.info(f"Done. All figures saved to {FIGURES_DIR}")


# ── Phase 5: CLI guard ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CIFAR-10-C/P visualisation suite (9 figures)"
    )
    parser.add_argument(
        "--data",
        default=str(DATA_PATH),
        help="Path to phase4_cifar10c_results.json "
             f"(default: {DATA_PATH})",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(FIGURES_DIR),
        help="Output directory for PNG figures "
             f"(default: {FIGURES_DIR})",
    )
    args = parser.parse_args()
    run_visualisation(Path(args.data), Path(args.figures_dir))
