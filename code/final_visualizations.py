"""
final_visualizations.py — Summary figures across the full project.

Figures
-------
final_accuracy.png
    Final CIFAR-10 test accuracy of the teacher (ResNet-50) and of every
    ResNet-18 student, grouped by distillation condition, with one marker per
    seed and the per-condition mean.

Inputs
------
- Teacher accuracy:  results/reference_meta.json   (Phase 1)
- Student accuracy:  ../phase4_results/phase4_<Condition>_<seed>.json  (Phase 4)

Any Phase 4 run file that contains ``conditions → <cond> → seed_<n> →
final_accuracy`` is picked up, so new conditions appear without code changes.

Usage (from Bi_project/code/):
    python final_visualizations.py
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, PHASE4_DIR, REF_META

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
INK_PRIMARY   = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED     = "#898781"
GRID_COLOR    = "#e1e0d9"
AXIS_COLOR    = "#c3c2b7"
SURFACE       = "#fcfcfb"

# Seed identity: colour + marker shape (shape keeps seeds apart without colour).
SEED_STYLE = {
    42:  {"color": "#2a78d6", "marker": "o"},
    123: {"color": "#eb6834", "marker": "s"},
    456: {"color": "#1baf7a", "marker": "^"},
}
FALLBACK_SEED_STYLE = {"color": INK_MUTED, "marker": "D"}

# Display order and labels; unknown conditions are appended alphabetically.
COND_ORDER  = ["vanilla", "uniform", "bi_acc", "bi_rep"]
COND_LABELS = {
    "vanilla": "Vanilla\n(no KD)",
    "uniform": "Uniform\nKD",
    "bi_acc":  "BI-Acc\nKD",
    "bi_rep":  "BI-Rep\nKD",
}

FONT = {"title": 13, "label": 11, "tick": 10, "annot": 9}
DPI  = 200

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_teacher_accuracy() -> float:
    with open(REF_META) as f:
        return float(json.load(f)["baseline_accuracy"])


def load_student_accuracies() -> dict:
    """Return {condition: {seed: final_accuracy}} from all Phase 4 run files."""
    acc = defaultdict(dict)
    for path in sorted(PHASE4_DIR.glob("phase4_*.json")):
        with open(path) as f:
            data = json.load(f)
        conditions = data.get("conditions")
        if not isinstance(conditions, dict):
            continue
        for cond, runs in conditions.items():
            for seed_key, run in runs.items():
                if not isinstance(run, dict) or "final_accuracy" not in run:
                    continue
                seed = int(seed_key.removeprefix("seed_"))
                if seed in acc[cond]:
                    logging.warning(f"Duplicate {cond}/seed {seed} in {path.name} — overriding")
                acc[cond][seed] = float(run["final_accuracy"])
    if not acc:
        raise FileNotFoundError(f"No Phase 4 run files with final_accuracy in {PHASE4_DIR}")
    return dict(acc)


def _ordered_conditions(conds) -> list:
    known = [c for c in COND_ORDER if c in conds]
    return known + sorted(c for c in conds if c not in COND_ORDER)


def log_accuracy_table(teacher_acc: float, student_acc: dict):
    seeds = sorted({s for runs in student_acc.values() for s in runs})
    header = f"{'model':<10}" + "".join(f"{'seed ' + str(s):>11}" for s in seeds) + f"{'mean ± std':>18}"
    lines = [header, "-" * len(header), f"{'teacher':<10}" + " " * 11 * len(seeds) + f"{teacher_acc * 100:>18.2f}"]
    for cond in _ordered_conditions(student_acc):
        runs = student_acc[cond]
        vals = np.array(list(runs.values())) * 100
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        cells = "".join(f"{runs[s] * 100:>11.2f}" if s in runs else f"{'—':>11}" for s in seeds)
        lines.append(f"{cond:<10}{cells}{f'{vals.mean():.2f} ± {std:.2f}':>18}")
    logging.info("Final test accuracy (%):\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_fig(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    logging.info(f"Saved {out}")


def fig_final_accuracy(teacher_acc: float, student_acc: dict):
    """Dot plot: teacher + every student seed per condition, with condition means."""
    conds = _ordered_conditions(student_acc)
    seeds = sorted({s for runs in student_acc.values() for s in runs})
    jitter = dict(zip(seeds, np.linspace(-0.16, 0.16, len(seeds)) if len(seeds) > 1 else [0.0]))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Teacher: single model, plus a reference line across the student groups
    teacher_pct = teacher_acc * 100
    ax.axhline(teacher_pct, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.plot(0, teacher_pct, marker="*", ms=16, color=INK_SECONDARY,
            mec=SURFACE, mew=1.5, ls="none", zorder=3, label="Teacher (ResNet-50)")
    ax.annotate(f"{teacher_pct:.2f}%", (0, teacher_pct), xytext=(0, 12),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=FONT["annot"], color=INK_PRIMARY)

    # Students: one marker per seed, mean as a horizontal bar
    plotted_seeds = set()
    for i, cond in enumerate(conds, start=1):
        runs = student_acc[cond]
        for seed, acc in sorted(runs.items()):
            style = SEED_STYLE.get(seed, FALLBACK_SEED_STYLE)
            ax.plot(i + jitter[seed], acc * 100, ls="none", ms=9, mec=SURFACE, mew=1.5,
                    zorder=3, label=None if seed in plotted_seeds else f"Seed {seed}", **style)
            plotted_seeds.add(seed)

        vals = np.array(list(runs.values())) * 100
        mean = vals.mean()
        ax.hlines(mean, i - 0.3, i + 0.3, color=INK_PRIMARY, lw=2, zorder=2,
                  label="Mean over seeds" if i == 1 else None)
        ax.annotate(f"{mean:.2f}%", (i, vals.max()), xytext=(0, 10),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=FONT["annot"], color=INK_PRIMARY)

    # Separate teacher column from the student group
    ax.axvline(0.5, color=AXIS_COLOR, lw=1, zorder=1)
    ax.text(0.5 + len(conds) / 2, 1.0, "ResNet-18 students", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=FONT["label"], color=INK_SECONDARY)

    ax.set_xticks(range(len(conds) + 1))
    ax.set_xticklabels(["Teacher\n(ResNet-50)"] + [COND_LABELS.get(c, c) for c in conds],
                       fontsize=FONT["tick"], color=INK_SECONDARY)
    ax.set_xlim(-0.6, len(conds) + 0.6)

    all_pct = [teacher_pct] + [a * 100 for runs in student_acc.values() for a in runs.values()]
    span = max(all_pct) - min(all_pct)
    ax.set_ylim(min(all_pct) - 0.25 * span, max(all_pct) + 0.3 * span)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.set_ylabel("Test accuracy (%)", fontsize=FONT["label"], color=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=FONT["tick"], colors=INK_SECONDARY)
    ax.tick_params(axis="x", length=0)

    ax.grid(axis="y", color=GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)

    ax.set_title("Final CIFAR-10 test accuracy after training", fontsize=FONT["title"],
                 color=INK_PRIMARY, loc="left", pad=22)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=FONT["annot"], labelcolor=INK_SECONDARY, numpoints=1)

    fig.tight_layout()
    _save_fig(fig, "final_accuracy.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    teacher_acc = load_teacher_accuracy()
    student_acc = load_student_accuracies()
    log_accuracy_table(teacher_acc, student_acc)
    fig_final_accuracy(teacher_acc, student_acc)


if __name__ == "__main__":
    main()
