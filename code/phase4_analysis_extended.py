"""
phase4_analysis_extended.py — Intra-class variance, Fisher Discriminant Ratio,
and Participation Ratio for teacher and Phase 4 student checkpoints.

Tests the hypothesis: SP-KD students have centroids closer to the teacher
but lower intra-class dispersion, yielding better effective margin despite
closer centroids. Vanilla students push centroids apart but have higher
intra-class variance, yielding worse margin.

Outputs:
    phase4_results/phase4_extended_analysis.json
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config
from bi_rep_extended import extract_features_with_labels
from data import get_calibration_loader, load_cifar10, load_or_build_calibration_indices
from model import load_model
from student_model import ResNet18_CIFAR

logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CRITICAL_PAIRS = {"cat_dog": (3, 5), "cat_deer": (3, 4), "dog_deer": (5, 4)}
CONDITION_MAP = {
    "biacc": "bi_acc", "birep": "bi_rep",
    "uniform": "uniform", "vanilla": "vanilla",
}
PHASE4_EXTENDED_ANALYSIS = config.PHASE4_DIR / "phase4_extended_analysis.json"


def _parse_checkpoint_name(path: Path) -> tuple[str, int]:
    m = re.match(r"^([a-z]+)_student_seed_(\d+)\.pt$", path.name)
    if not m:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return m.group(1), int(m.group(2))


def _load_teacher(device: torch.device) -> nn.Module:
    model = load_model(config.MODEL_CKPT, device)
    model.eval()
    return model


def _load_student(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = ResNet18_CIFAR(num_classes=10).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _class_stats(
    F: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int = 10,
) -> tuple[list[torch.Tensor], list[float]]:
    """Per-class centroids and intra-class variances (mean sq. dist from centroid)."""
    centroids, variances = [], []
    for k in range(n_classes):
        F_k = F[labels == k]
        if F_k.shape[0] == 0:
            centroids.append(torch.zeros(F.shape[1]))
            variances.append(0.0)
        else:
            mu_k = F_k.mean(dim=0)
            diffs = F_k - mu_k
            variances.append(float((diffs ** 2).sum(dim=1).mean().item()))
            centroids.append(mu_k)
    return centroids, variances


def _intra_class_variance(variances: list[float]) -> dict:
    result = {str(k): variances[k] for k in range(10)}
    result["mean"] = float(np.mean(variances))
    return result


def _fisher_discriminant_ratio(
    centroids: list[torch.Tensor],
    variances: list[float],
) -> tuple[list, dict, float]:
    """Full 10×10 FDR matrix (diagonal=0), critical pairs, mean over 45 pairs."""
    n = len(centroids)
    fdr = torch.zeros(n, n)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            dist_sq = float(((centroids[a] - centroids[b]) ** 2).sum().item())
            denom = variances[a] + variances[b]
            fdr[a, b] = dist_sq / denom if denom > 1e-12 else 0.0

    critical = {
        name: float(fdr[i, j].item()) for name, (i, j) in CRITICAL_PAIRS.items()
    }
    mean_fdr = float(
        sum(fdr[a, b].item() for a in range(n) for b in range(a + 1, n)) / 45.0
    )
    return fdr.tolist(), critical, mean_fdr


def _pr_single(mat: torch.Tensor) -> float:
    """Participation ratio: (Σ s²)² / Σ s⁴ where s are singular values."""
    if mat.shape[0] <= 1:
        return 1.0
    mat_c = mat - mat.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(mat_c)
    sv2 = sv ** 2
    denom = (sv2 ** 2).sum()
    return float((sv2.sum() ** 2 / denom).item()) if denom > 1e-12 else 1.0


@torch.no_grad()
def _participation_ratio(F: torch.Tensor, labels: torch.Tensor) -> dict:
    global_pr = _pr_single(F)
    per_class = {str(k): _pr_single(F[labels == k]) for k in range(10)}
    return {"global": global_pr, "per_class": per_class}


@torch.no_grad()
def _all_metrics(
    F: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict, list, dict, float, dict]:
    """Compute ICV, FDR, and PR from a feature matrix in one pass over class stats."""
    centroids, variances = _class_stats(F, labels)
    icv = _intra_class_variance(variances)
    fdr_matrix, fdr_critical, fdr_mean = _fisher_discriminant_ratio(centroids, variances)
    pr = _participation_ratio(F, labels)
    return icv, fdr_matrix, fdr_critical, fdr_mean, pr


def _per_class_recall_from_cm(cm_list: list) -> dict:
    cm = torch.tensor(cm_list)
    recall = {}
    for k in range(10):
        total = float(cm[k].sum().item())
        recall[str(k)] = float(cm[k, k].item() / total) if total > 0 else 0.0
    return recall


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_phase4_extended_analysis() -> dict:
    """Intra-class variance, FDR, and PR analysis of Phase 4 checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("─" * 60)
    logger.info(f"Phase 4 Extended Analysis  [device={device}]")

    with open(config.PHASE4_FINE_ANALYSIS) as f:
        fine_per_student = json.load(f).get("per_student", {})

    train_ds, _ = load_cifar10()
    calib_indices = load_or_build_calibration_indices(train_ds)
    calib_loader = get_calibration_loader(train_ds, calib_indices)

    logger.info("─" * 60)
    logger.info("Step 1 — Teacher")

    teacher = _load_teacher(device)
    F_teacher, labels = extract_features_with_labels(teacher, calib_loader, device)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info(f"  features: {tuple(F_teacher.shape)}")

    t_icv, t_fdr_mat, t_fdr_crit, t_fdr_mean, t_pr = _all_metrics(F_teacher, labels)
    logger.info(
        f"  ICV mean={t_icv['mean']:.4f}, "
        f"FDR mean={t_fdr_mean:.4f}, PR global={t_pr['global']:.2f}"
    )

    logger.info("─" * 60)
    logger.info("Step 2 — Per-student analysis")

    ckpt_paths = sorted(
        p for p in config.PHASE4_DIR.glob("*_student_seed_*.pt")
        if ":Zone.Identifier" not in p.name
    )
    logger.info(f"  found {len(ckpt_paths)} student checkpoint(s)")

    per_student: dict = {}

    for ckpt_path in ckpt_paths:
        condition, seed = _parse_checkpoint_name(ckpt_path)
        key = f"{condition}_seed_{seed}"
        logger.info(f"  [{key}] computing …")

        student = _load_student(ckpt_path, device)
        F_student, _ = extract_features_with_labels(student, calib_loader, device)
        del student
        if device.type == "cuda":
            torch.cuda.empty_cache()

        icv, fdr_mat, fdr_crit, fdr_mean, pr = _all_metrics(F_student, labels)

        fdr_ratio = {
            name: float(fdr_crit[name] / (t_fdr_crit[name] + 1e-12))
            for name in CRITICAL_PAIRS
        }
        fdr_ratio["mean"] = float(fdr_mean / (t_fdr_mean + 1e-12))

        recall: dict = {}
        if key in fine_per_student:
            recall = _per_class_recall_from_cm(
                fine_per_student[key]["confusion_matrix_student"]
            )
        else:
            logger.warning(f"  [{key}] not found in phase4_fine_analysis.json — skipping recall")

        per_student[key] = {
            "intra_class_variance": icv,
            "fdr_matrix": fdr_mat,
            "fdr_critical_pairs": fdr_crit,
            "fdr_mean_all_pairs": fdr_mean,
            "fdr_ratio_vs_teacher": fdr_ratio,
            "participation_ratio": pr,
            "per_class_recall": recall,
        }
        logger.info(
            f"  [{key}] ICV mean={icv['mean']:.4f}, "
            f"FDR mean={fdr_mean:.4f}, PR global={pr['global']:.2f}"
        )

    logger.info("─" * 60)
    logger.info("Step 3 — Aggregate")

    groups: dict[str, list] = defaultdict(list)
    for key, entry in per_student.items():
        raw_condition = key.split("_seed_")[0]
        groups[CONDITION_MAP.get(raw_condition, raw_condition)].append(entry)

    aggregate: dict = {}
    for agg_key, entries in groups.items():
        aggregate[agg_key] = {
            "mean_intra_class_var":     float(np.mean([e["intra_class_variance"]["mean"] for e in entries])),
            "mean_fdr_all_pairs":       float(np.mean([e["fdr_mean_all_pairs"] for e in entries])),
            "mean_fdr_ratio_cat_dog":   float(np.mean([e["fdr_ratio_vs_teacher"]["cat_dog"] for e in entries])),
            "mean_fdr_ratio_cat_deer":  float(np.mean([e["fdr_ratio_vs_teacher"]["cat_deer"] for e in entries])),
            "mean_fdr_ratio_dog_deer":  float(np.mean([e["fdr_ratio_vs_teacher"]["dog_deer"] for e in entries])),
            "mean_fdr_ratio_all":       float(np.mean([e["fdr_ratio_vs_teacher"]["mean"] for e in entries])),
            "mean_participation_ratio": float(np.mean([e["participation_ratio"]["global"] for e in entries])),
        }
        logger.info(
            f"  {agg_key}: ICV={aggregate[agg_key]['mean_intra_class_var']:.4f}, "
            f"FDR_mean={aggregate[agg_key]['mean_fdr_all_pairs']:.4f}, "
            f"PR={aggregate[agg_key]['mean_participation_ratio']:.2f}"
        )

    out = {
        "teacher": {
            "intra_class_variance": t_icv,
            "fdr_matrix": t_fdr_mat,
            "fdr_critical_pairs": t_fdr_crit,
            "fdr_mean_all_pairs": t_fdr_mean,
            "participation_ratio": t_pr,
        },
        "per_student": per_student,
        "aggregate": aggregate,
    }

    config.PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE4_EXTENDED_ANALYSIS, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"  Saved → {PHASE4_EXTENDED_ANALYSIS}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_phase4_extended_analysis()
