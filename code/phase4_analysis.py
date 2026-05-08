"""
phase4_analysis.py — Phase 4 fine-grained post-hoc analysis.

For each student checkpoint trained under weighted SP-KD:
    4.0  Setup: load teacher features and confusion matrix (cached once)
    4.1  Per-student:
         - Per-class CKA (teacher vs student avgpool features, 10 classes)
         - Class-pair cosine similarity matrices + diff
         - Confusion matrix delta vs teacher
    4.2  Aggregate summary by condition (bi_acc, bi_rep, uniform, vanilla)
    4.3  Save results

Outputs:
    phase4_results/phase4_fine_analysis.json
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
from bi_rep_extended import class_cosine_matrix, extract_features_with_labels
from data import (
    get_calibration_loader,
    get_test_loader,
    load_cifar10,
    load_or_build_calibration_indices,
)
from model import load_model
from phase2_metrics import linear_cka
from student_model import ResNet18_CIFAR

logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
# Silent-failure critical pairs from Phase 3 (indices into CIFAR10_CLASSES)
CRITICAL_PAIRS = {"cat_dog": (3, 5), "cat_deer": (3, 4), "dog_deer": (5, 4)}
# Checkpoint condition name → aggregate output key
CONDITION_MAP = {
    "biacc": "bi_acc", "birep": "bi_rep",
    "uniform": "uniform", "vanilla": "vanilla",
}


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


@torch.no_grad()
def _compute_confusion_matrix(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    cm = torch.zeros(10, 10, dtype=torch.long)
    for images, labels in loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        for t, p in zip(labels, preds):
            cm[t, p] += 1
    return cm


def _top5_confusion_increases(
    cm_t: torch.Tensor,
    cm_s: torch.Tensor,
) -> list:
    diff = (cm_s - cm_t).clone()
    diff.fill_diagonal_(0)
    top5_idx = diff.flatten().argsort(descending=True)[:5]
    result = []
    for idx in top5_idx:
        row, col = int(idx // 10), int(idx % 10)
        result.append([CIFAR10_CLASSES[row], CIFAR10_CLASSES[col], int(diff[row, col].item())])
    return result


def run_phase4_analysis() -> dict:
    """Fine-grained post-hoc analysis of Phase 4 student checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("─" * 60)
    logger.info(f"Step 4.0 — Setup  [device={device}]")

    train_ds, test_ds = load_cifar10()
    calib_indices = load_or_build_calibration_indices(train_ds)
    calib_loader  = get_calibration_loader(train_ds, calib_indices)
    test_loader   = get_test_loader(test_ds)

    teacher = _load_teacher(device)
    F_teacher, labels = extract_features_with_labels(teacher, calib_loader, device)
    S_teacher  = class_cosine_matrix(F_teacher, labels)
    CM_teacher = _compute_confusion_matrix(teacher, test_loader, device)
    logger.info(f"  teacher features: {tuple(F_teacher.shape)}, test CM sum={CM_teacher.sum().item()}")

    ckpt_paths = sorted(
        p for p in config.PHASE4_DIR.glob("*_student_seed_*.pt")
        if ":Zone.Identifier" not in p.name
    )
    logger.info(f"  found {len(ckpt_paths)} student checkpoint(s)")

    per_student: dict = {}

    logger.info("─" * 60)
    logger.info("Step 4.1 — Per-student analysis")

    for ckpt_path in ckpt_paths:
        condition, seed = _parse_checkpoint_name(ckpt_path)
        key = f"{condition}_seed_{seed}"
        logger.info(f"  [{key}] loading …")

        student = _load_student(ckpt_path, device)
        # Reuse teacher's labels — calibration set is identical for all models
        F_student, _ = extract_features_with_labels(student, calib_loader, device)

        # Per-class CKA
        per_class_cka: dict[str, float] = {}
        for k in range(10):
            mask = labels == k
            per_class_cka[str(k)] = float(linear_cka(F_teacher[mask], F_student[mask]))
        class_cka_variance = float(np.var(list(per_class_cka.values())))

        # Class-pair cosine similarity
        S_student    = class_cosine_matrix(F_student, labels)
        cosine_diff  = S_student - S_teacher
        cosine_diff_frob = float(torch.norm(cosine_diff, p="fro").item())
        critical_pairs = {
            name: float(cosine_diff[i, j].item())
            for name, (i, j) in CRITICAL_PAIRS.items()
        }

        # Confusion matrix delta
        CM_student = _compute_confusion_matrix(student, test_loader, device)
        top5 = _top5_confusion_increases(CM_teacher, CM_student)

        per_student[key] = {
            "per_class_cka":            per_class_cka,
            "class_cka_variance":       class_cka_variance,
            "student_class_cosine":     S_student.tolist(),
            "cosine_diff":              cosine_diff.tolist(),
            "cosine_diff_frob":         cosine_diff_frob,
            "critical_pairs":           critical_pairs,
            "confusion_matrix_teacher": CM_teacher.tolist(),
            "confusion_matrix_student": CM_student.tolist(),
            "top5_confusion_increases": top5,
        }
        logger.info(
            f"  [{key}] cka_var={class_cka_variance:.6f}, "
            f"cosine_frob={cosine_diff_frob:.4f}"
        )

    logger.info("─" * 60)
    logger.info("Step 4.2 — Aggregate summary")

    groups: dict[str, list] = defaultdict(list)
    for key, entry in per_student.items():
        raw_condition = key.split("_seed_")[0]
        groups[CONDITION_MAP.get(raw_condition, raw_condition)].append(entry)

    aggregate: dict = {}
    for agg_key, entries in groups.items():
        cm_deltas = []
        for e in entries:
            diff = (
                torch.tensor(e["confusion_matrix_student"])
                - torch.tensor(e["confusion_matrix_teacher"])
            ).abs()
            diff.fill_diagonal_(0)
            cm_deltas.append(float(diff.sum().item()))
        aggregate[agg_key] = {
            "mean_class_cka_var":   float(np.mean([e["class_cka_variance"] for e in entries])),
            "mean_cosine_frob":     float(np.mean([e["cosine_diff_frob"]   for e in entries])),
            "mean_confusion_delta": float(np.mean(cm_deltas)),
        }
        logger.info(
            f"  {agg_key}: cka_var={aggregate[agg_key]['mean_class_cka_var']:.6f}, "
            f"cosine_frob={aggregate[agg_key]['mean_cosine_frob']:.4f}, "
            f"cm_delta={aggregate[agg_key]['mean_confusion_delta']:.1f}"
        )

    logger.info("─" * 60)
    logger.info("Step 4.3 — Save results")

    out = {
        "teacher_class_cosine": S_teacher.tolist(),
        "per_student":          per_student,
        "aggregate":            aggregate,
    }
    config.PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PHASE4_FINE_ANALYSIS, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"  Saved → {config.PHASE4_FINE_ANALYSIS}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_phase4_analysis()
