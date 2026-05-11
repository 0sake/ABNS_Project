import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from data import get_test_loader, load_cifar10
from model import load_model
from student_model import ResNet18_CIFAR

logger = logging.getLogger(__name__)

PHASE4_CONFIDENCE_ANALYSIS = config.PHASE4_DIR / "phase4_confidence_analysis.json"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# cat=3, deer=4, dog=5
CRITICAL_PAIR_INDICES = {
    "cat_as_dog":  (3, 5),
    "dog_as_cat":  (5, 3),
    "cat_as_deer": (3, 4),
    "deer_as_cat": (4, 3),
    "dog_as_deer": (5, 4),
    "deer_as_dog": (4, 5),
}

CONDITION_MAP = {
    "biacc":   "bi_acc",
    "birep":   "bi_rep",
    "uniform": "uniform",
    "vanilla": "vanilla",
}

AGG_KEYS = [
    "mean_confidence_correct",
    "mean_maxprob_incorrect",
    "mean_entropy_overall",
    "mean_logit_margin",
    "calibration_gap",
]


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
def _inference_stats(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    all_logits, all_targets = [], []

    for images, labels in test_loader:
        logits = model(images.to(device)).cpu()
        all_logits.append(logits)
        all_targets.append(labels)

    logits  = torch.cat(all_logits)   # (N, 10)
    targets = torch.cat(all_targets)  # (N,)
    probs   = F.softmax(logits, dim=1)
    preds   = logits.argmax(dim=1)

    N = len(targets)
    correct_mask   = preds == targets
    incorrect_mask = ~correct_mask
    accuracy = correct_mask.float().mean().item()

    # Confidence
    true_class_probs = probs[torch.arange(N), targets]
    max_probs        = probs.max(dim=1).values

    mean_confidence_correct  = true_class_probs[correct_mask].mean().item()
    mean_maxprob_incorrect   = (
        max_probs[incorrect_mask].mean().item() if incorrect_mask.any() else 0.0
    )
    calibration_gap = mean_confidence_correct - accuracy

    # Entropy (clamp to avoid log(0))
    p_clamped = probs.clamp(min=1e-12)
    entropy   = -(p_clamped * p_clamped.log()).sum(dim=1)

    mean_entropy_correct   = entropy[correct_mask].mean().item()
    mean_entropy_incorrect = (
        entropy[incorrect_mask].mean().item() if incorrect_mask.any() else 0.0
    )
    mean_entropy_overall = entropy.mean().item()

    # Logit magnitude
    logit_norms      = logits.norm(dim=1)
    logit_maxes      = logits.max(dim=1).values
    true_class_logits = logits[torch.arange(N), targets]

    other_logits = logits.clone()
    other_logits[torch.arange(N), targets] = -float("inf")
    second_best  = other_logits.max(dim=1).values
    margin       = true_class_logits - second_best

    mean_logit_norm   = logit_norms.mean().item()
    mean_logit_max    = logit_maxes.mean().item()
    mean_logit_margin = (
        margin[correct_mask].mean().item() if correct_mask.any() else 0.0
    )

    # Per-class splits
    per_class_confidence_correct: dict[str, float] = {}
    per_class_entropy_correct: dict[str, float] = {}
    for k in range(10):
        mask_k = correct_mask & (targets == k)
        per_class_confidence_correct[str(k)] = (
            true_class_probs[mask_k].mean().item() if mask_k.any() else 0.0
        )
        per_class_entropy_correct[str(k)] = (
            entropy[mask_k].mean().item() if mask_k.any() else 0.0
        )

    # Critical pairs
    critical_pairs: dict[str, dict] = {}
    for name, (true_cls, pred_cls) in CRITICAL_PAIR_INDICES.items():
        mask = (targets == true_cls) & (preds == pred_cls)
        count = int(mask.sum().item())
        critical_pairs[name] = {
            "confidence": max_probs[mask].mean().item() if count > 0 else 0.0,
            "count": count,
        }

    return {
        "accuracy":                    accuracy,
        "mean_confidence_correct":     mean_confidence_correct,
        "mean_maxprob_incorrect":      mean_maxprob_incorrect,
        "calibration_gap":             calibration_gap,
        "mean_entropy_correct":        mean_entropy_correct,
        "mean_entropy_incorrect":      mean_entropy_incorrect,
        "mean_entropy_overall":        mean_entropy_overall,
        "mean_logit_norm":             mean_logit_norm,
        "mean_logit_max":              mean_logit_max,
        "mean_logit_margin":           mean_logit_margin,
        "per_class_confidence_correct": per_class_confidence_correct,
        "per_class_entropy_correct":   per_class_entropy_correct,
        "critical_pairs":              critical_pairs,
    }


def run_phase4_confidence_analysis() -> dict:
    """Confidence and entropy analysis of Phase 4 teacher and student checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    _, test_ds  = load_cifar10()
    test_loader = get_test_loader(test_ds)

    # Teacher — computed once
    logger.info("[teacher] loading …")
    teacher      = _load_teacher(device)
    teacher_stats = _inference_stats(teacher, test_loader, device)
    logger.info(
        f"  accuracy={teacher_stats['accuracy']:.4f}  "
        f"conf_correct={teacher_stats['mean_confidence_correct']:.4f}  "
        f"entropy_overall={teacher_stats['mean_entropy_overall']:.4f}"
    )
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Student checkpoints
    ckpt_paths = sorted(
        p for p in config.PHASE4_DIR.glob("*_student_seed_*.pt")
        if ":Zone.Identifier" not in p.name
    )
    if not ckpt_paths:
        logger.warning(f"No student checkpoints found in {config.PHASE4_DIR}")

    per_student: dict[str, dict] = {}
    for ckpt_path in ckpt_paths:
        condition, seed = _parse_checkpoint_name(ckpt_path)
        key = f"{condition}_seed_{seed}"
        logger.info(f"  [{key}] loading …")

        student = _load_student(ckpt_path, device)
        stats   = _inference_stats(student, test_loader, device)
        per_student[key] = stats
        logger.info(
            f"    accuracy={stats['accuracy']:.4f}  "
            f"conf_correct={stats['mean_confidence_correct']:.4f}  "
            f"margin={stats['mean_logit_margin']:.4f}"
        )
        del student
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate by condition
    groups: dict[str, list] = defaultdict(list)
    for key, entry in per_student.items():
        raw_cond  = key.split("_seed_")[0]
        canonical = CONDITION_MAP.get(raw_cond, raw_cond)
        groups[canonical].append(entry)

    aggregate = {
        cond: {
            k: float(sum(e[k] for e in entries) / len(entries))
            for k in AGG_KEYS
        }
        for cond, entries in groups.items()
    }

    out = {
        "teacher":     teacher_stats,
        "per_student": per_student,
        "aggregate":   aggregate,
    }

    config.PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE4_CONFIDENCE_ANALYSIS, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved → {PHASE4_CONFIDENCE_ANALYSIS}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_phase4_confidence_analysis()
