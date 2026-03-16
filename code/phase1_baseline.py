"""
phase1_baseline.py — Steps 1.3 & 1.4: model sanity check and
extraction of the intact-model reference representations.

Outputs:
    results/reference_representations.pt   (N × 2048 float tensor)
    results/reference_meta.json            (checksum, accuracy, metadata)
"""

import json
import logging
import time

import torch

from config import (
    ACCURACY_FAIL_THRESHOLD, ACCURACY_PASS_THRESHOLD, BASELINE_ACCURACY,
    BATCH_SIZE_CALIB, N_CALIBRATION, REF_META, REF_REPR, SEED,
)
from model import ResNet50_CIFAR, evaluate_accuracy
from utils import ActivationCapture, gap, tensor_checksum

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Step 1.3: Sanity check
# ──────────────────────────────────────────────

def run_sanity_check(
    model: ResNet50_CIFAR,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate the intact model on the full test set.

    Pass : accuracy ≥ ACCURACY_PASS_THRESHOLD  → proceed normally
    Warn : accuracy ∈ [FAIL, PASS)             → log warning, continue with caution
    Fail : accuracy < ACCURACY_FAIL_THRESHOLD  → raise RuntimeError

    Returns the measured accuracy.
    """
    logger.info("Step 1.3 — Running sanity check on full test set …")
    t0 = time.time()
    acc = evaluate_accuracy(model, test_loader, device)
    elapsed = time.time() - t0

    logger.info(f"  Test accuracy : {acc:.4f}  (target={BASELINE_ACCURACY}, "
                f"elapsed={elapsed:.1f}s)")

    if acc < ACCURACY_FAIL_THRESHOLD:
        raise RuntimeError(
            f"Sanity check FAILED: accuracy={acc:.4f} < {ACCURACY_FAIL_THRESHOLD}. "
            "Review normalisation constants or checkpoint keys."
        )
    elif acc < ACCURACY_PASS_THRESHOLD:
        logger.warning(
            f"Sanity check WARNING: accuracy={acc:.4f} is below pass threshold "
            f"({ACCURACY_PASS_THRESHOLD}). Proceeding with caution."
        )
    else:
        logger.info("  Sanity check PASSED ✓")

    return acc


# ──────────────────────────────────────────────
# Step 1.4: Reference representation extraction
# ──────────────────────────────────────────────

@torch.no_grad()
def extract_reference_representations(
    model: ResNet50_CIFAR,
    calib_loader: torch.utils.data.DataLoader,
    device: torch.device,
    baseline_accuracy: float,
    env: dict,
) -> torch.Tensor:
    """
    Extract the final pre-classifier representations (output of Global Average
    Pooling, shape N × 2048) from the INTACT model over the calibration set.

    Methodological decisions (per pipeline spec):
      • Final layer chosen over per-block layer: captures global impact.
      • Instance-level granularity preserved (no class-wise averaging) to
        maintain sensitivity to representational collapse and intra-class geometry.
      • Source: calibration set drawn from TRAIN split (avoids leaking test info
        into the pruning metric).

    Saves:
      REF_REPR  — the raw tensor
      REF_META  — JSON with checksum, accuracy, config snapshot

    Returns:
      F_intact : FloatTensor of shape (N, 2048)
    """
    if REF_REPR.exists() and REF_META.exists():
        logger.info("Reference representations already exist — loading from disk.")
        F_intact = torch.load(REF_REPR, map_location="cpu")
        with open(REF_META) as f:
            meta = json.load(f)
        logger.info(f"  Loaded F_intact {tuple(F_intact.shape)}, "
                    f"checksum={meta['checksum']}")
        return F_intact

    logger.info("Step 1.4 — Extracting reference representations …")
    model.eval()

    all_features = []

    # Hook on avgpool: captures the GAP output (the final pre-classifier vector)
    with ActivationCapture(model.avgpool) as cap:
        for batch_idx, (images, _) in enumerate(calib_loader):
            images = images.to(device)
            _ = model(images)                      # forward pass
            feat = cap.output                      # shape: (B, 2048, 1, 1)
            feat = gap(feat)                       # → (B, 2048)
            all_features.append(feat.cpu())

    F_intact = torch.cat(all_features, dim=0)      # (N, 2048)

    assert F_intact.shape == (N_CALIBRATION, 2048), (
        f"Unexpected shape: {F_intact.shape} (expected ({N_CALIBRATION}, 2048))"
    )

    checksum = tensor_checksum(F_intact)
    torch.save(F_intact, REF_REPR)

    meta = {
        "shape": list(F_intact.shape),
        "checksum": checksum,
        "baseline_accuracy": baseline_accuracy,
        "seed": SEED,
        "n_calibration": N_CALIBRATION,
        "batch_size": BATCH_SIZE_CALIB,
        "env": env,
    }
    with open(REF_META, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"  F_intact saved: shape={tuple(F_intact.shape)}, checksum={checksum}"
    )
    return F_intact


# ──────────────────────────────────────────────
# Convenience: run the full Phase 1
# ──────────────────────────────────────────────

def run_phase1(model, test_loader, calib_loader, device, env) -> tuple[float, torch.Tensor]:
    """
    Execute Phase 1 end-to-end.
    Returns (baseline_accuracy, F_intact).
    """
    baseline_acc = run_sanity_check(model, test_loader, device)
    F_intact = extract_reference_representations(
        model, calib_loader, device, baseline_acc, env
    )
    return baseline_acc, F_intact
