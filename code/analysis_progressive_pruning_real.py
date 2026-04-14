"""
analysis_progressive_pruning_real.py — Real progressive pruning analysis.

Ablates the first k blocks simultaneously (k = 1 → 16) under two strategies:
    Strategy 1: ascending BIacc  (least accuracy-impactful blocks first)
    Strategy 3: ascending BIrep  (least representation-impactful blocks first)

At each step k, measures:
    Real (GPU forward passes):
        acc_k       — test-set accuracy with k blocks ablated
        conf_mean   — mean top-1 softmax probability across all test samples
        H_mean      — mean Shannon entropy (nats) across all test samples
        H_Cl        — mean Shannon entropy restricted to correctly-predicted samples
        BIrep_k     — 1 − linear_cka(F_intact, F_ablated_k) on calibration set

    Simulated (from phase2_results, no GPU):
        cumulative_biacc   — sum of bi_acc for first k blocks in order
        cumulative_birep   — sum of bi_rep for first k blocks in order

Output saved to: results/progressive_pruning_real.json

Exposes:
    run_real_progressive_pruning(model, registry, test_loader, calib_loader,
                                 device, phase2_results)  → dict
"""

import json
import logging
from contextlib import ExitStack

import torch

from config import (
    DOWNSAMPLING_BLOCKS,
    REF_REPR,
    RESULTS_DIR,
    TARGET_BLOCKS,
)
from phase2_metrics import linear_cka
from utils import ActivationCapture, ablated_block, gap

logger = logging.getLogger(__name__)

PRUNING_REAL_RESULTS = RESULTS_DIR / "progressive_pruning_real.json"

_EPS = 1e-10


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in nats, per sample. probs: (N, C) — must be valid softmax."""
    return -(probs * torch.log(probs + _EPS)).sum(dim=1)   # (N,)


@torch.no_grad()
def _test_pass(model, test_loader, device):
    """
    Single full forward pass over the test set.

    Returns (acc, conf_mean, H_mean, H_Cl):
        acc       — fraction of correct predictions
        conf_mean — mean top-1 softmax probability
        H_mean    — mean Shannon entropy over all samples
        H_Cl      — mean Shannon entropy over correctly predicted samples (C_l)
                     C_l is recomputed fresh from current predictions.
    """
    logit_chunks = []
    label_chunks = []

    for images, labels in test_loader:
        images = images.to(device)
        logit_chunks.append(model(images).cpu())
        label_chunks.append(labels.cpu())

    logits = torch.cat(logit_chunks, dim=0)    # (N, C)
    labels = torch.cat(label_chunks, dim=0)    # (N,)

    probs = torch.softmax(logits, dim=1)        # (N, C)
    preds = probs.argmax(dim=1)                # (N,)
    correct = preds == labels                   # (N,) bool

    acc       = correct.float().mean().item()
    conf_mean = probs.max(dim=1).values.mean().item()

    H         = _entropy(probs)                # (N,)
    H_mean    = H.mean().item()
    H_Cl      = H[correct].mean().item() if correct.any() else 0.0

    return acc, conf_mean, H_mean, H_Cl


@torch.no_grad()
def _calib_pass(model, calib_loader, device) -> torch.Tensor:
    """
    Extract avgpool representations (N, 2048) from the calibration set
    in the model's current (possibly ablated) state.
    """
    feats = []
    with ActivationCapture(model.avgpool) as cap:
        for images, _ in calib_loader:
            images = images.to(device)
            _ = model(images)
            feats.append(gap(cap.output).cpu())
    return torch.cat(feats, dim=0)    # (N, 2048)


def _build_order(phase2_results: dict, metric: str) -> list:
    """Return TARGET_BLOCKS sorted by `metric` ascending."""
    scores = phase2_results[metric]
    return sorted(TARGET_BLOCKS, key=lambda b: scores[b])


def _compute_simulated(order: list, phase2_results: dict) -> dict:
    """
    Cumulative BIacc and BIrep sums along `order` (no GPU).

    Keys are "k1" … "k16".
    """
    bi_acc = phase2_results["bi_acc"]
    bi_rep = phase2_results["bi_rep"]
    out = {}
    cum_acc = 0.0
    cum_rep = 0.0
    for k, block in enumerate(order, start=1):
        cum_acc += bi_acc[block]
        cum_rep += bi_rep[block]
        out[f"k{k}"] = {
            "cumulative_biacc": round(cum_acc, 6),
            "cumulative_birep": round(cum_rep, 6),
        }
    return out


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def run_real_progressive_pruning(
    model,
    registry: dict,
    test_loader,
    calib_loader,
    device: torch.device,
    phase2_results: dict,
) -> dict:
    """
    Real progressive pruning: simultaneously ablate the first k blocks and
    measure actual accuracy / CKA disruption / confidence / entropy at each k.

    Strategies tested:
        strategy1 — blocks sorted by bi_acc ascending (least accuracy-impactful first)
        strategy3 — blocks sorted by bi_rep ascending (least representation-impactful first)

    F_intact is loaded from REF_REPR (never recomputed).
    All k context managers are entered via ExitStack before any forward pass,
    guaranteeing atomic ablation and safe restoration even on exception.

    Saves results to results/progressive_pruning_real.json and returns them.
    """
    logger.info("=" * 60)
    logger.info("Real Progressive Pruning Analysis")
    logger.info("=" * 60)

    # Load F_intact once — do not recompute
    logger.info(f"Loading F_intact from {REF_REPR} …")
    F_intact = torch.load(REF_REPR, map_location="cpu")
    logger.info(f"  F_intact shape: {tuple(F_intact.shape)}")

    strategies = {
        "strategy1": _build_order(phase2_results, "bi_acc"),
        "strategy3": _build_order(phase2_results, "bi_rep"),
    }

    results = {}

    for strategy_key, order in strategies.items():
        logger.info("─" * 60)
        logger.info(f"{strategy_key} order: {order}")

        real_results = {}
        simulated    = _compute_simulated(order, phase2_results)

        for k in range(1, len(order) + 1):
            blocks_k = order[:k]
            logger.info(f"  k={k:2d}: ablating {blocks_k}")

            # Enter all k context managers before any forward pass (ExitStack)
            with ExitStack() as stack:
                for block_name in blocks_k:
                    stack.enter_context(
                        ablated_block(
                            registry[block_name],
                            is_downsampling=(block_name in DOWNSAMPLING_BLOCKS),
                        )
                    )

                # Pass A — test set (accuracy, confidence, entropy)
                acc, conf_mean, H_mean, H_Cl = _test_pass(model, test_loader, device)

                # Pass B — calibration set (avgpool representations for CKA)
                F_ablated = _calib_pass(model, calib_loader, device)

            # BIrep_k = 1 − CKA(F_intact, F_ablated_k)
            cka_val = linear_cka(
                F_intact.to(device).float(),
                F_ablated.to(device).float(),
            )
            birep_k = 1.0 - float(cka_val)

            real_results[f"k{k}"] = {
                "acc":       round(acc,        6),
                "BIrep":     round(birep_k,    6),
                "H_mean":    round(H_mean,     6),
                "conf_mean": round(conf_mean,  6),
                "H_Cl":      round(H_Cl,       6),
            }
            logger.info(
                f"    acc={acc:.4f}  BIrep={birep_k:.4f}  "
                f"H={H_mean:.4f}  conf={conf_mean:.4f}  H_Cl={H_Cl:.4f}"
            )

        results[strategy_key] = {
            "order":     order,
            "real":      real_results,
            "simulated": simulated,
        }

    with open(PRUNING_REAL_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved → {PRUNING_REAL_RESULTS}")

    return results
