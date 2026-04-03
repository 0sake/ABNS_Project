"""
phase2_metrics.py — Phase 2: computation of the three Block Influence metrics
for all 16 ResNet-50 Bottleneck blocks.

Metrics:
    BIgeo  (Step 2.1) — Geometric Block Influence via cosine similarity
    BIacc  (Step 2.2) — Ablation Accuracy drop
    BIrep  (Step 2.3) — Ablation CKA (representational disruption)

Output: results/phase2_results.json
"""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import (
    BASELINE_ACCURACY, BATCH_SIZE_CALIB, CKA_EPSILON,
    DOWNSAMPLING_BLOCKS, MULTILAYER_STAGES,
    N_CALIBRATION, PHASE2_RESULTS, SEED, TARGET_BLOCKS,
)
from model import ResNet50_CIFAR, build_block_registry, evaluate_accuracy, get_stage_output_module
from utils import ActivationCapture, ablated_block, gap, tensor_checksum
from bi_rep_extended import run_extended_birep

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CKA implementation (Step 2.3)
# ──────────────────────────────────────────────

def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = CKA_EPSILON) -> float:
    """
    Compute Linear CKA between two feature matrices X, Y ∈ R^{N×D}.

    Following Kornblith et al. (2019):
        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F · ||Y^T Y||_F)

    Both matrices are mean-centred column-wise before computation.
    Mean-centring is non-negotiable: without it CKA measures mean activation
    magnitudes rather than representational geometry.

    An epsilon guard prevents division by zero in the case of representational
    collapse (all activations constant across samples).
    """
    # Mean-centring (column-wise)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram-like cross-covariance norm
    XtY = X.T @ Y                             # D × D
    XtX = X.T @ X
    YtY = Y.T @ Y

    numerator   = torch.norm(XtY, p="fro") ** 2
    denominator = torch.norm(XtX, p="fro") * torch.norm(YtY, p="fro") + eps

    if denominator < 1e-6:
        logger.warning("  CKA denominator near zero — possible representational collapse.")

    return (numerator / denominator).item()


# ──────────────────────────────────────────────
# Step 2.1: BIgeo
# ──────────────────────────────────────────────

@torch.no_grad()
def compute_bi_geo(
    model: ResNet50_CIFAR,
    registry: dict,
    calib_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Geometric Block Influence (BIgeo):

        BI_geo^(l) = 1 − (1/N) Σ_i CosSim(GAP(Input_l^(i)), GAP(Output_l^(i)))

    A score near 0 → block produces negligible transformation (redundant).
    A score near 1 → block performs a substantial transformation.

    For downsampling blocks the input is projected via the block's own
    downsample module before GAP + cosine similarity, so both sides share
    the same dimensionality.

    Implementation: passive measurement on the INTACT model via forward hooks.
    Hooks are registered/removed per block to prevent cross-block interference.
    """
    logger.info("─" * 60)
    logger.info("Step 2.1 — BIgeo computation")
    t_start = time.time()

    model.eval()
    bi_geo = {}

    for block_name in TARGET_BLOCKS:
        block = registry[block_name]
        is_ds = block_name in DOWNSAMPLING_BLOCKS

        cosine_sims = []

        with ActivationCapture(block) as cap:
            for images, _ in calib_loader:
                images = images.to(device)
                _ = model(images)

                x_in  = cap.input                       # raw block input
                x_out = cap.output                      # raw block output

                # For downsampling blocks project input to match output channels
                if is_ds and block.downsample is not None:
                    x_in = block.downsample(x_in)

                x_in  = gap(x_in)                       # (B, C)
                x_out = gap(x_out)                      # (B, C)

                sim = F.cosine_similarity(x_in, x_out, dim=1)  # (B,)
                cosine_sims.append(sim.cpu())

        mean_cos = torch.cat(cosine_sims).mean().item()
        bi_geo[block_name] = round(1.0 - mean_cos, 6)
        logger.info(f"  {block_name:12s}  BIgeo = {bi_geo[block_name]:.6f}")

    elapsed = time.time() - t_start
    logger.info(f"BIgeo done in {elapsed:.1f}s")
    return bi_geo


# ──────────────────────────────────────────────
# Step 2.2: BIacc
# ──────────────────────────────────────────────

@torch.no_grad()
def compute_bi_acc(
    model: ResNet50_CIFAR,
    registry: dict,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    baseline_acc: float,
) -> dict[str, float]:
    """
    Ablation Accuracy BI (BIacc):

        BI_acc^(l) = baseline_accuracy − accuracy_ablated^(l)

    Block ablation: the block's forward method is replaced with an identity
    (or downsample-only for transition blocks) via the `ablated_block` context
    manager, which guarantees restoration in a finally clause.

    Positive value → accuracy drop → block is important.
    Negative value → accuracy improved by removal → anomaly, flagged.
    Zero            → no measurable impact → logged as low-importance.

    Computational note: 40 forward passes per block × 16 blocks = 640 total
    test-set passes. Most expensive step in Phase 2.
    """
    logger.info("─" * 60)
    logger.info("Step 2.2 — BIacc computation")
    t_start = time.time()

    model.eval()
    bi_acc = {}
    anomalies = []

    for block_name in TARGET_BLOCKS:
        block = registry[block_name]
        is_ds = block_name in DOWNSAMPLING_BLOCKS

        t0 = time.time()
        with ablated_block(block, is_downsampling=is_ds):
            acc_abl = evaluate_accuracy(model, test_loader, device)

        delta = round(baseline_acc - acc_abl, 6)
        bi_acc[block_name] = delta
        elapsed_b = time.time() - t0

        tag = ""
        if delta < 0:
            tag = " ⚠ NEGATIVE (anomaly)"
            anomalies.append(block_name)
        elif delta == 0.0:
            tag = " (no measurable impact)"

        logger.info(
            f"  {block_name:12s}  acc_abl={acc_abl:.4f}  "
            f"ΔAcc={delta:+.6f}{tag}  [{elapsed_b:.1f}s]"
        )

    if anomalies:
        logger.warning(f"BIacc anomalies (negative Δ): {anomalies}")

    elapsed = time.time() - t_start
    logger.info(f"BIacc done in {elapsed:.1f}s")
    return bi_acc


# ──────────────────────────────────────────────
# Feature extraction helper (used in BIrep)
# ──────────────────────────────────────────────

@torch.no_grad()
def _extract_final_representations(
    model: ResNet50_CIFAR,
    calib_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Extract GAP features (N × 2048) from the model in its current state."""
    feats = []
    with ActivationCapture(model.avgpool) as cap:
        for images, _ in calib_loader:
            images = images.to(device)
            _ = model(images)
            feats.append(gap(cap.output).cpu())
    return torch.cat(feats, dim=0)


# ──────────────────────────────────────────────
# Step 2.3: BIrep + multi-layer propagation
# ──────────────────────────────────────────────

@torch.no_grad()
def compute_bi_rep(
    model: ResNet50_CIFAR,
    registry: dict,
    calib_loader: torch.utils.data.DataLoader,
    device: torch.device,
    F_intact: torch.Tensor,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Ablation CKA (BIrep):

        BI_rep^(l) = 1 − CKA(F_intact, F_ablated^(l))

    Compares final pre-classifier representations between the intact model
    and the model with block l ablated.

    0 → ablation produces no measurable representational change.
    1 → catastrophic representational disruption.

    After all 16 blocks are scored, the top-3 blocks by discrepancy
    (bi_rep − bi_acc) receive a multi-layer CKA propagation profile,
    extracted at the output of each of the four ResNet stages + avgpool.

    Returns:
        bi_rep            : {block_name: scalar BI score}
        bi_rep_multilayer : {block_name: {stage_name: CKA score}} for top-3
    """
    logger.info("─" * 60)
    logger.info("Step 2.3 — BIrep computation")
    t_start = time.time()

    model.eval()
    F_intact_dev = F_intact.to(device)  # keep on GPU for CKA computation
    bi_rep = {}
    collapse_flags = []

    for block_name in TARGET_BLOCKS:
        block = registry[block_name]
        is_ds = block_name in DOWNSAMPLING_BLOCKS

        t0 = time.time()
        with ablated_block(block, is_downsampling=is_ds):
            F_abl = _extract_final_representations(model, calib_loader, device)

        cka_score = linear_cka(F_intact_dev.cpu(), F_abl)
        bi_rep[block_name] = round(1.0 - cka_score, 6)

        # Range check
        if not (0.0 <= bi_rep[block_name] <= 1.0):
            logger.error(f"  {block_name}: BIrep={bi_rep[block_name]:.6f} OUT OF RANGE [0,1]!")
            collapse_flags.append(block_name)

        logger.info(
            f"  {block_name:12s}  CKA={cka_score:.6f}  "
            f"BIrep={bi_rep[block_name]:.6f}  [{time.time()-t0:.1f}s]"
        )

    # ── Multi-layer propagation for top-3 by discrepancy ──────────────────
    bi_rep_multilayer: dict[str, dict[str, float]] = {}

    # Requires bi_acc — we'll compute discrepancy placeholder using bi_rep only
    # (caller will pass bi_acc if available; here we trigger on highest bi_rep)
    top3 = sorted(bi_rep, key=lambda k: bi_rep[k], reverse=True)[:3]
    logger.info(f"Multi-layer CKA propagation for top-3: {top3}")

    for block_name in top3:
        block = registry[block_name]
        is_ds = block_name in DOWNSAMPLING_BLOCKS
        stage_cka = {}

        with ablated_block(block, is_downsampling=is_ds):
            for stage_name in MULTILAYER_STAGES:
                stage_mod = get_stage_output_module(model, stage_name)
                stage_feats = []

                with ActivationCapture(stage_mod) as cap:
                    for images, _ in calib_loader:
                        images = images.to(device)
                        _ = model(images)
                        stage_feats.append(gap(cap.output).cpu())

                F_stage_abl = torch.cat(stage_feats, dim=0)

                # Reference: extract intact-model stage representations once
                # (done lazily below — see _extract_stage_references)
                stage_cka[stage_name] = None   # placeholder, filled below

        bi_rep_multilayer[block_name] = stage_cka

    # Fill placeholders: compare against intact stage representations
    intact_stage_refs = _extract_stage_references(model, calib_loader, device)

    for block_name in top3:
        block = registry[block_name]
        is_ds = block_name in DOWNSAMPLING_BLOCKS

        with ablated_block(block, is_downsampling=is_ds):
            for stage_name in MULTILAYER_STAGES:
                stage_mod = get_stage_output_module(model, stage_name)
                stage_feats = []

                with ActivationCapture(stage_mod) as cap:
                    for images, _ in calib_loader:
                        images = images.to(device)
                        _ = model(images)
                        stage_feats.append(gap(cap.output).cpu())

                F_stage_abl = torch.cat(stage_feats, dim=0)
                F_stage_ref = intact_stage_refs[stage_name]
                cka_s = linear_cka(F_stage_ref, F_stage_abl)
                bi_rep_multilayer[block_name][stage_name] = round(cka_s, 6)
                logger.info(
                    f"  [multilayer] {block_name} @ {stage_name}: CKA={cka_s:.6f}"
                )

    elapsed = time.time() - t_start
    logger.info(f"BIrep done in {elapsed:.1f}s")

    if collapse_flags:
        logger.warning(f"Representational collapse detected in: {collapse_flags}")

    return bi_rep, bi_rep_multilayer


@torch.no_grad()
def _extract_stage_references(
    model: ResNet50_CIFAR,
    calib_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Extract intact-model representations at each MULTILAYER_STAGES stage."""
    from config import MULTILAYER_STAGES
    refs = {s: [] for s in MULTILAYER_STAGES}

    # Register hooks for all stages simultaneously for efficiency
    captures = {s: ActivationCapture(get_stage_output_module(model, s))
                for s in MULTILAYER_STAGES}

    for images, _ in calib_loader:
        images = images.to(device)
        _ = model(images)
        for s in MULTILAYER_STAGES:
            refs[s].append(gap(captures[s].output).cpu())

    for cap in captures.values():
        cap.remove()

    return {s: torch.cat(refs[s], dim=0) for s in MULTILAYER_STAGES}


# ──────────────────────────────────────────────
# Step 2.4: Phase completion checks
# ──────────────────────────────────────────────

def run_phase2_checks(
    bi_geo: dict, bi_acc: dict, bi_rep: dict
) -> list[str]:
    """
    Post-computation validation checks (Step 2.4).
    Returns a list of warning strings (empty if all checks pass).
    """
    warnings = []

    # Range checks
    for name, val in bi_geo.items():
        if val > 1.0:
            warnings.append(f"BIgeo[{name}]={val:.4f} > 1.0 (unexpected for converged network)")

    for name, val in bi_rep.items():
        if not (0.0 <= val <= 1.0):
            warnings.append(f"BIrep[{name}]={val:.4f} OUT OF RANGE [0,1]")

    for name, val in bi_acc.items():
        if val < -0.01:
            warnings.append(f"BIacc[{name}]={val:.4f} NEGATIVE (ablation improved accuracy)")

    # Monotonicity spot-check
    if bi_acc.get("layer4.2", 0) < bi_acc.get("layer1.0", 0):
        warnings.append(
            f"Monotonicity heuristic violated: BIacc[layer4.2]={bi_acc.get('layer4.2'):.4f} "
            f"< BIacc[layer1.0]={bi_acc.get('layer1.0'):.4f}. Manual inspection advised."
        )

    for w in warnings:
        logger.warning(f"  CHECK: {w}")

    if not warnings:
        logger.info("Phase 2 checks: all passed ✓")

    return warnings


# ──────────────────────────────────────────────
# Save Phase 2 results
# ──────────────────────────────────────────────

def save_phase2_results(
    bi_geo: dict,
    bi_acc: dict,
    bi_rep: dict,
    bi_rep_multilayer: dict,
    baseline_acc: float,
    device: torch.device,
    calib_indices_path: Path,
    env: dict,
    warnings: list[str],
) -> None:
    """Serialise all Phase 2 results to phase2_results.json."""
    payload = {
        "metadata": {
            "model": "ResNet50-CIFAR",
            "seed": SEED,
            "device": str(device),
            "calibration_set_indices_path": str(calib_indices_path),
            "n_calibration": N_CALIBRATION,
            "n_test": 10_000,
            "baseline_accuracy": baseline_acc,
            "env": env,
        },
        "bi_geo": bi_geo,
        "bi_acc": bi_acc,
        "bi_rep": bi_rep,
        "bi_rep_multilayer": bi_rep_multilayer,
        "phase2_warnings": warnings,
    }
    with open(PHASE2_RESULTS, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Phase 2 results saved → {PHASE2_RESULTS}")


# ──────────────────────────────────────────────
# Entry point: run full Phase 2
# ──────────────────────────────────────────────

def run_phase2(
    model: ResNet50_CIFAR,
    test_loader,
    calib_loader,
    device: torch.device,
    F_intact: torch.Tensor,
    baseline_acc: float,
    calib_indices_path: Path,
    env: dict,
) -> dict:
    """
    Execute all three metrics and post-checks.
    Returns the full results dict (also written to disk).
    """
    registry = build_block_registry(model)

    logger.info("=" * 60)
    logger.info("PHASE 2: Block Influence Metrics")
    logger.info("=" * 60)

    bi_geo = compute_bi_geo(model, registry, calib_loader, device)
    bi_acc = compute_bi_acc(model, registry, test_loader, device, baseline_acc)
    bi_rep, bi_rep_ml = compute_bi_rep(model, registry, calib_loader, device, F_intact)

    warnings = run_phase2_checks(bi_geo, bi_acc, bi_rep)
    save_phase2_results(
        bi_geo, bi_acc, bi_rep, bi_rep_ml,
        baseline_acc, device, calib_indices_path, env, warnings
    )

    extended = run_extended_birep(model, registry, calib_loader, device, F_intact)

    return {
        "bi_geo": bi_geo,
        "bi_acc": bi_acc,
        "bi_rep": bi_rep,
        "bi_rep_multilayer": bi_rep_ml,
        "bi_rep_gram": extended["bi_rep_gram"],
        "bi_rep_class": extended["bi_rep_class"],
        "warnings": warnings,
    }
