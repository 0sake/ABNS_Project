"""
phase3_analysis.py — Phase 3: Comparative analysis and interpretation.

Steps:
    3.1  Rank correlation (Kendall's tau) between all three metric pairs
    3.2  Top-k Jaccard similarity (k ∈ {3, 5})
    3.3  Silent failure analysis (BIrep ↑, BIacc ↓)
         3.3.1 Candidate identification (tercile thresholds + discrepancy score)
         3.3.2 Representational geometry (multi-layer CKA profile, per-class CKA)
         3.3.3 Decision confidence (softmax entropy, top-1 confidence, Wilcoxon)

Outputs:
    results/phase3_correlations.json
    results/phase3_jaccard.json
    results/phase3_silent_failure_{block_id}.json
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from config import (
    ALPHA, JACCARD_K_VALUES, PHASE2_RESULTS,
    PHASE3_CORRELATIONS, PHASE3_JACCARD, PHASE3_PRUNING,
    RESULTS_DIR, SECONDARY_DELTA_THRESHOLD, TARGET_BLOCKS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Step 3.1 — Rank Correlation Analysis (Kendall's tau)
# ═══════════════════════════════════════════════════════════════════════════

def compute_rank_correlations(
    bi_geo: dict, bi_acc: dict, bi_rep: dict
) -> dict:
    """
    Compute all three pairwise Kendall's tau values and associated p-values.

    Kendall's tau is preferred over Spearman's rho for its direct probabilistic
    interpretation (probability that a random block pair is ranked in the same
    order by both metrics) and its robustness with n=16.

    Returns a dict with keys: tau, p_value, significant for each pair.
    """
    metrics = {"bi_geo": bi_geo, "bi_acc": bi_acc, "bi_rep": bi_rep}
    pairs = [
        ("bi_geo", "bi_acc"),
        ("bi_geo", "bi_rep"),
        ("bi_acc", "bi_rep"),
    ]

    results = {}
    logger.info("─" * 60)
    logger.info("Step 3.1 — Kendall's tau rank correlation")

    for m1, m2 in pairs:
        v1 = [metrics[m1][b] for b in TARGET_BLOCKS]
        v2 = [metrics[m2][b] for b in TARGET_BLOCKS]

        tau, p_val = stats.kendalltau(v1, v2)
        significant = bool(p_val < ALPHA)

        key = f"{m1}_vs_{m2}"
        results[key] = {
            "tau": round(float(tau), 4),
            "p_value": round(float(p_val), 6),
            "significant": significant,
            "interpretation": _interpret_tau(m1, m2, tau, significant),
        }
        logger.info(
            f"  τ({m1}, {m2}) = {tau:.4f}  p={p_val:.4f}  "
            f"{'✓ sig' if significant else '✗ not sig'}"
        )

    _log_tau_interpretation(results)
    return results


def _interpret_tau(m1: str, m2: str, tau: float, sig: bool) -> str:
    """Map tau value + significance to an interpretation string."""
    strength = "strong" if abs(tau) > 0.6 else ("moderate" if abs(tau) > 0.3 else "weak")
    direction = "positive" if tau > 0 else "negative"
    sig_str = "statistically significant" if sig else "not significant"
    return f"{strength} {direction} correlation ({sig_str})"


def _log_tau_interpretation(results: dict) -> None:
    """Log the joint interpretation pattern (per pipeline spec Table)."""
    tau_acc_rep = results["bi_acc_vs_bi_rep"]["tau"]
    tau_geo_acc = results["bi_geo_vs_bi_acc"]["tau"]
    tau_geo_rep = results["bi_geo_vs_bi_rep"]["tau"]

    if all(abs(t) > 0.6 for t in [tau_acc_rep, tau_geo_acc, tau_geo_rep]):
        pattern = "All three metrics converge — CKA provides limited additional signal."
    elif tau_acc_rep < 0.3:
        pattern = "BIacc and BIrep fundamentally disagree — strong evidence for silent failure."
    elif abs(tau_acc_rep) > 0.6 and abs(tau_geo_acc) < 0.4:
        pattern = "Accuracy and CKA agree; ShortGPT captures a partially distinct signal."
    elif abs(tau_geo_rep) > 0.6 and abs(tau_geo_acc) < 0.4:
        pattern = "Geometric redundancy predicts representational importance better than accuracy."
    else:
        pattern = "Mixed pattern — inspect individual pairwise values."

    logger.info(f"  Joint interpretation: {pattern}")


# ═══════════════════════════════════════════════════════════════════════════
# Step 3.2 — Top-k Jaccard Similarity
# ═══════════════════════════════════════════════════════════════════════════

def compute_jaccard(
    bi_geo: dict, bi_acc: dict, bi_rep: dict,
    k_values: list = JACCARD_K_VALUES,
) -> dict:
    """
    For each k in k_values, compute the 3×3 pairwise Jaccard similarity matrix
    over the top-k most important blocks according to each metric.

    Handles ties: both tied blocks at the k-boundary are included in the set.

    Returns nested dict: {k: {pair_key: jaccard_value}}
    """
    metrics = {"bi_geo": bi_geo, "bi_acc": bi_acc, "bi_rep": bi_rep}
    metric_names = list(metrics.keys())
    pairs = [
        ("bi_geo", "bi_acc"),
        ("bi_geo", "bi_rep"),
        ("bi_acc", "bi_rep"),
    ]

    logger.info("─" * 60)
    logger.info("Step 3.2 — Top-k Jaccard similarity")

    results = {}

    for k in k_values:
        top_k_sets = {}
        for name, scores in metrics.items():
            top_k_sets[name] = _top_k_with_ties(scores, k)
            logger.info(f"  top-{k} {name}: {sorted(top_k_sets[name])}")

        jaccard_results = {}
        for m1, m2 in pairs:
            S1, S2 = top_k_sets[m1], top_k_sets[m2]
            j = _jaccard(S1, S2)
            key = f"{m1}_vs_{m2}"
            jaccard_results[key] = {
                "jaccard": round(j, 4),
                "intersection": sorted(S1 & S2),
                "union": sorted(S1 | S2),
                "k": k,
            }
            logger.info(f"  J(k={k}) {m1} vs {m2}: {j:.4f}  ∩={sorted(S1 & S2)}")

        results[str(k)] = jaccard_results

    return results


def _top_k_with_ties(scores: dict, k: int) -> set:
    """Return the top-k blocks by score, expanding the set at ties."""
    sorted_blocks = sorted(scores, key=lambda b: scores[b], reverse=True)
    threshold = scores[sorted_blocks[k - 1]] if len(sorted_blocks) >= k else -float("inf")
    return {b for b in scores if scores[b] >= threshold and
            sorted_blocks.index(b) < k or scores[b] == threshold}


def _jaccard(A: set, B: set) -> float:
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


# ═══════════════════════════════════════════════════════════════════════════
# Step 3.3 — Silent Failure Analysis
# ═══════════════════════════════════════════════════════════════════════════

def identify_silent_failure_candidates(
    bi_acc: dict, bi_rep: dict
) -> tuple[list[str], list[str], dict[str, float]]:
    """
    Step 3.3.1: Identify blocks with low BIacc + high BIrep.

    Thresholds are tercile-based (adaptive to empirical distribution):
        θ_acc = lower tercile of BIacc
        θ_rep = upper tercile of BIrep

    Candidates are ranked by discrepancy: Δ(l) = BIrep(l) − BIacc(l).
    Primary candidate: block with highest Δ.
    Secondary candidates: Δ > SECONDARY_DELTA_THRESHOLD (up to 2 additional).

    Returns:
        primary    : list of length 1 (primary candidate block name)
        secondary  : list of up to 2 secondary candidate block names
        delta_map  : {block_name: Δ score} for all blocks
    """
    logger.info("─" * 60)
    logger.info("Step 3.3.1 — Silent failure candidate identification")

    acc_vals = np.array([bi_acc[b] for b in TARGET_BLOCKS])
    rep_vals = np.array([bi_rep[b] for b in TARGET_BLOCKS])

    theta_acc = np.percentile(acc_vals, 33.3)   # lower tercile
    theta_rep = np.percentile(rep_vals, 66.7)   # upper tercile

    logger.info(f"  θ_acc (lower tercile) = {theta_acc:.6f}")
    logger.info(f"  θ_rep (upper tercile) = {theta_rep:.6f}")

    delta_map = {b: round(bi_rep[b] - bi_acc[b], 6) for b in TARGET_BLOCKS}
    sorted_by_delta = sorted(delta_map, key=lambda b: delta_map[b], reverse=True)

    candidates = [
        b for b in sorted_by_delta
        if bi_acc[b] < theta_acc and bi_rep[b] > theta_rep
    ]

    if not candidates:
        # Null result protocol: use highest-Δ block regardless of thresholds
        logger.warning(
            "No block satisfies the joint silent failure criteria. "
            "Null result protocol: using highest-Δ block for deep-dive analysis."
        )
        primary = [sorted_by_delta[0]]
        secondary = []
    else:
        primary = [candidates[0]]
        secondary = [
            b for b in candidates[1:]
            if delta_map[b] > SECONDARY_DELTA_THRESHOLD
        ][:2]

    logger.info(f"  Primary candidate  : {primary}")
    logger.info(f"  Secondary candidates: {secondary}")
    for b in primary + secondary:
        logger.info(
            f"    {b:12s}  BIacc={bi_acc[b]:.6f}  "
            f"BIrep={bi_rep[b]:.6f}  Δ={delta_map[b]:.6f}"
        )

    return primary, secondary, delta_map


@torch.no_grad()
def analyse_representational_geometry(
    block_name: str,
    bi_rep_multilayer: dict,
    model,
    registry: dict,
    calib_loader,
    device: torch.device,
    F_intact: torch.Tensor,
) -> dict:
    """
    Step 3.3.2: Per-class CKA and multi-layer propagation profile.

    Returns a dict with:
        propagation_profile : {stage: CKA_score}
        per_class_cka       : {class_idx: CKA_score}
    """
    from config import MULTILAYER_STAGES
    from utils import ActivationCapture, ablated_block, gap
    from phase2_metrics import linear_cka, _extract_stage_references
    from model import get_stage_output_module
    from data import get_class_conditional_loaders
    from torchvision import datasets
    from config import CIFAR10_MEAN, CIFAR10_STD
    from torchvision import transforms

    logger.info(f"  Step 3.3.2 — Geometry analysis for {block_name}")

    block = registry[block_name]
    is_ds = block_name in __import__("config").DOWNSAMPLING_BLOCKS

    # Propagation profile (already computed in Phase 2 for top-3)
    propagation = bi_rep_multilayer.get(block_name, {})
    if propagation:
        logger.info(f"    Propagation profile (from Phase 2): {propagation}")
    else:
        logger.warning(f"    No multilayer data for {block_name}; skipping propagation profile.")

    # Per-class CKA on calibration set
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10("./data", train=True, download=False, transform=tf)
    calib_indices = torch.load(__import__("config").CALIB_INDICES)
    class_loaders = get_class_conditional_loaders(train_ds, calib_indices)

    per_class_cka = {}
    for cls_idx, cls_loader in class_loaders.items():
        # Extract intact class representations
        cls_feats_intact = []
        with ActivationCapture(model.avgpool) as cap:
            for images, _ in cls_loader:
                images = images.to(device)
                _ = model(images)
                cls_feats_intact.append(gap(cap.output).cpu())
        F_cls_intact = torch.cat(cls_feats_intact, dim=0)

        # Extract ablated class representations
        with ablated_block(block, is_downsampling=is_ds):
            cls_feats_abl = []
            with ActivationCapture(model.avgpool) as cap:
                for images, _ in cls_loader:
                    images = images.to(device)
                    _ = model(images)
                    cls_feats_abl.append(gap(cap.output).cpu())
        F_cls_abl = torch.cat(cls_feats_abl, dim=0)

        cka_cls = linear_cka(F_cls_intact, F_cls_abl)
        per_class_cka[cls_idx] = round(cka_cls, 6)

    variance = float(np.var(list(per_class_cka.values())))
    logger.info(f"    Per-class CKA: {per_class_cka}")
    logger.info(f"    Class-CKA variance: {variance:.6f} "
                f"({'structured' if variance > 0.01 else 'uniform'} disruption)")

    return {
        "propagation_profile": propagation,
        "per_class_cka": {str(k): v for k, v in per_class_cka.items()},
        "class_cka_variance": variance,
    }


@torch.no_grad()
def analyse_decision_confidence(
    block_name: str,
    model,
    registry: dict,
    test_loader,
    device: torch.device,
    baseline_acc: float,
) -> dict:
    """
    Step 3.3.3: Softmax entropy + top-1 confidence analysis on correctly
    classified samples (set C_l).

    Tests whether representational disruption manifests as increased output
    uncertainty even when the classification decision is unchanged.

    Statistical test: one-sided Wilcoxon signed-rank test on paired entropy
    values (H_intact, H_ablated) for samples in C_l.
    Non-parametric test chosen because entropy is bounded and may deviate
    from normality.
    """
    from utils import ablated_block
    from model import evaluate_accuracy_with_logits

    logger.info(f"  Step 3.3.3 — Confidence analysis for {block_name}")

    block = registry[block_name]
    is_ds = block_name in __import__("config").DOWNSAMPLING_BLOCKS

    # Intact model: softmax probs + labels
    acc_intact, probs_intact, labels = evaluate_accuracy_with_logits(
        model, test_loader, device
    )

    # Ablated model: softmax probs
    with ablated_block(block, is_downsampling=is_ds):
        acc_abl, probs_abl, _ = evaluate_accuracy_with_logits(
            model, test_loader, device
        )

    preds_intact = probs_intact.argmax(dim=1)
    preds_abl    = probs_abl.argmax(dim=1)

    # C_l: samples where ablated model classifies correctly (same as intact)
    C_l = (preds_abl == labels) & (preds_intact == labels)
    # E_l: samples where ablation changed the prediction
    E_l = preds_abl != labels

    logger.info(
        f"    |C_l| = {C_l.sum().item()}  |E_l| = {E_l.sum().item()}  "
        f"acc_intact={acc_intact:.4f}  acc_abl={acc_abl:.4f}"
    )

    def shannon_entropy(probs: torch.Tensor) -> torch.Tensor:
        """Shannon entropy H = -Σ p log p, with clamp for numerical safety."""
        p = probs.clamp(min=1e-10)
        return -(p * p.log()).sum(dim=1)

    H_intact = shannon_entropy(probs_intact[C_l])
    H_abl    = shannon_entropy(probs_abl[C_l])
    delta_H  = (H_abl - H_intact)

    # Wilcoxon signed-rank test (one-sided: H_ablated > H_intact)
    stat, p_val = stats.wilcoxon(
        H_abl.numpy(), H_intact.numpy(),
        alternative="greater"
    )

    # Top-1 confidence comparison
    conf_intact = probs_intact[C_l].max(dim=1).values
    conf_abl    = probs_abl[C_l].max(dim=1).values
    _, p_conf   = stats.wilcoxon(
        conf_intact.numpy(), conf_abl.numpy(),
        alternative="greater"   # intact > ablated (confidence reduction)
    )

    result = {
        "block": block_name,
        "acc_intact": round(acc_intact, 6),
        "acc_ablated": round(acc_abl, 6),
        "bi_acc": round(acc_intact - acc_abl, 6),
        "n_C_l": int(C_l.sum().item()),
        "n_E_l": int(E_l.sum().item()),
        "entropy": {
            "mean_H_intact": round(float(H_intact.mean()), 6),
            "mean_H_ablated": round(float(H_abl.mean()), 6),
            "mean_delta_H": round(float(delta_H.mean()), 6),
            "median_delta_H": round(float(delta_H.median()), 6),
            "wilcoxon_stat": round(float(stat), 4),
            "wilcoxon_p": round(float(p_val), 6),
            "significant": bool(p_val < ALPHA),
        },
        "top1_confidence": {
            "mean_conf_intact": round(float(conf_intact.mean()), 6),
            "mean_conf_ablated": round(float(conf_abl.mean()), 6),
            "wilcoxon_p": round(float(p_conf), 6),
            "significant": bool(p_conf < ALPHA),
        },
    }

    logger.info(
        f"    Entropy: ΔH_mean={result['entropy']['mean_delta_H']:.6f}  "
        f"Wilcoxon p={p_val:.4f}  {'✓ sig' if p_val < ALPHA else '✗ not sig'}"
    )
    logger.info(
        f"    Confidence: intact={result['top1_confidence']['mean_conf_intact']:.4f}  "
        f"ablated={result['top1_confidence']['mean_conf_ablated']:.4f}"
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 3.4 — Synthesis & internal consistency check
# ═══════════════════════════════════════════════════════════════════════════

def run_consistency_check(corr_results: dict, jaccard_results: dict) -> list[str]:
    """
    Verify that rank correlation and Jaccard results are mutually consistent.
    A low tau(BIacc, BIrep) should be accompanied by a low Jaccard at k=3.
    """
    warnings = []
    tau_acc_rep = corr_results["bi_acc_vs_bi_rep"]["tau"]
    j3 = jaccard_results.get("3", {}).get("bi_acc_vs_bi_rep", {}).get("jaccard", None)

    if j3 is not None:
        if abs(tau_acc_rep) > 0.6 and j3 < 0.3:
            warnings.append(
                f"Inconsistency: high tau(BIacc,BIrep)={tau_acc_rep:.4f} "
                f"but low Jaccard@3={j3:.4f}. Investigate."
            )
        elif abs(tau_acc_rep) < 0.3 and j3 > 0.7:
            warnings.append(
                f"Inconsistency: low tau(BIacc,BIrep)={tau_acc_rep:.4f} "
                f"but high Jaccard@3={j3:.4f}. Investigate."
            )

    for w in warnings:
        logger.warning(f"  CONSISTENCY CHECK: {w}")
    if not warnings:
        logger.info("  Consistency check: passed ✓")
    return warnings


# ═══════════════════════════════════════════════════════════════════════════
# Step 3.5 — Progressive Pruning Analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyse_progressive_pruning(bi_acc: dict, bi_rep: dict) -> dict:
    """
    Pure post-hoc simulation of three progressive block pruning orderings.
    No GPU, no forward passes — uses only values already in phase2_results.json.

    Strategy 1: ascending BIacc  (least accuracy-impactful blocks first)
    Strategy 2: descending Δ = BIrep − BIacc  (silent-failure blocks first)
    Strategy 3: ascending BIrep  (least representation-impactful blocks first)

    For each step k, cumulative_X[k-1] = sum of metric X for the first k removed blocks.
    """
    logger.info("─" * 60)
    logger.info("Step 3.5 — Progressive pruning analysis")

    order1 = sorted(TARGET_BLOCKS, key=lambda b: bi_acc[b])
    delta  = {b: bi_rep[b] - bi_acc[b] for b in TARGET_BLOCKS}
    order2 = sorted(TARGET_BLOCKS, key=lambda b: delta[b], reverse=True)
    order3 = sorted(TARGET_BLOCKS, key=lambda b: bi_rep[b])

    def cumulative(order: list, metric: dict) -> list:
        running, vals = 0.0, []
        for b in order:
            running += metric[b]
            vals.append(round(running, 6))
        return vals

    results = {
        "strategy1_order": order1,
        "strategy2_order": order2,
        "strategy3_order": order3,
        "strategy1_cumulative_biacc": cumulative(order1, bi_acc),
        "strategy1_cumulative_birep": cumulative(order1, bi_rep),
        "strategy2_cumulative_biacc": cumulative(order2, bi_acc),
        "strategy2_cumulative_birep": cumulative(order2, bi_rep),
        "strategy3_cumulative_biacc": cumulative(order3, bi_acc),
        "strategy3_cumulative_birep": cumulative(order3, bi_rep),
    }

    for i, strat in enumerate(["strategy1", "strategy2", "strategy3"], 1):
        final_acc = results[f"{strat}_cumulative_biacc"][-1]
        final_rep = results[f"{strat}_cumulative_birep"][-1]
        logger.info(
            f"  Strategy {i}: final cumulative BIacc={final_acc:.4f}  BIrep={final_rep:.4f}"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_phase3(
    model,
    registry: dict,
    test_loader,
    calib_loader,
    device: torch.device,
    F_intact: torch.Tensor,
    baseline_acc: float,
) -> dict:
    """
    Execute Phase 3 end-to-end.
    Reads Phase 2 results from disk and runs all three analysis steps.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Comparative Analysis")
    logger.info("=" * 60)

    with open(PHASE2_RESULTS) as f:
        p2 = json.load(f)

    bi_geo    = p2["bi_geo"]
    bi_acc    = p2["bi_acc"]
    bi_rep    = p2["bi_rep"]
    bi_rep_ml = p2["bi_rep_multilayer"]

    # 3.1
    corr_results = compute_rank_correlations(bi_geo, bi_acc, bi_rep)
    with open(PHASE3_CORRELATIONS, "w") as f:
        json.dump(corr_results, f, indent=2)
    logger.info(f"Correlations saved → {PHASE3_CORRELATIONS}")

    # 3.2
    jaccard_results = compute_jaccard(bi_geo, bi_acc, bi_rep)
    with open(PHASE3_JACCARD, "w") as f:
        json.dump(jaccard_results, f, indent=2)
    logger.info(f"Jaccard saved → {PHASE3_JACCARD}")

    # 3.4 consistency check (early)
    run_consistency_check(corr_results, jaccard_results)

    # 3.3
    primary, secondary, delta_map = identify_silent_failure_candidates(bi_acc, bi_rep)

    sf_results = {}
    for block_name in primary + secondary:
        geo_analysis = analyse_representational_geometry(
            block_name, bi_rep_ml, model, registry, calib_loader, device, F_intact
        )
        conf_analysis = analyse_decision_confidence(
            block_name, model, registry, test_loader, device, baseline_acc
        )
        sf_result = {
            "block": block_name,
            "bi_acc": bi_acc[block_name],
            "bi_rep": bi_rep[block_name],
            "delta": delta_map[block_name],
            "geometry": geo_analysis,
            "confidence": conf_analysis,
        }
        sf_results[block_name] = sf_result

        out_path = RESULTS_DIR / f"phase3_silent_failure_{block_name.replace('.', '_')}.json"
        with open(out_path, "w") as f:
            json.dump(sf_result, f, indent=2)
        logger.info(f"Silent failure analysis saved → {out_path}")

    # Class-pair deep-dive for primary candidate (Step 3.3 extension)
    s_intact_raw = p2.get("s_intact_10x10")
    if s_intact_raw and primary:
        from bi_rep_extended import (
            compare_class_matrices, extract_features_with_labels, class_cosine_matrix,
        )
        from config import DOWNSAMPLING_BLOCKS
        from utils import ablated_block
        S_intact   = torch.tensor(s_intact_raw).to(device)
        block_cand = registry[primary[0]]
        is_ds_cand = primary[0] in DOWNSAMPLING_BLOCKS
        with torch.no_grad():
            with ablated_block(block_cand, is_downsampling=is_ds_cand):
                F_abl_cand, labels_cand = extract_features_with_labels(
                    model, calib_loader, device
                )
        S_abl_cand = class_cosine_matrix(F_abl_cand.to(device), labels_cand.to(device))
        compare_class_matrices(S_intact, S_abl_cand, primary[0])

    # 3.5 — Progressive pruning
    pruning_results = analyse_progressive_pruning(bi_acc, bi_rep)
    with open(PHASE3_PRUNING, "w") as f:
        json.dump(pruning_results, f, indent=2)
    logger.info(f"Pruning analysis saved → {PHASE3_PRUNING}")

    return {
        "correlations": corr_results,
        "jaccard": jaccard_results,
        "silent_failure": sf_results,
        "primary_candidate": primary,
        "secondary_candidates": secondary,
        "pruning": pruning_results,
    }
