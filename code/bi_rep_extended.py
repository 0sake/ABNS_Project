"""
bi_rep_extended.py — Extended BIrep with two additional representation metrics.

As suggested by the thesis advisor, this module complements the standard
Linear CKA (computed on the N×D feature matrix) with two alternative
representations of the same underlying idea:

    1. BIrep_gram  — CKA computed on the N×N object-object Gram matrix
                     (pairwise similarity structure between samples)

    2. BIrep_class — Structural similarity between the 10×10 class-class
                     cosine similarity matrices, computed from class mean
                     vectors in the 2048-dimensional feature space.

Each metric captures a different facet of representational disruption:

    Metric          What it captures                    Sensitive to
    ─────────────────────────────────────────────────────────────────
    BIrep           Instance-level feature geometry     Representational collapse
    BIrep_gram      Pairwise sample similarity structure Manifold topology
    BIrep_class     Inter-class angular structure        Decision boundary geometry

Usage:
    from bi_rep_extended import run_extended_birep
    results = run_extended_birep(model, registry, calib_loader, device, F_intact)

Output appended to phase2_results.json under keys:
    "bi_rep_gram"     : {block_name: float}
    "bi_rep_class"    : {block_name: float}
    "s_intact_10x10"  : [[float × 10] × 10]
"""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import (
    CKA_EPSILON, DOWNSAMPLING_BLOCKS,
    PHASE2_RESULTS, TARGET_BLOCKS,
)
from model import ResNet50_CIFAR
from utils import ActivationCapture, ablated_block, gap

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — Gram Matrix CKA  (object × object)
# ─────────────────────────────────────────────────────────────────────────────

def gram_matrix(F: torch.Tensor) -> torch.Tensor:
    """
    Compute the N×N Gram matrix K = F_c @ F_c.T after column-wise mean-centering.

    Mean-centering ensures K captures covariance structure (deviations from
    the mean), not raw dot-products dominated by mean activation magnitudes.

    Args:
        F : (N, D) feature matrix
    Returns:
        K : (N, N) symmetric Gram matrix
    """
    F_c = F - F.mean(dim=0, keepdim=True)
    return F_c @ F_c.T


def gram_cka(
    F_a: torch.Tensor,
    F_b: torch.Tensor,
    eps: float = CKA_EPSILON,
) -> float:
    """
    CKA between the N×N Gram matrices derived from F_a and F_b.

    This implements the HSIC-based CKA formulation from Kornblith et al. (2019):
        CKA(K_a, K_b) = HSIC(K_a, K_b) / sqrt(HSIC(K_a,K_a) * HSIC(K_b,K_b))

    where K = F @ F.T is the linear kernel (Gram matrix) and HSIC is
    estimated via double-centering.

    Relation to linear_cka(): the two are mathematically equivalent for
    linear kernels. Here we compute it explicitly on the Gram matrices to
    make the object-object comparison semantics transparent and to allow
    future extension to non-linear kernels (e.g. RBF).

    Args:
        F_a : (N, D) intact model features (mean-centered internally)
        F_b : (N, D) ablated model features
        eps : denominator guard against collapse
    Returns:
        CKA score ∈ [0, 1]
    """
    K_a = gram_matrix(F_a)    # (N, N)
    K_b = gram_matrix(F_b)    # (N, N)

    N = K_a.shape[0]

    # Double-centering: H = I - (1/N) 11^T  →  K_c = H K H
    ones = torch.ones(N, N, device=K_a.device) / N
    H    = torch.eye(N, device=K_a.device) - ones
    K_ac = H @ K_a @ H
    K_bc = H @ K_b @ H

    # HSIC = (1/(N-1)^2) * tr(K_ac K_bc)
    # The (N-1)^2 normalization cancels in the ratio, so we omit it.
    hsic_ab = (K_ac * K_bc).sum()
    hsic_aa = (K_ac * K_ac).sum()
    hsic_bb = (K_bc * K_bc).sum()

    denominator = (hsic_aa * hsic_bb).clamp(min=0).sqrt() + eps

    if denominator.item() < 1e-6:
        logger.warning("gram_cka: denominator near zero — possible representational collapse.")

    return (hsic_ab / denominator).clamp(0.0, 1.0).item()


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — Class×Class Cosine Similarity Matrix  (10 × 10)
# ─────────────────────────────────────────────────────────────────────────────

def class_mean_vectors(
    F: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int = 10,
) -> torch.Tensor:
    """
    Compute per-class mean feature vectors (class centroids).

    Args:
        F       : (N, D) feature matrix
        labels  : (N,)  ground-truth class indices
        n_classes : number of classes
    Returns:
        means : (n_classes, D) matrix of class centroids
    """
    D = F.shape[1]
    means = torch.zeros(n_classes, D, device=F.device)
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            means[c] = F[mask].mean(dim=0)
        else:
            logger.warning(f"class_mean_vectors: class {c} has no samples.")
    return means


def class_cosine_matrix(
    F: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int = 10,
) -> torch.Tensor:
    """
    Compute the 10×10 cosine similarity matrix between class mean vectors.

    Entry [i, j] = CosSim(mu_i, mu_j), where mu_c is the centroid of class c
    in the feature space. The diagonal is always 1.0.

    This matrix encodes the angular structure of the class directions:
    - High off-diagonal values → classes are represented as similar
    - Low values → classes are well-separated in feature space

    If ablating a block rotates/collapses the class directions, this matrix
    changes even when the final argmax prediction (accuracy) stays the same.

    Args:
        F       : (N, D) feature matrix
        labels  : (N,)  class labels
        n_classes : 10 for CIFAR-10
    Returns:
        S : (n_classes, n_classes) cosine similarity matrix
    """
    means = class_mean_vectors(F, labels, n_classes)   # (10, D)
    norms = means.norm(dim=1, keepdim=True).clamp(min=1e-8)
    means_normed = means / norms                        # unit vectors
    S = means_normed @ means_normed.T                   # (10, 10), symmetric, diag=1
    return S


def class_matrix_similarity(
    S_intact: torch.Tensor,
    S_ablated: torch.Tensor,
    eps: float = CKA_EPSILON,
) -> float:
    """
    Measure structural similarity between two 10×10 class cosine matrices.

    Uses the Frobenius inner product (cosine similarity between vectorised
    matrices), which is the natural measure: two class structures are similar
    if they agree on which class pairs are close/far in feature space.

    BIrep_class = 1 - class_matrix_similarity(S_intact, S_ablated)

    Args:
        S_intact  : (10, 10) class similarity matrix of the intact model
        S_ablated : (10, 10) class similarity matrix of the ablated model
        eps       : denominator guard
    Returns:
        similarity ∈ [-1, 1]  (in practice ∈ [0, 1] for converged networks)
    """
    va  = S_intact.flatten().float()
    vb  = S_ablated.flatten().float()
    num = (va * vb).sum()
    den = va.norm() * vb.norm() + eps
    return (num / den).item()


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features_with_labels(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single-pass extraction of GAP features and ground-truth labels.

    Returns:
        F      : (N, 2048) feature matrix (CPU)
        labels : (N,)      label tensor   (CPU)
    """
    all_feats, all_labels = [], []
    with ActivationCapture(model.avgpool) as cap:
        for images, labels in loader:
            images = images.to(device)
            _ = model(images)
            all_feats.append(gap(cap.output).cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main computation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_bi_rep_extended(
    model: ResNet50_CIFAR,
    registry: dict,
    calib_loader: torch.utils.data.DataLoader,
    device: torch.device,
    F_intact: torch.Tensor,
    labels_intact: torch.Tensor,
    n_classes: int = 10,
) -> dict:
    """
    Compute BIrep_gram and BIrep_class for all 16 blocks.

    For each block l:
        1. Short-circuit block l via ablated_block context manager
        2. Extract F_ablated (N×2048) from the ablated model
        3. BIrep_gram[l]  = 1 - gram_cka(F_intact, F_ablated)
        4. BIrep_class[l] = 1 - class_matrix_similarity(S_intact, S_ablated)

    S_intact (10×10) is computed once from F_intact and reused for all blocks.

    Args:
        model         : ResNet50_CIFAR in eval mode
        registry      : {block_name: module} from build_block_registry()
        calib_loader  : calibration DataLoader
        device        : compute device
        F_intact      : (N, 2048) reference features from Phase 1
        labels_intact : (N,) calibration set class labels
        n_classes     : 10 for CIFAR-10

    Returns:
        {
            "bi_rep_gram"  : {block_name: float ∈ [0,1]},
            "bi_rep_class" : {block_name: float ∈ [0,1]},
            "s_intact"     : [[float]×10]×10  (for saving to JSON),
        }
    """
    logger.info("=" * 60)
    logger.info("Extended BIrep — Gram CKA + Class-Class Matrix")
    logger.info("=" * 60)
    t_total = time.time()

    model.eval()
    F_dev = F_intact.to(device)
    L_dev = labels_intact.to(device)

    # Intact class cosine matrix — computed once
    S_intact = class_cosine_matrix(F_dev, L_dev, n_classes)
    logger.info(f"S_intact computed — off-diag mean: "
                f"{S_intact.fill_diagonal_(0).sum().item() / (n_classes*(n_classes-1)):.4f}")
    S_intact = class_cosine_matrix(F_dev, L_dev, n_classes)  # recompute with diag

    bi_rep_gram  = {}
    bi_rep_class = {}

    for block_name in TARGET_BLOCKS:
        block = registry[block_name]
        is_ds = block_name in DOWNSAMPLING_BLOCKS

        t0 = time.time()

        with ablated_block(block, is_downsampling=is_ds):
            F_abl, _ = extract_features_with_labels(model, calib_loader, device)

        F_abl_dev = F_abl.to(device)

        # BIrep_gram
        cka_g = gram_cka(F_dev, F_abl_dev)
        bi_rep_gram[block_name] = round(1.0 - cka_g, 6)

        # BIrep_class
        S_abl = class_cosine_matrix(F_abl_dev, L_dev, n_classes)
        sim_c = class_matrix_similarity(S_intact, S_abl)
        bi_rep_class[block_name] = round(1.0 - sim_c, 6)

        logger.info(
            f"  {block_name:12s}  "
            f"BIrep_gram={bi_rep_gram[block_name]:.6f}  "
            f"BIrep_class={bi_rep_class[block_name]:.6f}  "
            f"[{time.time()-t0:.1f}s]"
        )

    logger.info(f"Extended BIrep done in {time.time()-t_total:.1f}s")

    return {
        "bi_rep_gram":  bi_rep_gram,
        "bi_rep_class": bi_rep_class,
        "s_intact":     S_intact.cpu().tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: which class pairs changed most after ablation
# ─────────────────────────────────────────────────────────────────────────────

def compare_class_matrices(
    S_intact: torch.Tensor,
    S_ablated: torch.Tensor,
    block_name: str,
    class_names: list[str] | None = None,
) -> list[tuple]:
    """
    Return the top-5 class pairs whose cosine similarity changed most
    after ablation of block_name.

    Useful for interpreting BIrep_class: if 'cat' and 'dog' become more
    similar after ablation, that specific confusion is now quantified.

    Args:
        S_intact  : (10,10) intact class matrix
        S_ablated : (10,10) ablated class matrix
        block_name: name of ablated block (for logging)
        class_names: CIFAR-10 class name list
    Returns:
        List of (delta, class_i, class_j, before, after) tuples, sorted by |delta|
    """
    if class_names is None:
        class_names = ["airplane","automobile","bird","cat","deer",
                       "dog","frog","horse","ship","truck"]

    pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            before = S_intact[i, j].item()
            after  = S_ablated[i, j].item()
            delta  = after - before
            pairs.append((abs(delta), class_names[i], class_names[j], before, after, delta))

    pairs.sort(reverse=True)

    logger.info(f"[{block_name}] Top-5 class-pair similarity changes:")
    for abs_d, ci, cj, bef, aft, d in pairs[:5]:
        sign = "▲" if d > 0 else "▼"
        logger.info(f"  {ci:12s} ↔ {cj:12s}  {sign} {abs_d:.4f}  "
                    f"({bef:+.4f} → {aft:+.4f})")

    return pairs[:5]


# ─────────────────────────────────────────────────────────────────────────────
# Save: append to existing phase2_results.json
# ─────────────────────────────────────────────────────────────────────────────

def save_extended_results(
    extended: dict,
    phase2_path: Path = PHASE2_RESULTS,
) -> None:
    """
    Non-destructively append extended BIrep results to phase2_results.json.
    Existing keys (bi_geo, bi_acc, bi_rep) are preserved.
    """
    if not phase2_path.exists():
        raise FileNotFoundError(
            f"phase2_results.json not found at {phase2_path}. "
            "Run Phase 2 before computing extended metrics."
        )
    with open(phase2_path) as f:
        data = json.load(f)

    data["bi_rep_gram"]    = extended["bi_rep_gram"]
    data["bi_rep_class"]   = extended["bi_rep_class"]
    data["s_intact_10x10"] = extended["s_intact"]

    with open(phase2_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Extended results appended → {phase2_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner (called from run_pipeline.py or standalone)
# ─────────────────────────────────────────────────────────────────────────────

def run_extended_birep(
    model: ResNet50_CIFAR,
    registry: dict,
    calib_loader: torch.utils.data.DataLoader,
    device: torch.device,
    F_intact: torch.Tensor,
) -> dict:
    """
    Full runner for extended BIrep analysis.

    Collects labels from calib_loader, runs both metrics for all 16 blocks,
    logs the top-5 most disrupted class pairs for the primary silent failure
    candidate, and appends the results to phase2_results.json.

    To integrate with run_pipeline.py, add after Phase 2:
        from bi_rep_extended import run_extended_birep
        run_extended_birep(model, registry, calib_loader, device, F_intact)
    """
    # Collect calibration labels (one pass, no GPU needed)
    all_labels = []
    for _, labels in calib_loader:
        all_labels.append(labels)
    labels_intact = torch.cat(all_labels, dim=0)

    logger.info(f"Calibration labels: {labels_intact.shape[0]} samples, "
                f"classes {labels_intact.unique().tolist()}")

    # Compute
    extended = compute_bi_rep_extended(
        model, registry, calib_loader, device, F_intact, labels_intact
    )

    # Deep-dive on primary silent failure candidate
    if PHASE2_RESULTS.exists():
        with open(PHASE2_RESULTS) as f:
            p2 = json.load(f)
        bi_acc = p2.get("bi_acc", {})
        bi_rep = p2.get("bi_rep", {})
        if bi_acc and bi_rep:
            primary = max(TARGET_BLOCKS,
                          key=lambda b: bi_rep.get(b, 0) - bi_acc.get(b, 0))
            logger.info(f"\nClass-matrix deep-dive → primary candidate: {primary}")
            S_intact  = torch.tensor(extended["s_intact"])
            block = registry[primary]
            is_ds = primary in DOWNSAMPLING_BLOCKS
            with ablated_block(block, is_downsampling=is_ds):
                F_abl, _ = extract_features_with_labels(model, calib_loader, device)
            S_abl = class_cosine_matrix(
                F_abl.to(device), labels_intact.to(device)
            )
            compare_class_matrices(S_intact.to(device), S_abl, primary)

    # Save
    save_extended_results(extended)

    return extended
