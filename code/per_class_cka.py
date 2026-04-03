"""
per_class_cka.py
----------------
Computes linear CKA between intact and ablated avgpool representations
for each of the 10 CIFAR-10 classes, across all 16 TARGET_BLOCKS.

Public API
----------
run_per_class_cka(model, registry, class_loaders, device) -> dict
    Returns {block_name: {class_idx: cka_score}} and saves to
    PHASE3_PER_CLASS_CKA (JSON).
"""

import json

import torch
from tqdm import tqdm

from config import DOWNSAMPLING_BLOCKS, PHASE3_PER_CLASS_CKA, TARGET_BLOCKS
from phase2_metrics import linear_cka
from utils import ActivationCapture, ablated_block, gap

_CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def run_per_class_cka(model, registry: dict, class_loaders: dict, device: torch.device) -> dict:
    """Compute per-class linear CKA for every block in TARGET_BLOCKS.

    Extracts intact avgpool representations for all 10 classes once before
    iterating over blocks, avoiding redundant forward passes.

    Args:
        model: ResNet50_CIFAR already moved to *device*.
        registry: Mapping from block name (str) to nn.Module.
        class_loaders: Mapping from class index (int) to DataLoader,
            as returned by get_class_conditional_loaders().
        device: torch.device used for computation.

    Returns:
        Nested dict ``{block_name: {class_idx: cka_score}}`` where
        ``cka_score`` is a float in [0, 1].  Also persists the result
        to ``PHASE3_PER_CLASS_CKA`` as JSON.
    """
    model.eval()
    n_classes = len(class_loaders)

    # ------------------------------------------------------------------
    # Step A — extract intact representations for all classes (once)
    # ------------------------------------------------------------------
    F_intact: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for cls_idx in range(n_classes):
            cls_feats: list[torch.Tensor] = []
            with ActivationCapture(model.avgpool) as cap:
                for images, _ in class_loaders[cls_idx]:
                    images = images.to(device)
                    model(images)
                    cls_feats.append(gap(cap.output).cpu())
            F_intact[cls_idx] = torch.cat(cls_feats, dim=0)

    # ------------------------------------------------------------------
    # Step B — iterate over all blocks; for each, extract ablated reps
    #          and compute per-class linear CKA
    # ------------------------------------------------------------------
    results: dict[str, dict[int, float]] = {}

    with torch.no_grad():
        for block_name in tqdm(TARGET_BLOCKS, desc="per-class CKA"):
            block = registry[block_name]
            is_ds = block_name in DOWNSAMPLING_BLOCKS
            cls_scores: dict[int, float] = {}

            with ablated_block(block, is_downsampling=is_ds):
                for cls_idx in range(n_classes):
                    cls_feats_abl: list[torch.Tensor] = []
                    with ActivationCapture(model.avgpool) as cap:
                        for images, _ in class_loaders[cls_idx]:
                            images = images.to(device)
                            model(images)
                            cls_feats_abl.append(gap(cap.output).cpu())
                    F_abl = torch.cat(cls_feats_abl, dim=0)
                    score = linear_cka(
                        F_intact[cls_idx].to(device),
                        F_abl.to(device),
                    )
                    cls_scores[cls_idx] = round(float(score), 6)

            results[block_name] = cls_scores

    # ------------------------------------------------------------------
    # Step C — persist to disk
    # ------------------------------------------------------------------
    try:
        with open(PHASE3_PER_CLASS_CKA, "w") as f:
            json.dump(results, f, indent=2)
    except OSError as e:
        print(f"Warning: could not save per-class CKA results: {e}")

    return results
