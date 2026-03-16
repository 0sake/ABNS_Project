"""
data.py — CIFAR-10 data pipeline and calibration set construction.
Implements Step 1.2 of the pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    CIFAR10_MEAN, CIFAR10_STD,
    N_CALIBRATION, SAMPLES_PER_CLASS,
    BATCH_SIZE_INFERENCE, BATCH_SIZE_CALIB,
    NUM_WORKERS, SEED, CALIB_INDICES,
)
from utils import worker_init_fn, get_dataloader_generator

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────

def get_transform() -> transforms.Compose:
    """Standard CIFAR-10 normalisation (no augmentation — inference only)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


# ──────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────

def load_cifar10(data_dir: str = "./data") -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """
    Download (if needed) and return the full CIFAR-10 train and test sets
    with standard normalisation applied.
    """
    tf = get_transform()
    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=tf)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    logger.info(f"CIFAR-10 loaded — train: {len(train_ds)}, test: {len(test_ds)}")
    return train_ds, test_ds


# ──────────────────────────────────────────────
# Calibration set construction (Step 1.2)
# ──────────────────────────────────────────────

def build_calibration_indices(
    train_ds: datasets.CIFAR10,
    n_per_class: int = SAMPLES_PER_CLASS,
    seed: int = SEED,
    save_path: Path = CALIB_INDICES,
) -> torch.Tensor:
    """
    Construct a class-balanced calibration subset from the training set.

    Draws `n_per_class` samples per class (10 classes → N = n_per_class × 10).
    The resulting index tensor is saved for exact repeatability across runs.

    Returns:
        indices : LongTensor of shape (N,)
    """
    rng = np.random.default_rng(seed)
    targets = np.array(train_ds.targets)
    selected = []

    for cls in range(10):
        cls_indices = np.where(targets == cls)[0]
        chosen = rng.choice(cls_indices, size=n_per_class, replace=False)
        selected.extend(chosen.tolist())

    indices = torch.tensor(selected, dtype=torch.long)
    torch.save(indices, save_path)
    logger.info(
        f"Calibration set: {len(indices)} samples "
        f"({n_per_class}/class) → saved to {save_path.name}"
    )
    return indices


def load_or_build_calibration_indices(
    train_ds: datasets.CIFAR10,
    save_path: Path = CALIB_INDICES,
) -> torch.Tensor:
    """Load existing calibration indices or build and save them if absent."""
    if save_path.exists():
        indices = torch.load(save_path)
        logger.info(
            f"Loaded existing calibration indices ({len(indices)} samples) "
            f"from {save_path.name}"
        )
    else:
        logger.info("No calibration indices found — building fresh set.")
        indices = build_calibration_indices(train_ds, save_path=save_path)
    return indices


# ──────────────────────────────────────────────
# DataLoader factories
# ──────────────────────────────────────────────

def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    seed: int = SEED,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=get_dataloader_generator(seed) if shuffle else None,
        persistent_workers=NUM_WORKERS > 0,
    )


def get_test_loader(test_ds: datasets.CIFAR10) -> DataLoader:
    """Full 10 k-image test loader (no shuffle — used for BIacc)."""
    return _make_loader(test_ds, batch_size=BATCH_SIZE_INFERENCE)


def get_calibration_loader(
    train_ds: datasets.CIFAR10,
    calib_indices: torch.Tensor,
) -> DataLoader:
    """
    Calibration-set loader (balanced subset, no shuffle).
    Used for BIgeo and BIrep.
    """
    subset = Subset(train_ds, calib_indices.tolist())
    return _make_loader(subset, batch_size=BATCH_SIZE_CALIB)


def get_class_conditional_loaders(
    train_ds: datasets.CIFAR10,
    calib_indices: torch.Tensor,
) -> dict[int, DataLoader]:
    """
    Return one DataLoader per CIFAR-10 class, using only the calibration samples
    belonging to that class. Used for the per-class CKA analysis in Phase 3.
    """
    targets = np.array(train_ds.targets)
    calib_targets = targets[calib_indices.numpy()]
    loaders = {}

    for cls in range(10):
        cls_mask = calib_targets == cls
        cls_global_indices = calib_indices[cls_mask].tolist()
        subset = Subset(train_ds, cls_global_indices)
        loaders[cls] = _make_loader(subset, batch_size=BATCH_SIZE_CALIB)

    return loaders
