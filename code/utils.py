"""
utils.py — Deterministic environment, seed control, and shared utilities.
Implements Step 1.1 of the pipeline.
"""

import os
import random
import hashlib
import logging
import numpy as np
import torch
from contextlib import contextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Step 1.1: Deterministic Environment
# ──────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Fix all relevant RNG sources to guarantee reproducibility across runs.
    Covers: random, numpy, torch (CPU + CUDA), DataLoader workers.
    Sets cuDNN to deterministic mode (slight speed cost, required for validity).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic convolution algorithms — required for cross-run comparability
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforce determinism at the algorithm level (raises if a non-deterministic
    # op is encountered, making violations explicit rather than silent)
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        logger.warning("torch.use_deterministic_algorithms not available; skipping.")

    logger.info(f"Global seed set to {seed}. cuDNN deterministic=True, benchmark=False.")

def relax_determinism_for_training() -> None:
    """
    Override cuDNN determinism settings for Phase 4 training.
    set_seed() enforces deterministic=True / benchmark=False globally,
    which causes 5-10x slowdown during training. Seed-based reproducibility
    (random / numpy / torch / cuda) is preserved — only cuDNN convolution
    algorithm selection is relaxed.
    """
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    print("Override cuDNN determinism settings for Phase 4 training.")


def worker_init_fn(worker_id: int) -> None:
    """Per-worker seed initialiser for DataLoader reproducibility."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader_generator(seed: int = 42) -> torch.Generator:
    """Return a seeded Generator for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def log_environment() -> dict:
    """Log hardware and software context for reproducibility records."""
    import platform, torch

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    }
    logger.info("Environment: " + ", ".join(f"{k}={v}" for k, v in env.items()))
    return env


# ──────────────────────────────────────────────
# Tensor checksum utility
# ──────────────────────────────────────────────

def tensor_checksum(t: torch.Tensor) -> str:
    """SHA-256 hash of a tensor's raw bytes — used to verify reference integrity."""
    raw = t.cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


# ──────────────────────────────────────────────
# Context manager: safe block ablation
# ──────────────────────────────────────────────

@contextmanager
def ablated_block(block, is_downsampling: bool = False):
    """
    Context manager that temporarily short-circuits a ResNet Bottleneck block.

    Standard blocks: output = input (identity forward).
    Downsampling blocks (layer{2,3,4}.0): the downsampling projection is preserved
    (necessary to avoid shape mismatches in subsequent layers), while the three
    convolutional transformation layers are effectively zeroed by intercepting
    the residual output.

    Restoration is enforced in a `finally` clause, preventing state leakage
    even if an exception occurs during inference.
    """
    original_forward = block.forward

    if not is_downsampling:
        def patched_forward(x):
            return x
    else:
        # Preserve downsample projection; zero the conv branch
        def patched_forward(x):
            identity = block.downsample(x) if block.downsample is not None else x
            return identity

    block.forward = patched_forward
    try:
        yield
    finally:
        block.forward = original_forward


# ──────────────────────────────────────────────
# Hook-based activation capture
# ──────────────────────────────────────────────

class ActivationCapture:
    """
    Registers forward hooks on a module to capture input and output tensors.
    Hooks are removed automatically via .remove() or when used as a context manager.
    """

    def __init__(self, module: torch.nn.Module):
        self.input: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self._hook_in = module.register_forward_pre_hook(self._capture_input)
        self._hook_out = module.register_forward_hook(self._capture_output)

    def _capture_input(self, module, args):
        self.input = args[0].detach()

    def _capture_output(self, module, args, output):
        self.output = output.detach()

    def remove(self):
        self._hook_in.remove()
        self._hook_out.remove()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


def gap(tensor: torch.Tensor) -> torch.Tensor:
    """
    Global Average Pooling over spatial dims (H, W).
    Input shape: (N, C, H, W) → Output shape: (N, C).
    If tensor is already 2-D, returns it unchanged.
    """
    if tensor.dim() == 4:
        return tensor.mean(dim=[2, 3])
    return tensor
