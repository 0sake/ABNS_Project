"""
model.py — ResNet50 adapted for CIFAR-10.

Key modifications vs. the ImageNet ResNet50:
  • First convolution: 3×3 kernel, stride 1, padding 1  (instead of 7×7, stride 2)
  • MaxPool replaced with Identity                        (preserves spatial resolution)
  • FC output: 10 classes

Step 1.3: model acquisition, key-prefix stripping, sanity check.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Architecture
# ──────────────────────────────────────────────

class ResNet50_CIFAR(ResNet):
    """
    ResNet-50 with the standard CIFAR-10 head modifications.
    Inherits all Bottleneck logic from torchvision; only the stem is overridden.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],   # ResNet-50 block counts
            num_classes=num_classes,
        )
        # Replace 7×7 stem with 3×3 CIFAR stem
        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove maxpool (keep spatial resolution for 32×32 inputs)
        self.maxpool = nn.Identity()

    # torchvision's ResNet.forward already calls avgpool → flatten → fc,
    # so no override needed; the GAP output used for CKA is tapped via hook.


# ──────────────────────────────────────────────
# Weight loading
# ──────────────────────────────────────────────

def _strip_prefix(state_dict: dict, prefixes: tuple = ("module.", "net.")) -> dict:
    """
    Remove DataParallel / custom wrapper prefixes from a state_dict.
    Handles: 'module.layer1.0.conv1.weight' → 'layer1.0.conv1.weight'
    """
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
                break
        cleaned[new_k] = v
    return cleaned


def load_model(ckpt_path: Path, device: torch.device) -> ResNet50_CIFAR:
    """
    Instantiate ResNet50_CIFAR, load weights from ckpt_path, and run a
    basic sanity check on the state_dict keys.

    Returns the model in eval() mode on `device`.
    """
    model = ResNet50_CIFAR(num_classes=10)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Place your resnet50_cifar10.pth in checkpoints/ and retry."
        )

    # tensor first loaded in cpu
    raw = torch.load(ckpt_path, map_location="cpu")

    # Common checkpoint formats: plain state_dict or wrapped under a key
    if isinstance(raw, dict):
        sd = raw.get("state_dict", raw.get("model", raw))
    else:
        sd = raw  # assume it's already a state_dict

    sd = _strip_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    # then loaded into torch device
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded from {ckpt_path.name} → {device}")
    return model


# ──────────────────────────────────────────────
# Block registry helpers
# ──────────────────────────────────────────────

def build_block_registry(model: ResNet50_CIFAR) -> dict:
    """
    Return a dict {block_name: module} following the canonical TARGET_BLOCKS order.

    Examples:
        "layer1.0" → model.layer1[0]
        "layer2.3" → model.layer2[3]
    """
    registry = {}
    for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
        stage = getattr(model, stage_name)
        for idx, block in enumerate(stage):
            key = f"{stage_name}.{idx}"
            registry[key] = block
    return registry


def get_stage_output_module(model: ResNet50_CIFAR, stage_name: str) -> nn.Module:
    """
    Return the module whose output represents the end of a given stage.
    Used for multi-layer CKA propagation profiles.

    stage_name ∈ {"layer1", "layer2", "layer3", "layer4", "avgpool"}
    """
    if stage_name == "avgpool":
        return model.avgpool
    return getattr(model, stage_name)[-1]   # last block in the stage


# ──────────────────────────────────────────────
# Accuracy evaluation
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Full test-set accuracy evaluation.
    Returns accuracy ∈ [0, 1].
    """
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def evaluate_accuracy_with_logits(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Like evaluate_accuracy but also returns per-sample softmax probabilities
    and ground-truth labels — needed for the confidence analysis in Phase 3.

    Returns:
        accuracy     : float
        all_probs    : Tensor (N, 10)   softmax probabilities
        all_labels   : Tensor (N,)      ground-truth class indices
    """
    all_probs, all_labels = [], []
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    return correct / total, torch.cat(all_probs), torch.cat(all_labels)
