"""
student_model.py — ResNet-18 adapted for CIFAR-10.

Mirrors the stem modification in model.py (ResNet50_CIFAR):
  - conv1: 3×3, stride 1, padding 1  (no 7×7 downsampling)
  - maxpool: Identity                 (preserves 32×32 spatial resolution)
  - fc output: 10 classes

Stage output shapes for 32×32 CIFAR input (batch=N):
  layer1[1] → (N,  64, 32, 32)   matched to teacher layer1[2] → (N, 256, 32, 32)
  layer2[1] → (N, 128, 16, 16)   matched to teacher layer2[3] → (N, 512, 16, 16)
  layer3[1] → (N, 256,  8,  8)   matched to teacher layer3[5] → (N,1024,  8,  8)
  layer4[1] → (N, 512,  4,  4)   matched to teacher layer4[2] → (N,2048,  4,  4)

SP-KD Gram matrices are N×N, so channel depth differences are irrelevant.
"""

import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet


class ResNet18_CIFAR(ResNet):
    """ResNet-18 with CIFAR-10 stem (3×3 conv, no max-pool)."""

    def __init__(self, num_classes: int = 10):
        super().__init__(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=num_classes,
        )
        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.maxpool = nn.Identity()


def build_student(num_classes: int = 10) -> ResNet18_CIFAR:
    """Return a freshly initialised ResNet18_CIFAR (random weights)."""
    return ResNet18_CIFAR(num_classes=num_classes)
