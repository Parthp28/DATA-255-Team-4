"""ResNet-18 image classifier (D9: timm resnet18.a1_in1k, 224x224, 3-class head).

Owner: Kaushik. Grad-CAM-ready: `gradcam_target_layer` returns the last conv
block so src/interpretability/gradcam.py can hook it.
"""
from __future__ import annotations

import timm
import torch.nn as nn

from ..common.schema import N_CLASSES


def build_resnet(
    model_name: str = "resnet18.a1_in1k",
    pretrained: bool = True,
    num_classes: int = N_CLASSES,
) -> nn.Module:
    """Create the CNN. pretrained=True pulls the strong A1 ImageNet-1k recipe
    weights (D9); pretrained=False is used for offline/dummy-registry dev so the
    stack is testable with no network."""
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last residual block of layer4 -- the standard Grad-CAM target for ResNet."""
    return model.layer4[-1]
