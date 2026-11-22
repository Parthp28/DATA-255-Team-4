"""DeiT-S/16 classifier for E2, E4b and E4 (decision D9).

Backbone: timm ``deit_small_patch16_224.fb_in1k`` (ImageNet-1k supervised, 224 x 224), with the
1000-class head replaced by a 3-class head (0 DOWN, 1 NEUTRAL, 2 UP). Everything else keeps the
timm defaults: no dropout, no drop-path, class-token pooling, fused attention.

Interface
    * Input:  ``(B, 3, 224, 224)`` floating-point images, ALREADY normalized by the dataset
      pipeline (T-T3: the only transforms are ToTensor and Normalize). The model contains no
      normalization of its own. The constants the pipeline should use are in
      :func:`preprocessing_config`.
    * Output: ``(B, 3)`` raw logits (no softmax). Losses and metrics apply softmax themselves.
    * ``DeiTSmallClassifier.backbone`` is the timm ``VisionTransformer``, exposed for attention
      rollout and patch-to-region mapping. Rollout needs the attention maps, so it must turn
      fused attention off first (``block.attn.fused_attn = False``); that is a rollout concern
      and is not done here.

Not handled here (owned by the shared trainer, D10): AMP, gradient clipping, the optimizer and
schedule, early stopping, checkpointing, seeding. Seed before building the model: with
pretrained weights only the new 3-class head is randomly initialized, and that draw comes from
the global torch seed.

Parameter count, measured with timm 1.0.31: 22,050,664 for the 1000-class pretrained model
(the 22.1M in the plan) and 21,666,819 for this 3-class model (see ``tests/test_vit.py``).
"""
from typing import Dict

import timm
import torch
from torch import nn

MODEL_NAME = "deit_small_patch16_224.fb_in1k"
IMAGE_SIZE = 224
IN_CHANNELS = 3
NUM_CLASSES = 3


class DeiTSmallClassifier(nn.Module):
    """Thin wrapper around the timm DeiT-S/16 backbone with a 3-class head."""

    def __init__(self, pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super().__init__()
        # timm keeps the pretrained backbone and builds a fresh Linear(384, num_classes) head.
        self.backbone = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_images(x)
        return self.backbone(x)

    def extra_repr(self) -> str:
        return f"model={MODEL_NAME}, num_classes={self.num_classes}, image_size={IMAGE_SIZE}"


def _check_images(x: torch.Tensor) -> None:
    """Fail early with a clear message instead of a shape error deep inside timm."""
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"expected a torch.Tensor, got {type(x).__name__}")
    if x.dim() != 4:
        raise ValueError(f"expected a (B, {IN_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE}) tensor, got shape {tuple(x.shape)}")
    if x.shape[0] == 0:
        raise ValueError(f"empty batch: got shape {tuple(x.shape)}")
    if tuple(x.shape[1:]) != (IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected (B, {IN_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE}) images, got shape {tuple(x.shape)}")
    if not x.is_floating_point():
        raise ValueError(f"expected normalized floating-point images, got dtype {x.dtype}")


def build_deit_small(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> DeiTSmallClassifier:
    """Build the E2 / E4b / E4 model. ``pretrained=False`` skips the weight download (tests)."""
    return DeiTSmallClassifier(pretrained=pretrained, num_classes=num_classes)


def preprocessing_config() -> Dict[str, object]:
    """Normalization the dataset pipeline must apply, read from timm's pretrained config.

    Only ``mean`` and ``std`` are used by this project. ``crop_pct`` and ``interpolation`` in the
    timm config belong to its resize-and-crop evaluation transform; our images are rendered at
    224 x 224 and are never resized.
    """
    cfg = timm.models.get_pretrained_cfg(MODEL_NAME)
    return {"mean": tuple(cfg.mean), "std": tuple(cfg.std), "input_size": tuple(cfg.input_size)}


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Total (or trainable) parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
