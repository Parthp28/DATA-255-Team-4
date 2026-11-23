"""Attention rollout for the DeiT-S runs E2, E4b, E4 (Abnar and Zuidema, ACL 2020; plan 12, 15.1). Owner: Khushi.

For every block l the attention probabilities are averaged over heads (head fusion "mean") and the residual
connection is added as A_l' = (1 - r) A_l + r I with r = 0.5, then each row is renormalised. The rollout is
A_L' ... A_1'; its CLS row over the 14 x 14 patch tokens is the attribution. Patch-to-region mapping: each
patch's mass is spread evenly over its 16 x 16 pixels, so region masses come from the same pixel partition as
Grad-CAM and masking.

timm computes attention with a fused kernel that never materialises the probabilities. While capturing, the
blocks are switched to the explicit path (same maths), and their original setting, the hooks and the model's
mode are restored afterwards.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .regions import normalize_heat


def _backbone(model: nn.Module) -> nn.Module:
    return getattr(model, "backbone", model)       # src.models.vit.DeiTSmallClassifier wraps the timm model


def capture_attention(model: nn.Module, images: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Per-block attention probabilities [(B, heads, T, T)] and the logits of the same forward pass."""
    vit = _backbone(model)
    attns: list[torch.Tensor] = []
    handles, fused = [], []
    was_training = model.training
    model.eval()
    try:
        for blk in vit.blocks:
            fused.append(blk.attn.fused_attn)
            blk.attn.fused_attn = False
            handles.append(blk.attn.attn_drop.register_forward_hook(lambda _m, inp, _o: attns.append(inp[0].detach())))
        with torch.no_grad():
            logits = model(images)
    finally:
        for h in handles:
            h.remove()
        for blk, f in zip(vit.blocks, fused):
            blk.attn.fused_attn = f
        model.train(was_training)
    if len(attns) != len(vit.blocks):
        raise RuntimeError(f"captured {len(attns)} attention maps for {len(vit.blocks)} blocks")
    return attns, logits


def rollout_matrix(attns: list[torch.Tensor], residual: float = 0.5) -> torch.Tensor:
    """(B, T, T) rollout of head-averaged attention with the residual identity."""
    out = None
    for a in attns:
        a = a.float().mean(1)
        eye = torch.eye(a.shape[-1], device=a.device).expand_as(a)
        a = (1 - residual) * a + residual * eye
        a = a / a.sum(-1, keepdim=True)
        out = a if out is None else a @ out
    return out


def patch_to_pixels(cls_row: np.ndarray, grid: tuple[int, int], image_size: int) -> np.ndarray:
    """Spread each patch's value evenly over its pixels: (gh*gw,) -> (image_size, image_size); total preserved."""
    gh, gw = grid
    ph, pw = image_size // gh, image_size // gw
    if ph * gh != image_size or pw * gw != image_size:
        raise ValueError(f"patch grid {grid} does not tile a {image_size} px image")
    return np.kron(cls_row.reshape(gh, gw), np.ones((ph, pw))) / (ph * pw)


def attention_rollout(model: nn.Module, images: torch.Tensor, residual: float = 0.5
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (heat (B, H, W) summing to 1, predicted class (B,), softmax probabilities (B, C))."""
    vit = _backbone(model)
    attns, logits = capture_attention(model, images)
    roll = rollout_matrix(attns, residual)
    k = int(getattr(vit, "num_prefix_tokens", 1))
    cls = roll[:, 0, k:].cpu().numpy()
    grid = tuple(vit.patch_embed.grid_size)
    heat = np.stack([normalize_heat(patch_to_pixels(row, grid, images.shape[-1])) for row in cls])
    return heat, logits.argmax(1).cpu().numpy(), logits.float().softmax(1).cpu().numpy()
