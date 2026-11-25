"""Grad-CAM for the ResNet-18 runs E1, E3b, E3 (Selvaraju et al., ICCV 2017; plan 12, 15.1). Owner: Kaushik.

For each image the target is the model's own predicted class c (the same c that masking uses). With activations
A_k of the target layer (default `src.models.resnet.gradcam_target_layer`, the last block of layer4) and
gradients of logit_c with respect to them:

    alpha_k = mean over (h, w) of d logit_c / d A_k
    cam     = ReLU(sum_k alpha_k A_k), bilinearly upsampled to the input size, then scaled to sum 1

A map that is zero everywhere carries no attribution and becomes all NaN, so it is counted rather than hidden.
The model is run in eval mode and is left exactly as it was found (mode, hooks, parameter grads).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .regions import normalize_heat


def gradcam(model: nn.Module, images: torch.Tensor, layer: nn.Module | None = None,
            target: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (heat (B, H, W) summing to 1, target class (B,), softmax probabilities (B, C))."""
    if layer is None:
        from ..models.resnet import gradcam_target_layer
        layer = gradcam_target_layer(model)
    store: dict[str, torch.Tensor] = {}

    def keep(_m, _inp, out):
        store["a"] = out

    was_training = model.training
    old_grads = [p.grad for p in model.parameters()]
    handle = layer.register_forward_hook(keep)
    model.eval()
    try:
        with torch.enable_grad():
            x = images.detach().clone().requires_grad_(True)     # keeps a graph even for a frozen model
            logits = model(x)
            c = logits.argmax(1) if target is None else target.to(logits.device)
            sel = logits.gather(1, c[:, None]).sum()
            a = store["a"]
            (g,) = torch.autograd.grad(sel, a)
        alpha = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        heat = np.stack([normalize_heat(m) for m in cam.detach().cpu().numpy()])
        probs = logits.detach().float().softmax(1).cpu().numpy()
        return heat, c.detach().cpu().numpy(), probs
    finally:
        handle.remove()
        model.train(was_training)
        for p, gr in zip(model.parameters(), old_grads):
            p.grad = gr
