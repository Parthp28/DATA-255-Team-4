"""Paired consistency loss for E3b / E4b / E3 / E4 (decision D12).

Standalone interface: it takes two logit tensors and the labels, and returns the loss
terms. It does not know about datasets, samplers, or the trainer; whoever builds the
training step decides how a batch is split into the two views.

Pair convention
    ``logits_a[i]`` and ``logits_b[i]`` are the model outputs for the SAME window drawn in
    two different renderings (a != b, both from the seen set). ``labels[i]`` is that
    window's label. With 128 images per step this means 64 windows x 2 renderings, so both
    tensors have shape ``(64, 3)``.

Loss (Section 5.1 of the execution plan)
    total = 0.5 * [CE(a) + CE(b)] + lam * JS(p_a, p_b)

    * lam = 0 is the augmentation-only control (E3b, E4b): ``total`` is exactly the
      cross-entropy term, with the same value and the same gradients as a loss that never
      computed JS (test T-C6).
    * lam > 0 is the consistency intervention (E3, E4), lam in {0.3, 1, 3} (D11, D13).

JS divergence (D12)
    JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m),  m = 0.5 * (p + q),  natural log.
    Symmetric, in [0, ln 2], zero iff p == q. Gradient flows through BOTH branches: there is
    no teacher view and no ``detach``. Reported value is the mean over the pairs in the batch.

Numerics
    Everything is computed from logits through log-softmax and logsumexp, in at least float32
    (fp16/bf16 inputs are upcast; float32 and float64 keep their dtype), so AMP outputs cannot
    underflow a probability to zero and produce ``0 * log 0``.
"""
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

LN2 = math.log(2.0)


def _at_least_fp32(x: torch.Tensor) -> torch.Tensor:
    """Upcast fp16/bf16 logits to float32; leave float32 and float64 unchanged."""
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


class PairedLossOutput(NamedTuple):
    """Loss terms for one step. ``total`` is the only tensor to call ``backward()`` on."""

    total: torch.Tensor  # 0.5 * [CE(a) + CE(b)] + lam * JS
    ce: torch.Tensor  # 0.5 * [CE(a) + CE(b)]
    js: torch.Tensor  # mean JS over the pairs; detached for logging when lam == 0


def js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Jensen-Shannon divergence between softmax(logits_a) and softmax(logits_b), in nats.

    Args:
        logits_a, logits_b: raw logits of identical shape ``(N, C)``.
        reduction: ``"mean"`` (scalar), ``"none"`` (per-pair, shape ``(N,)``), or ``"sum"``.
    """
    if logits_a.shape != logits_b.shape or logits_a.dim() != 2:
        raise ValueError(f"expected two (N, C) tensors of equal shape, got {tuple(logits_a.shape)} and {tuple(logits_b.shape)}")
    if reduction not in ("mean", "none", "sum"):
        raise ValueError(f"unknown reduction: {reduction!r}")
    log_p = F.log_softmax(_at_least_fp32(logits_a), dim=-1)
    log_q = F.log_softmax(_at_least_fp32(logits_b), dim=-1)
    # log m = log(0.5 * (p + q)), computed stably in log space.
    log_m = torch.logsumexp(torch.stack([log_p, log_q]), dim=0) - LN2
    kl_pm = (log_p.exp() * (log_p - log_m)).sum(dim=-1)
    kl_qm = (log_q.exp() * (log_q - log_m)).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    if reduction == "mean":
        return js.mean()
    if reduction == "sum":
        return js.sum()
    return js


def paired_consistency_loss(
    logits_a: torch.Tensor, logits_b: torch.Tensor, labels: torch.Tensor, lam: float = 0.0
) -> PairedLossOutput:
    """0.5 * [CE(a) + CE(b)] + lam * JS(p_a, p_b) for one batch of pairs.

    Args:
        logits_a, logits_b: ``(N, C)`` logits for the two renderings of the same N windows.
        labels: ``(N,)`` integer class labels (0 DOWN, 1 NEUTRAL, 2 UP).
        lam: consistency weight, ``>= 0``. ``0`` reproduces the augmentation-only control.
    """
    if not (isinstance(lam, (int, float)) and math.isfinite(lam) and lam >= 0):
        raise ValueError(f"lam must be a finite number >= 0, got {lam!r}")
    if logits_a.shape != logits_b.shape or logits_a.dim() != 2:
        raise ValueError(f"expected two (N, C) tensors of equal shape, got {tuple(logits_a.shape)} and {tuple(logits_b.shape)}")
    if labels.shape != (logits_a.shape[0],):
        raise ValueError(f"labels must have shape ({logits_a.shape[0]},), got {tuple(labels.shape)}")
    labels = labels.long()
    ce = 0.5 * (F.cross_entropy(_at_least_fp32(logits_a), labels) + F.cross_entropy(_at_least_fp32(logits_b), labels))
    if lam == 0:
        # Exactly the E3b/E4b loss: JS is computed only for logging and stays out of the graph.
        with torch.no_grad():
            js = js_divergence(logits_a, logits_b)
        return PairedLossOutput(total=ce, ce=ce, js=js)
    js = js_divergence(logits_a, logits_b)
    return PairedLossOutput(total=ce + lam * js, ce=ce, js=js)


class PairedConsistencyLoss(nn.Module):
    """Module wrapper around :func:`paired_consistency_loss` with ``lam`` fixed at construction."""

    def __init__(self, lam: float = 0.0):
        super().__init__()
        if not (isinstance(lam, (int, float)) and math.isfinite(lam) and lam >= 0):
            raise ValueError(f"lam must be a finite number >= 0, got {lam!r}")
        self.lam = float(lam)

    def forward(self, logits_a: torch.Tensor, logits_b: torch.Tensor, labels: torch.Tensor) -> PairedLossOutput:
        return paired_consistency_loss(logits_a, logits_b, labels, self.lam)

    def extra_repr(self) -> str:
        return f"lam={self.lam}"
