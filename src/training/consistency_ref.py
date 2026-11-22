"""Reference loss functions for the shared trainer.

NOTE ON OWNERSHIP: the authoritative consistency loss lives in Khushi's
src/training/losses.py (D12: Jensen-Shannon divergence between paired softmax
outputs, gradient through both branches, symmetric & bounded). This module is
Kaushik's reference implementation so the trainer is runnable on the dummy
registry before the integration, and is the thing Kaushik cross-reviews against
(cross-review T-C). The trainer takes the consistency fn by injection, so
swapping in Khushi's version is a one-line change with no trainer edits.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric, bounded Jensen-Shannon divergence between the two softmax
    distributions. Gradient flows through BOTH branches (no stop-grad / no
    teacher rendering), per D12. Returns the batch mean."""
    log_pa = F.log_softmax(logits_a, dim=1)
    log_pb = F.log_softmax(logits_b, dim=1)
    pa, pb = log_pa.exp(), log_pb.exp()
    m = 0.5 * (pa + pb)
    log_m = m.clamp_min(1e-8).log()
    kl_a = F.kl_div(log_m, pa, reduction="batchmean")  # KL(pa || m)
    kl_b = F.kl_div(log_m, pb, reduction="batchmean")  # KL(pb || m)
    return 0.5 * (kl_a + kl_b)


def combined_loss(mode: str, lam: float, logits_a, labels, logits_b=None):
    """Dispatch the three trainer loss modes. Returns (total, parts_dict)."""
    if mode == "ce":
        ce = cross_entropy(logits_a, labels)
        return ce, {"ce": ce.item()}

    # paired modes
    ce_a = cross_entropy(logits_a, labels)
    ce_b = cross_entropy(logits_b, labels)
    ce = 0.5 * (ce_a + ce_b)
    if mode == "paired_ce":                       # E3b/E4b, lambda = 0
        return ce, {"ce": ce.item(), "js": 0.0}
    if mode == "ce_plus_js":                       # E3/E4 intervention
        js = js_divergence(logits_a, logits_b)
        return ce + lam * js, {"ce": ce.item(), "js": js.item()}
    raise ValueError(f"unknown loss mode {mode!r}")
