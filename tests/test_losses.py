"""Tests T-C1 to T-C6 for the Jensen-Shannon consistency loss (src/training/losses.py), plus input checks."""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.losses import LN2, PairedConsistencyLoss, js_divergence, paired_consistency_loss  # noqa: E402


def rand_logits(n=64, c=3, seed=0, scale=2.0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, c, generator=g) * scale).requires_grad_(requires_grad)


def reference_js(la, lb):
    """Straightforward float64 definition, used only to check the stable implementation."""
    p, q = F.softmax(la.double(), -1), F.softmax(lb.double(), -1)
    m = 0.5 * (p + q)
    return (0.5 * (p * (p / m).log()).sum(-1) + 0.5 * (q * (q / m).log()).sum(-1)).mean()


def test_c1_js_of_identical_distributions_is_zero():
    la = rand_logits(seed=1)
    assert js_divergence(la, la.clone()).item() == pytest.approx(0.0, abs=1e-7)
    # Logits that differ by a constant give the same softmax, so JS is still zero.
    assert js_divergence(la, la + 5.0).item() == pytest.approx(0.0, abs=1e-6)


def test_c2_js_of_different_distributions_is_positive():
    la, lb = rand_logits(seed=2), rand_logits(seed=3)
    js = js_divergence(la, lb, reduction="none")
    assert (js > 0).all()
    assert js_divergence(la, lb).item() == pytest.approx(reference_js(la, lb).item(), rel=1e-4)


def test_c3_js_is_symmetric():
    la, lb = rand_logits(seed=4), rand_logits(seed=5)
    assert torch.allclose(js_divergence(la, lb, "none"), js_divergence(lb, la, "none"), atol=1e-7)


def test_c4_js_is_bounded_by_ln2():
    la, lb = rand_logits(seed=6), rand_logits(seed=7)
    assert (js_divergence(la, lb, "none") <= LN2 + 1e-6).all()
    # Nearly disjoint distributions approach the bound without exceeding it or going non-finite.
    big = torch.tensor([[200.0, -200.0, -200.0]])
    other = torch.tensor([[-200.0, 200.0, -200.0]])
    js = js_divergence(big, other).item()
    assert math.isfinite(js) and js <= LN2 + 1e-6 and js == pytest.approx(LN2, abs=1e-5)
    # Half precision inputs (AMP) stay finite and bounded.
    jh = js_divergence(la.half(), lb.half(), "none")
    assert torch.isfinite(jh).all() and (jh <= LN2 + 1e-6).all()


def test_c5_gradient_reaches_both_branches_finite_and_nonzero():
    la, lb = rand_logits(seed=8, requires_grad=True), rand_logits(seed=9, requires_grad=True)
    js_divergence(la, lb).backward()
    for g in (la.grad, lb.grad):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    # Same through the full paired loss with lam > 0 (the CE term must not hide a missing JS path).
    la2, lb2 = rand_logits(seed=8, requires_grad=True), rand_logits(seed=9, requires_grad=True)
    labels = torch.randint(0, 3, (64,), generator=torch.Generator().manual_seed(0))
    out = paired_consistency_loss(la2, lb2, labels, lam=1.0)
    torch.autograd.grad(out.js, [la2, lb2], retain_graph=True)  # JS alone reaches both branches
    out.total.backward()
    assert torch.isfinite(la2.grad).all() and torch.isfinite(lb2.grad).all()


def test_c5c_js_gradient_matches_numerical_gradient_in_float64():
    # float64 inputs keep their dtype, so torch's finite-difference check is meaningful.
    g = torch.Generator().manual_seed(21)
    la = torch.randn(5, 3, generator=g, dtype=torch.double, requires_grad=True)
    lb = torch.randn(5, 3, generator=g, dtype=torch.double, requires_grad=True)
    assert js_divergence(la, lb).dtype == torch.float64
    assert torch.autograd.gradcheck(lambda a, b: js_divergence(a, b), (la, lb))


def test_c5b_js_gradient_is_zero_when_branches_agree():
    la = rand_logits(seed=10, requires_grad=True)
    lb = la.detach().clone().requires_grad_(True)
    js_divergence(la, lb).backward()
    assert la.grad.abs().max() < 1e-6 and lb.grad.abs().max() < 1e-6


def test_c6_lambda_zero_is_exactly_the_e3b_loss():
    labels = torch.randint(0, 3, (64,), generator=torch.Generator().manual_seed(1))
    la, lb = rand_logits(seed=11, requires_grad=True), rand_logits(seed=12, requires_grad=True)
    out = paired_consistency_loss(la, lb, labels, lam=0.0)
    e3b = 0.5 * (F.cross_entropy(la, labels) + F.cross_entropy(lb, labels))
    assert torch.equal(out.total, e3b) and torch.equal(out.total, out.ce)
    assert not out.js.requires_grad and out.js.item() > 0  # JS is logged but not optimized
    ga = torch.autograd.grad(out.total, [la, lb])
    gb = torch.autograd.grad(e3b, [la, lb])
    assert all(torch.equal(x, y) for x, y in zip(ga, gb))


def test_lambda_positive_adds_lambda_times_js():
    labels = torch.randint(0, 3, (64,), generator=torch.Generator().manual_seed(2))
    la, lb = rand_logits(seed=13), rand_logits(seed=14)
    base = paired_consistency_loss(la, lb, labels, lam=0.0)
    for lam in (0.3, 1.0, 3.0):
        out = paired_consistency_loss(la, lb, labels, lam=lam)
        assert torch.allclose(out.ce, base.ce) and torch.allclose(out.js, base.js)
        assert torch.allclose(out.total, base.ce + lam * base.js, atol=1e-6)
        assert out.total.item() > base.total.item()


def test_module_wrapper_matches_function():
    labels = torch.randint(0, 3, (16,), generator=torch.Generator().manual_seed(3))
    la, lb = rand_logits(16, seed=15), rand_logits(16, seed=16)
    assert torch.equal(PairedConsistencyLoss(lam=1.0)(la, lb, labels).total, paired_consistency_loss(la, lb, labels, 1.0).total)


def test_invalid_inputs_are_rejected():
    la, lb = rand_logits(4, seed=17), rand_logits(4, seed=18)
    labels = torch.zeros(4, dtype=torch.long)
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            paired_consistency_loss(la, lb, labels, lam=bad)
        with pytest.raises(ValueError):
            PairedConsistencyLoss(lam=bad)
    with pytest.raises(ValueError):
        paired_consistency_loss(la, lb[:3], labels, lam=1.0)
    with pytest.raises(ValueError):
        paired_consistency_loss(la, lb, labels[:3], lam=1.0)
    with pytest.raises(ValueError):
        js_divergence(la, lb, reduction="max")
