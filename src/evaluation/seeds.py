"""Cross-seed stability (Section 15.1). Owner: Parth.

Cross-seed disagreement (CSD) is a *reference* for normal training variability, not a formal noise
floor: the mean over seed pairs of the share of windows where two seeds disagree on R0.
"""
from __future__ import annotations

import itertools

import numpy as np


def mean_sd(values) -> tuple[float, float]:
    """Mean and sample SD (ddof = 1, n = 3 in the study)."""
    v = np.asarray(values, np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def cross_seed_disagreement(preds_by_seed) -> float:
    """preds_by_seed: list of aligned R0 argmax arrays (same window order)."""
    arrs = [np.asarray(p) for p in preds_by_seed]
    if len(arrs) < 2:
        return float("nan")
    if len({len(a) for a in arrs}) != 1:
        raise ValueError("seed predictions must be aligned on the same windows")
    return float(np.mean([np.mean(a != b) for a, b in itertools.combinations(arrs, 2)]))


def sign_agreement(deltas) -> bool:
    """True when every per-seed difference has the same (non-zero) sign."""
    d = np.sign(np.asarray(deltas, np.float64))
    return bool(len(d) > 0 and (d != 0).all() and (d == d[0]).all())
