"""Reliability metrics (Section 15.1). Owner: Parth."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 15


def _bins(conf: np.ndarray, n_bins: int) -> np.ndarray:
    # equal-width bins on top-1 confidence; bin m covers (m/n, (m+1)/n], conf = 1 goes in the last bin
    return np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)


def reliability_table(probs, y, n_bins: int = N_BINS) -> pd.DataFrame:
    probs, y = np.asarray(probs, np.float64), np.asarray(y, int)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    b = _bins(conf, n_bins)
    rows = []
    for m in range(n_bins):
        sel = b == m
        rows.append({"bin": m, "lo": m / n_bins, "hi": (m + 1) / n_bins, "n": int(sel.sum()),
                     "confidence": float(conf[sel].mean()) if sel.any() else float("nan"),
                     "accuracy": float(correct[sel].mean()) if sel.any() else float("nan")})
    return pd.DataFrame(rows)


def ece(probs, y, n_bins: int = N_BINS) -> float:
    """sum_m |B_m|/N |acc(B_m) - conf(B_m)| over 15 equal-width top-1 confidence bins."""
    t = reliability_table(probs, y, n_bins)
    t = t[t["n"] > 0]
    return float((t["n"] / t["n"].sum() * (t["accuracy"] - t["confidence"]).abs()).sum())


def brier(probs, y) -> float:
    """(1/N) sum_w sum_c (p_c(w) - 1[y(w) = c])^2 (multi-class, range [0, 2])."""
    probs, y = np.asarray(probs, np.float64), np.asarray(y, int)
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(((probs - onehot) ** 2).sum(axis=1)))
