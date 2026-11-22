"""Volatility-scaled 5-day return labels (Execution Plan D5).

    r       = Close[d+H] / Close[d] - 1
    sigma30 = sample SD (ddof=1) of the L-1 = 29 log returns inside the 30-row input window
    z       = r / (sigma30 * sqrt(H))
    label   = DOWN (0) if z <= t1, UP (2) if z > t2, else NEUTRAL (1)

(t1, t2) are the 1/3 and 2/3 quantiles of z over *training* windows only and are
applied unchanged to validation and test. sigma30 uses input rows only.
"""
from __future__ import annotations

import numpy as np

CLASS_NAMES = ("DOWN", "NEUTRAL", "UP")
DOWN, NEUTRAL, UP = 0, 1, 2


def forward_return(close_d: np.ndarray, close_dh: np.ndarray) -> np.ndarray:
    return np.asarray(close_dh, dtype=np.float64) / np.asarray(close_d, dtype=np.float64) - 1.0


def sigma_input(closes: np.ndarray) -> np.ndarray:
    """closes: (N, L) input-window closes -> (N,) SD of within-window log returns."""
    lr = np.diff(np.log(np.asarray(closes, dtype=np.float64)), axis=1)
    return lr.std(axis=1, ddof=1)


def z_score(r: np.ndarray, sigma: np.ndarray, horizon: int) -> np.ndarray:
    denom = np.asarray(sigma, dtype=np.float64) * np.sqrt(horizon)
    out = np.zeros_like(denom)
    np.divide(r, denom, out=out, where=denom > 0)   # flat window -> z = 0 (counted in report)
    return out


def fit_thresholds(z_train: np.ndarray) -> tuple[float, float]:
    z = np.asarray(z_train, dtype=np.float64)
    if z.size < 3:
        raise ValueError("need training windows to fit tertile thresholds")
    t1, t2 = np.quantile(z, [1.0 / 3.0, 2.0 / 3.0])
    return float(t1), float(t2)


def assign_labels(z: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    t1, t2 = thresholds
    z = np.asarray(z, dtype=np.float64)
    lab = np.full(z.shape, NEUTRAL, dtype=np.int64)
    lab[z <= t1] = DOWN
    lab[z > t2] = UP
    return lab
