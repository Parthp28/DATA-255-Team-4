"""B0 trivial baselines (Execution Plan 5.1, 15.2). Technical owner: Farzaan. CPU only; nothing is tuned.

"Above chance" in the report means above the best B0 (the highest Macro F1 of the three; scripts/evaluate.py),
with the interval stated. Three baselines, all computed from data_v1 alone:

B0-majority   the most frequent training label for every window; probabilities = training class frequencies
              (a constant score, so its AUROC is exactly 0.5). One run (seed 0).
B0-random     stratified random: each window's class is drawn from the training class frequencies; its
              probabilities are that drawn class. Seeds 0, 1, 2.
B0-momentum   momentum rule on the label's own construction run backwards in time:
                  z_past = (Close[d] / Close[d-5] - 1) / (sigma30 * sqrt(5))
              with the same sigma30 as the label (sample SD of the 29 input log returns, `sigma_30` in
              windows.parquet) and d the anchor (the last input row). DOWN if z_past <= m1, UP if z_past > m2,
              else NEUTRAL, where (m1, m2) are the 1/3 and 2/3 quantiles of z_past over TRAINING windows only.
              One run (seed 0).

Hard predictions are written as probabilities (1 - EPS) on the predicted class and EPS / 2 on the others, so
logits stay finite and pred_class equals argmax(logits) as the 10.3 validator requires. EPS does not change
Macro F1 or AUROC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common.schema import CLASS_ORDER, N_CLASSES, NUMERIC_RENDER_ID, PREDICTION_COLUMNS

B0_SEEDS = {"B0-majority": (0,), "B0-random": (0, 1, 2), "B0-momentum": (0,)}
B0_KINDS = {"B0-majority": "majority", "B0-random": "random", "B0-momentum": "momentum"}
EPS = 1e-6
HORIZON = 5        # D5: H = 5, so the momentum look-back is also 5 trading days
CLOSE = 3          # ohlcv.npy columns O, H, L, C, V


def train_class_freq(windows: pd.DataFrame) -> np.ndarray:
    y = windows.loc[windows["split"] == "train", "label"].to_numpy(dtype=np.int64)
    if y.size == 0:
        raise ValueError("no training windows: B0 frequencies and thresholds come from train only")
    return np.bincount(y, minlength=N_CLASSES) / y.size


def past_z(windows: pd.DataFrame, ohlcv: np.ndarray, lookback: int = HORIZON) -> np.ndarray:
    """Volatility-scaled return over the last `lookback` input days, ending at the anchor (no future rows)."""
    x = ohlcv[windows["row_idx"].to_numpy()]
    close = x[:, :, CLOSE].astype(np.float64)
    r = close[:, -1] / close[:, -1 - lookback] - 1.0
    denom = windows["sigma_30"].to_numpy(dtype=np.float64) * np.sqrt(lookback)
    out = np.zeros_like(denom)
    np.divide(r, denom, out=out, where=denom > 0)          # flat window -> 0, as in the label code
    return out


def momentum_thresholds(windows: pd.DataFrame, ohlcv: np.ndarray) -> tuple[float, float]:
    train = windows[windows["split"] == "train"]
    if train.empty:
        raise ValueError("no training windows for the momentum thresholds")
    m1, m2 = np.quantile(past_z(train, ohlcv), [1.0 / 3.0, 2.0 / 3.0])
    return float(m1), float(m2)


def _hard_logits(pred: np.ndarray) -> np.ndarray:
    p = np.full((pred.size, N_CLASSES), EPS / (N_CLASSES - 1))
    p[np.arange(pred.size), pred] = 1.0 - EPS
    return np.log(p)


def b0_logits(kind: str, windows: pd.DataFrame, rows: pd.DataFrame, ohlcv: np.ndarray | None, seed: int,
              split_index: int) -> np.ndarray:
    """Logits for `rows` (one split). Everything is fitted on the training rows of `windows` only."""
    freq = train_class_freq(windows)
    n = len(rows)
    if kind == "majority":
        return np.tile(np.log(np.clip(freq, EPS, None)), (n, 1))
    if kind == "random":
        rng = np.random.default_rng([seed, split_index])
        return _hard_logits(rng.choice(N_CLASSES, size=n, p=freq))
    if kind == "momentum":
        if ohlcv is None:
            raise ValueError("B0-momentum needs ohlcv.npy")
        m1, m2 = momentum_thresholds(windows, ohlcv)
        z = past_z(rows, ohlcv)
        pred = np.full(n, 1, dtype=np.int64)
        pred[z <= m1] = 0
        pred[z > m2] = 2
        return _hard_logits(pred)
    raise ValueError(f"unknown B0 kind {kind!r}")


def b0_predictions(experiment_id: str, run_id: str, seed: int, windows: pd.DataFrame, ohlcv: np.ndarray | None,
                   splits=("val",)) -> pd.DataFrame:
    """Prediction file (schema 10.3, render_id NUM) for one B0 run over every window of each split."""
    kind = B0_KINDS[experiment_id]
    parts = []
    for si, split in enumerate(splits):
        rows = windows[windows["split"] == split].sort_values("window_id")
        if rows.empty:
            raise ValueError(f"no {split} windows")
        lg = b0_logits(kind, windows, rows, ohlcv, seed, si)
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        df = pd.DataFrame({"run_id": run_id, "experiment_id": experiment_id, "seed": np.int64(seed), "split": split,
                           "window_id": rows["window_id"].to_numpy(), "render_id": NUMERIC_RENDER_ID,
                           "true_label": rows["label"].to_numpy(dtype=np.int64)})
        for j, c in enumerate(CLASS_ORDER):
            df[f"logit_{c}"] = lg[:, j]
        for j, c in enumerate(CLASS_ORDER):
            df[f"p_{c}"] = p[:, j]
        df["pred_class"] = lg.argmax(1).astype(np.int64)
        df["checkpoint"] = "none"            # B0 has no trained weights
        parts.append(df[PREDICTION_COLUMNS])
    return pd.concat(parts, ignore_index=True)
