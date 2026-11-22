"""Synthetic raw OHLCV frames in the cache format, for offline tests of G1/G2 code."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_raw(tickers: list[str], start: str = "2005-01-03", end: str = "2024-12-31",
                  seed: int = 0, drop_days: dict[str, list[int]] | None = None) -> dict[str, pd.DataFrame]:
    """Random-walk OHLCV on business days. ``drop_days`` removes row positions per ticker (gaps)."""
    dates = pd.bdate_range(start, end).strftime("%Y-%m-%d").to_numpy()
    rng = np.random.default_rng(seed)
    out = {}
    for t in tickers:
        n = len(dates)
        logp = np.log(50 + 50 * rng.random()) + np.cumsum(rng.normal(0, 0.015, n))
        c = np.exp(logp)
        o = c * np.exp(rng.normal(0, 0.005, n))
        h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.006, n)))
        lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
        v = rng.integers(1_000_000, 20_000_000, n).astype(np.float64)
        df = pd.DataFrame({"Date": dates, "Open": o, "High": h, "Low": lo, "Close": c, "Volume": v})
        if drop_days and t in drop_days:
            df = df.drop(index=drop_days[t]).reset_index(drop=True)
        out[t] = df
    return out


def synthetic_cfg(tickers: dict[str, list[str]] | None = None) -> dict:
    """A data config with the production boundaries and window parameters."""
    return {
        "data_version": "data_v1",
        "source": {"provider": "synthetic", "start": "2005-01-03", "end": "2024-12-31"},
        "tickers": tickers or {"stock": ["AAA", "BBB"], "etf": ["EEE"]},
        "window": {"input_len": 30, "horizon": 5, "stride": 5},
        "splits": {"train": ["2005-01-03", "2016-12-31"], "val": ["2017-01-01", "2019-12-31"],
                   "test": ["2020-01-01", "2024-12-31"]},
        "labels": {"classes": ["DOWN", "NEUTRAL", "UP"], "thresholds_from": "train"},
        "calendar_min_ticker_share": 0.5,
    }
