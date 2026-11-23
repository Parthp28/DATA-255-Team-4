"""Moving-block bootstrap over test anchor dates (D14; Section 15.1). Owner: Parth.

Blocks are runs of `block_len` consecutive trading days of the calendar of anchor dates. Each
resample draws ceil(T / block_len) block starts uniformly and keeps every row (all tickers and all
renderings) whose anchor date falls inside a drawn block, so whole dates always move together.
The statistic is recomputed on each resample; the 95% percentile interval is reported.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def calendar_positions(anchor_dates, calendar=None) -> tuple[np.ndarray, int]:
    """Map each row's anchor date to an integer trading-day position.

    `calendar`: sorted trading days covering the test period. Default: the business-day range
    spanned by the dates (a close proxy; pass the real calendar for production use).
    """
    d = pd.to_datetime(pd.Series(anchor_dates)).dt.normalize()
    if calendar is None:
        calendar = pd.bdate_range(d.min(), d.max())
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize()
    pos = cal.get_indexer(d)
    if (pos < 0).any():
        raise ValueError("anchor dates missing from the calendar")
    return pos.astype(np.int64), len(cal)


def block_bootstrap(stat: Callable[[np.ndarray], float], anchor_dates, n_boot: int = 1000,
                    block_len: int = 63, seed: int = 0, calendar=None, alpha: float = 0.05) -> dict:
    """stat(row_index_array) -> float, computed on the rows selected by one resample.

    Rows are repeated when their block is drawn more than once (bootstrap with replacement).
    """
    pos, T = calendar_positions(anchor_dates, calendar)
    order = np.argsort(pos, kind="mergesort")
    sorted_pos = pos[order]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_len))
    max_start = max(T - block_len, 0)
    point = float(stat(np.arange(len(pos))))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, n_blocks)
        lo = np.searchsorted(sorted_pos, starts, side="left")
        hi = np.searchsorted(sorted_pos, starts + block_len, side="left")
        idx = np.concatenate([order[a:z] for a, z in zip(lo, hi)]) if n_blocks else np.array([], int)
        vals[b] = stat(idx) if len(idx) else np.nan
    vals = vals[np.isfinite(vals)]
    return {"point": point, "ci_low": float(np.quantile(vals, alpha / 2)),
            "ci_high": float(np.quantile(vals, 1 - alpha / 2)), "n_boot": int(n_boot),
            "block_len": int(block_len), "samples": vals}
