"""Chronological split with strict purge/embargo (Execution Plan D3, D4, D6, Section 5.4).

A sample anchored at trading day d uses input days [d-L+1, d] and label days
[d+1, d+H]; its full information span is [d-L+1, d+H]. With B = first trading day
of an evaluation period, the strict rule (no trading day in both a training span
and an evaluation span) gives

    train side:  d + H <= B - 1      ->  d <= B - H - 1      (B - 6 for H = 5)
    eval side:   d - L + 1 >= B      ->  d >= B + L - 1      (B + 29 for L = 30)

so anchors B-H .. B+L-2 (34 anchors for L=30, H=5) are purged at each boundary.
Indices here are positions in the *global* trading calendar, so every ticker
uses the same boundary dates. Stride is applied after split assignment, within
each contiguous segment, aligned to the segment's first global anchor so anchor
dates line up across tickers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRAIN, VAL, TEST, PURGED = "train", "val", "test", "purged"
SPLITS = (TRAIN, VAL, TEST)


class LeakageError(AssertionError):
    """A split boundary rule is violated."""


@dataclass(frozen=True)
class SplitPlan:
    input_len: int
    horizon: int
    stride: int
    b_val: int          # calendar index of the first validation trading day
    b_test: int         # calendar index of the first test trading day
    n_days: int         # calendar length

    @property
    def segments(self) -> list[tuple[str, int, int]]:
        """(split, first_anchor_idx, last_anchor_idx) for each contiguous segment."""
        L, H = self.input_len, self.horizon
        return [
            (TRAIN, L - 1, self.b_val - H - 1),
            (PURGED, self.b_val - H, self.b_val + L - 2),
            (VAL, self.b_val + L - 1, self.b_test - H - 1),
            (PURGED, self.b_test - H, self.b_test + L - 2),
            (TEST, self.b_test + L - 1, self.n_days - 1 - H),
        ]


def plan_splits(cal_dates: np.ndarray, splits_cfg: dict, input_len: int, horizon: int,
                stride: int) -> SplitPlan:
    """Locate the boundaries B in the global calendar (array of 'YYYY-MM-DD')."""
    tr, va, te = splits_cfg["train"], splits_cfg["val"], splits_cfg["test"]
    if not (tr[1] < va[0] <= va[1] < te[0]):
        raise ValueError(f"split periods must be ordered and disjoint: {splits_cfg}")
    cal = np.asarray(cal_dates)
    b_val = int(np.searchsorted(cal, va[0], side="left"))
    b_test = int(np.searchsorted(cal, te[0], side="left"))
    if not (0 < b_val < b_test < len(cal)):
        raise ValueError("calendar does not cover all three split periods")
    plan = SplitPlan(input_len, horizon, stride, b_val, b_test, len(cal))
    for name, lo, hi in plan.segments:
        if hi < lo:
            raise ValueError(f"segment {name} is empty: [{lo}, {hi}]")
    return plan


def assign_segments(anchor_idx: np.ndarray, plan: SplitPlan) -> tuple[np.ndarray, np.ndarray]:
    """Return (split label, segment start index) for each anchor calendar index.

    Anchors outside every segment (too early / too late for a full span) get
    split '' and must be dropped by the caller.
    """
    a = np.asarray(anchor_idx)
    split = np.full(a.shape, "", dtype=object)
    start = np.full(a.shape, -1, dtype=np.int64)
    for name, lo, hi in plan.segments:
        m = (a >= lo) & (a <= hi)
        split[m] = name
        start[m] = lo
    return split, start


def stride_mask(anchor_idx: np.ndarray, seg_start: np.ndarray, stride: int) -> np.ndarray:
    return ((np.asarray(anchor_idx) - np.asarray(seg_start)) % stride) == 0


def expected_purged_per_boundary(input_len: int, horizon: int, stride: int) -> int:
    """Purged anchors per ticker per boundary after striding (7 for 30/5/5)."""
    n_daily = input_len + horizon - 1          # 34 daily anchors purged
    return -(-n_daily // stride)               # ceil


# --------------------------------------------------------------------------- checks
def check_leakage(windows: pd.DataFrame) -> None:
    """T-L1, T-L2, T-L3 on a windows table. Raises LeakageError on any violation.

    Dates are ISO strings, so lexicographic comparison is chronological.
    """
    w = windows[windows["split"].isin(SPLITS)]
    if w["window_id"].duplicated().any():
        raise LeakageError("duplicate window_id")
    ids = {s: set(w.loc[w["split"] == s, "window_id"]) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if ids[a] & ids[b]:
                raise LeakageError(f"T-L3: window_id shared by {a} and {b}")

    for early, late in ((TRAIN, VAL), (VAL, TEST), (TRAIN, TEST)):
        e, l = w[w["split"] == early], w[w["split"] == late]
        if e.empty or l.empty:
            continue
        # T-L2: all anchors of the earlier split precede all anchors of the later split
        if not e["anchor_date"].max() < l["anchor_date"].min():
            raise LeakageError(f"T-L2: {early} anchors overlap {late} anchors")
        # T-L1 per ticker: last label day of earlier split < first input day of later split
        e_end = e.groupby("ticker")["label_end_date"].max()
        l_start = l.groupby("ticker")["input_start_date"].min()
        both = e_end.index.intersection(l_start.index)
        bad = [t for t in both if not e_end[t] < l_start[t]]
        if bad:
            raise LeakageError(f"T-L1: {early} label span reaches {late} input span for {bad[:5]}")
        # global boundary form (boundaries are global dates)
        if not e["label_end_date"].max() < l["input_start_date"].min():
            raise LeakageError(f"T-L1: global {early}/{late} spans share a trading day")
