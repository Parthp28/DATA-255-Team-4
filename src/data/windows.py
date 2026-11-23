"""Cleaning, global calendar, rolling windows and the data_v1 window registry.

Schema (Execution Plan 10.1, ``data_v1/windows.parquet``), one row per window:
window_id ({ticker}_{anchor_date}), row_idx (row in ohlcv.npy, shape N x 30 x 5,
float32, columns O,H,L,C,V), ticker, asset_type, input_start_date, anchor_date,
label_end_date, split (train/val/test/purged), fwd_return, sigma_30, label_score,
label (0 DOWN, 1 NEUTRAL, 2 UP), ohlcv_hash (sha256 of the window's float32
bytes), data_version. Dates are ISO 'YYYY-MM-DD' strings (same as the dummy
registry the trainer was built against).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import labels as lab
from . import splits as sp

OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")
WINDOW_COLUMNS = [
    "window_id", "row_idx", "ticker", "asset_type", "input_start_date", "anchor_date",
    "label_end_date", "split", "fwd_return", "sigma_30", "label_score", "label",
    "ohlcv_hash", "data_version",
]


def ohlcv_hash(window: np.ndarray) -> str:
    """sha256 of the window's float32 bytes (C order)."""
    return hashlib.sha256(np.ascontiguousarray(window, dtype=np.float32).tobytes()).hexdigest()


# --------------------------------------------------------------------------- cleaning
def clean_ohlcv(df: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Sort, de-duplicate, restrict to [start, end], drop invalid rows, repair H/L drift.

    Adjusted series can carry float drift so that High < max(Open, Close) by a
    hair; such rows are repaired (High = max(O, H, C), Low = min(O, L, C)) and counted.
    """
    stats = {}
    d = df.copy()
    d["Date"] = d["Date"].astype(str).str.slice(0, 10)
    d = d.sort_values("Date")
    n0 = len(d)
    d = d.drop_duplicates("Date", keep="last")
    stats["duplicate_dates"] = n0 - len(d)
    d = d[(d["Date"] >= start) & (d["Date"] <= end)]
    num = d[list(OHLCV_FIELDS)].apply(pd.to_numeric, errors="coerce")
    bad = num.isna().any(axis=1) | (num[["Open", "High", "Low", "Close"]] <= 0).any(axis=1) \
        | (num["Volume"] < 0)
    stats["invalid_rows_dropped"] = int(bad.sum())
    d = pd.concat([d[["Date"]], num], axis=1)[~bad]
    hi = d[["Open", "High", "Close"]].max(axis=1)
    lo = d[["Open", "Low", "Close"]].min(axis=1)
    stats["hl_repaired"] = int(((hi != d["High"]) | (lo != d["Low"])).sum())
    d["High"], d["Low"] = hi, lo
    stats["rows"] = int(len(d))
    return d.reset_index(drop=True), stats


def global_calendar(frames: dict[str, pd.DataFrame], min_share: float) -> np.ndarray:
    """Trading days on which at least ``min_share`` of tickers have a row."""
    counts = pd.concat([f["Date"] for f in frames.values()]).value_counts()
    keep = counts[counts >= min_share * len(frames)].index
    return np.array(sorted(keep), dtype=object)


# --------------------------------------------------------------------------- build
@dataclass
class BuildResult:
    windows: pd.DataFrame
    ohlcv: np.ndarray
    thresholds: tuple[float, float]
    plan: sp.SplitPlan
    calendar: np.ndarray
    report: dict = field(default_factory=dict)


def build_windows(raw: dict[str, pd.DataFrame], asset_type: dict[str, str], cfg: dict) -> BuildResult:
    """Build the full window registry from raw per-ticker frames (pure function of the cache)."""
    src, wcfg = cfg["source"], cfg["window"]
    L, H, stride = int(wcfg["input_len"]), int(wcfg["horizon"]), int(wcfg["stride"])
    version = cfg["data_version"]

    cleaned, clean_stats = {}, {}
    for t, df in raw.items():
        cleaned[t], clean_stats[t] = clean_ohlcv(df, src["start"], src["end"])
    cal = global_calendar(cleaned, float(cfg.get("calendar_min_ticker_share", 0.5)))
    plan = sp.plan_splits(cal, cfg["splits"], L, H, stride)
    n = len(cal)
    offs = np.arange(-(L - 1), 1)

    parts, arrays, gap_stats = [], [], {}
    for t in raw:  # config order
        d = cleaned[t]
        on_cal = np.isin(d["Date"].to_numpy(), cal)
        d = d[on_cal]
        gidx = np.searchsorted(cal, d["Date"].to_numpy())
        present = np.zeros(n, dtype=bool)
        present[gidx] = True
        series = np.full((n, 5), np.nan, dtype=np.float64)
        series[gidx] = d[list(OHLCV_FIELDS)].to_numpy(dtype=np.float64)

        # full span [a-L+1, a+H] present  <=>  window sum of present == L+H
        csum = np.concatenate([[0], np.cumsum(present)])
        a_all = np.arange(L - 1, n - H)
        full = (csum[a_all + H + 1] - csum[a_all - L + 1]) == (L + H)
        a_valid = a_all[full]
        split, seg_start = sp.assign_segments(a_valid, plan)
        keep = (split != "") & sp.stride_mask(a_valid, seg_start, stride)
        a = a_valid[keep]
        split = split[keep]
        gap_stats[t] = {"off_calendar_rows": int((~on_cal).sum()),
                        "calendar_days_missing": int(n - present.sum()),
                        "anchors_lost_to_gaps": int((~full).sum())}
        if a.size == 0:
            continue

        win = series[a[:, None] + offs[None, :]]                 # (k, L, 5) float64
        close_d = series[a, 3]
        close_dh = series[a + H, 3]
        r = lab.forward_return(close_d, close_dh)
        sig = lab.sigma_input(win[:, :, 3])
        z = lab.z_score(r, sig, H)
        win32 = win.astype(np.float32)
        anchors = cal[a]
        parts.append(pd.DataFrame({
            "window_id": [f"{t}_{x}" for x in anchors],
            "ticker": t,
            "asset_type": asset_type[t],
            "input_start_date": cal[a - (L - 1)],
            "anchor_date": anchors,
            "label_end_date": cal[a + H],
            "split": split.astype(str),
            "fwd_return": r,
            "sigma_30": sig,
            "label_score": z,
            "ohlcv_hash": [ohlcv_hash(w) for w in win32],
            "data_version": version,
        }))
        arrays.append(win32)

    windows = pd.concat(parts, ignore_index=True)
    ohlcv = np.concatenate(arrays, axis=0)
    windows.insert(1, "row_idx", np.arange(len(windows), dtype=np.int64))

    train_z = windows.loc[windows["split"] == sp.TRAIN, "label_score"].to_numpy()
    thresholds = lab.fit_thresholds(train_z)
    windows["label"] = lab.assign_labels(windows["label_score"].to_numpy(), thresholds)
    windows = windows[WINDOW_COLUMNS]
    for c in ("input_start_date", "anchor_date", "label_end_date"):
        windows[c] = windows[c].astype(str)

    report = {
        "cleaning": clean_stats,
        "gaps": gap_stats,
        "calendar": {"days": int(n), "first": str(cal[0]), "last": str(cal[-1]),
                     "b_val": str(cal[plan.b_val]), "b_test": str(cal[plan.b_test])},
        "flat_sigma_windows": int((windows["sigma_30"] <= 0).sum()),
    }
    return BuildResult(windows, ohlcv, thresholds, plan, cal, report)


def window_from_raw(raw_clean: pd.DataFrame, anchor_date: str, input_len: int = 30) -> np.ndarray:
    """Rebuild one input window (float32, L x 5) straight from a cleaned raw frame."""
    dates = raw_clean["Date"].to_numpy()
    i = int(np.searchsorted(dates, anchor_date))
    if i >= len(dates) or dates[i] != anchor_date or i - input_len + 1 < 0:
        raise KeyError(anchor_date)
    rows = raw_clean.iloc[i - input_len + 1: i + 1][list(OHLCV_FIELDS)]
    return rows.to_numpy(dtype=np.float64).astype(np.float32)
