"""Shared fixtures for the G1/G2 tests (data, leakage, rendering). Owner: Farzaan.

Synthetic fixtures always run. Full-data fixtures run when the production
artifacts exist under $DATA255_DATA_DIR (default <repo>/data) and skip otherwise,
so a fresh clone without the shared Drive data still passes `pytest -q`.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import download as dl  # noqa: E402
from src.data.synthetic import synthetic_cfg, synthetic_raw  # noqa: E402
from src.data.windows import build_windows, clean_ohlcv, global_calendar  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA255_DATA_DIR", ROOT / "data"))
DATA_V1 = DATA_DIR / "data_v1"
RENDER_V1 = DATA_DIR / "render_v1"
RAW = DATA_DIR / "raw"

SYN_TICKERS = {"stock": ["AAA", "BBB"], "etf": ["EEE"]}
# BBB misses one trading day in 2010 (a gap) -> windows spanning it must be dropped
SYN_GAP = {"BBB": [1300]}


@lru_cache(maxsize=1)
def synthetic_build():
    cfg = synthetic_cfg(SYN_TICKERS)
    raw = synthetic_raw(SYN_TICKERS["stock"] + SYN_TICKERS["etf"], seed=7, drop_days=SYN_GAP)
    kind = {t: k for k, ts in SYN_TICKERS.items() for t in ts}
    res = build_windows(raw, kind, cfg)
    cleaned = {t: clean_ohlcv(df, cfg["source"]["start"], cfg["source"]["end"])[0] for t, df in raw.items()}
    return cfg, raw, cleaned, res


def have_full_data() -> bool:
    return (DATA_V1 / "windows.parquet").exists() and (DATA_V1 / "ohlcv.npy").exists() and RAW.exists()


def have_full_render() -> bool:
    return (RENDER_V1 / "render_registry.parquet").exists()


needs_full_data = pytest.mark.skipif(not have_full_data(), reason=f"production data_v1 not found in {DATA_DIR}")
needs_full_render = pytest.mark.skipif(not have_full_render(), reason=f"production render_v1 not found in {DATA_DIR}")


@lru_cache(maxsize=1)
def full_data():
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
    windows = pd.read_parquet(DATA_V1 / "windows.parquet")
    ohlcv = np.load(DATA_V1 / "ohlcv.npy", mmap_mode="r")
    thresholds = json.load(open(DATA_V1 / "thresholds.json"))
    tickers = cfg["tickers"]["stock"] + cfg["tickers"]["etf"]
    cleaned = {t: clean_ohlcv(dl.load_raw(str(RAW), t), cfg["source"]["start"], cfg["source"]["end"])[0]
               for t in tickers}
    cal = global_calendar(cleaned, float(cfg.get("calendar_min_ticker_share", 0.5)))
    return cfg, windows, ohlcv, thresholds, cleaned, cal
