"""T-D1 to T-D4 plus schema checks for data_v1 (Execution Plan 16.2; owner Farzaan)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from g1g2_fixtures import full_data, needs_full_data, synthetic_build
from src.data import labels as lab
from src.data.download import download_all, load_raw, verify_cache
from src.data.windows import WINDOW_COLUMNS, ohlcv_hash, window_from_raw

RNG = np.random.default_rng(255)


def _cal_index(cal: np.ndarray, dates: pd.Series) -> np.ndarray:
    idx = np.searchsorted(cal, dates.to_numpy())
    assert (cal[idx] == dates.to_numpy()).all(), "date not on the global calendar"
    return idx


def _check_schema(windows: pd.DataFrame, ohlcv: np.ndarray):
    assert list(windows.columns) == WINDOW_COLUMNS
    assert ohlcv.dtype == np.float32 and ohlcv.shape[1:] == (30, 5)
    assert len(windows) == ohlcv.shape[0]
    assert (windows["row_idx"].to_numpy() == np.arange(len(windows))).all()
    assert windows["window_id"].is_unique
    assert (windows["window_id"] == windows["ticker"] + "_" + windows["anchor_date"]).all()
    assert set(windows["split"]) <= {"train", "val", "test", "purged"}
    assert set(windows["label"]) <= {0, 1, 2}
    assert set(windows["asset_type"]) <= {"stock", "etf"}
    assert (windows["data_version"] == "data_v1").all()
    assert np.isfinite(ohlcv).all()
    o, h, l, c, v = (ohlcv[..., i] for i in range(5))
    assert (h >= np.maximum(o, c)).all() and (l <= np.minimum(o, c)).all() and (l > 0).all()
    assert (v >= 0).all()


def _check_d1_d3(windows, ohlcv, cleaned, cal, sample):
    s = _cal_index(cal, windows["input_start_date"])
    a = _cal_index(cal, windows["anchor_date"])
    e = _cal_index(cal, windows["label_end_date"])
    assert (a - s == 29).all(), "T-D1: input window is not 30 consecutive trading days"
    assert (e - a == 5).all(), "T-D3: label end is not 5 trading days after the anchor"
    for i in sample:  # T-D1: the stored rows are exactly the raw rows ending at the anchor
        r = windows.iloc[i]
        expect = window_from_raw(cleaned[r["ticker"]], r["anchor_date"])
        assert np.array_equal(np.asarray(ohlcv[r["row_idx"]]), expect)
        dates = cleaned[r["ticker"]]["Date"].to_numpy()
        j = int(np.searchsorted(dates, r["anchor_date"]))
        assert dates[j - 29] == r["input_start_date"] and dates[j + 5] == r["label_end_date"]


def _check_d2(windows, cleaned, sample):
    for i in sample:  # T-D2: r recomputed from raw closes
        r = windows.iloc[i]
        df = cleaned[r["ticker"]].set_index("Date")
        expect = df.loc[r["label_end_date"], "Close"] / df.loc[r["anchor_date"], "Close"] - 1.0
        assert r["fwd_return"] == pytest.approx(expect, rel=0, abs=1e-12)
        closes = df.loc[r["input_start_date"]:r["anchor_date"], "Close"].to_numpy()
        assert len(closes) == 30
        sig = lab.sigma_input(closes[None, :])[0]
        assert r["sigma_30"] == pytest.approx(sig, rel=1e-12)
        assert r["label_score"] == pytest.approx(r["fwd_return"] / (sig * np.sqrt(5)), rel=1e-9)


def _check_d4(windows, ohlcv):
    hashes = [ohlcv_hash(np.asarray(ohlcv[i])) for i in range(len(windows))]
    assert hashes == windows["ohlcv_hash"].tolist(), "T-D4: ohlcv_hash does not recompute"


# --------------------------------------------------------------------- synthetic
def test_schema_synthetic():
    _, _, _, res = synthetic_build()
    _check_schema(res.windows, res.ohlcv)


def test_td1_td3_synthetic():
    _, _, cleaned, res = synthetic_build()
    _check_d1_d3(res.windows, res.ohlcv, cleaned, res.calendar, range(len(res.windows)))


def test_td2_synthetic():
    _, _, cleaned, res = synthetic_build()
    _check_d2(res.windows, cleaned, RNG.choice(len(res.windows), 200, replace=False))


def test_td4_synthetic():
    _, _, _, res = synthetic_build()
    _check_d4(res.windows, res.ohlcv)


def test_gap_windows_dropped():
    """A window whose 35-day span covers a missing day for that ticker must not exist."""
    _, raw, _, res = synthetic_build()
    cal = res.calendar
    missing = sorted(set(cal) - set(raw["BBB"]["Date"]))
    assert len(missing) == 1
    gap = int(np.searchsorted(cal, missing[0]))
    b = res.windows[res.windows["ticker"] == "BBB"]
    a = np.searchsorted(cal, b["anchor_date"].to_numpy())
    assert not ((a - 29 <= gap) & (gap <= a + 5)).any()
    other = res.windows[res.windows["ticker"] == "AAA"]
    a2 = np.searchsorted(cal, other["anchor_date"].to_numpy())
    assert ((a2 - 29 <= gap) & (gap <= a2 + 5)).any(), "control ticker should keep windows there"


def test_labels_balanced_on_train_synthetic():
    _, _, _, res = synthetic_build()
    tr = res.windows[res.windows["split"] == "train"]["label"].value_counts(normalize=True)
    assert np.allclose(tr.sort_index().to_numpy(), 1 / 3, atol=0.01)


def test_download_cache_is_idempotent(tmp_path):
    calls = []

    def fake_fetch(t, start, end):
        calls.append(t)
        return pd.DataFrame({"Date": ["2020-01-02", "2020-01-03"], "Open": [1.0, 2.0], "High": [1.5, 2.5],
                             "Low": [0.5, 1.5], "Close": [1.2, 2.2], "Volume": [10.0, 20.0]})

    download_all(["XX", "YY"], str(tmp_path), "2020-01-01", "2020-12-31", pause=0, fetch=fake_fetch)
    download_all(["XX", "YY"], str(tmp_path), "2020-01-01", "2020-12-31", pause=0, fetch=fake_fetch)
    assert calls == ["XX", "YY"]
    verify_cache(["XX", "YY"], str(tmp_path))
    assert load_raw(str(tmp_path), "XX")["Close"].tolist() == [1.2, 2.2]
    (tmp_path / "XX.csv").write_text("Date,Open,High,Low,Close,Volume\n")
    with pytest.raises(RuntimeError, match="sha256"):
        verify_cache(["XX"], str(tmp_path))


# --------------------------------------------------------------------- full data
@needs_full_data
def test_schema_full():
    _, windows, ohlcv, _, _, _ = full_data()
    _check_schema(windows, np.asarray(ohlcv))
    assert windows["ticker"].nunique() == 50


@needs_full_data
def test_td1_td3_full():
    _, windows, ohlcv, _, cleaned, cal = full_data()
    _check_d1_d3(windows, ohlcv, cleaned, cal, RNG.choice(len(windows), 500, replace=False))


@needs_full_data
def test_td2_full():
    _, windows, _, _, cleaned, _ = full_data()
    _check_d2(windows, cleaned, RNG.choice(len(windows), 200, replace=False))


@needs_full_data
def test_td4_full():
    _, windows, ohlcv, _, _, _ = full_data()
    _check_d4(windows, ohlcv)
