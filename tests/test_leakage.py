"""T-L1 to T-L7 leakage tests for data_v1 (Execution Plan 5.4, 16.2; owner Farzaan).

Parth writes independent versions of T-L3 and T-L5 for the G1 sign-off; these are
the data owner's versions.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from g1g2_fixtures import full_data, needs_full_data, synthetic_build
from src.data import labels as lab
from src.data import splits as sp
from src.data.synthetic import synthetic_cfg, synthetic_raw
from src.data.windows import build_windows
from src.rendering.rasterizer import image_hash, render
from src.rendering.styles import RENDER_IDS, load_styles


def _tl1(windows):
    w = windows[windows["split"].isin(sp.SPLITS)]
    for early, late in (("train", "val"), ("val", "test")):
        e_end = w[w["split"] == early].groupby("ticker")["label_end_date"].max()
        l_start = w[w["split"] == late].groupby("ticker")["input_start_date"].min()
        assert (e_end < l_start.reindex(e_end.index)).all(), f"T-L1 {early}/{late}"


def _tl2(windows):
    a = {s: windows.loc[windows["split"] == s, "anchor_date"] for s in sp.SPLITS}
    assert a["train"].max() < a["val"].min() and a["val"].max() < a["test"].min()


def _tl3(windows):
    ids = [set(windows.loc[windows["split"] == s, "window_id"]) for s in sp.SPLITS]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])


def _tl4(windows, cal, plan):
    L, H, stride = plan.input_len, plan.horizon, plan.stride
    expect = sp.expected_purged_per_boundary(L, H, stride)
    assert expect == 7 and L + H - 1 == 34
    a = np.searchsorted(cal, windows["anchor_date"].to_numpy())
    mid = (plan.b_val + plan.b_test) // 2
    for B, side in ((plan.b_val, a < mid), (plan.b_test, a >= mid)):
        purged = windows[(windows["split"] == "purged") & side]
        pa = np.searchsorted(cal, purged["anchor_date"].to_numpy())
        assert ((pa >= B - H) & (pa <= B + L - 2)).all(), "purged anchor outside B-5..B+28"
        counts = purged.groupby("ticker").size()
        assert (counts == expect).all(), f"T-L4: purged per ticker at boundary {cal[B]}: {counts.to_dict()}"
        before = windows[(windows["split"] != "purged") & (a < B)]
        after = windows[(windows["split"] != "purged") & (a >= B) & (a < B + 400)]
        assert np.searchsorted(cal, before["anchor_date"]).max() <= B - H - 1     # d <= B-6
        assert np.searchsorted(cal, after["anchor_date"]).min() == B + L - 1      # d >= B+29, first kept


def _tl5(windows, thresholds):
    z = windows["label_score"].to_numpy()
    tr = windows["split"].to_numpy() == "train"
    t_train = lab.fit_thresholds(z[tr])
    assert t_train == pytest.approx(tuple(thresholds), abs=0)
    assert (lab.assign_labels(z, t_train) == windows["label"].to_numpy()).all()
    t_all = lab.fit_thresholds(z[windows["split"].isin(sp.SPLITS).to_numpy()])
    assert t_all != pytest.approx(t_train, abs=1e-6), "T-L5: thresholds should change if val/test leak in"


def _tl6(windows, cal, plan):
    """Planted-leak canary: a boundary-crossing window labelled train must make T-L1 fail."""
    sp.check_leakage(windows)                       # clean data passes
    bad = windows.copy()
    a = np.searchsorted(cal, bad["anchor_date"].to_numpy())
    i = np.flatnonzero((bad["split"].to_numpy() == "purged") & (a == plan.b_val - plan.horizon))[0]
    bad.loc[bad.index[i], "split"] = "train"         # anchor B-5: label span reaches B
    with pytest.raises(sp.LeakageError, match="T-L1"):
        sp.check_leakage(bad)
    with pytest.raises(AssertionError):
        _tl1(bad)
    dup = windows.copy()                             # same window in two splits
    j = np.flatnonzero(dup["split"].to_numpy() == "val")[0]
    extra = dup.iloc[[j]].assign(split="test")
    with pytest.raises(sp.LeakageError):
        sp.check_leakage(__import__("pandas").concat([dup, extra]))


# --------------------------------------------------------------------- synthetic
def test_tl1_tl2_tl3_synthetic():
    _, _, _, res = synthetic_build()
    _tl1(res.windows); _tl2(res.windows); _tl3(res.windows)
    sp.check_leakage(res.windows)


def test_tl4_synthetic():
    _, _, _, res = synthetic_build()
    _tl4(res.windows, res.calendar, res.plan)


def test_tl4_daily_stride_purges_34():
    cfg = synthetic_cfg({"stock": ["AAA"], "etf": []})
    cfg["window"]["stride"] = 1
    res = build_windows(synthetic_raw(["AAA"], seed=1), {"AAA": "stock"}, cfg)
    assert (res.windows["split"] == "purged").sum() == 2 * 34


def test_tl5_synthetic():
    _, _, _, res = synthetic_build()
    _tl5(res.windows, res.thresholds)


def test_tl6_canary_synthetic():
    _, _, _, res = synthetic_build()
    _tl6(res.windows, res.calendar, res.plan)


def test_tl7_row_after_anchor_does_not_change_inputs():
    """Changing row d+1 leaves the window array (E5 input) and every rendering byte-identical."""
    cfg, raw, _, res = synthetic_build()
    kind = {"AAA": "stock", "BBB": "stock", "EEE": "etf"}
    styles, _, _ = load_styles()
    targets = res.windows[(res.windows["ticker"] == "AAA") & res.windows["split"].isin(sp.SPLITS)]
    targets = targets.iloc[[5, len(targets) // 2, -3]]
    for _, row in targets.iterrows():
        raw2 = copy.deepcopy(raw)
        df = raw2["AAA"]
        k = int(np.flatnonzero(df["Date"].to_numpy() == row["anchor_date"])[0]) + 1   # row d+1
        df.loc[k, ["Open", "High", "Low", "Close"]] *= 1.37
        df.loc[k, "Volume"] *= 5
        res2 = build_windows(raw2, kind, cfg)
        r2 = res2.windows.set_index("window_id").loc[row["window_id"]]
        w1, w2 = res.ohlcv[row["row_idx"]], res2.ohlcv[int(r2["row_idx"])]
        assert w1.tobytes() == w2.tobytes()
        assert r2["ohlcv_hash"] == row["ohlcv_hash"]
        for rid in RENDER_IDS:
            assert image_hash(render(w1, styles[rid])) == image_hash(render(w2, styles[rid]))
        # the change is visible to the window anchored at d+1 (sanity: perturbation was real)
        nxt = res2.windows[(res2.windows["ticker"] == "AAA")]
        assert not res.windows.set_index("window_id")["ohlcv_hash"].reindex(nxt["window_id"]).equals(
            nxt.set_index("window_id")["ohlcv_hash"])


# --------------------------------------------------------------------- full data
@needs_full_data
def test_tl1_tl2_tl3_full():
    _, windows, *_ = full_data()
    _tl1(windows); _tl2(windows); _tl3(windows)
    sp.check_leakage(windows)


@needs_full_data
def test_tl4_full():
    cfg, windows, _, _, _, cal = full_data()
    w = cfg["window"]
    plan = sp.plan_splits(cal, cfg["splits"], w["input_len"], w["horizon"], w["stride"])
    _tl4(windows, cal, plan)


@needs_full_data
def test_tl5_full():
    _, windows, _, thresholds, _, _ = full_data()
    _tl5(windows, (thresholds["t1"], thresholds["t2"]))


@needs_full_data
def test_tl6_canary_full():
    cfg, windows, _, _, _, cal = full_data()
    w = cfg["window"]
    _tl6(windows, cal, sp.plan_splits(cal, cfg["splits"], w["input_len"], w["horizon"], w["stride"]))
