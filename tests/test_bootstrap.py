"""T-E9 (owner Parth): the block bootstrap keeps whole dates and covers about 95% on i.i.d. data."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.bootstrap import block_bootstrap, calendar_positions  # noqa: E402


def test_e9a_resamples_keep_whole_dates():
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=200), 5)   # 5 tickers per date
    seen = []

    def stat(idx):
        seen.append(idx)
        return float(len(idx))

    block_bootstrap(stat, dates, n_boot=50, block_len=20, seed=0)
    d = pd.Series(dates)
    full = d.value_counts()
    for idx in seen[1:]:
        counts = d.iloc[idx].value_counts()
        # each drawn date appears a multiple of its 5 rows (whole dates, possibly repeated)
        assert ((counts % full.loc[counts.index]) == 0).all()


def test_e9b_blocks_are_contiguous_trading_days():
    cal = pd.bdate_range("2020-01-01", periods=100)
    pos, T = calendar_positions(cal[::5], cal)
    assert T == 100 and pos.tolist() == list(range(0, 100, 5))


def test_e9c_coverage_near_95pct_on_iid_data():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=630)
    hits, trials = 0, 150
    for t in range(trials):
        x = rng.normal(0.0, 1.0, len(dates))
        res = block_bootstrap(lambda idx, x=x: float(x[idx].mean()), dates, n_boot=400, block_len=63, seed=t)
        hits += res["ci_low"] <= 0.0 <= res["ci_high"]
    assert 0.88 <= hits / trials <= 0.99


def test_bootstrap_point_estimate_uses_all_rows():
    dates = pd.bdate_range("2021-01-01", periods=100)
    x = np.arange(100.0)
    res = block_bootstrap(lambda idx: float(x[idx].mean()), dates, n_boot=100, block_len=10, seed=1)
    assert res["point"] == x.mean() and res["ci_low"] < res["point"] < res["ci_high"]
