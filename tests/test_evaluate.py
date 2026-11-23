"""scripts/evaluate.py on synthetic fixtures: every output table (Section 10.7), planted values, A1-A8,
the prediction-file checks, and the metrics_long.csv contract consumed by scripts/select_lambda.py (D13).
Synthetic fixtures are test inputs, not results."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import evaluate as ev  # noqa: E402
import select_lambda as sl  # noqa: E402
from make_synthetic_predictions import (PLANTED, fixture_dataset, fixture_run_id, make_all,  # noqa: E402
                                        planted_run)
from src.evaluation.robustness import FR_SEEN_RENDERS  # noqa: E402

PLAN_COLUMNS = {
    "metrics_long.csv": ["run_id", "experiment_id", "seed", "split", "render_set", "metric", "value"],
    "per_render.csv": ["experiment_id", "seed", "render_id", "FR", "PD", "dF1", "class_shift"],
    "seeds.csv": ["experiment_id", "metric", "mean", "sd", "s0", "s1", "s2", "cross_seed_disagreement"],
    "comparisons.csv": ["comparison_id", "metric", "exp_a", "exp_b", "delta_mean", "delta_sd", "delta_s0",
                        "delta_s1", "delta_s2", "ci_low", "ci_high", "n_boot", "block_len", "sign_agree"],
}


def write_data(root, windows, registry):
    d = Path(root) / "data"
    d.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(d / "windows.parquet", index=False)
    registry.to_parquet(d / "render_registry.parquet", index=False)
    return d


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("evalfx")
    windows, registry = fixture_dataset()
    data = write_data(root, windows, registry)
    files = make_all(windows, root / "pred")
    tables = ev.evaluate(files, data, "test", root / "eval", allow_synthetic=True, n_boot=200)
    return {"root": root, "data": data, "files": files, "tables": tables, "windows": windows, "registry": registry}


def test_all_tables_have_plan_columns_and_synthetic_stamp(fx):
    for name, cols in PLAN_COLUMNS.items():
        assert list(fx["tables"][name].columns) == cols, name
    assert (fx["root"] / "eval" / "SYNTHETIC_NOT_RESULTS.txt").exists()


def test_planted_flip_rates_recovered_and_r0_is_exactly_zero(fx):
    pr, ml = fx["tables"]["per_render.csv"], fx["tables"]["metrics_long.csv"]
    n = (fx["windows"]["split"] == "test").sum()
    for exp, spec in PLANTED.items():
        if spec["flips"] is None:
            continue
        rows = pr[pr["experiment_id"] == exp]
        assert (rows.loc[rows["render_id"] == "R0", ["FR", "PD", "dF1", "class_shift"]] == 0).all().all()
        for r, f in spec["flips"].items():
            assert rows.loc[rows["render_id"] == r, "FR"].iloc[0] == pytest.approx(round(f * n) / n)
            long = ml[(ml["experiment_id"] == exp) & (ml["render_set"] == r) & (ml["metric"] == "flip_rate")]
            assert np.allclose(long["value"], round(f * n) / n)


def test_seen_row_is_the_mean_over_r1_r2_r4(fx):
    """select_lambda.py trusts the 'seen' flip-rate row; it must exclude R0 (Section 15.1)."""
    assert FR_SEEN_RENDERS == ("R1", "R2", "R4")
    ml = fx["tables"]["metrics_long.csv"]
    fr = ml[ml["metric"] == "flip_rate"].pivot_table(index="run_id", columns="render_set", values="value")
    np.testing.assert_allclose(fr["seen"], fr[["R1", "R2", "R4"]].mean(axis=1))
    np.testing.assert_allclose(fr["held_out"], fr[["R3", "R5", "R6"]].mean(axis=1))
    assert not np.allclose(fr["seen"], fr[["R0", "R1", "R2", "R4"]].mean(axis=1))


def test_metric_names_cover_the_plan(fx):
    ml = fx["tables"]["metrics_long.csv"]
    r0 = set(ml.loc[ml["render_set"] == "R0", "metric"])
    for m in ("accuracy", "macro_f1", "macro_auroc", "precision_down", "recall_up", "f1_neutral", "ece", "brier",
              "flip_rate", "prob_dispersion", "delta_macro_f1", "class_shift", "max_pred_share"):
        assert m in r0, m
    assert set(ml.loc[ml["render_set"] == "NUM", "experiment_id"]) >= {"E5", "B0-majority"}
    assert set(ml.loc[ml["render_set"] == "all", "metric"]) == {"any_flip_rate"}


def test_comparisons_a1_to_a8_with_block_bootstrap_intervals(fx):
    c = fx["tables"]["comparisons.csv"]
    assert set(c["comparison_id"]) == {f"A{i}" for i in range(1, 9)}
    assert (c["ci_low"] <= c["ci_high"]).all() and (c["n_boot"] == 200).all() and (c["block_len"] == 63).all()
    a1 = c[(c["comparison_id"] == "A1") & (c["metric"] == "FR_seen")].iloc[0]
    assert (a1["exp_a"], a1["exp_b"]) == ("E1", "E3b") and a1["delta_mean"] < 0
    assert c.loc[c["comparison_id"] == "A8", "exp_a"].str.startswith("B0").all()


def test_seeds_table_reports_cross_seed_disagreement(fx):
    s = fx["tables"]["seeds.csv"]
    row = s[(s["experiment_id"] == "E1") & (s["metric"] == "FR_seen")].iloc[0]
    assert row["sd"] >= 0 and 0 <= row["cross_seed_disagreement"] <= 1
    assert {"B0-majority", "B0-random", "B0-momentum", "E5"} <= set(s["experiment_id"])


def test_synthetic_fixtures_need_the_explicit_flag(fx, tmp_path):
    with pytest.raises(SystemExit, match="synthetic"):
        ev.evaluate(fx["files"][:1], fx["data"], "test", tmp_path)


def test_prediction_file_checks_reject_bad_files(fx):
    df = pd.read_parquet([f for f in fx["files"] if f.name == f"{fixture_run_id('E1', 0)}.parquet"][0])
    w, reg = fx["windows"], fx["registry"]
    ev.check_prediction_file(df, w, reg)
    with pytest.raises(ValueError, match="incomplete coverage"):
        ev.check_prediction_file(df.iloc[1:], w, reg)
    bad = df.copy()
    bad.loc[0, "p_up"] += 0.01
    with pytest.raises(ValueError, match="sum to 1"):
        ev.check_prediction_file(bad, w, reg)
    bad = df.copy()
    bad.loc[0, "pred_class"] = (bad.loc[0, "pred_class"] + 1) % 3
    with pytest.raises(ValueError, match="argmax"):
        ev.check_prediction_file(bad, w, reg)
    with pytest.raises(ValueError, match="one prediction file"):
        ev.check_prediction_file(pd.concat([df, df.assign(run_id="other")]), w, reg)
    bad = df.copy()
    bad["true_label"] = (bad["true_label"] + 1) % 3
    with pytest.raises(ValueError, match="true_label"):
        ev.check_prediction_file(bad, w, reg)


def test_metrics_long_feeds_select_lambda_d13(tmp_path):
    """End to end on validation fixtures: evaluate.py -> metrics_long.csv -> select_lambda.py (D13)."""
    windows, registry = fixture_dataset()
    data = write_data(tmp_path, windows, registry)
    runs, files = tmp_path / "runs", []
    specs = [("E3b", 0.0, PLANTED["E3b"]["flips"])] + [("E3", lam, {r: f * k for r, f in PLANTED["E3"]["flips"].items()})
                                                      for lam, k in ((0.3, 1.0), (1.0, 0.6), (3.0, 0.2))]
    for exp, lam, flips in specs:
        rid = f"{exp}_s0_{hashlib.sha256(f'{exp}{lam}'.encode()).hexdigest()[:8]}"
        df = planted_run(windows, exp, 0, splits=("val",), flips=flips, run_id=rid)
        files.append(tmp_path / f"{rid}.parquet")
        df.to_parquet(files[-1], index=False)
        (runs / rid).mkdir(parents=True)
        (runs / rid / "run.json").write_text(json.dumps({"run_id": rid, "experiment_id": exp, "seed": 0,
                                                         "consistency_lambda": lam, "status": "complete",
                                                         "git_dirty": False}))
    tables = ev.evaluate(files, data, "val", tmp_path / "eval", runs_dir=runs, allow_synthetic=True, n_boot=10)
    rows = sl.read_metrics_csv(tmp_path / "eval" / "metrics_long.csv")
    inputs, counts = sl.extract_inputs(rows, ["E3"], sl.runs_dir_resolver(runs), filter_input=True)
    x = inputs["E3"]
    assert len(x.candidates) == 3 and counts["excluded"] > 0          # held-out / R3 / R5 / R6 rows never read
    ml = tables["metrics_long.csv"]
    ctrl = ml[(ml["experiment_id"] == "E3b") & (ml["metric"] == "macro_f1") & ml["render_set"].isin(["R0", "R1", "R2", "R4"])]
    assert x.baseline_val_macro_f1 == pytest.approx(ctrl["value"].mean())
    sel = sl.select_lambda("E3", x.baseline_val_macro_f1, x.candidates, baseline_run_id=x.baseline_run_id)
    assert sel.selected_lambda in (0.3, 1.0, 3.0) and sel.rule in ("eligible", "fallback")
