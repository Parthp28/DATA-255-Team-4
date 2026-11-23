"""DEVELOPMENT ONLY: schema-valid synthetic windows, registry, and prediction files with PLANTED properties.

Builds the evaluation library before any real prediction exists (execution plan Section 9.4). Planted
flip fractions make expected metric values known in advance (e.g. T-E3). Everything here carries the
synthetic data_version / render_version, which evaluate.py refuses without --allow-synthetic and which
can never be validated or reported.

    python scripts/make_synthetic_predictions.py --out outputs/synthetic_dev
      -> outputs/synthetic_dev/data/{windows,render_registry}.parquet
         outputs/synthetic_dev/predictions/<run_id>.parquet   (run_id = <experiment>_s<seed>_<sha8>)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.schema import (NUMERIC_RENDER_ID, PREDICTION_COLUMNS, RENDER_COLUMNS, RENDER_IDS,  # noqa: E402
                               SEEN_RENDERS, WINDOW_COLUMNS)

SYNTHETIC_DATA_VERSION = "synthetic_dev"
SYNTHETIC_RENDER_VERSION = "synthetic_render_dev"
LOGIT_COLS = ("logit_down", "logit_neutral", "logit_up")
PROB_COLS = ("p_down", "p_neutral", "p_up")

# Planted per-rendering flip fractions (fixture design, not expectations about real models).
PLANTED = {
    "E1": {"signal": 0.8, "flips": {"R1": 0.10, "R2": 0.20, "R3": 0.25, "R4": 0.10, "R5": 0.30, "R6": 0.40}},
    "E2": {"signal": 0.8, "flips": {"R1": 0.05, "R2": 0.10, "R3": 0.15, "R4": 0.05, "R5": 0.20, "R6": 0.25}},
    "E3b": {"signal": 0.8, "flips": {"R1": 0.05, "R2": 0.05, "R3": 0.20, "R4": 0.05, "R5": 0.25, "R6": 0.35}},
    "E4b": {"signal": 0.8, "flips": {"R1": 0.03, "R2": 0.03, "R3": 0.10, "R4": 0.03, "R5": 0.15, "R6": 0.20}},
    "E3": {"signal": 0.7, "flips": {"R1": 0.02, "R2": 0.02, "R3": 0.15, "R4": 0.02, "R5": 0.20, "R6": 0.30}},
    "E4": {"signal": 0.7, "flips": {"R1": 0.01, "R2": 0.01, "R3": 0.08, "R4": 0.01, "R5": 0.10, "R6": 0.15}},
    "E5": {"signal": 0.6, "flips": None},
}


def fixture_run_id(exp: str, seed: int) -> str:
    """Team run_id format <experiment>_s<seed>_<sha8>; the hash marks a fixture, not a config."""
    return f"{exp}_s{seed}_{hashlib.sha256(f'synthetic-fixture:{exp}:{seed}'.encode()).hexdigest()[:8]}"


def fixture_dataset(n_per_split=(60, 30, 45), tickers=("SYNA", "SYNB", "SYNE"), seed=0,
                    data_version=SYNTHETIC_DATA_VERSION, render_version=SYNTHETIC_RENDER_VERSION):
    """Window registry (10.1) and render registry (10.2) for evaluation fixtures (no images needed).

    Anchors advance in 5-business-day steps, split chronologically, so the block bootstrap sees real dates."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2015-01-02", periods=5 * sum(n_per_split) // len(tickers) + 80)
    rows, i = [], 0
    for split, n in zip(("train", "val", "test"), n_per_split):
        for _ in range(n):
            t = tickers[i % len(tickers)]
            a = 35 + 5 * (i // len(tickers))
            rows.append({"window_id": f"{t}_{cal[a].date()}", "row_idx": i, "ticker": t,
                         "asset_type": "etf" if t.endswith("E") else "stock",
                         "input_start_date": str(cal[a - 29].date()), "anchor_date": str(cal[a].date()),
                         "label_end_date": str(cal[a + 5].date()), "split": split,
                         "fwd_return": float(rng.normal(0, 0.02)), "sigma_30": 0.01, "label_score": 0.0,
                         "label": int(rng.integers(0, 3)),
                         "ohlcv_hash": hashlib.sha256(f"w{i}".encode()).hexdigest(), "data_version": data_version})
            i += 1
    windows = pd.DataFrame(rows)[WINDOW_COLUMNS]
    reg = [{"window_id": w.window_id, "render_id": r, "split": w.split, "label": int(w.label),
            "ohlcv_hash": w.ohlcv_hash,
            "render_role": "reference" if r == "R0" else ("seen" if r in SEEN_RENDERS else "held_out"),
            "image_path": f"{w.split}_{r}/{w.window_id}.png",
            "image_hash": hashlib.sha256(f"{w.window_id}:{r}".encode()).hexdigest(), "render_version": render_version}
           for w in windows.itertuples() for r in RENDER_IDS]
    return windows, pd.DataFrame(reg)[RENDER_COLUMNS]


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _frame(run_id, exp, seed, split, wids, render_id, y, logits):
    p = _softmax(logits)
    df = pd.DataFrame({"run_id": run_id, "experiment_id": exp, "seed": np.int64(seed), "split": split,
                       "window_id": wids, "render_id": render_id, "true_label": y.astype("int64")})
    for j, c in enumerate(LOGIT_COLS):
        df[c] = logits[:, j].astype("float64")
    for j, c in enumerate(PROB_COLS):
        df[c] = p[:, j]
    df["pred_class"] = logits.argmax(1).astype("int64")
    df["checkpoint"] = f"outputs/runs/{run_id}/best.pt"
    return df[PREDICTION_COLUMNS]


def planted_run(windows, exp, seed, splits=("val", "test"), flips=None, signal=0.8, run_id=None):
    """Image-model run: for rendering r a planted fraction flips[r] of windows gets its R0 logits rotated by
    one class (guaranteed argmax change); every other window keeps its R0 logits exactly. flips=None -> E5/B0."""
    parts = []
    run_id = run_id or fixture_run_id(exp, seed)
    for si, split in enumerate(splits):
        w = windows[windows["split"] == split].sort_values("window_id")
        wids, y = w["window_id"].to_numpy(), w["label"].to_numpy()
        rng = np.random.default_rng([seed, si, sum(map(ord, exp))])
        base = rng.normal(0, 1, (len(w), 3))
        base[np.arange(len(w)), y] += signal
        if flips is None:
            parts.append(_frame(run_id, exp, seed, split, wids, NUMERIC_RENDER_ID, y, base))
            continue
        for r in RENDER_IDS:
            lg = base.copy()
            if r != "R0":
                k = int(round(flips[r] * len(w)))
                idx = rng.choice(len(w), k, replace=False)
                lg[idx] = np.roll(base[idx], 1, axis=1)
                same = lg[idx].argmax(1) == base[idx].argmax(1)
                lg[idx[same]] = np.roll(base[idx[same]], 2, axis=1)
            parts.append(_frame(run_id, exp, seed, split, wids, r, y, lg))
    return pd.concat(parts, ignore_index=True)


def b0_runs(windows, splits=("val", "test")):
    """Trivial-baseline-shaped fixtures (majority, stratified random x 3, momentum-like)."""
    out = []
    for kind, seeds in (("majority", [0]), ("random", [0, 1, 2]), ("momentum", [0])):
        exp = f"B0-{kind}"
        for s in seeds:
            parts = []
            for si, split in enumerate(splits):
                w = windows[windows["split"] == split].sort_values("window_id")
                y = w["label"].to_numpy()
                rng = np.random.default_rng([s, si, len(kind)])
                if kind == "majority":
                    lg = np.tile(np.log([0.3, 0.4, 0.3]), (len(w), 1))
                elif kind == "random":
                    lg = np.log(np.eye(3)[rng.integers(0, 3, len(w))] * 0.97 + 0.01)
                else:
                    lg = rng.normal(0, 1, (len(w), 3))
                    lg[np.arange(len(w)), y] += 0.3
                parts.append(_frame(fixture_run_id(exp, s), exp, s, split, w["window_id"].to_numpy(),
                                    NUMERIC_RENDER_ID, y, lg))
            out.append(pd.concat(parts, ignore_index=True))
    return out


def make_all(windows, out_dir, seeds=(0, 1, 2)):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    frames = [planted_run(windows, e, s, flips=spec["flips"], signal=spec["signal"])
              for e, spec in PLANTED.items() for s in seeds] + b0_runs(windows)
    for df in frames:
        p = out / f"{df['run_id'].iloc[0]}.parquet"
        df.to_parquet(p, index=False)
        paths.append(p)
    (out / "SYNTHETIC_FIXTURES_NOT_RESULTS.txt").write_text(
        "Synthetic test fixtures with planted properties. Not results.\n")
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/synthetic_dev")
    a = ap.parse_args(argv)
    windows, registry = fixture_dataset()
    data = Path(a.out) / "data"
    data.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(data / "windows.parquet", index=False)
    registry.to_parquet(data / "render_registry.parquet", index=False)
    paths = make_all(windows, Path(a.out) / "predictions")
    print(f"wrote fixture data to {data} and {len(paths)} synthetic prediction files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
