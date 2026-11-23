"""Run one B0 trivial baseline on CPU and write its prediction file and run.json (Execution Plan 5.1, 15.2).

    python scripts/run_b0.py --config configs/frozen/B0-majority.yaml --data-root data/render_v1
    python scripts/run_b0.py --config configs/frozen/B0-random.yaml   --data-root data/render_v1 --seed 1
    python scripts/run_b0.py --config configs/frozen/B0-momentum.yaml --data-root data/render_v1

Technical owner: Farzaan. B0 has nothing to tune, so its configs can be frozen before any run:

    python scripts/freeze_config.py --config experiments/B0-momentum.yaml --rule "B0: no tuning (plan 5.1)" \
        --out configs/frozen/B0-momentum.yaml

As in scripts/train.py, test-split predictions are written only for a verified frozen config; any other config
predicts validation only. The run record follows plan 10.5 and enters evaluation only after the owner signs it
with scripts/validate_run.py. Seeds follow the plan: majority and momentum seed 0, stratified random 0, 1, 2.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src import tracking  # noqa: E402
from src.baselines import B0_KINDS, B0_SEEDS, b0_predictions, momentum_thresholds, train_class_freq  # noqa: E402
from src.common.config import ExperimentConfig  # noqa: E402
from src.common.schema import validate_prediction_frame  # noqa: E402
from src.evaluation.frozen import frozen_config_info  # noqa: E402
from src.evaluation.gate import owner_of  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="experiments/B0-*.yaml or configs/frozen/B0-*.yaml")
    ap.add_argument("--data-root", required=True, help="directory with windows.parquet and ohlcv.npy")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)

    frozen_info = frozen_config_info(a.config)
    splits = ("val", "test") if frozen_info else ("val",)
    cfg = ExperimentConfig.from_yaml(a.config)
    if cfg.experiment_id not in B0_KINDS:
        raise SystemExit(f"{cfg.experiment_id} is not a B0 baseline ({sorted(B0_KINDS)})")
    if a.seed is not None:
        cfg = dataclasses.replace(cfg, seed=a.seed)
    if cfg.seed not in B0_SEEDS[cfg.experiment_id]:
        raise SystemExit(f"{cfg.experiment_id}: seed {cfg.seed} is not in the plan's seeds {B0_SEEDS[cfg.experiment_id]}")

    t0 = time.time()
    windows = pd.read_parquet(os.path.join(a.data_root, "windows.parquet"))
    ohlcv = np.load(os.path.join(a.data_root, "ohlcv.npy")) if B0_KINDS[cfg.experiment_id] == "momentum" else None
    run_id = f"{cfg.experiment_id}_s{cfg.seed}_{cfg.sha256()[:8]}"
    preds = b0_predictions(cfg.experiment_id, run_id, cfg.seed, windows, ohlcv, splits)
    validate_prediction_frame(preds, image_model=False)
    for s in splits:
        want = set(windows.loc[windows["split"] == s, "window_id"])
        if set(preds.loc[preds["split"] == s, "window_id"]) != want:
            raise SystemExit(f"{run_id}: {s} predictions do not cover every {s} window")

    val = preds[preds["split"] == "val"]
    val_f1 = float(f1_score(val["true_label"], val["pred_class"], average="macro", labels=[0, 1, 2], zero_division=0))
    run_dir = os.path.join(a.out, "runs", run_id)
    pred_dir = os.path.join(a.out, "predictions")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, f"{run_id}.parquet")
    preds.to_parquet(pred_path, index=False)

    fitted = {"train_class_freq": [round(float(x), 6) for x in train_class_freq(windows)]}
    if ohlcv is not None:
        fitted["momentum_thresholds"] = [round(x, 6) for x in momentum_thresholds(windows, ohlcv)]
    minutes = (time.time() - t0) / 60.0
    metrics = {"best_val_f1": val_f1, "best_epoch": 0, "epochs_run": 0, "prediction_paths": [pred_path],
               "checkpoint_path": None, "technical_owner": owner_of(cfg.experiment_id), "gpu_minutes": 0.0,
               **tracking.environment_info("cpu"), "pretrained": False, "history": []}
    record = tracking.write_run_json(run_dir, run_id, cfg, "complete", metrics, {"wall_minutes": round(minutes, 3)},
                                     extra={"prediction_splits": list(splits), "validation_status": "pending",
                                            "max_steps": None, "b0_fitted_on_train": fitted, **(frozen_info or {})})
    tracking.append_index_csv(os.path.join(a.out, "runs", "index.csv"), record)
    print(f"[{run_id}] done | val_macroF1={val_f1:.3f} | rows={len(preds)} | pred={pred_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
