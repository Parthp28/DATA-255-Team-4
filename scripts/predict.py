"""Seed-0 promotion: test-split predictions for a run whose configuration is now frozen. Plan owner: Kaushik.

    python scripts/predict.py --frozen configs/frozen/E1.yaml --data-root data/render_v1
    python scripts/predict.py --frozen configs/frozen/E3.yaml --data-root data/render_v1 --run-dir outputs/runs/<run_id>

Plan 4 / 14.4: the selected validation run (seed 0) becomes the final seed-0 run. This adds its test-split
predictions (every window x R0-R6, or every window for E5) from its own best.pt, but only if
  * the frozen config verifies (configs/frozen/, freeze_record intact), and
  * the run trained exactly that scientific configuration (same science hash; seed, phase label and
    frozen flag aside), finished, is not rejected/superseded, and is not a --max-steps debug run.
The run directory defaults to outputs/runs/<freeze_record.source_run_id>. run.json gains the frozen
provenance, status becomes "promoted", and validation_status resets to "pending" so the technical owner
re-validates the updated prediction file (scripts/validate_run.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.common.config import ExperimentConfig  # noqa: E402
from src.common.schema import RENDER_IDS, validate_prediction_frame  # noqa: E402
from src.evaluation.frozen import FrozenConfigError, frozen_config_info, run_matches_frozen, verify_frozen  # noqa: E402
from src.training.datasets import ChartImageDataset, NumericWindowDataset  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402

import train as train_script  # noqa: E402


def promote(frozen, data_root, run_dir=None, runs_root="outputs/runs", device="cpu", test_mode=False) -> dict:
    rec_f = verify_frozen(frozen)
    run_dir = Path(run_dir) if run_dir else Path(runs_root) / str(rec_f.get("source_run_id"))
    rec = json.loads((run_dir / "run.json").read_text())
    info = frozen_config_info(frozen)
    if not run_matches_frozen({**rec, **info}):
        raise FrozenConfigError(f"{rec['run_id']} was not trained with the configuration frozen in {frozen}")
    if rec.get("status") not in ("complete", "promoted") or rec.get("validation_status") in ("rejected", "superseded"):
        raise FrozenConfigError(f"{rec['run_id']}: status {rec.get('status')}/{rec.get('validation_status')} cannot be promoted")
    if rec.get("max_steps") and not test_mode:
        raise FrozenConfigError(f"{rec['run_id']} is a --max-steps debug run; debug runs are never promoted")
    cfg = ExperimentConfig.from_yaml(str(frozen))
    torch.manual_seed(0)
    trainer = Trainer(train_script.build_model(cfg, pretrained=False), cfg, torch.device(device), str(run_dir))
    ckpt = rec["checkpoint_path"]
    trainer.load_checkpoint(ckpt)                                    # weights come from the run's own best.pt
    windows = pd.read_parquet(os.path.join(data_root, "windows.parquet"))
    if cfg.is_numeric:
        ohlcv = np.load(os.path.join(data_root, "ohlcv.npy"))
        test = trainer.predict(NumericWindowDataset(windows, ohlcv, "test"), rec["run_id"], "test", ckpt)
        validate_prediction_frame(test, image_model=False)
    else:
        reg = pd.read_parquet(os.path.join(data_root, "render_registry.parquet"))
        ds = ChartImageDataset(reg, os.path.join(data_root, "shards"), "test", RENDER_IDS, cfg.image_size)
        test = trainer.predict(ds, rec["run_id"], "test", ckpt)
        validate_prediction_frame(test, expected_pairs=ds.pairs(), image_model=True)
    pred_path = rec["prediction_paths"][0]
    old = pd.read_parquet(pred_path)
    preds = pd.concat([old[old["split"] != "test"], test], ignore_index=True)
    preds.to_parquet(pred_path)
    was = rec.get("validation_status")
    rec.update(info, prediction_splits=sorted(set(preds["split"])), status="promoted", validation_status="pending",
               promoted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    rec.pop("validation", None)
    if was == "RESULT_VALIDATED":
        rec["validation_note"] = "reset to pending after test inference; re-validate the updated prediction file"
    (run_dir / "run.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frozen", required=True, help="configs/frozen/<experiment>.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run-dir", help="default: outputs/runs/<freeze_record.source_run_id>")
    ap.add_argument("--runs-root", default="outputs/runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--test-mode", action="store_true", help="unit tests only: allow --max-steps runs")
    a = ap.parse_args(argv)
    rec = promote(a.frozen, a.data_root, a.run_dir, a.runs_root, a.device, a.test_mode)
    print(f"{rec['run_id']}: promoted with test predictions -> {rec['prediction_paths'][0]} (validation_status pending)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
