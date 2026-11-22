"""Train one frozen experiment end-to-end (Execution Plan 9.2, scripts/train.py).

Reads a frozen config + a registry, trains with the shared Trainer, exports a
prediction file covering every window x R0-R6, validates completeness, and writes
run.json + index.csv. Works identically for E1 (single), E3b/E3 (paired), and E5
(numeric).

    python scripts/train.py --config configs/experiments/E3.yaml \
        --data-root data/dummy --out outputs --no-pretrained --max-steps 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.config import ExperimentConfig  # noqa: E402
from src.common.schema import RENDER_IDS, validate_prediction_frame  # noqa: E402
from src.models.resnet import build_resnet  # noqa: E402
from src.models.numeric import Numeric1DCNN  # noqa: E402
from src.training.datasets import (  # noqa: E402
    ChartImageDataset, PairedChartDataset, NumericWindowDataset,
)
from src.training.trainer import Trainer  # noqa: E402
from src import tracking  # noqa: E402


def build_model(cfg: ExperimentConfig, pretrained: bool):
    if cfg.is_numeric:
        return Numeric1DCNN()
    return build_resnet(cfg.model, pretrained=pretrained)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(args.device)

    windows = pd.read_parquet(os.path.join(args.data_root, "windows.parquet"))
    shard_root = os.path.join(args.data_root, "shards")
    run_id = f"{cfg.experiment_id}_s{cfg.seed}_{cfg.sha256()[:8]}"
    ckpt_path = os.path.join(args.out, "runs", run_id, "best.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    model = build_model(cfg, pretrained=args.pretrained)
    trainer = Trainer(model, cfg, device, os.path.dirname(ckpt_path))

    t0 = time.time()
    if cfg.is_numeric:
        ohlcv = np.load(os.path.join(args.data_root, "ohlcv.npy"))
        train_ds = NumericWindowDataset(windows, ohlcv, "train")
        val_eval = NumericWindowDataset(windows, ohlcv, "val")
        train_loader = DataLoader(train_ds, batch_size=cfg.images_per_step, shuffle=True)
        res = trainer.fit(train_loader, val_eval, own_render_ids=["NUM"],
                          checkpoint_path=ckpt_path, max_steps=args.max_steps)
        trainer.load_checkpoint(ckpt_path)
        preds = pd.concat([
            trainer.predict(NumericWindowDataset(windows, ohlcv, s), run_id, s, ckpt_path)
            for s in ("val", "test")], ignore_index=True)
        validate_prediction_frame(preds, image_model=False)
    else:
        reg = pd.read_parquet(os.path.join(args.data_root, "render_registry.parquet"))
        if cfg.pairing == "pair_distinct":
            train_ds = PairedChartDataset(reg, shard_root, "train", cfg.train_renders,
                                          cfg.image_size, seed=cfg.seed)
            train_loader = DataLoader(train_ds, batch_size=cfg.windows_per_step, shuffle=True)
        else:
            train_ds = ChartImageDataset(reg, shard_root, "train", cfg.train_renders, cfg.image_size)
            train_loader = DataLoader(train_ds, batch_size=cfg.images_per_step, shuffle=True)
        val_eval = ChartImageDataset(reg, shard_root, "val", cfg.train_renders, cfg.image_size)
        res = trainer.fit(train_loader, val_eval, own_render_ids=cfg.train_renders,
                          checkpoint_path=ckpt_path, max_steps=args.max_steps)
        trainer.load_checkpoint(ckpt_path)
        preds_list, expected = [], []
        for s in ("val", "test"):
            full = ChartImageDataset(reg, shard_root, s, RENDER_IDS, cfg.image_size)
            preds_list.append(trainer.predict(full, run_id, s, ckpt_path))
            expected += full.pairs()
        preds = pd.concat(preds_list, ignore_index=True)
        validate_prediction_frame(preds, expected_pairs=expected, image_model=True)

    pred_dir = os.path.join(args.out, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, f"{run_id}.parquet")
    preds.to_parquet(pred_path)

    minutes = (time.time() - t0) / 60.0
    metrics = {"best_val_f1": res.best_val_f1, "best_epoch": res.best_epoch,
               "epochs_run": res.epochs_run, "prediction_paths": [pred_path],
               "checkpoint_path": ckpt_path, "technical_owner": "kaushik",
               "gpu_minutes": round(minutes, 3), "gpu_model": str(device)}
    record = tracking.write_run_json(os.path.dirname(ckpt_path), run_id, cfg,
                                     "complete", metrics, {"wall_minutes": round(minutes, 3)})
    tracking.append_index_csv(os.path.join(args.out, "runs", "index.csv"), record)
    print(f"[{run_id}] done | val_macroF1={res.best_val_f1:.3f} | "
          f"rows={len(preds)} | pred={pred_path}")


if __name__ == "__main__":
    main()
