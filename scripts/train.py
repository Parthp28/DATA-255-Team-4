"""Train one frozen experiment end-to-end (Execution Plan 9.2, scripts/train.py).

Reads a frozen config + a registry, trains with the shared Trainer, exports a
prediction file covering every window x R0-R6, validates completeness, and writes
run.json + index.csv. Works identically for E1 (single), E3b/E3 (paired), and E5
(numeric).

    python scripts/train.py --config experiments/E3.yaml \
        --data-root data/dummy --out outputs --no-pretrained --max-steps 4

Tuning / lambda candidates (D11, D13) and E3b/E4b/E3/E4 inheriting the frozen E1/E2 lr and weight decay:
    python scripts/train.py --config experiments/E1.yaml --set lr=1e-4 weight_decay=0.01 --data-root data/render_v1
    python scripts/train.py --config experiments/E3.yaml --inherit configs/frozen/E1.yaml --set consistency_lambda=0.3 ...
Frozen configs (configs/frozen/) accept only --seed. An interrupted run resumes from last.pt (--no-resume to restart).
"""
from __future__ import annotations

import argparse
import dataclasses
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
from src.models import vit  # noqa: E402
from src.models.numeric import Numeric1DCNN  # noqa: E402
from src.training.datasets import (  # noqa: E402
    ChartImageDataset, PairedChartDataset, NumericWindowDataset,
)
from src.training.trainer import Trainer  # noqa: E402
from src import tracking  # noqa: E402
from src.evaluation.frozen import frozen_config_info, verify_frozen  # noqa: E402
from src.evaluation.gate import D16_GPU, is_d16_gpu  # noqa: E402
from src.evaluation.gate import owner_of  # noqa: E402

TUNABLE = ("lr", "weight_decay", "consistency_lambda")   # D11 grid (lr, wd) and D13 candidates (lambda)
INHERITED = ("lr", "weight_decay")                       # E3b/E3 <- frozen E1, E4b/E4 <- frozen E2 (plan 5.1)


def build_model(cfg: ExperimentConfig, pretrained: bool):
    if cfg.is_numeric:
        return Numeric1DCNN()
    if cfg.model.startswith("deit"):
        if cfg.model != vit.MODEL_NAME:
            raise ValueError(f"DeiT model must be {vit.MODEL_NAME} (D9), got {cfg.model}")
        return vit.build_deit_small(pretrained=pretrained)        # Khushi's wrapper (3-class head)
    return build_resnet(cfg.model, pretrained=pretrained)


def apply_overrides(cfg: ExperimentConfig, sets, inherit, frozen: bool):
    """--inherit copies lr / weight decay from a verified frozen parent; --set changes tunable fields only.
    Returns (config, provenance) and re-runs ExperimentConfig validation."""
    if frozen and (sets or inherit):
        raise SystemExit("a frozen config is final: only --seed may change (no --set / --inherit)")
    changes, prov = {}, {}
    if inherit:
        rec = verify_frozen(inherit)
        parent = ExperimentConfig.from_yaml(inherit)
        changes.update({k: getattr(parent, k) for k in INHERITED})
        prov = {"inherit_path": str(inherit).replace(os.sep, "/"), "inherit_config_sha256": rec["config_sha256"]}
    for item in sets or []:
        k, _, v = item.partition("=")
        if k not in TUNABLE:
            raise SystemExit(f"--set may only change {TUNABLE}, not {k!r}")
        changes[k] = float(v)
    if changes:
        prov["overrides"] = {k: changes[k] for k in sorted(changes)}
    return dataclasses.replace(cfg, **changes), prov


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help=f"tunable fields only: {TUNABLE}")
    ap.add_argument("--inherit", help="verified frozen parent config (configs/frozen/E1.yaml or E2.yaml)")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing last.pt and start over")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    args = ap.parse_args(argv)

    # Plan 14.4: no test-split number from a config that is not a verified frozen config
    # (configs/frozen/, scripts/freeze_config.py). Tuning / lambda / dev runs predict validation only.
    frozen_info = frozen_config_info(args.config)
    pred_splits = ("val", "test") if frozen_info else ("val",)
    cfg, provenance = apply_overrides(ExperimentConfig.from_yaml(args.config), args.set, args.inherit,
                                      bool(frozen_info))
    if cfg.weight_decay is None:
        raise SystemExit(f"{cfg.experiment_id}: weight_decay is not set. For E5 it is an open pre-E5 specification "
                         f"decision (D11 lists only lr); record it in the config before running")
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(args.device)
    if device.type == "cuda":                      # D16: reported image models on the RTX 5090 only; E5 on CPU
        name = torch.cuda.get_device_name(device)
        if cfg.is_numeric:
            raise SystemExit(f"{cfg.experiment_id}: E5 runs on CPU (D16); refusing to run it on {name}")
        if not is_d16_gpu(name):
            raise SystemExit(f"{cfg.experiment_id}: GPU {name!r} is not the D16 GPU ({D16_GPU}); refusing to train")
        torch.cuda.reset_peak_memory_stats(device)

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
        train_loader = DataLoader(train_ds, batch_size=cfg.images_per_step, shuffle=True,
                                  num_workers=args.num_workers)
        t_fit = time.time()
        res = trainer.fit(train_loader, val_eval, own_render_ids=["NUM"],
                          checkpoint_path=ckpt_path, max_steps=args.max_steps, resume=not args.no_resume)
        fit_seconds = time.time() - t_fit
        trainer.load_checkpoint(ckpt_path)
        preds = pd.concat([
            trainer.predict(NumericWindowDataset(windows, ohlcv, s), run_id, s, ckpt_path)
            for s in pred_splits], ignore_index=True)
        validate_prediction_frame(preds, image_model=False)
    else:
        reg = pd.read_parquet(os.path.join(args.data_root, "render_registry.parquet"))
        if cfg.pairing == "pair_distinct":
            train_ds = PairedChartDataset(reg, shard_root, "train", cfg.train_renders,
                                          cfg.image_size, seed=cfg.seed)
            train_loader = DataLoader(train_ds, batch_size=cfg.windows_per_step, shuffle=True,
                                      num_workers=args.num_workers)
        else:
            train_ds = ChartImageDataset(reg, shard_root, "train", cfg.train_renders, cfg.image_size)
            train_loader = DataLoader(train_ds, batch_size=cfg.images_per_step, shuffle=True,
                                      num_workers=args.num_workers)
        val_eval = ChartImageDataset(reg, shard_root, "val", cfg.train_renders, cfg.image_size)
        t_fit = time.time()
        res = trainer.fit(train_loader, val_eval, own_render_ids=cfg.train_renders,
                          checkpoint_path=ckpt_path, max_steps=args.max_steps, resume=not args.no_resume)
        fit_seconds = time.time() - t_fit
        trainer.load_checkpoint(ckpt_path)
        preds_list, expected = [], []
        for s in pred_splits:
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
    env = tracking.environment_info(device)
    metrics = {"best_val_f1": res.best_val_f1, "best_epoch": res.best_epoch,
               "epochs_run": res.epochs_run, "prediction_paths": [pred_path],
               "checkpoint_path": ckpt_path, "last_checkpoint_path": os.path.join(os.path.dirname(ckpt_path), "last.pt"),
               "technical_owner": owner_of(cfg.experiment_id),               # plan 5.1 ownership table
               "gpu_minutes": round(minutes, 3) if device.type == "cuda" else 0.0, **env,
               "pretrained": bool(args.pretrained) and not cfg.is_numeric, "history": trainer.history,
               # batch policy (D10: 128 images per step; paired = 64 windows x 2), measured speed and memory
               "physical_batch_images": cfg.images_per_step, "effective_batch_images": cfg.images_per_step,
               "grad_accumulation_steps": 1, "amp_active": trainer.use_amp, "fit_seconds": round(fit_seconds, 1),
               "train_steps": len(trainer.step_losses),
               "train_images_per_s_incl_val": round(len(trainer.step_losses) * cfg.images_per_step / fit_seconds, 1)
               if fit_seconds > 0 else None,
               "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3)
               if device.type == "cuda" else None}
    record = tracking.write_run_json(os.path.dirname(ckpt_path), run_id, cfg,
                                     "complete", metrics, {"wall_minutes": round(minutes, 3)},
                                     extra={"prediction_splits": list(pred_splits),
                                            "validation_status": "pending",   # owner signs via validate_run.py
                                            "max_steps": args.max_steps,      # set => debug run, never reported
                                            **provenance, **(frozen_info or {})})
    tracking.append_index_csv(os.path.join(args.out, "runs", "index.csv"), record)
    print(f"[{run_id}] done | val_macroF1={res.best_val_f1:.3f} | "
          f"rows={len(preds)} | pred={pred_path}")


if __name__ == "__main__":
    main()
