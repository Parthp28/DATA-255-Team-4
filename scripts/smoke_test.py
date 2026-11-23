"""G3 smoke test for one experiment config (P00: ResNet-18, H00: DeiT-S). Owners: Kaushik (P00), Khushi (H00).

    python scripts/smoke_test.py --config experiments/E1.yaml --data-root data/render_v1      # P00 (lane P)
    python scripts/smoke_test.py --config experiments/E2.yaml --data-root data/render_v1      # H00 (lane H)

Gate G3 asks the GPU smoke jobs to pass on real data: one-batch overfit, resume, prediction export (plan
16.1, T-T5, T-T6). This script runs them through the shared stack (scripts/train.build_model, Trainer,
datasets) on a small subset and writes outputs/smoke/<experiment>/smoke_report.json:
  1. finite_step       one training step: finite loss, finite and non-zero gradients
  2. overfit           one batch of 32 images (paired: 16 windows x 2) reaches 100% train accuracy within
                       --overfit-steps (default 200) at the config lr, constant schedule
  3. resume            an interrupted run resumed from last.pt reproduces the uninterrupted run's losses
                       (rtol 1e-3, the AMP tolerance) and final weights
  4. prediction_export best.pt predicts every subset window x R0-R6 (or NUM) and passes the team validator
plus the compute environment read from the runtime (GPU model, CUDA/cuDNN, torch, timm), AMP, and peak GPU
memory. Exit code 0 only if every check passes. Smoke runs are never reported or promoted.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src import tracking  # noqa: E402
from src.common.config import ExperimentConfig  # noqa: E402
from src.common.schema import RENDER_IDS, validate_prediction_frame  # noqa: E402
from src.training.datasets import ChartImageDataset, NumericWindowDataset, PairedChartDataset  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402

import train as train_script  # noqa: E402


class Data:
    """A seeded subset of the registry: `n_train` train windows and `n_eval` validation windows."""

    def __init__(self, cfg, root, n_train, n_eval, seed=0):
        self.cfg, self.root = cfg, root
        self.windows = pd.read_parquet(os.path.join(root, "windows.parquet"))
        rng = np.random.default_rng(seed)
        pick = lambda s, n: sorted(rng.choice(sorted(self.windows.loc[self.windows["split"] == s, "window_id"]),  # noqa: E731
                                              min(n, int((self.windows["split"] == s).sum())), replace=False))
        self.train_ids, self.val_ids = pick("train", n_train), pick("val", n_eval)
        if cfg.is_numeric:
            self.ohlcv = np.load(os.path.join(root, "ohlcv.npy"))
        else:
            reg = pd.read_parquet(os.path.join(root, "render_registry.parquet"))
            self.reg = reg[reg["window_id"].isin(set(self.train_ids) | set(self.val_ids))]
            self.shards = os.path.join(root, "shards")

    def _w(self, ids):
        return self.windows[self.windows["window_id"].isin(set(ids))]

    def train_ds(self):
        c = self.cfg
        if c.is_numeric:
            return NumericWindowDataset(self._w(self.train_ids), self.ohlcv, "train")
        if c.pairing == "pair_distinct":
            return PairedChartDataset(self.reg, self.shards, "train", c.train_renders, c.image_size, seed=c.seed)
        return ChartImageDataset(self.reg, self.shards, "train", c.train_renders, c.image_size)

    def eval_ds(self, renders):
        if self.cfg.is_numeric:
            return NumericWindowDataset(self._w(self.val_ids), self.ohlcv, "val")
        return ChartImageDataset(self.reg, self.shards, "val", renders, self.cfg.image_size)

    def own_renders(self):
        return ["NUM"] if self.cfg.is_numeric else list(self.cfg.train_renders)


def new_trainer(cfg, device, pretrained, out):
    torch.manual_seed(0)
    np.random.seed(0)
    return Trainer(train_script.build_model(cfg, pretrained), cfg, device, str(out))


def check_finite_step(cfg, data, device, pretrained, out):
    t = new_trainer(cfg, device, pretrained, out)
    batch = next(iter(DataLoader(data.train_ds(), batch_size=8, shuffle=False)))
    t.model.train()
    with torch.amp.autocast("cuda", enabled=t.use_amp):
        loss, parts = t._batch_loss(batch)
    t.scaler.scale(loss).backward()
    grads = [p.grad for p in t.model.parameters() if p.requires_grad and p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    nonzero = sum(float(g.abs().sum()) for g in grads) > 0
    return {"loss": float(loss.detach()), "parts": parts, "params_with_grad": len(grads),
            "pass": bool(math.isfinite(float(loss.detach())) and finite and nonzero)}


def check_overfit(cfg, data, device, pretrained, out, steps):
    t = new_trainer(cfg, device, pretrained, out)
    bs = 16 if cfg.pairing == "pair_distinct" else 32                 # 32 images either way (T-T5)
    batch = next(iter(DataLoader(data.train_ds(), batch_size=bs, shuffle=False)))
    opt = torch.optim.AdamW(t.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    t0, reached = time.time(), None
    for i in range(steps):
        t.model.train()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=t.use_amp):
            loss, _ = t._batch_loss(batch)
        t.scaler.scale(loss).backward()
        t.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(t.model.parameters(), cfg.grad_clip)
        t.scaler.step(opt)
        t.scaler.update()
        if (i + 1) % 25 == 0 or i + 1 == steps:
            acc = train_accuracy(t, batch)
            if acc == 1.0:
                reached = i + 1
                break
    acc = train_accuracy(t, batch)
    return {"images": int(bs * (2 if cfg.pairing == "pair_distinct" else 1)), "max_steps": steps,
            "steps_to_100pct": reached, "train_accuracy": acc, "final_loss": float(loss.detach()),
            "seconds": round(time.time() - t0, 2), "pass": acc == 1.0}


@torch.no_grad()
def train_accuracy(t, batch) -> float:
    t.model.eval()
    y = batch["label"].to(t.device)
    with torch.amp.autocast("cuda", enabled=t.use_amp):
        if t.cfg.pairing == "pair_distinct":
            preds = [t.model(batch[k].to(t.device)).argmax(1) for k in ("image_a", "image_b")]
            return float(torch.cat([(p == y) for p in preds]).float().mean())
        return float((t.model(batch[t.input_key].to(t.device)).argmax(1) == y).float().mean())


def check_resume(cfg, data, device, pretrained, out):
    small = dataclasses.replace(cfg, epochs_max=2, patience=2)
    bs = small.windows_per_step if small.pairing == "pair_distinct" else small.images_per_step
    bs = max(2, min(bs, len(data.train_ids) // 2))

    def run(d, stop=None, resume=True):
        t = new_trainer(small, device, pretrained, d)
        loader = DataLoader(data.train_ds(), batch_size=bs, shuffle=True)
        t.fit(loader, data.eval_ds(data.own_renders()), data.own_renders(), str(d / "best.pt"),
              resume=resume, stop_after_epochs=stop)
        return t

    det = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    try:
        full = run(out / "full", resume=False)
        first = run(out / "cut", stop=1)
        resumed = run(out / "cut")
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = det
    n = len(first.step_losses)
    ref, got = np.array(full.step_losses[n:]), np.array(resumed.step_losses)
    same_losses = ref.shape == got.shape and bool(np.allclose(got, ref, rtol=1e-3, atol=1e-4))
    weights = all(torch.allclose(a.float(), b.float(), rtol=1e-3, atol=1e-4)
                  for a, b in zip(full.model.state_dict().values(), resumed.model.state_dict().values()))
    return {"steps_epoch0": n, "steps_resumed": int(got.size), "max_abs_loss_diff": float(np.abs(got - ref).max()) if
            ref.shape == got.shape and got.size else None, "weights_match": weights,
            "last_pt": str(out / "cut" / "last.pt"), "pass": same_losses and weights and (out / "cut" / "last.pt").exists()}, resumed


def check_export(cfg, data, trainer, out):
    ckpt = str(out / "cut" / "best.pt")
    trainer.load_checkpoint(ckpt)
    ds = data.eval_ds(["NUM"] if cfg.is_numeric else RENDER_IDS)
    preds = trainer.predict(ds, f"SMOKE-{cfg.experiment_id}", "val", ckpt)
    try:
        if cfg.is_numeric:
            validate_prediction_frame(preds, image_model=False)
        else:
            validate_prediction_frame(preds, expected_pairs=ds.pairs(), image_model=True)
        ok = len(preds) == len(data.val_ids) * (1 if cfg.is_numeric else len(RENDER_IDS))
        err = ""
    except ValueError as exc:
        ok, err = False, str(exc)
    path = out / "smoke_predictions.parquet"
    preds.to_parquet(path, index=False)
    return {"rows": int(len(preds)), "path": str(path), "error": err, "pass": ok}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="tunable fields, as in train.py")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    ap.add_argument("--overfit-steps", type=int, default=200)
    ap.add_argument("--train-windows", type=int, default=64)
    ap.add_argument("--eval-windows", type=int, default=16)
    a = ap.parse_args(argv)
    cfg, prov = train_script.apply_overrides(ExperimentConfig.from_yaml(a.config), a.set, None, False)
    if cfg.weight_decay is None:
        raise SystemExit(f"{cfg.experiment_id}: weight_decay is not set (open pre-E5 decision); pass --set for a smoke")
    device = torch.device(a.device)
    out = Path(a.out or ROOT / "outputs" / "smoke" / cfg.experiment_id)
    out.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    data = Data(cfg, a.data_root, a.train_windows, a.eval_windows)
    t0 = time.time()
    report = {"experiment_id": cfg.experiment_id, "model": cfg.model, "config": str(a.config).replace("\\", "/"),
              "config_sha256": cfg.sha256(), "pretrained": bool(a.pretrained) and not cfg.is_numeric,
              "amp": bool(cfg.amp) and device.type == "cuda", **tracking.environment_info(device), **prov,
              "subset": {"train_windows": len(data.train_ids), "eval_windows": len(data.val_ids)}, "checks": {}}
    report["checks"]["finite_step"] = check_finite_step(cfg, data, device, a.pretrained, out / "finite")
    report["checks"]["overfit"] = check_overfit(cfg, data, device, a.pretrained, out / "overfit", a.overfit_steps)
    report["checks"]["resume"], resumed = check_resume(cfg, data, device, a.pretrained, out)
    report["checks"]["prediction_export"] = check_export(cfg, data, resumed, out)
    if device.type == "cuda":
        report["peak_gpu_memory_gib"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3)
    report["seconds"] = round(time.time() - t0, 1)
    report["pass"] = all(c["pass"] for c in report["checks"].values())
    (out / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    for name, c in report["checks"].items():
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {name}")
    print(f"SMOKE {'PASS' if report['pass'] else 'FAIL'} ({report['gpu_model']}) -> {out / 'smoke_report.json'}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
