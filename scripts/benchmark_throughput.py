"""Training / inference throughput and memory benchmark (P00, H00; plan assumption A1). Owner: Kaushik.

    python scripts/benchmark_throughput.py --config experiments/E1.yaml --data-root data/render_v1          # P00
    python scripts/benchmark_throughput.py --config experiments/E2.yaml --data-root data/render_v1 --probe-batch  # H00

Builds the experiment's model through scripts/train.build_model (weights are not loaded: they do not change
speed) and times full training steps on random 224 x 224 batches (forward + backward + AdamW, fp16 AMP and
grad clipping on CUDA), inference, and, with --data-root, the PNG loader (ChartImageDataset). --probe-batch
tries {128, 96, 64, 32} images per step and reports the largest that fits; if 128 does not fit, keep 128
images per step through gradient accumulation (plan 19). A1 assumes about 700 (ResNet-18) and 250 (DeiT-S)
train images/s on a T4; the queue is rescaled if the measurement is off by more than 30%. The GPU model is
read from the runtime. Writes outputs/smoke/<experiment>/throughput.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src import tracking  # noqa: E402
from src.common.config import ExperimentConfig  # noqa: E402

import train as train_script  # noqa: E402

A1_TRAIN_IMG_S = {"resnet18.a1_in1k": 700.0, "deit_small_patch16_224.fb_in1k": 250.0}


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def bench_train(model, dev, batch, steps, warmup, amp, grad_clip):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    x = torch.randn(batch, 3, 224, 224, device=dev)
    y = torch.randint(0, 3, (batch,), device=dev)
    model.train()
    t0 = None
    for i in range(warmup + steps):
        if i == warmup:
            _sync(dev)
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            loss = F.cross_entropy(model(x).float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
    _sync(dev)
    return batch * steps / (time.perf_counter() - t0)


@torch.no_grad()
def bench_eval(model, dev, batch, steps, amp):
    model.eval()
    x = torch.randn(batch, 3, 224, 224, device=dev)
    with torch.amp.autocast("cuda", enabled=amp):
        model(x)
        _sync(dev)
        t0 = time.perf_counter()
        for _ in range(steps):
            model(x)
    _sync(dev)
    return batch * steps / (time.perf_counter() - t0)


def bench_loader(data_root, batch, n_batches, workers):
    import os

    import pandas as pd
    from torch.utils.data import DataLoader

    from src.training.datasets import ChartImageDataset
    reg = pd.read_parquet(os.path.join(data_root, "render_registry.parquet"))
    ds = ChartImageDataset(reg, os.path.join(data_root, "shards"), "train", ["R0"], 224)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers)
    t0, n = time.perf_counter(), 0
    for i, b in enumerate(dl):
        n += len(b["label"])
        if i + 1 >= n_batches:
            break
    return n / (time.perf_counter() - t0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", help="also time the PNG loader")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--probe-batch", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = ExperimentConfig.from_yaml(a.config)
    if cfg.is_numeric:
        raise SystemExit("E5 runs on CPU; the image throughput benchmark does not apply")
    dev = torch.device(a.device)
    amp = bool(cfg.amp) and dev.type == "cuda"
    model = train_script.build_model(cfg, pretrained=False).to(dev)
    rep = {"experiment_id": cfg.experiment_id, "model": cfg.model, "amp": amp, **tracking.environment_info(dev),
           "probe": {}}
    fit = None
    for b in ([128, 96, 64, 32] if a.probe_batch else [a.batch]):
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
            ips = bench_train(model, dev, b, a.steps, a.warmup, amp, cfg.grad_clip)
            mem = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 3) if dev.type == "cuda" else None
            rep["probe"][str(b)] = {"train_img_s": round(ips, 1), "peak_mem_gib": mem}
            fit = fit or b
            if not a.probe_batch:
                break
        except torch.cuda.OutOfMemoryError:
            rep["probe"][str(b)] = {"oom": True}
    if fit is None:
        raise SystemExit("no tested batch size fits in memory")
    rep.update(images_per_step_fits=fit, grad_accumulation_needed=fit < cfg.images_per_step,
               train_img_s=rep["probe"][str(fit)]["train_img_s"],
               eval_img_s=round(bench_eval(model, dev, fit, max(5, a.steps // 2), amp), 1))
    assumed = A1_TRAIN_IMG_S.get(cfg.model)
    if assumed:
        ratio = rep["train_img_s"] / assumed
        rep.update(a1_assumed_train_img_s=assumed, ratio_to_a1=round(ratio, 3), rescale_queue=abs(ratio - 1) > 0.30)
    if a.data_root:
        rep["loader_img_s"] = round(bench_loader(a.data_root, fit, 10, a.workers), 1)
        rep["loader_bound"] = rep["loader_img_s"] < rep["train_img_s"]
    out = Path(a.out or ROOT / "outputs" / "smoke" / cfg.experiment_id / "throughput.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "probe"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
