"""Shared loading for the interpretability scripts: a run's image model and the D15 sample images with their region
masks.

Plan D15: interpretability uses the seed-0 checkpoints of the frozen configurations of E1, E2, E3b, E4b, E3 and E4.
Any other run is refused outside --test-mode. Every image is read from the PNG shards the model was evaluated on.
Its decoded hash must equal the registry's image_hash and a fresh render from ohlcv.npy, so the region masks taken
from the draw list line up with the pixels exactly.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..common.config import ExperimentConfig
from ..evaluation.frozen import run_matches_frozen
from ..evaluation.gate import has_image_backbone
from ..rendering.rasterizer import image_hash, render
from ..rendering.styles import load_styles
from .regions import region_masks

INTERPRETABILITY_EXPERIMENTS = ("E1", "E2", "E3b", "E4b", "E3", "E4")


class InterpretabilityRefused(RuntimeError):
    pass


def build_image_model(cfg: ExperimentConfig) -> torch.nn.Module:
    if cfg.model.startswith("deit"):
        from ..models.vit import build_deit_small
        return build_deit_small(pretrained=False)
    from ..models.resnet import build_resnet
    return build_resnet(cfg.model, pretrained=False)


def method_for(cfg: ExperimentConfig) -> str:
    return "rollout" if cfg.model.startswith("deit") else "gradcam"


def load_run(run_dir, device, test_mode: bool = False):
    """(run record, config, model in eval mode with the run's best.pt) for an interpretability run."""
    rec = json.loads((Path(run_dir) / "run.json").read_text())
    rid = rec.get("run_id", "?")
    if rec.get("experiment_id") not in INTERPRETABILITY_EXPERIMENTS or not has_image_backbone(rec):
        raise InterpretabilityRefused(f"{rid}: interpretability covers {INTERPRETABILITY_EXPERIMENTS} only (D15)")
    if not test_mode:
        if int(rec.get("seed", -1)) != 0:
            raise InterpretabilityRefused(f"{rid}: D15 uses seed-0 checkpoints")
        if rec.get("status") not in ("complete", "promoted") or not run_matches_frozen(rec):
            raise InterpretabilityRefused(f"{rid}: not a complete run of its frozen configuration")
    fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
    cfg = ExperimentConfig(**{k: rec[k] for k in fields if k in rec})
    model = build_image_model(cfg)
    ckpt = torch.load(rec["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return rec, cfg, model.to(device).eval()


def sample_items(data_root, window_ids, render_ids, styles_path=None):
    """Yield (window_id, render_id, uint8 image, region masks, style), each image hash-checked."""
    root = Path(data_root)
    reg = pd.read_parquet(root / "render_registry.parquet")
    windows = pd.read_parquet(root / "windows.parquet").set_index("window_id")
    ohlcv = np.load(root / "ohlcv.npy", mmap_mode="r")
    styles = load_styles(styles_path)[0] if styles_path else load_styles()[0]
    want = reg[reg["window_id"].isin(set(window_ids)) & reg["render_id"].isin(render_ids)]
    missing = len(window_ids) * len(render_ids) - len(want)
    if missing:
        raise InterpretabilityRefused(f"{missing} sampled (window, rendering) images are not in the registry")
    for r in want.sort_values(["render_id", "window_id"]).itertuples(index=False):
        img = np.asarray(Image.open(os.path.join(root, "shards", r.image_path)).convert("RGB"), dtype=np.uint8)
        win = np.asarray(ohlcv[int(windows.at[r.window_id, "row_idx"])])
        style = styles[r.render_id]
        h = image_hash(img)
        if h != r.image_hash or image_hash(render(win, style)) != h:
            raise InterpretabilityRefused(f"{r.window_id} {r.render_id}: image does not match the registry "
                                          f"and a fresh render; region masks would not line up")
        yield r.window_id, r.render_id, img, region_masks(win, style), style


def sample_sha256(window_ids) -> str:
    return hashlib.sha256("\n".join(sorted(window_ids)).encode()).hexdigest()
