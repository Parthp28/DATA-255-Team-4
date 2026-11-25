"""Mini render_v1 for the interpretability tests, drawn with the production renderer (src.rendering) so the
region masks line up with the PNG pixels. Random-walk prices; not data, not results."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.schema import RENDER_IDS  # noqa: E402
from src.data.labels import sigma_input  # noqa: E402
from src.rendering.rasterizer import image_hash, render  # noqa: E402
from src.rendering.styles import load_styles  # noqa: E402


def random_windows(n: int, seed: int = 0, trend: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    drift = np.zeros(n) if trend is None else trend
    steps = rng.normal(0, 0.01, (n, 30)) + drift[:, None]
    c = 100 * np.exp(np.cumsum(steps, axis=1))
    o = c * np.exp(rng.normal(0, 0.004, (n, 30)))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, (n, 30))))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, (n, 30))))
    v = rng.lognormal(12, 0.4, (n, 30))
    return np.stack([o, h, lo, c, v], -1).astype(np.float32)


def mini_render_root(tmp, n: int = 24, seed: int = 0, labels=None, ohlcv=None, renders=RENDER_IDS) -> Path:
    styles = load_styles()[0]
    x = random_windows(n, seed) if ohlcv is None else ohlcv
    n = len(x)
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n) if labels is None else np.asarray(labels)
    split = np.where(np.arange(n) < n // 3, "train", "test")
    root = Path(tmp) / "render_v1"
    rows = []
    for i in range(n):
        wid = f"T{i:03d}_2021-01-01"
        for rid in renders:
            img = render(x[i], styles[rid])
            rel = f"{split[i]}_{rid}/{wid}.png"
            (root / "shards" / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(root / "shards" / rel)
            rows.append({"window_id": wid, "render_id": rid, "split": split[i], "label": int(y[i]),
                         "ohlcv_hash": hashlib.sha256(x[i].tobytes()).hexdigest(), "render_role": styles[rid].role,
                         "image_path": rel, "image_hash": image_hash(img), "render_version": "render_v1"})
    pd.DataFrame(rows).to_parquet(root / "render_registry.parquet", index=False)
    w = pd.DataFrame({"window_id": [f"T{i:03d}_2021-01-01" for i in range(n)], "row_idx": np.arange(n),
                      "split": split, "label": y, "sigma_30": sigma_input(x[:, :, 3].astype(np.float64)),
                      "data_version": "data_v1"})
    w.to_parquet(root / "windows.parquet", index=False)
    np.save(root / "ohlcv.npy", x)
    return root


def fake_run(tmp, experiment_id: str, model: torch.nn.Module, model_name: str, seed: int = 0) -> Path:
    """A run directory with run.json and best.pt for an untrained model (tests only, test mode)."""
    rid = f"{experiment_id}_s{seed}_test0000"
    d = Path(tmp) / "runs" / rid
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "best_f1": 0.0}, d / "best.pt")
    paired = experiment_id in ("E3b", "E3", "E4b", "E4")
    rec = {"run_id": rid, "experiment_id": experiment_id, "model": model_name, "seed": seed, "status": "complete",
           "train_renders": ["R0", "R1", "R2", "R4"] if paired else ["R0"],
           "pairing": "pair_distinct" if paired else "none",
           "loss": {"E3": "ce_plus_js", "E4": "ce_plus_js"}.get(experiment_id, "paired_ce" if paired else "ce"),
           "consistency_lambda": 1.0 if experiment_id in ("E3", "E4") else 0.0,
           "checkpoint_path": str(d / "best.pt")}
    (d / "run.json").write_text(json.dumps(rec))
    return d
