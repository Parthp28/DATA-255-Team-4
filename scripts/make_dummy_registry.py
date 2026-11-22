"""Generate a synthetic dummy registry (windows + render registry + PNG shards).

Mirrors the frozen schema (10.1-10.2) so the trainer stack is fully runnable with
no network, no GPU, and before Farzaan's real data_v1/render_v1 exist. This is the
"develop against the dummy registry" path in the plan (Kaushik, Inputs row 9.2).

Labels and prices here are RANDOM -- this proves the pipeline, not any finance.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.schema import (  # noqa: E402
    RENDER_IDS, SEEN_RENDERS, WINDOW_COLUMNS, RENDER_COLUMNS,
)

_BG = {"R0": (255, 255, 255), "R1": (225, 232, 240), "R2": (255, 255, 255),
       "R3": (248, 248, 248), "R4": (255, 255, 255), "R5": (255, 255, 255),
       "R6": (250, 250, 250)}


def _render_png(window: np.ndarray, render_id: str, path: str, size: int = 224):
    img = Image.new("RGB", (size, size), _BG[render_id])
    d = ImageDraw.Draw(img)
    if render_id != "R2":  # R2 == grid removed
        for gx in range(0, size, size // 6):
            d.line([(gx, 0), (gx, size)], fill=(230, 230, 230))
    closes = window[:, 3]
    lo, hi = closes.min(), closes.max()
    norm = (closes - lo) / max(hi - lo, 1e-8)
    up_color = (30, 160, 60) if render_id != "R4" else (40, 90, 200)   # R4 palette swap
    pts = [(int(i * size / len(norm)), int(size - 10 - norm[i] * (size - 20)))
           for i in range(len(norm))]
    d.line(pts, fill=up_color, width=(3 if render_id == "R5" else 1))  # R5 candle width
    img.save(path)
    return hashlib.sha256(np.asarray(img).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dummy")
    ap.add_argument("--n", type=int, default=36, help="total windows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.n
    # random-walk OHLCV windows (N, 30, 5): O,H,L,C,V
    ohlcv = np.zeros((n, 30, 5), dtype=np.float32)
    for i in range(n):
        steps = rng.normal(0, 1, size=30).cumsum() + 100
        o = steps + rng.normal(0, 0.5, 30)
        c = steps + rng.normal(0, 0.5, 30)
        hi = np.maximum(o, c) + rng.uniform(0, 1, 30)
        lo = np.minimum(o, c) - rng.uniform(0, 1, 30)
        v = rng.uniform(1e5, 1e6, 30)
        ohlcv[i] = np.stack([o, hi, lo, c, v], axis=1)

    splits = np.array((["train"] * int(n * 0.6)) + (["val"] * int(n * 0.2)) +
                      (["test"] * (n - int(n * 0.6) - int(n * 0.2))))
    labels = rng.integers(0, 3, size=n)

    shard_root = os.path.join(args.out, "shards")
    win_rows, reg_rows = [], []
    for i in range(n):
        split = splits[i]
        wid = f"TKR{i:03d}_2020-01-{(i % 28) + 1:02d}"
        ohash = hashlib.sha256(ohlcv[i].tobytes()).hexdigest()
        win_rows.append({
            "window_id": wid, "row_idx": i, "ticker": f"TKR{i % 5}",
            "asset_type": "stock" if i % 3 else "etf",
            "input_start_date": "2020-01-01", "anchor_date": "2020-02-10",
            "label_end_date": "2020-02-15", "split": split,
            "fwd_return": float(rng.normal()), "sigma_30": float(abs(rng.normal())),
            "label_score": float(rng.normal()), "label": int(labels[i]),
            "ohlcv_hash": ohash, "data_version": "data_v1",
        })
        for rid in RENDER_IDS:
            folder = os.path.join(shard_root, f"{split}_{rid}")
            os.makedirs(folder, exist_ok=True)
            rel = f"{split}_{rid}/{wid}.png"
            ihash = _render_png(ohlcv[i], rid, os.path.join(shard_root, rel))
            reg_rows.append({
                "window_id": wid, "render_id": rid, "split": split,
                "label": int(labels[i]), "ohlcv_hash": ohash,
                "render_role": "seen" if rid in SEEN_RENDERS else "held_out",
                "image_path": rel, "image_hash": ihash, "render_version": "render_v1",
            })

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(win_rows)[WINDOW_COLUMNS].to_parquet(os.path.join(args.out, "windows.parquet"))
    pd.DataFrame(reg_rows)[RENDER_COLUMNS].to_parquet(os.path.join(args.out, "render_registry.parquet"))
    np.save(os.path.join(args.out, "ohlcv.npy"), ohlcv)
    print(f"dummy registry written to {args.out}: {n} windows x {len(RENDER_IDS)} renders")


if __name__ == "__main__":
    main()
