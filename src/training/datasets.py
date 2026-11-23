"""ChartDataset + numeric dataset (Execution Plan 9.2, 10.1-10.3).

- ChartImageDataset   : one (window, render) -> normalized 224x224 image tensor.
- PairedChartDataset  : one window -> two distinct renderings (E3b/E3/E4b/E4),
                        batched at 64 windows so images_per_step stays 128.
- NumericWindowDataset: one window -> (5, 30) OHLCV tensor for E5.

No torchvision dependency: PIL -> numpy -> tensor + ImageNet normalization by hand.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..common.schema import NUMERIC_RENDER_ID
from .samplers import PairSampler

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0      # HWC in [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD           # per-channel ImageNet norm
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()  # CHW


class ChartImageDataset(Dataset):
    """Flat (window, render) index for a split. Used for E1 training (render set
    = {R0}) and for full R0-R6 inference/prediction on val/test."""

    def __init__(self, registry: pd.DataFrame, shard_root: str, split: str,
                 render_ids: list[str], image_size: int = 224):
        rows = registry[(registry["split"] == split) &
                        (registry["render_id"].isin(render_ids))]
        self.rows = rows.reset_index(drop=True)
        self.shard_root = shard_root
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def pairs(self) -> list[tuple[str, str]]:
        """(window_id, render_id) coverage list -- feeds the 10.3 validator."""
        return list(zip(self.rows["window_id"], self.rows["render_id"]))

    def __getitem__(self, i: int) -> dict:
        r = self.rows.iloc[i]
        img = _load_image(os.path.join(self.shard_root, r["image_path"]), self.image_size)
        return {
            "image": img,
            "label": int(r["label"]),
            "window_id": r["window_id"],
            "render_id": r["render_id"],
        }


class PairedChartDataset(Dataset):
    """One item == one window drawn as two DISTINCT seen renderings.

    DataLoader(batch_size=windows_per_step=64) therefore yields 128 images/step
    arranged as two aligned views (a, b) of the same 64 windows."""

    def __init__(self, registry: pd.DataFrame, shard_root: str, split: str,
                 render_ids: list[str], image_size: int = 224, seed: int = 0):
        rows = registry[(registry["split"] == split) &
                        (registry["render_id"].isin(render_ids))]
        # map window_id -> {render_id: image_path, label}
        self.by_window: dict[str, dict] = {}
        for _, r in rows.iterrows():
            w = self.by_window.setdefault(r["window_id"], {"label": int(r["label"]), "renders": {}})
            w["renders"][r["render_id"]] = r["image_path"]
        self.window_ids = sorted(self.by_window)
        self.shard_root = shard_root
        self.image_size = image_size
        self.sampler = PairSampler(render_ids, seed=seed)

    def __len__(self) -> int:
        return len(self.window_ids)

    def __getitem__(self, i: int) -> dict:
        wid = self.window_ids[i]
        w = self.by_window[wid]
        ra, rb = self.sampler.draw_pair(list(w["renders"].keys()), key=i)
        img_a = _load_image(os.path.join(self.shard_root, w["renders"][ra]), self.image_size)
        img_b = _load_image(os.path.join(self.shard_root, w["renders"][rb]), self.image_size)
        return {
            "image_a": img_a, "image_b": img_b,
            "label": w["label"], "window_id": wid,
            "render_id_a": ra, "render_id_b": rb,
        }


def numeric_input(window: np.ndarray) -> np.ndarray:
    """E5 input exactly as Execution Plan 5.1: "prices divided by Close[d] minus 1, volume divided by
    window mean then log1p; window-only scaling". window: (30, 5) O,H,L,C,V with the anchor d last.
    Returns (5, 30) float32. No statistic outside the window is used (no global fit, no later rows)."""
    w = np.asarray(window, dtype=np.float64)
    out = np.empty_like(w)
    out[:, :4] = w[:, :4] / w[-1, 3] - 1.0                 # O, H, L, C relative to Close[d]
    vmean = w[:, 4].mean()
    out[:, 4] = np.log1p(w[:, 4] / vmean) if vmean > 0 else 0.0
    return out.T.astype(np.float32)


class NumericWindowDataset(Dataset):
    """Raw OHLCV windows for E5, scaled per window by numeric_input() (plan 5.1; no global
    statistic is fit). Returns (5, 30) tensors."""

    def __init__(self, windows: pd.DataFrame, ohlcv: np.ndarray, split: str):
        self.rows = windows[windows["split"] == split].reset_index(drop=True)
        self.ohlcv = ohlcv  # (N, 30, 5) float32

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows.iloc[i]
        seq = torch.from_numpy(numeric_input(self.ohlcv[int(r["row_idx"])])).contiguous()  # (5, 30)
        return {
            "seq": seq, "label": int(r["label"]),
            "window_id": r["window_id"], "render_id": NUMERIC_RENDER_ID,
        }
