"""Rendering registry render_v1 and PNG shards (Execution Plan 10.2).

``render_registry.parquet`` has one row per (window, render_id) for every
train/val/test window (purged windows are never rendered):

window_id, render_id, split, label, ohlcv_hash (copied from data_v1),
render_role (reference / seen / held_out), image_path (inside the shard root,
``{split}_{render_id}/{window_id}.png``), image_hash (sha256 of the decoded
uint8 array), render_version.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .rasterizer import image_hash, render
from .styles import RENDER_IDS, Style

RENDER_COLUMNS = ["window_id", "render_id", "split", "label", "ohlcv_hash", "render_role",
                  "image_path", "image_hash", "render_version"]


def image_relpath(split: str, render_id: str, window_id: str) -> str:
    return f"{split}_{render_id}/{window_id}.png"


def shard_names(splits=("train", "val", "test")) -> list[str]:
    return [f"{s}_{r}" for s in splits for r in RENDER_IDS]


def write_png(img: np.ndarray, path: str) -> None:
    tmp = path + ".tmp"
    Image.fromarray(img, mode="RGB").save(tmp, format="PNG", compress_level=6)
    os.replace(tmp, path)


def read_png(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


# ----------------------------------------------------------------- worker (spawn-safe)
_W: dict = {}


def _init_worker(ohlcv_path: str, styles: dict[str, Style], shard_root: str, verify: bool):
    _W["ohlcv"] = np.load(ohlcv_path, mmap_mode="r")
    _W["styles"] = styles
    _W["root"] = shard_root
    _W["verify"] = verify


def render_chunk(rows: list[tuple[int, str, str]]) -> list[tuple[str, str, str, str]]:
    """rows: (row_idx, window_id, split) -> [(window_id, render_id, image_path, image_hash)]."""
    out = []
    for row_idx, wid, split in rows:
        win = np.asarray(_W["ohlcv"][row_idx])
        for rid in RENDER_IDS:
            img = render(win, _W["styles"][rid])
            h = image_hash(img)
            rel = image_relpath(split, rid, wid)
            path = os.path.join(_W["root"], rel)
            write_png(img, path)
            if _W["verify"] and image_hash(read_png(path)) != h:
                raise RuntimeError(f"PNG round-trip mismatch: {rel}")
            out.append((wid, rid, rel, h))
    return out
