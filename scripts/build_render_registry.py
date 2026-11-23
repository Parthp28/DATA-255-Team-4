"""Render every train/val/test window in R0-R6 and write render_v1 (G2; owner Farzaan).

    python scripts/build_render_registry.py --styles configs/render_styles.yaml \
        [--data-dir data] [--workers 8] [--tar]

Reads data/data_v1/{windows.parquet, ohlcv.npy}; writes

    data/render_v1/render_registry.parquet   (Execution Plan 10.2)
    data/render_v1/region_geometry.json      boxes, grid rows/cols per render_id
    data/render_v1/shards/{split}_{rid}/{window_id}.png   (21 shard folders)
    data/render_v1/windows.parquet, ohlcv.npy  copies, so data/render_v1 is a complete
                                              --data-root for scripts/train.py
    data/render_v1/figure1_sheet.png          one window per row in R0..R6, for review
    data/render_v1/render_manifest.json       hashes, counts, git commit
    data/render_v1_tar/{split}_{rid}.tar + SHA256SUMS   (with --tar; for Drive/Kaggle)

Every PNG is decoded right after writing and its hash compared with the in-memory render.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.build_dataset import git_state  # noqa: E402
from src.data.download import sha256_file  # noqa: E402
from src.rendering import registry as rg  # noqa: E402
from src.rendering.rasterizer import render  # noqa: E402
from src.rendering.regions import region_geometry  # noqa: E402
from src.rendering.styles import RENDER_IDS, load_styles  # noqa: E402

CODE_FILES = ["src/rendering/rasterizer.py", "src/rendering/styles.py", "src/rendering/registry.py"]


def figure1_sheet(windows: pd.DataFrame, ohlcv: np.ndarray, styles, path: str, n: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    pick = []
    for s in ("train", "val", "test"):
        idx = windows.index[windows["split"] == s].to_numpy()
        pick += list(rng.choice(idx, size=max(1, n // 3 + (s == "train")), replace=False))
    pad = 6
    c = 224
    sheet = np.full((len(pick) * (c + pad) + pad, len(RENDER_IDS) * (c + pad) + pad, 3), 90, np.uint8)
    for r, i in enumerate(pick):
        win = np.asarray(ohlcv[int(windows.loc[i, "row_idx"])])
        for k, rid in enumerate(RENDER_IDS):
            y, x = pad + r * (c + pad), pad + k * (c + pad)
            sheet[y:y + c, x:x + c] = render(win, styles[rid])
    Image.fromarray(sheet).save(path)
    return [windows.loc[i, "window_id"] for i in pick]


def make_tars(shard_root: str, tar_dir: str) -> dict[str, str]:
    os.makedirs(tar_dir, exist_ok=True)
    sums = {}
    for name in rg.shard_names():
        path = os.path.join(tar_dir, f"{name}.tar")
        with tarfile.open(path + ".tmp", "w", format=tarfile.PAX_FORMAT) as tf:
            folder = os.path.join(shard_root, name)
            tf.add(folder, arcname=name, recursive=False)
            for fn in sorted(os.listdir(folder)):
                tf.add(os.path.join(folder, fn), arcname=f"{name}/{fn}")
        os.replace(path + ".tmp", path)
        sums[f"{name}.tar"] = sha256_file(path)
    with open(os.path.join(tar_dir, "SHA256SUMS"), "w") as fh:
        for k, v in sums.items():
            fh.write(f"{v}  {k}\n")
    return sums


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--styles", default="configs/render_styles.yaml")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--data-version", default="data_v1")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunk", type=int, default=64, help="windows per task")
    ap.add_argument("--limit", type=int, default=None, help="render only the first N windows (debug)")
    ap.add_argument("--no-verify", action="store_true", help="skip PNG read-back hash check")
    ap.add_argument("--tar", action="store_true", help="also pack each shard into a .tar for upload")
    args = ap.parse_args()

    styles, render_version, styles_sha = load_styles(args.styles)
    data_dir = os.path.join(args.data_dir, args.data_version)
    out_dir = os.path.join(args.data_dir, render_version)
    shard_root = os.path.join(out_dir, "shards")
    windows = pd.read_parquet(os.path.join(data_dir, "windows.parquet"))
    ohlcv_path = os.path.join(data_dir, "ohlcv.npy")
    ohlcv = np.load(ohlcv_path, mmap_mode="r")

    todo = windows[windows["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    if args.limit:
        todo = todo.iloc[:args.limit]
    for name in rg.shard_names():
        os.makedirs(os.path.join(shard_root, name), exist_ok=True)

    rows = list(zip(todo["row_idx"].astype(int), todo["window_id"], todo["split"]))
    chunks = [rows[i:i + args.chunk] for i in range(0, len(rows), args.chunk)]
    t0 = time.time()
    results = []
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=rg._init_worker,
                  initargs=(ohlcv_path, styles, shard_root, not args.no_verify)) as pool:
        for k, part in enumerate(pool.imap(rg.render_chunk, chunks), 1):
            results.extend(part)
            if k % 50 == 0 or k == len(chunks):
                done = len(results) // len(RENDER_IDS)
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(rows)} windows  ({rate:.0f} win/s)", flush=True)

    res = pd.DataFrame(results, columns=["window_id", "render_id", "image_path", "image_hash"])
    meta = todo.set_index("window_id")[["split", "label", "ohlcv_hash"]]
    reg = res.join(meta, on="window_id")
    reg["render_role"] = reg["render_id"].map({rid: styles[rid].role for rid in RENDER_IDS})
    reg["render_version"] = render_version
    reg["label"] = reg["label"].astype(np.int64)
    order = {rid: i for i, rid in enumerate(RENDER_IDS)}
    reg = reg.assign(_w=reg["window_id"].map({w: i for i, w in enumerate(todo["window_id"])}),
                     _r=reg["render_id"].map(order)).sort_values(["_w", "_r"])
    reg = reg[rg.RENDER_COLUMNS].reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    reg_path = os.path.join(out_dir, "render_registry.parquet")
    reg.to_parquet(reg_path, index=False)
    with open(os.path.join(out_dir, "region_geometry.json"), "w") as fh:
        json.dump({rid: region_geometry(styles[rid]) for rid in RENDER_IDS}, fh, indent=2)
    shutil.copyfile(os.path.join(data_dir, "windows.parquet"), os.path.join(out_dir, "windows.parquet"))
    shutil.copyfile(ohlcv_path, os.path.join(out_dir, "ohlcv.npy"))
    sheet_ids = figure1_sheet(windows, ohlcv, styles, os.path.join(out_dir, "figure1_sheet.png"))

    tar_sums = make_tars(shard_root, os.path.join(args.data_dir, f"{render_version}_tar")) if args.tar else None
    code_sha = hashlib.sha256(b"".join(open(os.path.join(ROOT, f), "rb").read() for f in CODE_FILES)).hexdigest()
    manifest = {
        "render_version": render_version,
        "data_version": args.data_version,
        "styles_path": os.path.relpath(os.path.abspath(args.styles), ROOT),
        "styles_sha256": styles_sha,
        "renderer_code_sha256": code_sha,
        "windows_parquet_sha256": sha256_file(os.path.join(data_dir, "windows.parquet")),
        "render_registry_sha256": sha256_file(reg_path),
        "n_windows": int(len(todo)),
        "n_images": int(len(reg)),
        "images_per_shard": {k: int(v) for k, v in
                             reg.assign(s=reg["image_path"].str.split("/").str[0])["s"].value_counts().sort_index().items()},
        "png_roundtrip_verified": not args.no_verify,
        "figure1_windows": sheet_ids,
        "tar_sha256": tar_sums,
        "seconds": round(time.time() - t0, 1),
        **git_state(),
    }
    with open(os.path.join(out_dir, "render_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"render_v1: {len(reg)} images for {len(todo)} windows in {manifest['seconds']} s -> {out_dir}")


if __name__ == "__main__":
    main()
