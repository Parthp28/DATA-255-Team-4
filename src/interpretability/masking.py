"""Region masking with area-matched random controls (plan 12, 15.1, 10.6; tests T-M1 to T-M5). Owner: Farzaan.
One engine serves both architectures (CNN occlusion and ViT patch masking).

    delta_p = p_c(image) - p_c(image with the region filled by its background colour),  c = original prediction

`flipped` records whether the masked image's prediction differs from c. Each region is compared with an
area-matched random mask in the same panel (`panel_box`). The control is built from 4 x 4 blocks of non-ink pixels
and trimmed to the region's pixel count, so it never removes ink and matches the area within 2% (T-M5). Only
regions that hold something besides background are masked (candles, volume, grid, axes); an empty region, such as
the grid in R2, gives delta_p = 0 exactly (T-M3).
"""
from __future__ import annotations

import numpy as np
import torch

from ..training.datasets import IMAGENET_MEAN, IMAGENET_STD
from .regions import INK_REGIONS, panel_box

MASKED_REGIONS = ("candles", "volume", "grid", "axes")
FILL_MODES = ("background", "random_area_matched")
MASKING_COLUMNS = ["run_id", "window_id", "render_id", "region", "fill_mode", "area_frac", "pred_class_orig",
                   "p_pred_orig", "p_pred_masked", "delta_p", "flipped"]
BLOCK = 4


def fill(img: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    """Copy of the uint8 (H, W, 3) image with `mask` pixels set to `color`."""
    out = img.copy()
    out[mask] = np.asarray(color, dtype=np.uint8)
    return out


def random_area_matched(region: np.ndarray, ink: np.ndarray, box, rng: np.random.Generator) -> np.ndarray:
    """Random mask of exactly region.sum() non-ink pixels inside `box`, grown from 4 x 4 blocks."""
    target = int(region.sum())
    out = np.zeros_like(region)
    if target == 0:
        return out
    x0, y0, x1, y1 = box
    free = np.zeros_like(region)
    free[y0:y1 + 1, x0:x1 + 1] = True
    free &= ~ink
    if free.sum() < target:
        raise ValueError(f"panel has {int(free.sum())} non-ink pixels, fewer than the region's {target}")
    ys = np.arange(y0, y1 + 1, BLOCK)
    xs = np.arange(x0, x1 + 1, BLOCK)
    cells = [(y, x) for y in ys for x in xs]
    for i in rng.permutation(len(cells)):
        y, x = cells[i]
        out[y:y + BLOCK, x:x + BLOCK] |= free[y:y + BLOCK, x:x + BLOCK]
        if out.sum() >= target:
            break
    extra = int(out.sum()) - target
    if extra > 0:                                      # trim the last block to the exact pixel count
        on = np.flatnonzero(out)
        out.flat[rng.choice(on, size=extra, replace=False)] = False
    return out


def to_tensor(imgs: list[np.ndarray]) -> torch.Tensor:
    """uint8 (H, W, 3) images -> normalised (B, 3, H, W), exactly as the training datasets do it."""
    arr = np.stack(imgs).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(0, 3, 1, 2)).contiguous()


@torch.no_grad()
def predict_proba(model, imgs: list[np.ndarray], device, batch: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(imgs), batch):
        out.append(model(to_tensor(imgs[i:i + batch]).to(device)).float().softmax(1).cpu().numpy())
    return np.concatenate(out)


def mask_variants(img: np.ndarray, masks: dict[str, np.ndarray], style, rng: np.random.Generator):
    """[(region, fill_mode, area_frac, masked image)] for one image: each masked region and its control."""
    ink = np.zeros(img.shape[:2], dtype=bool)
    for r in INK_REGIONS:
        ink |= masks[r]
    out = []
    for region in MASKED_REGIONS:
        m = masks[region]
        ctrl = random_area_matched(m, ink, panel_box(region, style), rng)
        for mode, mm in (("background", m), ("random_area_matched", ctrl)):
            out.append((region, mode, float(mm.sum()) / mm.size, fill(img, mm, style.background)))
    return out


def masking_rows(model, run_id: str, items, device, seed: int = 0, batch: int = 64) -> list[dict]:
    """`items`: iterable of (window_id, render_id, uint8 image, region masks, style). One forward of the
    original image and of every masked variant; rows follow plan 10.6."""
    rows = []
    for k, (wid, rid, img, masks, style) in enumerate(items):
        rng = np.random.default_rng([seed, k])
        variants = mask_variants(img, masks, style, rng)
        # A variant whose pixels equal the original (empty region, or a control that only touched background)
        # reuses the original probabilities: delta_p is then exactly 0 (T-M3), not batch-position float noise.
        changed = [i for i, v in enumerate(variants) if not np.array_equal(v[3], img)]
        p = predict_proba(model, [img] + [variants[i][3] for i in changed], device, batch)
        q_all = np.repeat(p[:1], len(variants), axis=0)
        q_all[changed] = p[1:]
        c = int(p[0].argmax())
        for (region, mode, area, _), q in zip(variants, q_all):
            rows.append({"run_id": run_id, "window_id": wid, "render_id": rid, "region": region, "fill_mode": mode,
                         "area_frac": area, "pred_class_orig": c, "p_pred_orig": float(p[0, c]),
                         "p_pred_masked": float(q[c]), "delta_p": float(p[0, c] - q[c]),
                         "flipped": bool(int(q.argmax()) != c)})
    return rows
