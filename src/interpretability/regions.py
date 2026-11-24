"""Region partition of a rendered chart, the D15 sample and the attribution rows of plan 10.6. Owner: Farzaan
(region geometry); used by Grad-CAM (Kaushik), rollout (Khushi) and masking (Farzaan).

Each pixel of the 224 x 224 canvas belongs to exactly one region (T-M1). Pixels are assigned in draw order, so the
last primitive drawn on a pixel decides, which matches the rasterizer:

    candles          candle bodies and wicks (price panel ink)
    volume           volume bars (volume panel ink)
    grid             visible grid-line pixels (R0, R1, R3-R6; none in R2)
    axes             visible axis and tick pixels (R3 only)
    plot_background  background pixels inside the plot box (both panels and the gap between them)
    margins          background pixels outside the plot box
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..rendering.rasterizer import draw_list
from ..rendering.styles import Style, geometry

REGIONS = ("candles", "volume", "grid", "axes", "plot_background", "margins")
INK_REGIONS = ("candles", "volume")
_TAG_REGION = {"body": "candles", "wick": "candles", "volume": "volume", "grid": "grid", "axis": "axes"}
ATTRIBUTION_COLUMNS = ["run_id", "window_id", "render_id", "method", "region", "attribution_mass", "area_frac",
                       "mass_ratio"]


def region_masks(window: np.ndarray, style: Style) -> dict[str, np.ndarray]:
    """{region: bool (canvas, canvas)} for one window drawn in `style`; the masks partition the canvas."""
    c = style.canvas
    code = np.full((c, c), -1, dtype=np.int8)                 # -1 = background
    index = {r: i for i, r in enumerate(REGIONS)}
    for p in draw_list(window, style):
        if p.tag == "background":
            code[p.y0:p.y1 + 1, p.x0:p.x1 + 1] = -1
            continue
        if p.tag not in _TAG_REGION:
            raise ValueError(f"unknown draw-list tag {p.tag!r}")
        code[p.y0:p.y1 + 1, p.x0:p.x1 + 1] = index[_TAG_REGION[p.tag]]
    x0, y0, x1, y1 = geometry(style).plot_box
    inside = np.zeros((c, c), dtype=bool)
    inside[y0:y1 + 1, x0:x1 + 1] = True
    out = {r: code == index[r] for r in REGIONS[:4]}
    out["plot_background"] = (code == -1) & inside
    out["margins"] = (code == -1) & ~inside
    return out


def panel_box(region: str, style: Style) -> tuple[int, int, int, int]:
    """Inclusive [x0, y0, x1, y1] box that an area-matched control for `region` is drawn from."""
    g = geometry(style)
    c = style.canvas
    return {"candles": g.price_box, "volume": g.volume_box, "grid": g.plot_box}.get(region, (0, 0, c - 1, c - 1))


def interpretability_sample(windows: pd.DataFrame, n: int = 2000, seed: int = 0, split: str = "test") -> list[str]:
    """D15: `n` windows of `split`, stratified by true class (proportional allocation, largest remainder), the
    same windows for every model. Returns sorted window_ids."""
    w = windows[windows["split"] == split]
    if len(w) <= n:
        return sorted(w["window_id"])
    counts = w["label"].value_counts().sort_index()
    exact = counts / counts.sum() * n
    take = np.floor(exact).astype(int)
    for lab in (exact - take).sort_values(ascending=False).index[: n - int(take.sum())]:
        take[lab] += 1
    rng = np.random.default_rng(seed)
    out = []
    for lab, k in take.items():
        ids = np.sort(w.loc[w["label"] == lab, "window_id"].to_numpy())
        out += list(rng.choice(ids, size=int(k), replace=False))
    return sorted(out)


def attribution_rows(run_id: str, window_id: str, render_id: str, method: str, heat: np.ndarray,
                     masks: dict[str, np.ndarray]) -> list[dict]:
    """Plan 10.6 attribution rows for one image. `heat` is a non-negative (canvas, canvas) map summing to 1
    (or all NaN when the method gives no signal, which is recorded rather than hidden)."""
    total = heat.size
    rows = []
    for region in REGIONS:
        m = masks[region]
        area = float(m.sum()) / total
        mass = float(heat[m].sum()) if np.isfinite(heat).all() else float("nan")
        rows.append({"run_id": run_id, "window_id": window_id, "render_id": render_id, "method": method,
                     "region": region, "attribution_mass": mass, "area_frac": area,
                     "mass_ratio": mass / area if area > 0 else float("nan")})
    return rows


def normalize_heat(heat: np.ndarray) -> np.ndarray:
    """Clip at 0 and scale to sum 1; an all-zero map becomes all NaN (no attribution signal)."""
    h = np.clip(np.asarray(heat, dtype=np.float64), 0, None)
    s = h.sum()
    return h / s if s > 0 else np.full_like(h, np.nan)
