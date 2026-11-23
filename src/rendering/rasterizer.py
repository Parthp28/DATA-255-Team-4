"""Deterministic NumPy candlestick rasterizer (production renderer for render_v1).

The renderer's only inputs are the 30 x 5 float32 window (O, H, L, C, V) and a
Style; it never sees dates, tickers or later rows. Rendering is two steps:

1. ``draw_list(window, style)`` -> list of primitives. The only primitive kind is
   an axis-aligned filled rectangle ("rect"); there are no text primitives.
   Every price (O/H/L/C) goes through the single ``price_to_row`` function.
2. ``rasterize(prims, canvas)`` -> (224, 224, 3) uint8, painting rects in order.

No anti-aliasing, so pixels can be read back exactly by the equivalence tests.
"""
from __future__ import annotations

import hashlib
import math
from typing import NamedTuple

import numpy as np

from .styles import Geometry, Style, geometry

PRIMITIVE_KINDS = ("rect",)


class Prim(NamedTuple):
    kind: str          # always "rect"
    x0: int            # inclusive box
    y0: int
    x1: int
    y1: int
    color: tuple
    tag: str           # background / grid / axis / volume / wick / body
    idx: int           # candle index for volume / wick / body, else -1


def price_to_row(p: float, lo: float, hi: float, top: int, height: int) -> int:
    """The one price-to-pixel map used for every O/H/L/C in every style.

    hi -> top row, lo -> bottom row (top + height - 1). A flat window maps to the middle row.
    """
    if hi <= lo:
        return top + (height - 1) // 2
    frac = (float(hi) - float(p)) / (float(hi) - float(lo))
    return top + int(math.floor(frac * (height - 1) + 0.5))


def volume_px(v: float, vmax: float, height: int) -> int:
    if vmax <= 0:
        return 0
    return int(math.floor(float(v) / float(vmax) * height + 0.5))


def _span(center: int, width: int) -> tuple[int, int]:
    left = center - width // 2
    return left, left + width - 1


def draw_list(window: np.ndarray, style: Style) -> list[Prim]:
    win = np.asarray(window, dtype=np.float64)
    if win.shape != (30, 5) or not np.isfinite(win).all():
        raise ValueError(f"window must be a finite 30 x 5 array, got {win.shape}")
    g: Geometry = geometry(style, len(win))
    c = style.canvas
    prims = [Prim("rect", 0, 0, c - 1, c - 1, style.background, "background", -1)]

    x0, y0, x1, y1 = g.plot_box
    if style.grid:
        for y in g.grid_y:
            prims.append(Prim("rect", x0, y, x1, y, style.grid_color, "grid", -1))
        for x in g.grid_x:
            prims.append(Prim("rect", x, y0, x, y1, style.grid_color, "grid", -1))
    if style.axes:
        ax, by = x0 - 1, y1 + 1                      # spines sit just outside the plot area
        prims.append(Prim("rect", ax, y0, ax, by, style.axis_color, "axis", -1))
        prims.append(Prim("rect", ax, by, x1, by, style.axis_color, "axis", -1))
        for y in g.grid_y:
            prims.append(Prim("rect", ax - style.tick_len, y, ax - 1, y, style.axis_color, "axis", -1))
        for x in g.grid_x:
            prims.append(Prim("rect", x, by + 1, x, by + style.tick_len, style.axis_color, "axis", -1))

    o, h, l, cl, v = (win[:, k] for k in range(5))
    lo, hi = float(l.min()), float(h.max())
    vmax = float(v.max())
    for i, cx in enumerate(g.centers):              # volume bars (gray in every style)
        n = volume_px(v[i], vmax, g.vol_h)
        if n > 0:
            a, b = _span(cx, g.volume_px)
            prims.append(Prim("rect", a, g.vol_bottom - n + 1, b, g.vol_bottom, style.volume_color, "volume", i))
    rows = lambda p: price_to_row(p, lo, hi, g.price_top, g.price_h)  # noqa: E731
    for i, cx in enumerate(g.centers):              # wicks
        color = style.up_color if cl[i] >= o[i] else style.down_color
        a, b = _span(cx, style.wick_px)
        prims.append(Prim("rect", a, rows(h[i]), b, rows(l[i]), color, "wick", i))
    for i, cx in enumerate(g.centers):              # bodies
        color = style.up_color if cl[i] >= o[i] else style.down_color
        a, b = _span(cx, g.body_px)
        top, bot = rows(max(o[i], cl[i])), rows(min(o[i], cl[i]))
        prims.append(Prim("rect", a, top, b, bot, color, "body", i))
    return prims


def rasterize(prims: list[Prim], canvas: int = 224) -> np.ndarray:
    img = np.empty((canvas, canvas, 3), dtype=np.uint8)
    for p in prims:
        if p.kind != "rect":
            raise ValueError(f"unsupported primitive {p.kind}")
        if not (0 <= p.x0 <= p.x1 < canvas and 0 <= p.y0 <= p.y1 < canvas):
            raise ValueError(f"primitive outside canvas: {p}")
        img[p.y0:p.y1 + 1, p.x0:p.x1 + 1] = p.color
    return img


def render(window: np.ndarray, style: Style) -> np.ndarray:
    return rasterize(draw_list(window, style), style.canvas)


def image_hash(img: np.ndarray) -> str:
    """sha256 of the decoded uint8 array (H, W, 3), C order."""
    return hashlib.sha256(np.ascontiguousarray(img, dtype=np.uint8).tobytes()).hexdigest()
