"""Region geometry per render_id (``render_v1/region_geometry.json``, Execution Plan 10.2).

Boxes are inclusive pixel boxes [x0, y0, x1, y1] on the 224 x 224 canvas. Grid
rows/cols are listed for styles that draw a grid. Per-image background and ink
masks come from the draw list (``image_masks``), so masking code never has to
guess what is data and what is decoration.
"""
from __future__ import annotations

import numpy as np

from .rasterizer import draw_list
from .styles import Style, geometry


def region_geometry(style: Style) -> dict:
    g = geometry(style)
    c = style.canvas
    x0, y0, x1, y1 = g.plot_box
    out = {
        "render_id": style.render_id,
        "role": style.role,
        "canvas": [c, c],
        "plot": list(g.plot_box),
        "price_panel": list(g.price_box),
        "volume_panel": list(g.volume_box),
        "panel_gap": [x0, g.price_bottom + 1, x1, g.vol_top - 1],
        "margins": {"top": [0, 0, c - 1, y0 - 1], "bottom": [0, y1 + 1, c - 1, c - 1],
                    "left": [0, y0, x0 - 1, y1], "right": [x1 + 1, y0, c - 1, y1]},
        "axes": ([x0 - 1 - style.tick_len, y0, x1, y1 + 1 + style.tick_len] if style.axes else None),
        "grid_rows": list(g.grid_y) if style.grid else [],
        "grid_cols": list(g.grid_x) if style.grid else [],
        "candle_centers": list(g.centers),
        "body_px": g.body_px,
        "volume_px": g.volume_px,
        "background": list(style.background),
        "ink_colors": {"up": list(style.up_color), "down": list(style.down_color),
                       "volume": list(style.volume_color)},
    }
    return out


def image_masks(window: np.ndarray, style: Style) -> dict[str, np.ndarray]:
    """Boolean (224, 224) masks from the draw list: ink (candles+volume), decoration, background."""
    c = style.canvas
    owner = np.zeros((c, c), dtype=np.uint8)          # 0 bg, 1 decoration, 2 ink
    for p in draw_list(window, style):
        code = 0 if p.tag == "background" else (1 if p.tag in ("grid", "axis") else 2)
        owner[p.y0:p.y1 + 1, p.x0:p.x1 + 1] = code
    return {"background": owner == 0, "decoration": owner == 1, "ink": owner == 2}
