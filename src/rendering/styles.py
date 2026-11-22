"""Rendering style table (render_v1) and the per-style pixel geometry.

Styles are loaded from ``configs/render_styles.yaml``; each style is R0 plus the
overrides of its single factor. All geometry is integer pixels so the rasterizer
is exact (no anti-aliasing) and the equivalence tests can read pixels back.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

RENDER_IDS = ("R0", "R1", "R2", "R3", "R4", "R5", "R6")
SEEN = ("R0", "R1", "R2", "R4")
HELD_OUT = ("R3", "R5", "R6")
DEFAULT_STYLES = Path(__file__).resolve().parents[2] / "configs" / "render_styles.yaml"


def _rnd(x: float) -> int:
    """Round half up (deterministic across platforms, unlike banker's rounding)."""
    return int(math.floor(x + 0.5))


@dataclass(frozen=True)
class Style:
    render_id: str
    role: str
    canvas: int
    background: tuple
    grid: bool
    grid_rows: int
    grid_cols: int
    grid_color: tuple
    up_color: tuple
    down_color: tuple
    volume_color: tuple
    body_frac: float
    volume_frac: float
    wick_px: int
    margin: int
    price_frac: float
    volume_frac_h: float
    axes: bool
    axis_color: tuple
    tick_len: int


@dataclass(frozen=True)
class Geometry:
    """Integer pixel layout. Boxes are inclusive (x0, y0, x1, y1)."""
    x0: int
    y0: int
    w: int
    h: int
    price_top: int
    price_bottom: int
    vol_top: int
    vol_bottom: int
    centers: tuple            # wick column of each of the 30 candles
    body_px: int
    volume_px: int
    grid_y: tuple
    grid_x: tuple

    @property
    def plot_box(self):
        return (self.x0, self.y0, self.x0 + self.w - 1, self.y0 + self.h - 1)

    @property
    def price_box(self):
        return (self.x0, self.price_top, self.x0 + self.w - 1, self.price_bottom)

    @property
    def volume_box(self):
        return (self.x0, self.vol_top, self.x0 + self.w - 1, self.vol_bottom)

    @property
    def price_h(self):
        return self.price_bottom - self.price_top + 1

    @property
    def vol_h(self):
        return self.vol_bottom - self.vol_top + 1


def geometry(style: Style, n_candles: int = 30) -> Geometry:
    m, c = style.margin, style.canvas
    w = h = c - 2 * m
    price_h = _rnd(style.price_frac * h)
    vol_h = _rnd(style.volume_frac_h * h)
    slot = w / n_candles
    centers = tuple(m + int(math.floor((i + 0.5) * slot)) for i in range(n_candles))
    return Geometry(
        x0=m, y0=m, w=w, h=h,
        price_top=m, price_bottom=m + price_h - 1,
        vol_top=m + h - vol_h, vol_bottom=m + h - 1,
        centers=centers,
        body_px=max(1, _rnd(style.body_frac * slot)),
        volume_px=max(1, _rnd(style.volume_frac * slot)),
        grid_y=tuple(m + _rnd(k * h / (style.grid_rows + 1)) for k in range(1, style.grid_rows + 1)),
        grid_x=tuple(m + _rnd(k * w / (style.grid_cols + 1)) for k in range(1, style.grid_cols + 1)),
    )


def load_styles(path: str | Path = DEFAULT_STYLES) -> tuple[dict[str, Style], str, str]:
    """Return ({render_id: Style}, render_version, sha256 of the YAML file)."""
    raw = Path(path).read_bytes()
    spec = yaml.safe_load(raw)
    names = {f.name for f in fields(Style)}
    styles = {}
    for rid in RENDER_IDS:
        merged = dict(spec["base"])
        merged.update(spec["styles"].get(rid) or {})
        merged["render_id"] = rid
        merged["canvas"] = spec["canvas"]
        unknown = set(merged) - names
        if unknown:
            raise ValueError(f"{rid}: unknown style keys {unknown}")
        for k, v in list(merged.items()):
            if isinstance(v, list):
                merged[k] = tuple(v)
        styles[rid] = Style(**merged)
    roles = {rid: s.role for rid, s in styles.items()}
    if [r for r in RENDER_IDS if roles[r] in ("reference", "seen")] != list(SEEN) or \
            [r for r in RENDER_IDS if roles[r] == "held_out"] != sorted(HELD_OUT):
        raise ValueError(f"seen/held-out roles differ from D8: {roles}")
    return styles, spec["render_version"], hashlib.sha256(raw).hexdigest()
