"""T-R1 to T-R8 rendering equivalence tests for render_v1 (Execution Plan 5.2, 16.2; owner Farzaan).

Synthetic tests always run; full-registry tests run when data/render_v1 exists.
Full-registry pixel checks decode stored PNGs (a random sample, size set by
$DATA255_RENDER_SAMPLE, default 1000) so the suite stays under the 5-minute budget;
the build script already decoded and hash-checked every PNG it wrote.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from g1g2_fixtures import RENDER_V1, ROOT, full_data, needs_full_render, synthetic_build
from src.rendering import registry as rg
from src.rendering.rasterizer import PRIMITIVE_KINDS, draw_list, image_hash, price_to_row, render
from src.rendering.regions import image_masks, region_geometry
from src.rendering.styles import HELD_OUT, RENDER_IDS, SEEN, geometry, load_styles

STYLES, RENDER_VERSION, _ = load_styles()
RNG = np.random.default_rng(2024)
SAMPLE = int(os.environ.get("DATA255_RENDER_SAMPLE", "1000"))


def _syn_windows(n=200):
    _, _, _, res = synthetic_build()
    w = res.windows[res.windows["split"] != "purged"]
    pick = RNG.choice(len(w), n, replace=False)
    return w.iloc[pick], res.ohlcv


# ------------------------------------------------------------- pixel spot check (T-R6)
def _spot_check(win: np.ndarray, img: np.ndarray, style):
    """Wick extremes and body edges read back from pixels match price_to_row of H/L/O/C."""
    g = geometry(style)
    lo, hi = float(win[:, 2].min()), float(win[:, 1].max())
    inks = np.array([style.up_color, style.down_color], dtype=np.uint8)
    panel = img[g.price_top:g.price_bottom + 1]
    for i, cx in enumerate(g.centers):
        o, h, l, c = (float(x) for x in win[i, :4])
        color = np.array(style.up_color if c >= o else style.down_color, dtype=np.uint8)
        col = panel[:, cx]
        ink_rows = np.flatnonzero((col[:, None, :] == inks[None]).all(-1).any(-1))
        assert ink_rows.size, f"{style.render_id} candle {i}: no ink in wick column"
        assert (col[ink_rows] == color).all(), "candle color must encode Close >= Open"
        top = price_to_row(h, lo, hi, g.price_top, g.price_h)
        bot = price_to_row(l, lo, hi, g.price_top, g.price_h)
        assert g.price_top + ink_rows.min() == top and g.price_top + ink_rows.max() == bot
        assert len(ink_rows) == bot - top + 1, "wick must be continuous"
        # body edge column carries ink exactly from row(max(O,C)) to row(min(O,C))
        edge = cx - g.body_px // 2
        brow = np.flatnonzero((panel[:, edge] == color).all(-1))
        assert g.price_top + brow.min() == price_to_row(max(o, c), lo, hi, g.price_top, g.price_h)
        assert g.price_top + brow.max() == price_to_row(min(o, c), lo, hi, g.price_top, g.price_h)


# --------------------------------------------------------------------- styles
def test_style_roles_match_d8():
    assert RENDER_VERSION == "render_v1"
    assert [r for r in RENDER_IDS if STYLES[r].role in ("reference", "seen")] == list(SEEN)
    assert sorted(r for r in RENDER_IDS if STYLES[r].role == "held_out") == sorted(HELD_OUT)
    assert STYLES["R0"].role == "reference"


def test_r0_geometry_matches_spec():
    g = geometry(STYLES["R0"])
    assert (g.w, g.h, g.x0) == (208, 208, 8)
    assert g.price_h == 156 and g.vol_h == 42           # 75% / 20% of 208
    assert len(g.grid_y) == 5 and len(g.grid_x) == 6
    assert geometry(STYLES["R6"]).w == 176 and geometry(STYLES["R6"]).x0 == 24
    assert len(set(g.centers)) == 30 and list(g.centers) == sorted(g.centers)


def test_ink_colors_distinct_from_decoration():
    for s in STYLES.values():
        deco = {s.background, s.grid_color, s.volume_color, s.axis_color}
        assert s.up_color not in deco and s.down_color not in deco and s.up_color != s.down_color


def test_each_style_changes_one_factor():
    """Pixels that differ from R0 lie only where that style's factor lives."""
    wins, ohlcv = _syn_windows(20)
    for _, row in wins.iterrows():
        win = ohlcv[row["row_idx"]]
        r0 = render(win, STYLES["R0"])
        m0 = image_masks(win, STYLES["R0"])
        diff = {rid: (render(win, STYLES[rid]) != r0).any(-1) for rid in RENDER_IDS}
        assert not diff["R0"].any()
        assert (diff["R1"] == m0["background"]).all()                         # background only
        assert diff["R2"].any() and not (diff["R2"] & ~m0["decoration"]).any()  # grid pixels only
        assert not (diff["R3"] & ~image_masks(win, STYLES["R3"])["decoration"]).any()
        g = geometry(STYLES["R0"])
        x0, y0, x1, y1 = g.plot_box
        assert not diff["R3"][y0:y1 + 1, x0:x1 + 1].any(), "R3 spines/ticks must stay outside the plot"
        assert not (diff["R4"] & ~m0["ink"]).any()                             # candle colors only
        volume = np.zeros_like(m0["ink"])
        for p in draw_list(win, STYLES["R0"]):
            if p.tag == "volume":
                volume[p.y0:p.y1 + 1, p.x0:p.x1 + 1] = True
        assert not (diff["R4"] & volume).any(), "volume bars stay gray in R4"
        assert not diff["R5"][g.vol_top:].any(), "R5 changes candle bodies only"
        r5_cols = np.flatnonzero(diff["R5"].any(0))
        assert all(min(abs(c - cx) for cx in g.centers) <= g.body_px // 2 for c in r5_cols)


# --------------------------------------------------------------------- T-R5, T-R6, T-R8
def test_tr5_thirty_candles_in_date_order():
    wins, ohlcv = _syn_windows(50)
    for _, row in wins.iterrows():
        for rid in RENDER_IDS:
            prims = draw_list(ohlcv[row["row_idx"]], STYLES[rid])
            for tag in ("wick", "body"):
                ps = [p for p in prims if p.tag == tag]
                assert [p.idx for p in ps] == list(range(30))
                xs = [p.x0 for p in ps]
                assert xs == sorted(xs) and len(set(xs)) == 30


def test_tr6_one_price_function_and_pixel_spot_check():
    wins, ohlcv = _syn_windows(200)
    for _, row in wins.iterrows():
        win = ohlcv[row["row_idx"]]
        for rid in RENDER_IDS:
            _spot_check(win, render(win, STYLES[rid]), STYLES[rid])


def test_tr6_wick_rows_identical_across_same_geometry_styles():
    """Same O/H/L/C -> same rows for every style sharing R0's plot geometry (R0-R5)."""
    wins, ohlcv = _syn_windows(30)
    for _, row in wins.iterrows():
        win = ohlcv[row["row_idx"]]
        ref = [(p.y0, p.y1) for p in draw_list(win, STYLES["R0"]) if p.tag == "wick"]
        for rid in ("R1", "R2", "R3", "R4", "R5"):
            assert [(p.y0, p.y1) for p in draw_list(win, STYLES[rid]) if p.tag == "wick"] == ref


def test_tr8_no_text_and_no_metadata_inputs():
    assert PRIMITIVE_KINDS == ("rect",)
    assert list(inspect.signature(render).parameters) == ["window", "style"]
    assert list(inspect.signature(draw_list).parameters) == ["window", "style"]
    wins, ohlcv = _syn_windows(10)
    for _, row in wins.iterrows():
        for rid in RENDER_IDS:
            assert {p.kind for p in draw_list(ohlcv[row["row_idx"]], STYLES[rid])} == {"rect"}
    # identical arrays from different tickers/dates -> identical images
    win = ohlcv[wins.iloc[0]["row_idx"]].copy()
    assert image_hash(render(win, STYLES["R0"])) == image_hash(render(win.copy(), STYLES["R0"]))


def test_tr4_shape_dtype():
    wins, ohlcv = _syn_windows(5)
    for rid in RENDER_IDS:
        img = render(ohlcv[wins.iloc[0]["row_idx"]], STYLES[rid])
        assert img.shape == (224, 224, 3) and img.dtype == np.uint8


def test_flat_window_renders():
    win = np.ones((30, 5), dtype=np.float32)
    win[:, 4] = 0
    for rid in RENDER_IDS:
        assert render(win, STYLES[rid]).shape == (224, 224, 3)


def test_region_geometry_boxes():
    for rid in RENDER_IDS:
        geo = region_geometry(STYLES[rid])
        for key in ("plot", "price_panel", "volume_panel"):
            x0, y0, x1, y1 = geo[key]
            assert 0 <= x0 <= x1 < 224 and 0 <= y0 <= y1 < 224
        assert (geo["axes"] is not None) == (rid == "R3")
        assert (len(geo["grid_rows"]) == 0) == (rid == "R2")


# --------------------------------------------------------------------- T-R7
def test_tr7_same_hash_across_processes(tmp_path):
    wins, ohlcv = _syn_windows(8)
    arr = np.stack([ohlcv[i] for i in wins["row_idx"]])
    np.save(tmp_path / "w.npy", arr)
    code = ("import sys, numpy as np; sys.path.insert(0, sys.argv[1]);"
            "from src.rendering.rasterizer import render, image_hash;"
            "from src.rendering.styles import load_styles, RENDER_IDS;"
            "S=load_styles()[0]; a=np.load(sys.argv[2]);"
            "print('\\n'.join(image_hash(render(w, S[r])) for w in a for r in RENDER_IDS))")
    out = subprocess.run([sys.executable, "-c", code, str(ROOT), str(tmp_path / "w.npy")],
                         capture_output=True, text=True, check=True).stdout.split()
    here = [image_hash(render(w, STYLES[r])) for w in arr for r in RENDER_IDS]
    assert out == here


# --------------------------------------------------------------------- registry (synthetic)
def _synthetic_registry(tmp_path, n=12):
    wins, _ = _syn_windows(n)
    _, _, _, res = synthetic_build()
    np.save(tmp_path / "ohlcv.npy", res.ohlcv)
    root = tmp_path / "shards"
    for name in rg.shard_names():
        (root / name).mkdir(parents=True)
    rg._init_worker(str(tmp_path / "ohlcv.npy"), STYLES, str(root), True)
    out = rg.render_chunk(list(zip(wins["row_idx"].astype(int), wins["window_id"], wins["split"])))
    reg = pd.DataFrame(out, columns=["window_id", "render_id", "image_path", "image_hash"])
    reg = reg.join(wins.set_index("window_id")[["split", "label", "ohlcv_hash"]], on="window_id")
    reg["render_role"] = reg["render_id"].map({r: STYLES[r].role for r in RENDER_IDS})
    reg["render_version"] = RENDER_VERSION
    return reg[rg.RENDER_COLUMNS], wins, root


def _check_registry(reg, windows):
    assert list(reg.columns) == rg.RENDER_COLUMNS
    assert not reg.duplicated(["window_id", "render_id"]).any()
    per = reg.groupby("window_id")
    assert (per["render_id"].apply(lambda s: sorted(s) == list(RENDER_IDS))).all()
    for col in ("ohlcv_hash", "label", "split"):                 # T-R1..T-R3
        assert (per[col].nunique() == 1).all(), f"{col} differs across renderings of a window"
    w = windows.set_index("window_id")
    for col in ("ohlcv_hash", "label", "split"):
        assert (reg[col].to_numpy() == w.loc[reg["window_id"], col].to_numpy()).all()
    expect = reg["split"] + "_" + reg["render_id"] + "/" + reg["window_id"] + ".png"
    assert (reg["image_path"] == expect).all()
    assert (reg["render_role"] == reg["render_id"].map({r: STYLES[r].role for r in RENDER_IDS})).all()
    assert (reg["render_version"] == "render_v1").all()


def test_tr1_tr3_registry_synthetic(tmp_path):
    reg, wins, root = _synthetic_registry(tmp_path)
    _check_registry(reg, wins)
    for _, r in reg.iterrows():                                  # T-R4 on decoded PNGs
        img = rg.read_png(str(root / r["image_path"]))
        assert img.shape == (224, 224, 3) and img.dtype == np.uint8
        assert image_hash(img) == r["image_hash"]


# --------------------------------------------------------------------- full registry
@needs_full_render
def test_tr1_tr3_full_registry():
    _, windows, *_ = full_data()
    reg = pd.read_parquet(RENDER_V1 / "render_registry.parquet")
    _check_registry(reg, windows)
    kept = windows[windows["split"] != "purged"]
    assert set(reg["window_id"]) == set(kept["window_id"]), "every non-purged window is rendered"
    assert len(reg) == 7 * len(kept)


@needs_full_render
def test_tr4_tr6_full_sample_pngs():
    _, windows, ohlcv, *_ = full_data()
    reg = pd.read_parquet(RENDER_V1 / "render_registry.parquet")
    row_idx = windows.set_index("window_id")["row_idx"]
    pick = reg.iloc[RNG.choice(len(reg), min(SAMPLE, len(reg)), replace=False)]
    for _, r in pick.iterrows():
        img = rg.read_png(str(RENDER_V1 / "shards" / r["image_path"]))
        assert img.shape == (224, 224, 3) and img.dtype == np.uint8          # T-R4
        assert image_hash(img) == r["image_hash"]
        win = np.asarray(ohlcv[row_idx[r["window_id"]]])
        assert image_hash(render(win, STYLES[r["render_id"]])) == r["image_hash"]  # T-R7 vs stored
    wids = RNG.choice(reg["window_id"].unique(), 200, replace=False)            # T-R6, 200 windows
    sub = reg[reg["window_id"].isin(wids)]
    for _, r in sub.iterrows():
        img = rg.read_png(str(RENDER_V1 / "shards" / r["image_path"]))
        _spot_check(np.asarray(ohlcv[row_idx[r["window_id"]]]), img, STYLES[r["render_id"]])


@needs_full_render
def test_render_manifest_consistent():
    man = json.load(open(RENDER_V1 / "render_manifest.json"))
    from src.data.download import sha256_file
    assert man["render_registry_sha256"] == sha256_file(str(RENDER_V1 / "render_registry.parquet"))
    assert man["windows_parquet_sha256"] == sha256_file(str(RENDER_V1.parent / "data_v1" / "windows.parquet"))
    assert sha256_file(str(RENDER_V1 / "windows.parquet")) == man["windows_parquet_sha256"]
    assert man["png_roundtrip_verified"] is True
    _, _, styles_sha = load_styles()
    assert man["styles_sha256"] == styles_sha
