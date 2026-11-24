"""Region partition and masking, tests T-M1 to T-M5 (plan 16.2; owner Farzaan). CPU only; synthetic windows."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from interp_fixtures import random_windows  # noqa: E402
from src.common.schema import RENDER_IDS  # noqa: E402
from src.interpretability.link import masking_summary  # noqa: E402
from src.interpretability.masking import (MASKED_REGIONS, fill, mask_variants, masking_rows,  # noqa: E402
                                          random_area_matched, to_tensor)
from src.interpretability.regions import INK_REGIONS, REGIONS, interpretability_sample, panel_box, region_masks  # noqa: E402
from src.rendering.rasterizer import render  # noqa: E402
from src.rendering.styles import load_styles  # noqa: E402

STYLES = load_styles()[0]
WINDOWS = random_windows(6, seed=3)


@pytest.mark.parametrize("rid", RENDER_IDS)
def test_tm1_region_masks_partition_the_canvas(rid):
    for w in WINDOWS:
        m = region_masks(w, STYLES[rid])
        stack = np.stack([m[r] for r in REGIONS]).astype(int)
        assert (stack.sum(0) == 1).all(), "every pixel in exactly one region"
        img = render(w, STYLES[rid])
        bg = np.all(img == np.asarray(STYLES[rid].background, dtype=np.uint8), axis=-1)
        assert (m["plot_background"] | m["margins"])[~bg].sum() == 0          # background regions are background
        assert m["candles"].any() and m["volume"].any()
    assert not region_masks(WINDOWS[0], STYLES["R2"])["grid"].any()           # R2 draws no grid
    assert region_masks(WINDOWS[0], STYLES["R3"])["axes"].any() and not region_masks(WINDOWS[0], STYLES["R0"])["axes"].any()


@pytest.mark.parametrize("rid", RENDER_IDS)
def test_tm2_fill_equals_the_rendering_background(rid):
    w, st = WINDOWS[1], STYLES[rid]
    img = render(w, st)
    m = region_masks(w, st)
    for region in MASKED_REGIONS:
        out = fill(img, m[region], st.background)
        assert (out[m[region]] == np.asarray(st.background, dtype=np.uint8)).all()
        assert (out[~m[region]] == img[~m[region]]).all()                     # nothing else changes
    inkless = fill(img, m["candles"] | m["volume"], st.background)
    assert not np.array_equal(inkless, img)                                    # ink removal is visible


def test_tm3_masking_the_absent_grid_in_r2_changes_nothing():
    w, st = WINDOWS[2], STYLES["R2"]
    img = render(w, st)
    m = region_masks(w, st)
    model = nn.Sequential(nn.Conv2d(3, 4, 7, stride=7), nn.Flatten(), nn.LazyLinear(3))
    model(to_tensor([img]))
    rows = masking_rows(model, "r", [("w", "R2", img, m, st)], torch.device("cpu"))
    grid = [r for r in rows if r["region"] == "grid"]
    assert len(grid) == 2 and all(r["delta_p"] == 0.0 and r["area_frac"] == 0.0 and not r["flipped"] for r in grid)


@pytest.mark.parametrize("rid", ["R0", "R1", "R3", "R5", "R6"])
def test_tm5_random_controls_avoid_ink_and_match_area_within_2pct(rid):
    st = STYLES[rid]
    for k, w in enumerate(WINDOWS):
        m = region_masks(w, st)
        ink = m["candles"] | m["volume"]
        for region in MASKED_REGIONS:
            ctrl = random_area_matched(m[region], ink, panel_box(region, st), np.random.default_rng(k))
            assert not (ctrl & ink).any()
            a, b = int(m[region].sum()), int(ctrl.sum())
            assert abs(b - a) <= 0.02 * max(a, 1)
            x0, y0, x1, y1 = panel_box(region, st)
            assert not ctrl[:y0].any() and not ctrl[y1 + 1:].any() and not ctrl[:, :x0].any() and not ctrl[:, x1 + 1:].any()


def test_mask_variants_cover_every_region_and_both_fill_modes():
    st = STYLES["R0"]
    img = render(WINDOWS[0], st)
    v = mask_variants(img, region_masks(WINDOWS[0], st), st, np.random.default_rng(0))
    assert sorted((r, m) for r, m, _, _ in v) == sorted((r, m) for r in MASKED_REGIONS
                                                          for m in ("background", "random_area_matched"))


class InkReader(nn.Module):
    """Linear model on a 28 x 28 grayscale view of the chart, trained below by gradient descent."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(28 * 28, 3)

    def forward(self, x):
        return self.fc(self.pool(x.mean(1, keepdim=True)).flatten(1))


def test_tm4_ink_masking_beats_other_regions_on_a_trained_model():
    """Trend windows (label UP for a rising window, DOWN for a falling one) are learnable from the candles alone.
    After training, removing the candles must move the prediction more than removing the grid or the axes, and
    more than its own area-matched control."""
    torch.manual_seed(0)
    n = 160
    rng = np.random.default_rng(0)
    y = rng.choice([0, 2], n)
    x = random_windows(n, seed=11, trend=np.where(y == 2, 0.012, -0.012))
    st = STYLES["R0"]
    imgs = [render(w, st) for w in x]
    X, Y = to_tensor(imgs), torch.as_tensor(y)
    model = InkReader()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        nn.functional.cross_entropy(model(X), Y).backward()
        opt.step()
    assert (model(X).argmax(1) == Y).float().mean() > 0.95
    items = [(f"w{i}", "R0", imgs[i], region_masks(x[i], st), st) for i in range(40)]
    s = masking_summary(pd.DataFrame(masking_rows(model, "r", items, torch.device("cpu"))))
    bg = s[s["fill_mode"] == "background"].set_index("region")["mean_abs_delta_p"]
    assert bg["candles"] > max(bg["grid"], bg["axes"], bg["volume"])
    assert s[(s["region"] == "candles") & (s["fill_mode"] == "background")]["region_minus_control"].iloc[0] > 0


def test_d15_sample_is_class_stratified_deterministic_and_test_only():
    rng = np.random.default_rng(0)
    w = pd.DataFrame({"window_id": [f"w{i:05d}" for i in range(12000)], "label": rng.choice(3, 12000, p=[.3, .3, .4]),
                      "split": rng.choice(["train", "val", "test"], 12000, p=[.5, .2, .3])})
    ids = interpretability_sample(w, n=2000)
    assert len(ids) == 2000 == len(set(ids)) and ids == interpretability_sample(w, n=2000)
    t = w.set_index("window_id").loc[ids]
    assert (t["split"] == "test").all()
    share = t["label"].value_counts(normalize=True).sort_index().to_numpy()
    full = w[w["split"] == "test"]["label"].value_counts(normalize=True).sort_index().to_numpy()
    np.testing.assert_allclose(share, full, atol=1 / 2000 + 1e-9)
    assert INK_REGIONS == ("candles", "volume")
