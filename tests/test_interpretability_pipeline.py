"""run_attribution.py and run_masking.py end to end on a mini render_v1 drawn with the production renderer, the
D15 run gate, the image-hash guard, and the attribution-to-flip link (review: Farzaan, Kaushik, Khushi; Parth).
CPU only; untrained weights; not results."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import run_attribution  # noqa: E402
import run_masking  # noqa: E402
from interp_fixtures import fake_run, mini_render_root  # noqa: E402
from src.interpretability.link import attribution_flip_auroc, masking_summary  # noqa: E402
from src.interpretability.masking import MASKED_REGIONS, MASKING_COLUMNS  # noqa: E402
from src.interpretability.regions import ATTRIBUTION_COLUMNS, REGIONS  # noqa: E402
from src.interpretability.runs import InterpretabilityRefused, load_run  # noqa: E402
from src.models.resnet import build_resnet  # noqa: E402
from src.models.vit import build_deit_small  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return mini_render_root(tmp_path_factory.mktemp("interp"), n=18)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    torch.manual_seed(0)
    tmp = tmp_path_factory.mktemp("runs")
    return {"E1": fake_run(tmp, "E1", build_resnet("resnet18.a1_in1k", pretrained=False), "resnet18.a1_in1k"),
            "E4": fake_run(tmp, "E4", build_deit_small(pretrained=False), "deit_small_patch16_224.fb_in1k")}


@pytest.mark.parametrize("exp,method", [("E1", "gradcam"), ("E4", "rollout")])
def test_attribution_rows_follow_10_6_and_each_image_sums_to_one(root, runs, tmp_path, exp, method):
    df, meta = run_attribution.run(runs[exp], root, tmp_path, renders=("R0", "R2"), n=6, batch=4, device=CPU,
                                   test_mode=True)
    n_test = 12                                                    # 18 windows, the first third is train
    assert list(df.columns) == ATTRIBUTION_COLUMNS and set(df["method"]) == {method}
    assert meta["n_windows"] == 6 and len(df) == 6 * 2 * len(REGIONS) and n_test > 6
    per_image = df.groupby(["window_id", "render_id"])
    np.testing.assert_allclose(per_image["attribution_mass"].sum(), 1.0, rtol=1e-6)
    np.testing.assert_allclose(per_image["area_frac"].sum(), 1.0, rtol=1e-9)
    assert (df[df["render_id"] == "R2"].query("region == 'grid'")["area_frac"] == 0).all()
    saved = json.loads((tmp_path / f"{meta['run_id']}.json").read_text())
    assert saved["sample_sha256"] == meta["sample_sha256"] and saved["method"] == method
    again, meta2 = run_attribution.run(runs[exp], root, tmp_path / "b", renders=("R0", "R2"), n=6, batch=3,
                                       device=CPU, test_mode=True)
    assert meta2["sample_sha256"] == meta["sample_sha256"]                  # same windows for every model
    np.testing.assert_allclose(again["attribution_mass"], df["attribution_mass"], atol=1e-6)   # batch-size free


def test_masking_rows_follow_10_6(root, runs, tmp_path):
    df, meta = run_masking.run(runs["E1"], root, tmp_path, renders=("R0", "R2"), n=6, device=CPU, test_mode=True)
    assert list(df.columns) == MASKING_COLUMNS and len(df) == 6 * 2 * len(MASKED_REGIONS) * 2
    np.testing.assert_allclose(df["delta_p"], df["p_pred_orig"] - df["p_pred_masked"], atol=1e-7)
    assert (df.query("render_id == 'R2' and region == 'grid'")["delta_p"] == 0).all()      # T-M3 end to end
    assert (df.query("render_id == 'R0' and region == 'axes'")["delta_p"] == 0).all()      # R0 has no axes
    s = masking_summary(df)
    assert {"mean_abs_delta_p", "flip_rate", "region_minus_control"} <= set(s.columns)


def test_d15_gate_refuses_other_runs(runs, tmp_path):
    with pytest.raises(InterpretabilityRefused, match="frozen configuration"):
        load_run(runs["E1"], CPU)                                            # not frozen -> refused outside tests
    rec = json.loads((runs["E1"] / "run.json").read_text())
    for k, (patch, msg) in enumerate((({"seed": 1}, "seed-0"), ({"experiment_id": "E5", "model": "numeric1d"}, "covers"))):
        d = tmp_path / f"case{k}"
        d.mkdir()
        (d / "run.json").write_text(json.dumps({**rec, **patch}))
        with pytest.raises(InterpretabilityRefused, match=msg):
            load_run(d, CPU)


def test_a_changed_png_is_refused(runs, tmp_path):
    root = mini_render_root(tmp_path, n=9)
    reg = pd.read_parquet(root / "render_registry.parquet")
    r = reg[(reg["split"] == "test") & (reg["render_id"] == "R0")].iloc[0]
    p = root / "shards" / r["image_path"]
    img = np.asarray(Image.open(p)).copy()
    img[5, 5] = [1, 2, 3]
    Image.fromarray(img).save(p)
    with pytest.raises(InterpretabilityRefused, match="does not match"):
        run_masking.run(runs["E1"], root, tmp_path / "o", n=6, device=CPU, test_mode=True)


def test_attribution_to_flip_auroc():
    wids = [f"w{i}" for i in range(8)]
    grid_mass = np.linspace(0.01, 0.4, 8)
    att = pd.DataFrame([{"run_id": "r", "method": "gradcam", "window_id": w, "render_id": "R0", "region": g,
                         "attribution_mass": (m if g == "grid" else 0.1)}
                        for w, m in zip(wids, grid_mass) for g in ("grid", "candles", "plot_background", "margins")])
    rows = []
    for i, w in enumerate(wids):
        for rid in ("R0", "R1", "R2", "R4", "R5", "R6"):
            flip = (rid == "R2" and i >= 5)                      # the windows that look most at the grid flip
            rows.append({"run_id": "r", "window_id": w, "render_id": rid, "pred_class": 1 if flip else 0})
    link = attribution_flip_auroc(att, pd.DataFrame(rows)).set_index(["region", "render_id"])
    assert link.loc[("grid", "R2"), "auroc"] == 1.0 and link.loc[("grid", "R2"), "n_flips"] == 3
    assert np.isnan(link.loc[("candles", "R5"), "auroc"])     # no flips -> undefined, reported as NaN
