"""B0 trivial baselines (review: Farzaan, technical owner; Parth for the gate wiring). CPU only; synthetic
windows; not results."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import freeze_config  # noqa: E402
import freeze_results  # noqa: E402
import run_b0  # noqa: E402
import validate_run as vr  # noqa: E402
from src.baselines import B0_SEEDS, EPS, b0_predictions, momentum_thresholds, past_z, train_class_freq  # noqa: E402
from src.common.schema import NUMERIC_RENDER_ID, validate_prediction_frame  # noqa: E402
from src.data.labels import sigma_input  # noqa: E402

EXP = ROOT / "experiments"


def synthetic_root(tmp, n=300, seed=0):
    """windows.parquet + ohlcv.npy in the data_v1 layout with random-walk closes (not real data)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, 30)), axis=1))
    ohlcv = np.stack([close, close * 1.01, close * 0.99, close, rng.lognormal(10, 0.3, (n, 30))], -1).astype(np.float32)
    split = np.array(["train"] * (n // 2) + ["val"] * (n // 4) + ["test"] * (n - n // 2 - n // 4))
    w = pd.DataFrame({"window_id": [f"T_{i:04d}" for i in range(n)], "row_idx": np.arange(n), "split": split,
                      "label": rng.choice(3, n, p=[0.3, 0.45, 0.25]),
                      "sigma_30": sigma_input(ohlcv[:, :, 3].astype(np.float64)), "data_version": "data_v1"})
    root = Path(tmp) / "data"
    root.mkdir(parents=True, exist_ok=True)
    w.to_parquet(root / "windows.parquet", index=False)
    np.save(root / "ohlcv.npy", ohlcv)
    return root, w, ohlcv


def test_majority_uses_train_frequencies_only(tmp_path):
    _, w, x = synthetic_root(tmp_path)
    freq = train_class_freq(w)
    df = b0_predictions("B0-majority", "r", 0, w, None, ("val", "test"))
    validate_prediction_frame(df, image_model=False)
    np.testing.assert_allclose(df[["p_down", "p_neutral", "p_up"]].to_numpy(), np.tile(freq, (len(df), 1)), atol=1e-9)
    assert (df["pred_class"] == freq.argmax()).all() and (df["render_id"] == NUMERIC_RENDER_ID).all()
    moved = w.copy()
    moved.loc[moved["split"] != "train", "label"] = 0                    # val/test labels must not matter
    same = b0_predictions("B0-majority", "r", 0, moved, None, ("val", "test"))
    np.testing.assert_array_equal(same["pred_class"], df["pred_class"])


def test_stratified_random_is_seeded_and_follows_train_frequencies(tmp_path):
    _, w, _ = synthetic_root(tmp_path, n=6000)
    a = b0_predictions("B0-random", "r", 0, w, None, ("val", "test"))
    assert a.equals(b0_predictions("B0-random", "r", 0, w, None, ("val", "test")))
    assert (a["pred_class"] != b0_predictions("B0-random", "r", 1, w, None, ("val", "test"))["pred_class"]).any()
    share = np.bincount(a["pred_class"], minlength=3) / len(a)
    np.testing.assert_allclose(share, train_class_freq(w), atol=0.03)
    p = a[["p_down", "p_neutral", "p_up"]].to_numpy()
    np.testing.assert_allclose(p.max(1), 1 - EPS, rtol=1e-6)            # hard prediction, finite logits
    assert np.isfinite(a[["logit_down", "logit_neutral", "logit_up"]].to_numpy()).all()


def test_momentum_rule_is_the_label_construction_run_backwards(tmp_path):
    _, w, x = synthetic_root(tmp_path)
    z = past_z(w, x)
    c = x[:, :, 3].astype(np.float64)
    np.testing.assert_allclose(z, (c[:, -1] / c[:, -6] - 1) / (w["sigma_30"].to_numpy() * np.sqrt(5)))
    m1, m2 = momentum_thresholds(w, x)
    np.testing.assert_allclose([m1, m2], np.quantile(z[w["split"] == "train"], [1 / 3, 2 / 3]))
    df = b0_predictions("B0-momentum", "r", 0, w, x, ("val", "test")).set_index("window_id")
    zz = pd.Series(z, index=w["window_id"]).loc[df.index]
    want = np.where(zz <= m1, 0, np.where(zz > m2, 2, 1))
    np.testing.assert_array_equal(df["pred_class"].to_numpy(), want)
    x2 = x.copy()
    x2[(w["split"] != "train").to_numpy()] *= 1.7                         # val/test prices leave thresholds alone
    assert momentum_thresholds(w, x2) == (m1, m2)
    flat = w.iloc[:1].assign(sigma_30=0.0)
    assert past_z(flat, x)[0] == 0.0                                      # flat window -> 0, as in the label code


def test_run_b0_writes_val_only_unless_frozen_and_passes_the_owner_gate(tmp_path):
    root, w, _ = synthetic_root(tmp_path)
    out = tmp_path / "out"
    assert run_b0.main(["--config", str(EXP / "B0-momentum.yaml"), "--data-root", str(root), "--out", str(out)]) == 0
    dev = json.loads(next((out / "runs").glob("*/run.json")).read_text())
    assert dev["prediction_splits"] == ["val"] and dev["technical_owner"] == "Farzaan" and dev["gpu_model"] == "cpu"
    assert "momentum_thresholds" in dev["b0_fitted_on_train"]
    with pytest.raises(SystemExit, match="not in the plan's seeds"):
        run_b0.main(["--config", str(EXP / "B0-majority.yaml"), "--data-root", str(root), "--out", str(out),
                     "--seed", "1"])
    with pytest.raises(SystemExit, match="not a B0 baseline"):
        run_b0.main(["--config", str(EXP / "E1.yaml"), "--data-root", str(root), "--out", str(out)])

    frozen_dir = tmp_path / "frozen"
    for exp in B0_SEEDS:
        freeze_config.main(["--config", str(EXP / f"{exp}.yaml"), "--rule", "B0: no tuning (plan 5.1)",
                            "--out", str(frozen_dir / f"{exp}.yaml")])
        for seed in B0_SEEDS[exp]:
            run_b0.main(["--config", str(frozen_dir / f"{exp}.yaml"), "--data-root", str(root), "--out", str(out),
                         "--seed", str(seed)])
    finals = [r for r in (json.loads(p.read_text()) for p in (out / "runs").glob("*/run.json"))
              if r.get("frozen_config_path")]
    assert sorted((r["experiment_id"], r["seed"]) for r in finals) == sorted(
        (e, s) for e, seeds in B0_SEEDS.items() for s in seeds)
    for r in finals:
        assert r["prediction_splits"] == ["val", "test"]
        signed = vr.validate(out / "runs" / r["run_id"], "Farzaan", root, test_mode=True)
        assert signed["validation_status"] == "RESULT_VALIDATED"
        strict, _ = vr.run_checks(signed, root, test_mode=False)
        assert strict["pretrained_backbone"] is True and strict["test_predictions_from_frozen_config"] is True
    with pytest.raises(vr.ValidationRefused, match="technical owner"):
        vr.validate(out / "runs" / finals[0]["run_id"], "Parth", root, test_mode=True)

    entries, missing = freeze_results.build_manifest(out / "runs", frozen_dir, list(B0_SEEDS), test_mode=True)
    assert not missing and len(entries) == 5
    rec = finals[0]
    rec = json.loads((out / "runs" / rec["run_id"] / "run.json").read_text())
    fpath = Path(rec["frozen_config_path"])
    why = freeze_results.problems(rec, fpath, rec["frozen_config_sha256"], test_mode=False)
    assert not any("D16" in x or "D9" in x for x in why)                  # B0 runs on CPU, no backbone
    man = tmp_path / "freeze.json"
    assert freeze_results.main(["--runs-dir", str(out / "runs"), "--frozen-dir", str(frozen_dir), "--out", str(man),
                                "--experiments", "B0-majority", "--include-b0", "--write", "--test-mode"]) == 0
    assert len(json.loads(man.read_text())["runs"]) == 5


def test_image_models_still_need_the_t4_and_pretrained_weights():
    from src.evaluation.gate import has_image_backbone
    assert has_image_backbone({"experiment_id": "E3"}) and has_image_backbone({"experiment_id": "E2"})
    assert not has_image_backbone({"experiment_id": "E5"}) and not has_image_backbone({"experiment_id": "B0-random"})
