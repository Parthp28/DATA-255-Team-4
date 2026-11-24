"""Training-pipeline regressions on the dummy registry, CPU only, --no-pretrained (nothing downloaded):
DeiT-S through the shared trainer (review: Khushi, Kaushik), tuning / inheritance overrides, the E5
weight-decay decision gate, and the D11 path select -> freeze -> promote -> seeds -> validate -> results
freeze -> evaluate (review: Kaushik, Parth). Dummy data: software checks, not results."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

import evaluate as ev  # noqa: E402
import freeze_config  # noqa: E402
import freeze_results  # noqa: E402
import predict as predict_script  # noqa: E402
import select_config  # noqa: E402
import train as train_script  # noqa: E402
import validate_run as vr  # noqa: E402
from src.common.schema import RENDER_IDS  # noqa: E402
from src.evaluation.frozen import FrozenConfigError, verify_frozen  # noqa: E402

EXP = ROOT / "experiments"


@pytest.fixture(scope="module")
def dummy(tmp_path_factory):
    root = tmp_path_factory.mktemp("dummy")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_dummy_registry.py"), "--out", str(root), "--n", "36"],
                   check=True, capture_output=True)
    return root


def train(cfg, dummy, out, *extra):
    argv = ["--config", str(cfg), "--data-root", str(dummy), "--out", str(out), "--no-pretrained", "--device", "cpu",
            *extra]
    train_script.main(argv)
    recs = [json.loads(p.read_text()) for p in sorted((Path(out) / "runs").glob("*/run.json"))]
    return recs


def test_every_experiment_config_encodes_the_plan_matrix():
    from src.common.config import ExperimentConfig
    expect = {"E1": ("resnet18.a1_in1k", ["R0"], "none", "ce"),
              "E2": ("deit_small_patch16_224.fb_in1k", ["R0"], "none", "ce"),
              "E3b": ("resnet18.a1_in1k", ["R0", "R1", "R2", "R4"], "pair_distinct", "paired_ce"),
              "E4b": ("deit_small_patch16_224.fb_in1k", ["R0", "R1", "R2", "R4"], "pair_distinct", "paired_ce"),
              "E3": ("resnet18.a1_in1k", ["R0", "R1", "R2", "R4"], "pair_distinct", "ce_plus_js"),
              "E4": ("deit_small_patch16_224.fb_in1k", ["R0", "R1", "R2", "R4"], "pair_distinct", "ce_plus_js"),
              "E5": ("numeric1d", ["NUM"], "none", "ce")}
    for e, (model, renders, pairing, loss) in expect.items():
        c = ExperimentConfig.from_yaml(str(EXP / f"{e}.yaml"))
        assert (c.experiment_id, c.model, c.train_renders, c.pairing, c.loss) == (e, model, renders, pairing, loss)
        assert (c.images_per_step, c.epochs_max, c.patience, c.grad_clip, c.amp, c.image_size) == (128, 10, 3, 1.0, True, 224)
        assert c.early_stop_metric == "val_macro_f1_own_renders" and c.frozen is False and c.phase == "dev"
        assert c.windows_per_step == (64 if pairing == "pair_distinct" else 128)          # 128 images per step
        assert (c.consistency_lambda > 0) == (loss == "ce_plus_js")
    assert ExperimentConfig.from_yaml(str(EXP / "E5.yaml")).weight_decay is None          # open pre-E5 decision


def test_e2_deit_trains_through_the_shared_trainer(dummy, tmp_path):
    (rec,) = train(EXP / "E2.yaml", dummy, tmp_path, "--max-steps", "1")
    assert rec["model"] == "deit_small_patch16_224.fb_in1k" and rec["technical_owner"] == "Khushi"
    assert rec["gpu_model"] == "cpu" and rec["device"] == "cpu" and rec["pretrained"] is False
    assert rec["torch_version"] and rec["timm_version"] and rec["gpu_minutes"] == 0.0
    assert Path(rec["last_checkpoint_path"]).exists() and rec["prediction_splits"] == ["val"]
    preds = pd.read_parquet(rec["prediction_paths"][0])
    assert set(preds["render_id"]) == set(RENDER_IDS)
    assert ((preds[["p_down", "p_neutral", "p_up"]].sum(axis=1) - 1).abs() < 1e-5).all()   # 3-class softmax
    assert not [k for k in rec if "executor" in k or k in ("lane", "queue_id")]


def test_e4_paired_deit_inherits_frozen_e2_and_sets_lambda(dummy, tmp_path):
    fz = tmp_path / "configs" / "frozen" / "E2.yaml"
    freeze_config.main(["--config", str(EXP / "E2.yaml"), "--set", "lr=5e-5", "weight_decay=0.01", "--rule", "test",
                        "--out", str(fz)])
    (rec,) = train(EXP / "E4.yaml", dummy, tmp_path / "out", "--inherit", str(fz), "--set", "consistency_lambda=0.3",
                   "--max-steps", "1")
    assert (rec["lr"], rec["weight_decay"], rec["consistency_lambda"]) == (5e-5, 0.01, 0.3)
    assert rec["inherit_config_sha256"] == verify_frozen(fz)["config_sha256"]
    assert rec["overrides"] == {"consistency_lambda": 0.3, "lr": 5e-5, "weight_decay": 0.01}
    assert rec["technical_owner"] == "Khushi" and rec["pairing"] == "pair_distinct"


def test_overrides_are_restricted(dummy, tmp_path):
    with pytest.raises(SystemExit, match="may only change"):
        train(EXP / "E1.yaml", dummy, tmp_path, "--set", "epochs_max=3", "--max-steps", "1")
    with pytest.raises(FrozenConfigError):
        train(EXP / "E3.yaml", dummy, tmp_path, "--inherit", str(EXP / "E1.yaml"), "--max-steps", "1")
    fz = tmp_path / "frozen" / "E1.yaml"
    freeze_config.main(["--config", str(EXP / "E1.yaml"), "--rule", "test", "--out", str(fz)])
    with pytest.raises(SystemExit, match="frozen config is final"):
        train(fz, dummy, tmp_path, "--set", "lr=1e-3", "--max-steps", "1")
    with pytest.raises(ValueError):                        # E3 needs lambda > 0 (team config validation)
        train(EXP / "E3.yaml", dummy, tmp_path, "--set", "consistency_lambda=0", "--max-steps", "1")


def test_e5_refuses_to_run_until_its_weight_decay_is_decided(dummy, tmp_path):
    with pytest.raises(SystemExit, match="open pre-E5"):
        train(EXP / "E5.yaml", dummy, tmp_path)


def test_d11_select_freeze_promote_seeds_validate_freeze_evaluate(dummy, tmp_path):
    out = tmp_path / "out"
    for lr in ("1e-3", "3e-3"):                            # E5 D11 grid; weight decay set for this test only
        train(EXP / "E5.yaml", dummy, out, "--set", f"lr={lr}", "weight_decay=0.01")
    tune = [json.loads(p.read_text()) for p in (out / "runs").glob("*/run.json")]
    assert len(tune) == 2 and all(r["prediction_splits"] == ["val"] for r in tune)
    frozen = tmp_path / "configs" / "frozen" / "E5.yaml"
    select_config.main(["--experiment", "E5", "--runs-dir", str(out / "runs"), "--out", str(frozen),
                        "--log-dir", str(tmp_path / "sel"), "--test-mode"])
    best = max(tune, key=lambda r: (r["best_val_f1"], [-ord(c) for c in r["run_id"]]))
    frec = verify_frozen(frozen)
    assert frec["source_run_id"] == best["run_id"] and frec["selected_hyperparameters"]["lr"] == best["lr"]
    assert json.loads((tmp_path / "sel" / "SEL-E5.json").read_text())["metrics_used"] == "validation only"
    other = next(r for r in tune if r["run_id"] != best["run_id"])
    with pytest.raises(FrozenConfigError, match="not trained with"):
        predict_script.promote(frozen, dummy, run_dir=out / "runs" / other["run_id"])
    s0 = predict_script.promote(frozen, dummy, runs_root=out / "runs")
    assert s0["status"] == "promoted" and s0["prediction_splits"] == ["test", "val"]
    for seed in ("1", "2"):
        train(frozen, dummy, out, "--seed", seed)
    finals = {r["seed"]: r for r in (json.loads(p.read_text()) for p in (out / "runs").glob("*/run.json"))
              if r.get("frozen_config_path")}
    assert sorted(finals) == [0, 1, 2]
    assert len({r["frozen_config_science_sha256"] for r in finals.values()}) == 1      # one config, three seeds
    for r in finals.values():
        vr.validate(out / "runs" / r["run_id"], "Farzaan", dummy, test_mode=True)
    entries, missing = freeze_results.build_manifest(out / "runs", frozen.parent, ["E5"], test_mode=True)
    assert not missing and sorted(e["seed"] for e in entries) == [0, 1, 2]
    assert not [k for e in entries for k in e if "executor" in k or k == "lane"]
    _, strict = freeze_results.build_manifest(out / "runs", frozen.parent, ["E5"])
    assert strict == ["E5: frozen in test mode"]
    s1 = json.loads((out / "runs" / finals[1]["run_id"] / "run.json").read_text())
    why = freeze_results.problems(s1, frozen, frec["config_sha256"], test_mode=False)
    assert not any("D16" in w for w in why)              # plan 5.1: E5 runs on CPU; only image models need the D16 GPU
    other_gpu = {**s1, "gpu_model": "NVIDIA A100-SXM4-40GB"}
    assert any("D16" in w for w in freeze_results.problems(other_gpu, frozen, frec["config_sha256"], test_mode=False))
    man = tmp_path / "results_freeze.json"
    assert freeze_results.main(["--runs-dir", str(out / "runs"), "--frozen-dir", str(frozen.parent), "--out", str(man),
                                "--experiments", "E5", "--write", "--test-mode"]) == 0
    tables = ev.evaluate(ev.files_from_freeze(man), dummy, "test", tmp_path / "eval", runs_dir=out / "runs", n_boot=10)
    assert set(tables["metrics_long.csv"]["run_id"]) == {e["run_id"] for e in entries}


def test_d11_refuses_incomplete_or_mixed_candidate_sets(tmp_path):
    from src.common.config import ExperimentConfig
    runs = tmp_path / "runs"
    for i, (lr, wd, ep) in enumerate([(1e-4, 0.01, 10), (1e-4, 0.05, 10), (3e-4, 0.01, 10)]):
        c = ExperimentConfig.from_yaml(str(EXP / "E1.yaml"))
        rec = {**c.to_dict(), "lr": lr, "weight_decay": wd, "epochs_max": ep, "run_id": f"E1_s0_{i:08d}",
               "status": "complete", "best_val_f1": 0.4 + i / 100, "git_dirty": False, "max_steps": None}
        (runs / rec["run_id"]).mkdir(parents=True)
        (runs / rec["run_id"] / "run.json").write_text(json.dumps(rec))
    with pytest.raises(select_config.SelectionError, match="expected 4"):
        select_config.select(runs, "E1")
    for rid, bad in (("E1_s0_dirty000", {"git_dirty": True}), ("E1_s0_debug000", {"max_steps": 2})):
        rec = json.loads((runs / "E1_s0_00000000" / "run.json").read_text())
        rec.update(run_id=rid, lr=3e-4, weight_decay=0.05, **bad)
        (runs / rid).mkdir()
        (runs / rid / "run.json").write_text(json.dumps(rec))
    with pytest.raises(select_config.SelectionError, match="dirty git tree.*debug run|debug run.*dirty git tree"):
        select_config.select(runs, "E1")                   # never selected outside --test-mode
    for rid in ("E1_s0_dirty000", "E1_s0_debug000"):
        (runs / rid / "run.json").unlink()
        (runs / rid).rmdir()
    rec = json.loads((runs / "E1_s0_00000002" / "run.json").read_text())
    rec.update(run_id="E1_s0_00000003", weight_decay=0.05, lr=3e-4, epochs_max=8)
    (runs / rec["run_id"]).mkdir()
    (runs / rec["run_id"] / "run.json").write_text(json.dumps(rec))
    with pytest.raises(select_config.SelectionError, match="differ in more than"):
        select_config.select(runs, "E1")
