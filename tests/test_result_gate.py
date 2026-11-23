"""Scientific result validation (run.json validation_status), the evaluation gate, and frozen-config safety.

No GPU: train.py runs on CPU on the team dummy registry with --no-pretrained and a step cap. Fixture data
is relabelled data_v1 so the real-data gate applies; nothing here is a result.
"""
import dataclasses
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
import validate_run as vr  # noqa: E402
from make_synthetic_predictions import fixture_dataset, planted_run  # noqa: E402
from src.common.config import ExperimentConfig  # noqa: E402
from src.evaluation.frozen import (FrozenConfigError, config_hash_at_seed0, frozen_config_info,  # noqa: E402
                                   run_matches_frozen, verify_frozen)
from src.evaluation.gate import EvaluationRefused, check_evaluable, owner_of  # noqa: E402

FLIPS = {r: 0.1 for r in ("R1", "R2", "R3", "R4", "R5", "R6")}


def make_run(root, cfg: ExperimentConfig, splits=("val",), extra=None):
    """A completed run in the team's run.json layout (scripts/train.py + src/tracking.py)."""
    windows, registry = fixture_dataset(data_version="data_v1", render_version="render_v1")
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(data / "windows.parquet", index=False)
    registry.to_parquet(data / "render_registry.parquet", index=False)
    rid = f"{cfg.experiment_id}_s{cfg.seed}_{cfg.sha256()[:8]}"
    pred = root / "predictions" / f"{rid}.parquet"
    pred.parent.mkdir(parents=True, exist_ok=True)
    planted_run(windows, cfg.experiment_id, cfg.seed, splits=splits, flips=FLIPS, run_id=rid).to_parquet(pred, index=False)
    rec = {"run_id": rid, **cfg.to_dict(), "config_sha256": cfg.sha256(), "git_commit": "abc", "git_dirty": False,
           "status": "complete", "wall_minutes": 1.0, "best_val_f1": 0.41, "best_epoch": 2, "epochs_run": 5,
           "prediction_paths": [str(pred)], "checkpoint_path": "best.pt", "technical_owner": "kaushik",
           "gpu_minutes": 1.0, "gpu_model": "cuda", "prediction_splits": list(splits),
           "validation_status": "pending", "max_steps": None, **(extra or {})}
    run_dir = root / "runs" / rid
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(rec))
    return {"data": data, "pred": pred, "run_dir": run_dir, "runs": root / "runs", "rid": rid}


def evaluate(r, tmp_path, split="val", allow_synthetic=False):
    return ev.evaluate([r["pred"]], r["data"], split, tmp_path / "eval", runs_dir=r["runs"],
                       allow_synthetic=allow_synthetic, n_boot=10)


def e1():
    return ExperimentConfig.from_yaml(str(ROOT / "experiments" / "E1.yaml"))


# ============================================================================ RESULT_VALIDATED gate
def test_pending_run_is_refused_then_accepted_after_owner_validation(tmp_path):
    r = make_run(tmp_path, e1())
    with pytest.raises(SystemExit, match="'pending', not RESULT_VALIDATED"):
        evaluate(r, tmp_path)
    with pytest.raises(SystemExit, match="pending"):
        evaluate(r, tmp_path, allow_synthetic=True)          # the development flag cannot admit real data
    with pytest.raises(vr.ValidationRefused, match="technical owner"):
        vr.validate(r["run_dir"], "Khushi", r["data"])        # E1 belongs to Kaushik (Section 5.1)
    rec = vr.validate(r["run_dir"], "Kaushik", r["data"])      # strict mode: clean tree, full data, data_v1
    assert rec["validation_status"] == "RESULT_VALIDATED" and rec["validation"]["technical_owner"] == "Kaushik"
    assert len(rec["validation"]["prediction_sha256"]) == 64
    assert len(evaluate(r, tmp_path)["metrics_long.csv"]) > 0


def test_changed_rejected_or_superseded_results_are_refused(tmp_path):
    r = make_run(tmp_path, e1())
    vr.validate(r["run_dir"], "Kaushik", r["data"])
    original = r["pred"].read_bytes()
    pd.read_parquet(r["pred"]).iloc[::-1].to_parquet(r["pred"], index=False)   # same rows, new bytes
    with pytest.raises(SystemExit, match="changed after RESULT_VALIDATED"):
        evaluate(r, tmp_path)
    r["pred"].write_bytes(original)
    evaluate(r, tmp_path)
    vr.validate(r["run_dir"], "Kaushik", reject="diverged at epoch 2")
    with pytest.raises(SystemExit, match="rejected"):
        evaluate(r, tmp_path)
    vr.validate(r["run_dir"], "Kaushik", supersede="E1 selection rejected at G4")
    rec = json.loads((r["run_dir"] / "run.json").read_text())
    assert rec["status"] == "superseded"                       # select_lambda.py skips superseded runs
    with pytest.raises(SystemExit, match="superseded"):
        evaluate(r, tmp_path)


@pytest.mark.parametrize("extra,failed", [({"max_steps": 2}, "full_data_no_step_cap"),
                                          ({"git_dirty": True}, "clean_git_tree"),
                                          ({"gpu_executor": "someone"}, "no_logistics_fields"),
                                          ({"lane": "P"}, "no_logistics_fields")])
def test_validation_refuses_debug_dirty_and_logistics(tmp_path, extra, failed):
    r = make_run(tmp_path, e1(), extra=extra)
    with pytest.raises(vr.ValidationRefused, match=failed):
        vr.validate(r["run_dir"], "Kaushik", r["data"])


def test_logistics_fields_never_pass_the_evaluation_gate(tmp_path):
    r = make_run(tmp_path, e1())
    rec = vr.validate(r["run_dir"], "Kaushik", r["data"])
    with pytest.raises(EvaluationRefused, match="logistics"):
        check_evaluable({**rec, "backup_executor": "x"}, r["pred"], False)


def test_owner_table_follows_the_plan():
    assert [owner_of(e) for e in ("E1", "E3b", "E3", "E2", "E4b", "E4", "E5", "B0-random")] == \
        ["Kaushik"] * 3 + ["Khushi"] * 3 + ["Farzaan"] * 2


# ============================================================================ frozen-config safety
def freeze(tmp_path, *args):
    out = tmp_path / "configs" / "frozen" / "E1.yaml"
    freeze_config.main(["--config", str(ROOT / "experiments" / "E1.yaml"), "--set", "lr=1e-4", "weight_decay=0.01",
                        "--rule", "D11 test", "--out", str(out), *args])
    return out


def test_test_predictions_need_a_verified_frozen_config(tmp_path):
    r = make_run(tmp_path / "a", e1(), splits=("val", "test"))           # unfrozen config with test rows
    with pytest.raises(vr.ValidationRefused, match="test_predictions_from_frozen_config"):
        vr.validate(r["run_dir"], "Kaushik", r["data"])
    fz = freeze(tmp_path)
    cfg = dataclasses.replace(ExperimentConfig.from_yaml(str(fz)), seed=1)
    r = make_run(tmp_path / "b", cfg, splits=("val", "test"), extra=frozen_config_info(fz))
    rec = vr.validate(r["run_dir"], "Kaushik", r["data"])
    assert rec["validation"]["checks"]["test_predictions_from_frozen_config"] is True
    assert len(evaluate(r, tmp_path, split="test")["metrics_long.csv"]) > 0
    fz.write_text(fz.read_text().replace("lr: 0.0001", "lr: 0.0003"))   # edit after freezing
    with pytest.raises(FrozenConfigError, match="edited after freezing"):
        verify_frozen(fz)


def test_frozen_files_are_immutable_verified_and_carry_provenance(tmp_path):
    fz = freeze(tmp_path)
    rec = verify_frozen(fz)
    raw = yaml.safe_load(fz.read_text())
    assert raw["frozen"] is True and raw["phase"] == "final" and raw["seed"] == 0
    for k in ("config_sha256", "experiment_id", "data_version", "render_version", "git_commit",
              "selected_hyperparameters", "rule", "frozen_at"):
        assert k in rec
    assert rec["selected_hyperparameters"]["lr"] == 1e-4
    assert rec["config_sha256"] == config_hash_at_seed0(ExperimentConfig.from_yaml(str(fz)))
    freeze(tmp_path)                                                   # identical content: no-op
    with pytest.raises(FrozenConfigError, match="immutable"):
        freeze_config.main(["--config", str(ROOT / "experiments" / "E1.yaml"), "--set", "lr=3e-4",
                            "--rule", "x", "--out", str(fz)])
    with pytest.raises(FrozenConfigError, match="frozen/"):
        freeze_config.main(["--config", str(ROOT / "experiments" / "E1.yaml"), "--rule", "x",
                            "--out", str(tmp_path / "elsewhere" / "E1.yaml")])
    with pytest.raises(SystemExit, match="may only change"):
        freeze_config.main(["--config", str(ROOT / "experiments" / "E1.yaml"), "--set", "epochs_max=3",
                            "--rule", "x", "--out", str(tmp_path / "frozen" / "E1.yaml")])
    assert frozen_config_info(ROOT / "experiments" / "E1.yaml") is None          # experiments/ is never frozen
    (tmp_path / "configs" / "frozen" / "E3.yaml").write_text((ROOT / "experiments" / "E3.yaml").read_text())
    with pytest.raises(FrozenConfigError, match="not a frozen config"):
        frozen_config_info(tmp_path / "configs" / "frozen" / "E3.yaml")          # fail closed inside frozen/


def test_freeze_from_selected_run(tmp_path):
    r = make_run(tmp_path, dataclasses.replace(e1(), lr=1e-4))
    out = tmp_path / "frozen" / "E1.yaml"
    freeze_config.main(["--from-run", str(r["run_dir"]), "--rule", "D11", "--out", str(out)])
    rec = verify_frozen(out)
    assert rec["source_run_id"] == r["rid"] and rec["selected_hyperparameters"]["lr"] == 1e-4
    run = json.loads((r["run_dir"] / "run.json").read_text())
    assert run_matches_frozen({**run, "frozen": True, "phase": "final", "frozen_config_sha256": rec["config_sha256"]})
    r2 = make_run(tmp_path / "s1", dataclasses.replace(e1(), seed=1))
    with pytest.raises(SystemExit, match="seed-0"):
        freeze_config.main(["--from-run", str(r2["run_dir"]), "--rule", "D11", "--out", str(tmp_path / "frozen" / "X.yaml")])


# ============================================================================ train.py guard (CPU, dummy registry)
@pytest.fixture(scope="module")
def dummy(tmp_path_factory):
    root = tmp_path_factory.mktemp("dummy")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_dummy_registry.py"), "--out", str(root), "--n", "36"],
                   check=True, capture_output=True)
    return root


def train(config, dummy, out):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(config), "--data-root",
                           str(dummy), "--out", str(out), "--no-pretrained", "--max-steps", "1", "--device", "cpu"],
                          capture_output=True, text=True, cwd=ROOT)


def run_record(out):
    (d,) = list((out / "runs").glob("*_s*_*"))
    return json.loads((d / "run.json").read_text())


def test_train_py_unfrozen_config_predicts_validation_only(dummy, tmp_path):
    p = train(ROOT / "experiments" / "E1.yaml", dummy, tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    rec = run_record(tmp_path)
    preds = pd.read_parquet(rec["prediction_paths"][0])
    assert set(preds["split"]) == {"val"} and rec["prediction_splits"] == ["val"]
    assert rec["validation_status"] == "pending" and rec["max_steps"] == 1 and "frozen_config_sha256" not in rec
    assert not [k for k in rec if "executor" in k or k in ("lane", "queue_id")]


def test_train_py_frozen_config_predicts_test_and_records_provenance(dummy, tmp_path):
    cfg_yaml = tmp_path / "E5.yaml"
    cfg_yaml.write_text(yaml.safe_dump({"experiment_id": "E5", "model": "numeric1d", "loss": "ce", "lr": 1e-3,
                                        "weight_decay": 0.01, "images_per_step": 16, "epochs_max": 2}))
    fz = tmp_path / "configs" / "frozen" / "E5.yaml"
    freeze_config.main(["--config", str(cfg_yaml), "--rule", "D11 test", "--out", str(fz)])
    p = train(fz, dummy, tmp_path / "out")
    assert p.returncode == 0, p.stderr[-2000:]
    rec = run_record(tmp_path / "out")
    assert set(pd.read_parquet(rec["prediction_paths"][0])["split"]) == {"val", "test"}
    assert rec["frozen_config_sha256"] == verify_frozen(fz)["config_sha256"] and run_matches_frozen(rec)
    fz.write_text(fz.read_text().replace("lr: 0.001", "lr: 0.003"))
    p = train(fz, dummy, tmp_path / "out2")
    assert p.returncode != 0 and "edited after freezing" in p.stderr
