"""D16 (amended): reported image-model runs on the NVIDIA GeForce RTX 5090 only, E5/B0 on CPU; and git_dirty
counts code, not the pipeline's own outputs (review: Parth; Kaushik for train.py / tracking.py). CPU only."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402
import torch  # noqa: E402

import freeze_results  # noqa: E402
import train as train_script  # noqa: E402
from src import tracking  # noqa: E402
from src.evaluation.gate import D16_GPU, is_d16_gpu  # noqa: E402

EXP = ROOT / "experiments"


@pytest.mark.parametrize("name,ok", [
    ("NVIDIA GeForce RTX 5090", True), ("nvidia geforce rtx 5090", True), ("NVIDIA  GeForce  RTX 5090 ", True),
    ("GeForce RTX 5090", True), ("Tesla T4", False), ("NVIDIA GeForce RTX 5090 Laptop GPU", False),
    ("NVIDIA GeForce RTX 5080", False), ("NVIDIA A100-SXM4-40GB", False), ("cpu", False), ("", False), (None, False)])
def test_d16_device_identity(name, ok):
    assert is_d16_gpu(name) is ok
    assert D16_GPU == "NVIDIA GeForce RTX 5090"


def record(exp, gpu):
    return {"experiment_id": exp, "gpu_model": gpu, "git_dirty": False, "max_steps": None, "data_version": "data_v1",
            "pretrained": True, "validation_status": "RESULT_VALIDATED", "status": "complete"}


@pytest.mark.parametrize("exp,gpu,d16_problem", [
    ("E1", "NVIDIA GeForce RTX 5090", False), ("E4", "NVIDIA GeForce RTX 5090", False),
    ("E2", "Tesla T4", True), ("E3b", "cpu", True), ("E3", "NVIDIA GeForce RTX 5090 Laptop GPU", True),
    ("E5", "cpu", False), ("B0-momentum", "cpu", False), ("E5", "NVIDIA GeForce RTX 5090", True),
    ("B0-random", "Tesla T4", True)])
def test_results_freeze_applies_d16(tmp_path, exp, gpu, d16_problem):
    why = freeze_results.problems(record(exp, gpu), tmp_path / "x.yaml", "sha", test_mode=False)
    assert any("D16" in w for w in why) is d16_problem, why


def test_train_refuses_image_models_off_the_d16_gpu_and_e5_on_any_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "Tesla T4")
    with pytest.raises(SystemExit, match="not the D16 GPU"):
        train_script.main(["--config", str(EXP / "E1.yaml"), "--data-root", str(tmp_path), "--device", "cuda",
                           "--out", str(tmp_path / "o")])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "NVIDIA GeForce RTX 5090")
    with pytest.raises(SystemExit, match="E5 runs on CPU"):
        train_script.main(["--config", str(EXP / "E5.yaml"), "--set", "weight_decay=0.01", "--data-root",
                           str(tmp_path), "--device", "cuda", "--out", str(tmp_path / "o")])


def git(repo, *args):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo, check=True,
                   capture_output=True)


def test_git_dirty_ignores_pipeline_outputs_but_not_code(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "init")
    assert tracking.git_info(repo)["git_dirty"] is False
    for p in ("outputs/runs/E1_s0_x/run.json", "outputs/predictions/E1_s0_x.parquet", "outputs/smoke/E1/r.json",
              "configs/frozen/E1.yaml"):
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text("{}")
    info = tracking.git_info(repo)
    assert info["git_dirty"] is False and len(info["git_commit"]) == 40      # a run's own outputs are not code
    (repo / "src" / "b.py").write_text("y = 2\n")                            # untracked code -> dirty
    assert tracking.git_info(repo)["git_dirty"] is True
    (repo / "src" / "b.py").unlink()
    (repo / "src" / "a.py").write_text("x = 2\n")                            # modified tracked code -> dirty
    assert tracking.git_info(repo)["git_dirty"] is True


def test_git_dirty_fails_closed_without_git(tmp_path):
    assert tracking.git_info(tmp_path)["git_dirty"] is True                  # not a repository -> cannot be clean


def test_cpu_environment_info_has_no_gpu_fields():
    info = tracking.environment_info("cpu")
    assert info["gpu_model"] == "cpu" and "gpu_total_memory_gib" not in info
    json.dumps(info)
