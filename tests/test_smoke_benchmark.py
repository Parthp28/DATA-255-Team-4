"""P00/H00 tooling on CPU with the dummy registry (review: Kaushik for P00 / ResNet, Khushi for H00 / DeiT).

The real P00/H00 run on the D16 GPU with pretrained weights and real data after G2; here every check runs
end to end on CPU with --no-pretrained. The numeric E5 model proves the overfit criterion can pass; the
image models run a few overfit steps so the suite stays fast, and their other checks must pass."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402

import benchmark_throughput  # noqa: E402
import smoke_test  # noqa: E402

EXP = ROOT / "experiments"


@pytest.fixture(scope="module")
def dummy(tmp_path_factory):
    root = tmp_path_factory.mktemp("dummy")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_dummy_registry.py"), "--out", str(root), "--n", "48"],
                   check=True, capture_output=True)
    return root


def smoke(cfg, dummy, out, *extra):
    rc = smoke_test.main(["--config", str(EXP / cfg), "--data-root", str(dummy), "--out", str(out), "--device", "cpu",
                          "--no-pretrained", *extra])
    return rc, json.loads((out / "smoke_report.json").read_text())


def test_numeric_smoke_passes_every_check_including_the_200_step_overfit(dummy, tmp_path):
    rc, rep = smoke("E5.yaml", dummy, tmp_path, "--set", "weight_decay=0.01")
    assert rc == 0 and rep["pass"] and set(rep["checks"]) == {"finite_step", "overfit", "resume", "prediction_export"}
    assert rep["checks"]["overfit"]["train_accuracy"] == 1.0 and rep["checks"]["overfit"]["images"] == 32
    assert rep["checks"]["resume"]["weights_match"] and rep["gpu_model"] == "cpu" and rep["device"] == "cpu"


@pytest.mark.parametrize("cfg,owner", [("E1.yaml", "P00 ResNet-18"), ("E2.yaml", "H00 DeiT-S"),
                                       ("E3.yaml", "paired ResNet + JS"), ("E4.yaml", "paired DeiT + JS")])
def test_image_smoke_paths_run_end_to_end(dummy, tmp_path, cfg, owner):
    rc, rep = smoke(cfg, dummy, tmp_path, "--overfit-steps", "2", "--train-windows", "16", "--eval-windows", "4")
    c = rep["checks"]
    assert c["finite_step"]["pass"] and c["resume"]["pass"] and c["prediction_export"]["pass"], owner
    assert c["prediction_export"]["rows"] == 4 * 7 and c["overfit"]["images"] == 32
    assert rc == (0 if rep["pass"] else 1)                   # 2 overfit steps cannot reach 100%: exit code says so
    assert rep["pretrained"] is False and rep["amp"] is False


def test_smoke_rerun_into_the_same_out_dir_starts_clean(dummy, tmp_path):
    """Re-running P00/H00 after a failure is routine: the second run must not resume from the first run's
    last.pt, and a --set change must not be refused as a configuration mismatch."""
    args = ("--overfit-steps", "2", "--train-windows", "16", "--eval-windows", "4")
    _, first = smoke("E1.yaml", dummy, tmp_path, *args)
    _, again = smoke("E1.yaml", dummy, tmp_path, *args)
    _, changed = smoke("E1.yaml", dummy, tmp_path, *args, "--set", "lr=1e-4")
    for rep in (first, again, changed):
        r = rep["checks"]["resume"]
        assert r["pass"] and r["steps_epoch0"] > 0 and r["steps_resumed"] > 0
    assert again["checks"]["resume"]["max_abs_loss_diff"] == first["checks"]["resume"]["max_abs_loss_diff"]


def test_smoke_refuses_e5_without_a_weight_decay_decision(dummy, tmp_path):
    with pytest.raises(SystemExit, match="open pre-E5"):
        smoke("E5.yaml", dummy, tmp_path)


@pytest.mark.parametrize("cfg", ["E1.yaml", "E2.yaml"])
def test_benchmark_reports_throughput_metadata_on_cpu(dummy, tmp_path, cfg):
    out = tmp_path / "tp.json"
    benchmark_throughput.main(["--config", str(EXP / cfg), "--data-root", str(dummy), "--batch", "2", "--steps", "1",
                               "--warmup", "0", "--workers", "0", "--device", "cpu", "--out", str(out)])
    rep = json.loads(out.read_text())
    assert rep["train_img_s"] > 0 and rep["eval_img_s"] > 0 and rep["loader_img_s"] > 0
    assert rep["gpu_model"] == "cpu" and rep["amp"] is False and "ratio_to_a1" in rep and "rescale_queue" in rep
