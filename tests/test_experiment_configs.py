"""Every experiments/*.yaml against the plan's experiment matrix (5.1), the shared recipe D10, the tuning
grid D11 and the lambda grid D12/D13 (review: each experiment's owner; E5 Farzaan). Reads YAML only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402
import yaml  # noqa: E402

import select_config  # noqa: E402
from src.common.config import ExperimentConfig  # noqa: E402
from src.common.schema import SEEN_RENDERS  # noqa: E402

EXP = ROOT / "experiments"
RESNET, DEIT = "resnet18.a1_in1k", "deit_small_patch16_224.fb_in1k"
SEEN = list(SEEN_RENDERS)  # R0, R1, R2, R4 (D8)

# Plan 5.1: architecture, training renderings, pairing, loss; the only things that differ between experiments.
MATRIX = {
    "E1": (RESNET, ["R0"], "none", "ce"),
    "E2": (DEIT, ["R0"], "none", "ce"),
    "E3b": (RESNET, SEEN, "pair_distinct", "paired_ce"),
    "E4b": (DEIT, SEEN, "pair_distinct", "paired_ce"),
    "E3": (RESNET, SEEN, "pair_distinct", "ce_plus_js"),
    "E4": (DEIT, SEEN, "pair_distinct", "ce_plus_js"),
    "E5": ("numeric1d", ["NUM"], "none", "ce"),
}
D10 = {"images_per_step": 128, "epochs_max": 10, "patience": 3, "amp": True, "grad_clip": 1.0, "image_size": 224,
       "early_stop_metric": "val_macro_f1_own_renders"}
D11_LR = {"E1": {1e-4, 3e-4}, "E2": {5e-5, 1e-4}, "E5": {1e-3, 3e-3}}
D11_WD = {"E1": {0.01, 0.05}, "E2": {0.01, 0.05}}
D12_LAMBDA = {0.3, 1.0, 3.0}
PARENT = {"E3b": "E1", "E3": "E1", "E4b": "E2", "E4": "E2"}     # lr and weight decay inherited at freeze


def raw(exp):
    return yaml.safe_load((EXP / f"{exp}.yaml").read_text())


def test_the_directory_holds_exactly_the_plan_experiments():
    assert sorted(p.stem for p in EXP.glob("*.yaml")) == sorted(
        list(MATRIX) + ["B0-majority", "B0-random", "B0-momentum"])


@pytest.mark.parametrize("exp", list(MATRIX))
def test_matrix_and_shared_recipe(exp):
    r = raw(exp)
    model, renders, pairing, loss = MATRIX[exp]
    assert (r["experiment_id"], r["model"], r["train_renders"], r["pairing"], r["loss"]) == (exp, model, renders,
                                                                                            pairing, loss)
    assert {k: r[k] for k in D10} == D10, f"{exp} departs from the shared recipe D10"
    assert (r["seed"], r["phase"], r["frozen"], r["data_version"], r["render_version"]) == (
        0, "dev", False, "data_v1", "render_v1")
    assert set(r) <= set(ExperimentConfig.__dataclass_fields__), "unknown keys would be dropped silently"


@pytest.mark.parametrize("exp", ["E1", "E2", "E5"])
def test_base_values_sit_on_the_d11_grid(exp):
    r = raw(exp)
    assert r["lr"] in D11_LR[exp]
    if exp in D11_WD:
        assert r["weight_decay"] in D11_WD[exp]
    assert select_config.EXPECTED[exp] == len(D11_LR[exp]) * len(D11_WD.get(exp, {None}))


def test_paired_controls_and_lambda():
    for exp in ("E3b", "E4b"):
        assert raw(exp)["consistency_lambda"] == 0.0
    for exp in ("E3", "E4"):
        assert raw(exp)["consistency_lambda"] in D12_LAMBDA
    for exp in ("E1", "E2", "E5"):
        assert raw(exp)["consistency_lambda"] == 0.0
    for child, parent in PARENT.items():
        assert raw(child)["model"] == raw(parent)["model"]
        assert (raw(child)["lr"], raw(child)["weight_decay"]) == (raw(parent)["lr"], raw(parent)["weight_decay"])


def test_e5_is_the_numeric_control_with_its_one_open_decision():
    r = raw("E5")
    assert r["weight_decay"] is None, "E5 weight decay is an open pre-E5 decision; set it deliberately, then freeze"
    cfg = ExperimentConfig(**{**r, "weight_decay": 0.01})
    assert cfg.is_numeric and cfg.windows_per_step == 128


@pytest.mark.parametrize("exp", list(MATRIX))
def test_every_config_constructs(exp):
    r = raw(exp)
    if r["weight_decay"] is None:
        r["weight_decay"] = 0.0
    ExperimentConfig(**r)
