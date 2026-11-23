"""Shared-trainer regressions (review: Kaushik; loss wiring also Khushi): last.pt + exact resume (T-T6),
non-finite-loss guard, canonical JS loss as the default, early stopping on the mean over own renderings
(plan 5.1), and the E5 numeric input of plan 5.1. CPU only; tiny data; not results."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from make_synthetic_predictions import fixture_dataset  # noqa: E402
from src.common.config import ExperimentConfig  # noqa: E402
from src.models.numeric import Numeric1DCNN  # noqa: E402
from src.models.resnet import build_resnet  # noqa: E402
from src.training import consistency_ref  # noqa: E402
from src.training.datasets import ChartImageDataset, NumericWindowDataset, PairedChartDataset, numeric_input  # noqa: E402
from src.training.trainer import Trainer, canonical_loss  # noqa: E402

E5 = dict(experiment_id="E5", model="numeric1d", loss="ce", lr=1e-3, weight_decay=0.01,
          images_per_step=16, epochs_max=3, patience=3)


def numeric_data():
    windows, _ = fixture_dataset(data_version="data_v1", render_version="render_v1")
    rng = np.random.default_rng(1)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(windows), 30)), axis=1))
    o, v = c * np.exp(rng.normal(0, 0.003, c.shape)), rng.uniform(1e5, 1e6, c.shape)
    ohlcv = np.stack([o, np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998, c, v], axis=2).astype(np.float32)
    return windows, ohlcv


def run_numeric(out, resume=True, stop_after=None, epochs=3):
    windows, ohlcv = numeric_data()
    cfg = ExperimentConfig(**{**E5, "epochs_max": epochs})
    torch.manual_seed(0)
    np.random.seed(0)
    model = Numeric1DCNN()
    t = Trainer(model, cfg, torch.device("cpu"), str(out))
    loader = DataLoader(NumericWindowDataset(windows, ohlcv, "train"), batch_size=16, shuffle=True)
    res = t.fit(loader, NumericWindowDataset(windows, ohlcv, "val"), ["NUM"], str(out / "best.pt"),
                resume=resume, stop_after_epochs=stop_after)
    return t, res


def test_last_pt_resume_reproduces_the_uninterrupted_run(tmp_path):
    full, rf = run_numeric(tmp_path / "full", resume=False)
    first, r1 = run_numeric(tmp_path / "cut", stop_after=1)              # session dies after epoch 0
    assert r1.epochs_run == 1 and (tmp_path / "cut" / "last.pt").exists()
    resumed, r2 = run_numeric(tmp_path / "cut", resume=True)
    n = len(first.step_losses)
    np.testing.assert_allclose(resumed.step_losses, full.step_losses[n:], rtol=1e-6)
    for a, b in zip(full.model.state_dict().values(), resumed.model.state_dict().values()):
        assert torch.allclose(a.float(), b.float(), atol=1e-6)
    assert r2.epochs_run == rf.epochs_run == 3 and r2.best_epoch == rf.best_epoch
    assert [h["epoch"] for h in resumed.history] == [0, 1, 2]


def test_last_pt_holds_full_state_and_refuses_another_config(tmp_path):
    run_numeric(tmp_path, stop_after=1)
    ck = torch.load(tmp_path / "last.pt", weights_only=False)
    assert {"model", "optimizer", "scheduler", "scaler", "rng", "history", "config", "epoch", "step",
            "best_f1", "best_epoch", "patience_left"} <= set(ck)
    assert ck["epoch"] == 1 and ck["step"] > 0
    windows, ohlcv = numeric_data()
    other = ExperimentConfig(**{**E5, "lr": 3e-3})
    t = Trainer(Numeric1DCNN(), other, torch.device("cpu"), str(tmp_path))
    with pytest.raises(ValueError, match="different configuration"):
        t.fit(DataLoader(NumericWindowDataset(windows, ohlcv, "train"), batch_size=16),
              NumericWindowDataset(windows, ohlcv, "val"), ["NUM"], str(tmp_path / "best.pt"))


def test_non_finite_loss_fails_fast(tmp_path):
    windows, ohlcv = numeric_data()
    model = Numeric1DCNN()
    model.forward = lambda x: torch.full((x.shape[0], 3), float("nan"), requires_grad=True)
    t = Trainer(model, ExperimentConfig(**E5), torch.device("cpu"), str(tmp_path))
    with pytest.raises(FloatingPointError, match="non-finite"):
        t.fit(DataLoader(NumericWindowDataset(windows, ohlcv, "train"), batch_size=16),
              NumericWindowDataset(windows, ohlcv, "val"), ["NUM"], str(tmp_path / "best.pt"))


def test_trainer_uses_khushis_canonical_loss_and_matches_the_reference():
    t = Trainer(Numeric1DCNN(), ExperimentConfig(**E5), torch.device("cpu"), "unused")
    assert t.loss_fn is canonical_loss
    g = torch.Generator().manual_seed(0)
    la, lb = torch.randn(8, 3, generator=g), torch.randn(8, 3, generator=g)
    y = torch.randint(0, 3, (8,), generator=g)
    for mode, lam in (("ce", 0.0), ("paired_ce", 0.0), ("ce_plus_js", 1.0), ("ce_plus_js", 3.0)):
        ours, parts = canonical_loss(mode, lam, la, y, lb)
        ref, _ = consistency_ref.combined_loss(mode, lam, la, y, lb)
        assert torch.allclose(ours, ref, atol=1e-6), mode
    la.requires_grad_(True)
    total, parts = canonical_loss("paired_ce", 0.0, la, y, lb)
    assert parts["js"] > 0 and torch.equal(total, 0.5 * (torch.nn.functional.cross_entropy(la, y)
                                                          + torch.nn.functional.cross_entropy(lb, y)))


def test_early_stopping_metric_is_the_mean_of_per_render_macro_f1(monkeypatch):
    t = Trainer(Numeric1DCNN(), ExperimentConfig(**E5), torch.device("cpu"), "unused")
    # R0: perfect on 3 classes; R1: predicts one class for everything
    rows = [("w%d" % i, "R0", i % 3, None, None, i % 3) for i in range(9)] + \
           [("w%d" % i, "R1", i % 3, None, None, 0) for i in range(9)]
    monkeypatch.setattr(t, "_infer", lambda dataset, render_ids: rows)
    from sklearn.metrics import f1_score
    per = [f1_score([r[2] for r in rows if r[1] == k], [r[5] for r in rows if r[1] == k], average="macro")
           for k in ("R0", "R1")]
    pooled = f1_score([r[2] for r in rows], [r[5] for r in rows], average="macro")
    assert t.evaluate_macro_f1(None, ["R0", "R1"]) == pytest.approx(np.mean(per))
    assert not np.isclose(np.mean(per), pooled)


def test_e5_numeric_input_is_exactly_plan_5_1():
    w = np.zeros((30, 5), np.float32)
    w[:, 0], w[:, 1], w[:, 2], w[:, 3] = 9.0, 11.0, 8.0, np.linspace(9.0, 10.0, 30)
    w[:, 4] = np.arange(1, 31, dtype=np.float32) * 100
    x = numeric_input(w)
    assert x.shape == (5, 30) and x.dtype == np.float32
    np.testing.assert_allclose(x[3, -1], 0.0, atol=1e-7)                     # Close[d] / Close[d] - 1
    np.testing.assert_allclose(x[0], 9.0 / 10.0 - 1, rtol=1e-6)
    np.testing.assert_allclose(x[1], 11.0 / 10.0 - 1, rtol=1e-6)
    np.testing.assert_allclose(x[4], np.log1p(w[:, 4] / w[:, 4].mean()), rtol=1e-6)
    windows, ohlcv = numeric_data()
    ds = NumericWindowDataset(windows, ohlcv, "train")
    np.testing.assert_allclose(ds[0]["seq"].numpy(), numeric_input(ohlcv[int(ds.rows.iloc[0]["row_idx"])]))


@pytest.fixture(scope="module")
def dummy(tmp_path_factory):
    root = tmp_path_factory.mktemp("dummy")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_dummy_registry.py"), "--out", str(root), "--n", "36"],
                   check=True, capture_output=True)
    return root


def test_paired_resnet_resume_through_the_shared_trainer(dummy, tmp_path):
    """E3 path (pairs + canonical JS) on the dummy registry: resume continues exactly."""
    reg = pd.read_parquet(dummy / "render_registry.parquet")
    cfg = ExperimentConfig(experiment_id="E3", model="resnet18.a1_in1k", train_renders=["R0", "R1", "R2", "R4"],
                           pairing="pair_distinct", loss="ce_plus_js", consistency_lambda=1.0, images_per_step=8,
                           epochs_max=2, patience=2)

    def run(out, stop=None, resume=True):
        torch.manual_seed(0)
        t = Trainer(build_resnet(cfg.model, pretrained=False), cfg, torch.device("cpu"), str(out))
        ds = PairedChartDataset(reg, str(dummy / "shards"), "train", cfg.train_renders, 224, seed=0)
        val = ChartImageDataset(reg, str(dummy / "shards"), "val", cfg.train_renders, 224)
        t.fit(DataLoader(ds, batch_size=cfg.windows_per_step, shuffle=True), val, cfg.train_renders,
              str(out / "best.pt"), resume=resume, stop_after_epochs=stop)
        return t

    full = run(tmp_path / "full", resume=False)
    first = run(tmp_path / "cut", stop=1)
    resumed = run(tmp_path / "cut")
    np.testing.assert_allclose(resumed.step_losses, full.step_losses[len(first.step_losses):], rtol=1e-5)
