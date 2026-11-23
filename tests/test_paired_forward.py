"""Paired step = one 128-image forward (review: Kaushik, Khushi). Plan 5.1: paired runs use 64 windows x 2
renderings = 128 images per step, the same images per step as E1/E2. With BatchNorm (ResNet-18) the batch
statistics and the running-stat updates must therefore come from one forward over a and b together, not from
two 64-image forwards. CPU only; tiny tensors; not results."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.common.config import ExperimentConfig  # noqa: E402
from src.training.trainer import Trainer, canonical_loss  # noqa: E402

N = 4  # windows per step in this test (64 in the real config)


class SpyBN(nn.Module):
    """Conv + BatchNorm + linear head that records the batch size of every forward call."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, stride=4)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Linear(4, 3)
        self.calls: list[int] = []

    def forward(self, x):
        self.calls.append(x.shape[0])
        return self.head(self.bn(self.conv(x)).mean((2, 3)))


def paired_cfg(loss="ce_plus_js", lam=1.0):
    return ExperimentConfig(experiment_id="E3", model="resnet18.a1_in1k", train_renders=["R0", "R1", "R2", "R4"],
                            pairing="pair_distinct", loss=loss, consistency_lambda=lam, images_per_step=2 * N)


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"image_a": torch.randn(N, 3, 16, 16, generator=g), "image_b": torch.randn(N, 3, 16, 16, generator=g),
            "label": torch.randint(0, 3, (N,), generator=g)}


def test_paired_step_is_a_single_forward_over_all_images():
    model = SpyBN()
    t = Trainer(model, paired_cfg(), torch.device("cpu"), "unused")
    t.model.train()
    t._batch_loss(batch())
    assert model.calls == [2 * N]                     # one forward of 2N images, not two of N
    assert int(model.bn.num_batches_tracked) == 1     # one running-stat update per step, as in E1


def test_batchnorm_statistics_cover_a_and_b_together():
    model = SpyBN()
    t = Trainer(model, paired_cfg(), torch.device("cpu"), "unused")
    t.model.train()
    b = batch(1)
    t._batch_loss(b)
    with torch.no_grad():
        feats = model.conv(torch.cat([b["image_a"], b["image_b"]]))
        joint_mean = feats.mean((0, 2, 3))
        expected = (1 - model.bn.momentum) * torch.zeros(4) + model.bn.momentum * joint_mean
    torch.testing.assert_close(model.bn.running_mean, expected)


def test_split_keeps_a_and_b_aligned_with_their_images():
    """In eval mode the joint forward must give exactly the per-view logits, so a/b are not swapped or mixed."""
    torch.manual_seed(0)
    model = SpyBN().eval()
    for loss, lam in (("paired_ce", 0.0), ("ce_plus_js", 1.0), ("ce_plus_js", 3.0)):
        t = Trainer(model, paired_cfg(loss, lam), torch.device("cpu"), "unused")
        b = batch(2)
        total, parts = t._batch_loss(b)
        with torch.no_grad():
            la, lb = model(b["image_a"]), model(b["image_b"])
        ref, ref_parts = canonical_loss(loss, lam, la, b["label"], lb)
        torch.testing.assert_close(total, ref)
        assert parts == ref_parts


def test_unpaired_runs_are_unchanged():
    model = SpyBN()
    cfg = ExperimentConfig(experiment_id="E1", model="resnet18.a1_in1k", train_renders=["R0"], images_per_step=2 * N)
    t = Trainer(model, cfg, torch.device("cpu"), "unused")
    t._batch_loss({"image": torch.randn(2 * N, 3, 16, 16), "label": torch.zeros(2 * N, dtype=torch.long)})
    assert model.calls == [2 * N]
