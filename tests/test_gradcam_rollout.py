"""Grad-CAM (review: Kaushik) and attention rollout (review: Khushi) on CPU: planted-answer models, the reference
maths, and real ResNet-18 / DeiT-S graphs left unchanged. Untrained weights; not results."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.interpretability.gradcam import gradcam  # noqa: E402
from src.interpretability.regions import normalize_heat  # noqa: E402
from src.interpretability.rollout import attention_rollout, capture_attention, patch_to_pixels, rollout_matrix  # noqa: E402
from src.models.resnet import build_resnet, gradcam_target_layer  # noqa: E402
from src.models.vit import build_deit_small  # noqa: E402


class PlantedCNN(nn.Module):
    """Channel k of the target layer is the image restricted to region k, and logit k sums channel k, so Grad-CAM
    for class k must put all of its mass inside region k."""

    def __init__(self, regions: torch.Tensor):
        super().__init__()
        self.register_buffer("regions", regions)        # (3, H, W) disjoint 0/1 masks
        self.target = nn.Identity()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        a = self.target(self.scale * x.mean(1, keepdim=True) * self.regions[None])
        return a.sum((2, 3))


def planted():
    h = w = 32
    reg = torch.zeros(3, h, w)
    reg[0, :, :8], reg[1, :, 8:16], reg[2, :, 16:] = 1, 1, 1      # region 2 is largest -> predicted
    return PlantedCNN(reg), reg


def test_gradcam_puts_all_mass_on_the_planted_region():
    model, reg = planted()
    x = torch.rand(4, 3, 32, 32) + 0.1
    heat, c, probs = gradcam(model, x, layer=model.target)
    assert (c == 2).all() and probs.shape == (4, 3)
    np.testing.assert_allclose(heat.sum((1, 2)), 1.0, rtol=1e-6)
    np.testing.assert_allclose(heat[:, reg[2].bool().numpy()].sum(1), 1.0, rtol=1e-6)
    heat0, c0, _ = gradcam(model, x, layer=model.target, target=torch.zeros(4, dtype=torch.long))
    assert (c0 == 0).all()
    np.testing.assert_allclose(heat0[:, reg[0].bool().numpy()].sum(1), 1.0, rtol=1e-6)


def test_gradcam_on_resnet18_matches_a_reference_and_leaves_the_model_alone():
    torch.manual_seed(0)
    model = build_resnet("resnet18.a1_in1k", pretrained=False).train()
    layer = gradcam_target_layer(model)
    x = torch.randn(2, 3, 224, 224)
    heat, c, _ = gradcam(model, x)
    assert heat.shape == (2, 224, 224)
    assert model.training and not layer._forward_hooks and all(p.grad is None for p in model.parameters())
    model.eval()
    feats = {}
    h = layer.register_forward_hook(lambda _m, _i, o: feats.setdefault("a", o))
    logits = model(x)
    h.remove()
    g = torch.autograd.grad(logits.gather(1, torch.as_tensor(c)[:, None]).sum(), feats["a"])[0]
    cam = torch.relu((g.mean((2, 3), keepdim=True) * feats["a"]).sum(1, keepdim=True))
    cam = nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[:, 0].detach().numpy()
    ref = np.stack([normalize_heat(m) for m in cam])
    np.testing.assert_allclose(np.nan_to_num(heat), np.nan_to_num(ref), atol=1e-7)


def test_gradcam_without_signal_is_nan_not_uniform():
    model, _ = planted()
    heat, _, _ = gradcam(model, torch.zeros(1, 3, 32, 32), layer=model.target)
    assert np.isnan(heat).all()


def test_rollout_matrix_matches_the_definition():
    g = torch.Generator().manual_seed(0)
    attns = [torch.rand(2, 3, 5, 5, generator=g).softmax(-1) for _ in range(4)]
    ref = None
    for a in attns:
        m = 0.5 * a.mean(1).double().numpy() + 0.5 * np.eye(5)
        m = m / m.sum(-1, keepdims=True)
        ref = m if ref is None else m @ ref
    np.testing.assert_allclose(rollout_matrix(attns).numpy(), ref, rtol=1e-5)
    np.testing.assert_allclose(rollout_matrix(attns).sum(-1).numpy(), 1.0, rtol=1e-5)


def test_uniform_attention_gives_a_uniform_map_and_patches_keep_their_mass():
    t = 1 + 196
    roll = rollout_matrix([torch.full((1, 6, t, t), 1.0 / t)] * 12)
    heat = normalize_heat(patch_to_pixels(roll[0, 0, 1:].numpy(), (14, 14), 224))
    np.testing.assert_allclose(heat, 1.0 / 224 ** 2, rtol=1e-5)
    row = np.zeros(196)
    row[15] = 1.0                                                  # patch (1, 1)
    px = patch_to_pixels(row, (14, 14), 224)
    assert px.sum() == 1.0 and px[16:32, 16:32].sum() == 1.0


def test_rollout_on_deit_s_restores_the_fused_path_and_changes_no_logits():
    torch.manual_seed(0)
    model = build_deit_small(pretrained=False).train()
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        before = model(x)
    model.train()
    attns, logits = capture_attention(model, x)
    assert len(attns) == 12 and attns[0].shape == (2, 6, 197, 197)
    torch.testing.assert_close(logits, before, rtol=1e-4, atol=1e-5)          # explicit path == fused path
    heat, c, probs = attention_rollout(model, x)
    assert heat.shape == (2, 224, 224) and probs.shape == (2, 3)
    np.testing.assert_allclose(heat.sum((1, 2)), 1.0, rtol=1e-6)
    vit = model.backbone
    assert model.training and all(b.attn.fused_attn for b in vit.blocks)
    assert not any(b.attn.attn_drop._forward_hooks for b in vit.blocks)
