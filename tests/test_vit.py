"""CPU tests for the DeiT-S/16 classifier (src/models/vit.py).

All tests build the model with ``pretrained=False`` so they need no network. The pretrained
weight check is optional: set ``DATA255_TEST_PRETRAINED=1`` (needs the Hugging Face cache or a
one-time ~88 MB download).

Deliberately NOT here (T4 stage): the 32-image / 200-step overfit, the 128-image fp16 AMP memory
test, checkpoint/resume, and throughput.
"""
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("timm")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.vit import (  # noqa: E402
    IMAGE_SIZE,
    MODEL_NAME,
    NUM_CLASSES,
    DeiTSmallClassifier,
    build_deit_small,
    count_parameters,
    preprocessing_config,
)
from src.training.losses import paired_consistency_loss  # noqa: E402

# Measured with timm 1.0.31 (see the module docstring); the plan quotes 22.1M for the pretrained model.
PARAMS_3_CLASS = 21_666_819
PARAMS_1000_CLASS = 22_050_664
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Float32 reduction order differs between batch shapes; measured max deviation on CPU is ~2e-6.
ATOL = RTOL = 1e-4


def images(n, seed=0):
    return torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE, generator=torch.Generator().manual_seed(seed))


def labels(n, seed=0):
    return torch.randint(0, NUM_CLASSES, (n,), generator=torch.Generator().manual_seed(seed))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_deit_small(pretrained=False).eval()


def fresh_model(seed=0):
    torch.manual_seed(seed)
    return build_deit_small(pretrained=False)


def test_forward_shape_dtype_and_finite_logits(model):
    with torch.no_grad():
        out = model(images(4))
    assert out.shape == (4, NUM_CLASSES) and out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_output_is_raw_backbone_logits_and_model_does_not_normalize(model):
    x = images(2, seed=1)
    with torch.no_grad():
        assert torch.equal(model(x), model.backbone(x))  # no extra transform around the backbone
    # Structural check that nothing normalizes inside the model: no mean/std buffers, one child.
    assert list(model.named_buffers()) == []
    assert [n for n, _ in model.named_children()] == ["backbone"]
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    cfg = preprocessing_config()
    assert cfg["mean"] == IMAGENET_MEAN and cfg["std"] == IMAGENET_STD and cfg["input_size"] == (3, 224, 224)


def test_architecture_and_patch_token_configuration(model):
    b = model.backbone
    assert type(b).__name__ == "VisionTransformer" and MODEL_NAME == "deit_small_patch16_224.fb_in1k"
    assert (b.embed_dim, len(b.blocks), b.blocks[0].attn.num_heads) == (384, 12, 6)
    assert b.patch_embed.patch_size == (16, 16) and b.patch_embed.img_size == (224, 224)
    assert b.patch_embed.grid_size == (14, 14) and b.patch_embed.num_patches == 196
    assert b.num_prefix_tokens == 1 and b.cls_token is not None and getattr(b, "dist_token", None) is None
    assert tuple(b.pos_embed.shape) == (1, 197, 384)  # 196 patches + 1 class token
    assert b.global_pool == "token"
    assert isinstance(b.head, torch.nn.Linear) and (b.head.in_features, b.head.out_features) == (384, 3)
    # timm defaults preserved: no dropout, no drop-path.
    assert b.head_drop.p == 0.0 and b.pos_drop.p == 0.0
    assert all(float(getattr(blk.drop_path1, "drop_prob", 0.0)) == 0.0 for blk in b.blocks)
    with torch.no_grad():
        tokens = b.forward_features(images(1))
    assert tuple(tokens.shape) == (1, 197, 384)


def test_parameter_count_is_measured_and_pinned(model):
    assert count_parameters(model) == PARAMS_3_CLASS == count_parameters(model, trainable_only=True)
    # Swapping the 1000-class head (384*1000 + 1000) for the 3-class head (384*3 + 3) is the only difference.
    assert PARAMS_1000_CLASS - PARAMS_3_CLASS == (384 * 1000 + 1000) - (384 * 3 + 3)


def test_backward_gives_finite_gradients_to_every_parameter():
    m = fresh_model().train()
    F.cross_entropy(m(images(4)), labels(4)).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert missing == []
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.backbone.head.weight.grad.abs().sum() > 0 and m.backbone.patch_embed.proj.weight.grad.abs().sum() > 0


def test_eval_is_deterministic_and_has_no_side_effects(model):
    x = images(3, seed=2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        a, b = model(x), model(x)
    assert torch.equal(a, b)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    # Same seed, same architecture: identical weights and outputs.
    with torch.no_grad():
        assert torch.equal(fresh_model(7).eval()(x), fresh_model(7).eval()(x))


def test_batch_splitting_equivalence_within_float32_tolerance(model):
    # Paired training concatenates two views into one batch; the logits must not depend on how
    # the batch is split (ViT uses LayerNorm, so there are no batch statistics).
    x = images(16, seed=3)
    with torch.no_grad():
        full = model(x)
        halves = torch.cat([model(x[:8]), model(x[8:])])
        singles = torch.cat([model(x[i : i + 1]) for i in range(16)])
        reversed_ = model(x.flip(0)).flip(0)
    for other in (halves, singles, reversed_):
        torch.testing.assert_close(full, other, atol=ATOL, rtol=RTOL)
    # The tolerance is tight enough to still distinguish different inputs.
    assert (full - model(images(16, seed=4)).detach()).abs().max() > 100 * ATOL


def test_train_mode_batch_split_matches_too():
    m = fresh_model().train()  # default dropout and drop-path are 0, so train mode is also split-invariant
    x = images(8, seed=5)
    with torch.no_grad():
        torch.testing.assert_close(m(x), torch.cat([m(x[:4]), m(x[4:])]), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(2, 3, 256, 256),  # wrong resolution
        torch.zeros(2, 3, 224, 192),  # not square
        torch.zeros(2, 1, 224, 224),  # grayscale
        torch.zeros(2, 4, 224, 224),  # RGBA
        torch.zeros(3, 224, 224),  # missing batch dimension
        torch.zeros(0, 3, 224, 224),  # empty batch
        torch.zeros(2, 3, 224, 224, dtype=torch.uint8),  # un-normalized integer pixels
        torch.zeros(2, 3, 224, 224, dtype=torch.long),
    ],
)
def test_invalid_inputs_raise_value_error(model, bad):
    with pytest.raises(ValueError):
        model(bad)


def test_non_tensor_input_raises_value_error(model):
    with pytest.raises(ValueError):
        model([[0.0]])


def test_cpu_training_smoke_loss_falls_with_the_planned_optimizer_settings():
    # Lightweight stand-in for the T4 overfit test: 8 images, AdamW, gradient clip 1.0 (D10).
    m = fresh_model().train()
    x, y = images(8, seed=6), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert all(map(lambda v: v == v, losses))  # no NaN
    assert losses[0] > 0.8  # starts near ln 3 = 1.10 for a 3-class head
    assert losses[-1] < 0.1 * losses[0]  # measured: ~1.07 -> ~0.001 (large margin)
    m.eval()
    with torch.no_grad():
        assert (m(x).argmax(-1) == y).all()


def test_compatible_with_merged_paired_consistency_loss():
    # 2N images = N windows x 2 renderings, one forward pass, split into the two views.
    n = 4
    x = images(2 * n, seed=8)
    y = labels(n, seed=8)
    m = fresh_model().train()
    logits = m(x)
    la, lb = logits[:n], logits[n:]
    assert la.shape == lb.shape == (n, NUM_CLASSES)
    out0 = paired_consistency_loss(la, lb, y, lam=0.0)
    assert torch.equal(out0.total, out0.ce)  # lambda = 0: exactly the augmentation-only control
    out1 = paired_consistency_loss(la, lb, y, lam=1.0)
    assert torch.isfinite(out1.total) and out1.total > out1.ce
    grads = {}
    for name, out in (("lam0", out0), ("lam1", out1)):
        m.zero_grad()
        out.total.backward(retain_graph=True)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        grads[name] = m.backbone.head.weight.grad.clone()
    assert not torch.allclose(grads["lam0"], grads["lam1"])  # the JS term reaches the model weights


@pytest.mark.skipif(os.environ.get("DATA255_TEST_PRETRAINED") != "1", reason="set DATA255_TEST_PRETRAINED=1 to load official weights")
def test_pretrained_weights_load_and_head_supports_three_classes():
    import timm

    torch.manual_seed(0)
    m3 = build_deit_small(pretrained=True).eval()
    ref = timm.create_model(MODEL_NAME, pretrained=True).eval()  # official 1000-class model
    assert count_parameters(ref) == PARAMS_1000_CLASS and count_parameters(m3) == PARAMS_3_CLASS
    # Every backbone tensor equals the official one; only the classification head differs.
    sd3, sdr = m3.backbone.state_dict(), ref.state_dict()
    head_keys = {k for k in sd3 if k.startswith("head.")}
    assert head_keys == {"head.weight", "head.bias"} and sd3["head.weight"].shape == (3, 384)
    assert all(torch.equal(sd3[k], sdr[k]) for k in sd3 if k not in head_keys)
    # The weights are really pretrained: they differ from a random initialization.
    rand = fresh_model().backbone.state_dict()
    assert not torch.equal(sd3["blocks.0.attn.qkv.weight"], rand["blocks.0.attn.qkv.weight"])
    # Official preprocessing configuration of the loaded model.
    cfg = m3.backbone.pretrained_cfg
    assert cfg["architecture"] == "deit_small_patch16_224" and cfg["tag"] == "fb_in1k"
    assert tuple(cfg["mean"]) == IMAGENET_MEAN and tuple(cfg["std"]) == IMAGENET_STD and tuple(cfg["input_size"]) == (3, 224, 224)
    # Features before the head are identical to the official model, so the 1000-class features carry over.
    x = images(2, seed=9)
    with torch.no_grad():
        torch.testing.assert_close(m3.backbone.forward_features(x), ref.forward_features(x), atol=ATOL, rtol=RTOL)
        assert m3(x).shape == (2, 3) and torch.isfinite(m3(x)).all()
