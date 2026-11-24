# GPU smoke tests P00 and H00 (gate G3)

Execution plan Sections 14.3 and 16.1. Run these only after G1/G2 pass and `data/render_v1/` is present on the GPU machine.
D16 (`docs/compute_environment.md`): every reported image-model run, and therefore P00/H00, runs on the same NVIDIA GeForce RTX 5090; the smoke report's `d16_gpu` check enforces it. Smoke runs are never reported or promoted.

| Job | Owner | Commands | Planned time (T4 estimate) |
|---|---|---|---|
| P00 | Kaushik | `python scripts/smoke_test.py --config experiments/E1.yaml --data-root data/render_v1` then `python scripts/benchmark_throughput.py --config experiments/E1.yaml --data-root data/render_v1` | about 10 min |
| H00 | Khushi | `python scripts/smoke_test.py --config experiments/E2.yaml --data-root data/render_v1` then `python scripts/benchmark_throughput.py --config experiments/E2.yaml --data-root data/render_v1 --probe-batch` | about 10 min |

Pretrained ImageNet-1k weights are loaded by default (D9), so the session needs internet access, or a cached copy of
the timm weights.

## Pass criteria (`outputs/smoke/<experiment>/smoke_report.json`, exit code 0)

`smoke_test.py` runs four checks on a 64-window training subset and a 16-window validation subset.

| Check | Passes when |
|---|---|
| `finite_step` | One training step gives a finite loss and finite, non-zero gradients. |
| `overfit` | One batch of 32 images (paired configs: 16 windows × 2) reaches 100% training accuracy within 200 steps at the config learning rate (T-T5). |
| `resume` | A run interrupted after epoch 0 and resumed from `last.pt` reproduces the uninterrupted run's step losses (rtol 1e-3) and final weights (T-T6). |
| `prediction_export` | `best.pt` predicts every subset window × R0–R6, and the predictions pass `validate_prediction_frame`. |

The report also records:
- `gpu_model`, CUDA and cuDNN versions, and torch and timm versions, all read from the runtime;
- whether AMP was on;
- peak GPU memory.

`benchmark_throughput.py` writes `outputs/smoke/<experiment>/throughput.json`:
- train and inference images per second, and loader images per second;
- `ratio_to_a1` against assumption A1 (about 700 images/s for ResNet-18 and 250 for DeiT-S, planning figures made for a T4).

Read the results this way:
- **`rescale_queue: true`** means the measurement is more than 30% off A1, so the queue estimates must be rescaled.
- **H00's `--probe-batch`** reports the largest of {128, 96, 64, 32} images per step that fits. If 128 does not fit, keep
  128 images per step through gradient accumulation (Section 19); the plan flags DeiT below 175 images/s.
- **`loader_bound: true`** means the PNG loader is slower than the GPU. Raise the `--num-workers` value passed to
  `scripts/train.py`.

## CPU rehearsal (no GPU)

On the team dummy registry, both smoke paths pass all four checks on CPU even without pretrained weights:
- `smoke_test.py ... --no-pretrained --device cpu`
- ResNet-18 overfits at step 50 and DeiT-S at step 125.
- Resume differences are 0.0.

`tests/test_smoke_benchmark.py` keeps the code paths covered.
