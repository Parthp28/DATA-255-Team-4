# Financial Vision Models: Price Patterns or Rendering Shortcuts?

A counterfactual robustness study of ResNet-18 and DeiT-S on candlestick charts of 50 U.S. stocks and ETFs: 30-day OHLCV windows, 5-day DOWN / NEUTRAL / UP targets, and 7 information-equivalent renderings of every window.

DATA 255 Deep Learning and Computer Vision, Fall 2026, San José State University.
Team: Parth Patel, Khushi Donda, Farzaan Dalwai, Kaushik Hariharan.

## Research questions

When CNN and Vision Transformer models predict future-return classes from chart images, do their predictions depend on the price structure, or partly on how the chart is drawn? We hold the financial information fixed, change only the rendering, and measure how often predictions change; we then test whether multi-style training and an explicit consistency loss reduce that sensitivity.

## Pipeline

```text
yfinance OHLCV (cached, hashed)
   │
   ▼
30-day windows ─► labels (train-only tertiles) ─► chronological split with 35-day purge/embargo
   │
   ▼
Counterfactual renderer (R0 reference; R1, R2, R4 seen; R3, R5, R6 held out)
   │
   ├─► ResNet-18:  E1 (R0) ─► E3b (paired styles, CE) ─► E3 (paired styles, CE + JS consistency)
   ├─► DeiT-S/16:  E2 (R0) ─► E4b (paired styles, CE) ─► E4 (paired styles, CE + JS consistency)
   └─► 1D CNN on raw 30 × 5 OHLCV: E5        B0: majority, stratified random, momentum
   │
   ▼
Evaluation: Macro F1, AUROC, flip rate, probability dispersion, ECE, Brier,
            cross-seed disagreement, block bootstrap, Grad-CAM, rollout, masking
```

## Results

Filled from `outputs/tables/` after the results freeze (`outputs/results_freeze.json`). Mean ± SD over 3 seeds on the 2020 to 2024 test split.

| Experiment | Model | Training renders | Macro F1 | AUROC | Flip rate (seen) | Flip rate (held-out) | ECE |
|---|---|---|---|---|---|---|---|
| B0 | trivial baselines | none | | | n/a | n/a | |
| E1 | ResNet-18 | R0 | | | | | |
| E2 | DeiT-S/16 | R0 | | | | | |
| E3b | ResNet-18 | paired seen styles | | | | | |
| E4b | DeiT-S/16 | paired seen styles | | | | | |
| E3 | ResNet-18 | paired + consistency | | | | | |
| E4 | DeiT-S/16 | paired + consistency | | | | | |
| E5 | 1D CNN | numeric OHLCV | | | n/a | n/a | |

## Why: design decisions

**Why the augmentation controls train on pairs with λ = 0.** E3b and E4b see exactly the same rendering pairs as E3 and E4, so the only difference is the consistency term; a one-image control would confound the loss with data volume.

**Why a 35-day purge/embargo.** Each sample spans 30 input days and 5 label days, so no trading day may appear in both a training span and an evaluation span; that requires a 35-day gap between the last training anchor and the first evaluation anchor.

**Why pre-rendered images.** Every rendering is drawn once by a deterministic rasterizer and hashed, so training reads exactly the registered images and the equivalence tests cover what the models saw.

**Why masking over heatmaps.** Grad-CAM and attention rollout show where attribution concentrates; masking a region against an area-matched random control tests whether the decision actually depends on it.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/build_dataset.py --config configs/data.yaml
python scripts/build_render_registry.py --styles configs/render_styles.yaml
pytest -q
python scripts/train.py --config configs/frozen/E1.yaml --seed 0
python scripts/evaluate.py --data-root data/render_v1 --runs-from outputs/results_freeze.json --split test
python scripts/make_report_assets.py
```

`scripts/run_queue.py` runs a CSV of frozen training commands in priority and dependency order with checkpoint resume.

## Compute environment

Final reported image-model experiments are planned on NVIDIA T4 GPUs with fp16 mixed precision; PyTorch and timm versions will be recorded in every `run.json`. Planned recipe: AdamW, cosine schedule with 1 warmup epoch, at most 10 epochs, early stopping with patience 3, 128 images per step, 224 × 224 inputs, seeds 0, 1, 2; the checkpoint with the best validation Macro F1 will be used for testing.

## Repository layout

```text
configs/   data, rendering styles, experiment configs, frozen configs
src/       data, rendering, training, models, evaluation, interpretability
scripts/   dataset build, training, prediction, evaluation, report assets
tests/     data, leakage, rendering, training, metrics, masking
outputs/   run records, predictions, evaluation tables, figures
report/    IEEE LaTeX source
docs/      lessons learned, Q&A and ablation sheet, course GenAI log
```

## Contributions

Planned technical responsibilities. This section will be revised to describe completed work at the results freeze.

- **Farzaan Dalwai:** data curation, preprocessing, temporal splitting and leakage controls, counterfactual rendering pipeline, numeric baseline methodology, masking analysis.
- **Kaushik Hariharan:** shared training infrastructure, ResNet-18 methodology and experiments, augmentation and consistency CNN experiments, Grad-CAM analysis.
- **Khushi Donda:** Vision Transformer methodology and experiments, consistency objective, λ-selection methodology, attention-rollout analysis, robustness interpretation.
- **Parth Patel:** evaluation methodology, robustness metrics, calibration, statistical analysis, experiment integration, result synthesis and report integration.
