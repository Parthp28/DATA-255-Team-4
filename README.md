# Financial Vision Models: Price Patterns or Rendering Shortcuts?

A counterfactual robustness study of ResNet-18 and DeiT-S on candlestick charts of 50 U.S. stocks and ETFs. Each sample is a 30-day OHLCV window with a 5-day DOWN / NEUTRAL / UP target, and every window is drawn in 7 information-equivalent renderings.

DATA 255 Deep Learning and Computer Vision, Fall 2026, San José State University.
Team: Parth Patel, Khushi Donda, Farzaan Dalwai, Kaushik Hariharan.

**Final report:** IEEE Computer Society LaTeX source in [`report/`](report/) (`main.tex`, `references.bib`, and generated tables and figures in `report/generated/`).

## Research question

When CNN and vision-transformer models predict return classes from chart images, do their predictions depend on the price structure or on how the chart is drawn? We hold the financial information fixed, change only the rendering, and measure how often predictions change. We then test whether multi-rendering training and a consistency loss reduce that sensitivity.

## Data and renderings

- **Universe and source:** 40 large-cap stocks and 10 ETFs, with adjusted daily OHLCV from 2005-01-03 to 2024-12-31. Every raw file is hashed.
- **Windows:** 30 input days, a 5-day horizon and a stride of 5.
- **Labels:** the volatility-scaled 5-day return, split at tertile thresholds fitted on training windows only (t1 = −0.278, t2 = 0.520).
- **Split:** chronological, with train 2005–2016, validation 2017–2019 and test 2020–2024. The purge is derived from each sample's 35-day information span. Counts: 29,900 train, 7,200 validation, 12,250 test and 700 purged windows.
- **Renderings:** a 224 × 224 RGB reference rendering (R0) plus six single-factor counterfactuals drawn from the same 30 × 5 array.
  - Seen in training: R1 background tint, R2 grid removed, R4 candle palette.
  - Held out from all training and tuning: R3 axes added, R5 narrower candle bodies, R6 wider margins.
  - In total 345,450 images, verified for hashes, determinism, factor isolation and the absence of text.

## Experiments

| Experiment | Model | Training renderings | Loss |
|---|---|---|---|
| B0 | majority, stratified random (3 seeds), momentum rule | none | none |
| E1 / E2 | ResNet-18 / DeiT-S | R0 | cross-entropy |
| E3b / E4b | ResNet-18 / DeiT-S | pairs of seen renderings | paired CE, λ = 0 (control) |
| E3 / E4 | ResNet-18 / DeiT-S | pairs of seen renderings | paired CE + λ · Jensen-Shannon, λ = 3 |
| E5 | 1D CNN on the raw 30 × 5 array | none | cross-entropy |

- **Recipe:** one shared recipe for all image models: AdamW, cosine decay, at most 10 epochs with early stopping, 128 images per step and fp16.
- **Selection:** on validation data only. The learning rate and weight decay come from a 4-run grid per architecture. E3b/E3 inherit E1's settings and E4b/E4 inherit E2's. λ is chosen from {0.3, 1, 3} by the predeclared rule.
- **Seeds:** every final experiment uses seeds 0, 1 and 2.

## Results (test split 2020–2024, mean ± SD over 3 seeds)

| Experiment | Macro F1 | AUROC | Flip rate (seen) | Flip rate (held-out) | ECE |
|---|---|---|---|---|---|
| B0 majority | 0.167 | 0.500 | n/a | n/a | 0.001 |
| B0 stratified random | 0.332 ± 0.002 | 0.499 ± 0.001 | n/a | n/a | 0.668 ± 0.002 |
| B0 momentum | 0.341 | 0.501 | n/a | n/a | 0.659 |
| E1 ResNet-18, R0 | 0.359 ± 0.003 | 0.530 ± 0.003 | 0.547 ± 0.027 | 0.387 ± 0.031 | 0.090 ± 0.067 |
| E2 DeiT-S, R0 | 0.346 ± 0.005 | 0.523 ± 0.007 | 0.198 ± 0.063 | 0.304 ± 0.076 | 0.112 ± 0.128 |
| E3b ResNet-18, pairs | 0.350 ± 0.007 | 0.527 ± 0.003 | 0.279 ± 0.009 | 0.389 ± 0.025 | 0.123 ± 0.050 |
| E4b DeiT-S, pairs | 0.353 ± 0.003 | 0.522 ± 0.004 | 0.157 ± 0.063 | 0.358 ± 0.064 | 0.229 ± 0.188 |
| E3 ResNet-18, pairs + JS | 0.355 ± 0.011 | 0.529 ± 0.013 | 0.226 ± 0.040 | 0.412 ± 0.030 | 0.199 ± 0.136 |
| E4 DeiT-S, pairs + JS | 0.351 ± 0.007 | 0.521 ± 0.011 | 0.132 ± 0.047 | 0.376 ± 0.058 | 0.247 ± 0.161 |
| E5 numeric 1D CNN | 0.352 ± 0.006 | 0.538 ± 0.004 | n/a | n/a | 0.031 ± 0.013 |

- **Predictive performance was modest.** Every learned model was close to the trivial baselines. The numeric model matched the image models: the interval for the E1 − E5 Macro F1 difference included zero.
- **Predictions were rendering-sensitive.** The R0-only ResNet changed its predicted class for 96.8% of test windows under at least one information-preserving rendering.
- **DeiT-S and paired training reduced sensitivity to the seen appearance factors.** Held-out geometric changes (R5, R6) still flipped 33–59% of predictions in every image model.
- **Uncertainty:** intervals are 95% moving-block bootstrap intervals, which cover test-sample uncertainty only. Seed agreement is reported separately in the report.

Full tables, ablations A1–A8, calibration, cross-seed disagreement and interpretability are in the report. The evaluation outputs are in [`outputs/eval/`](outputs/eval/), the frozen run set in [`outputs/results_freeze.json`](outputs/results_freeze.json), and the validation-only selection logs in [`outputs/selection/`](outputs/selection/).

## Compute environment

- **GPU runs:** all reported image-model experiments ran on a single NVIDIA GeForce RTX 5090 (32 GB) with fp16 mixed precision, PyTorch 2.14.1, CUDA 13.0 and timm 1.0.31 ([`docs/compute_environment.md`](docs/compute_environment.md)).
- **CPU runs:** B0 and E5.
- **Totals:** 38 training runs (including tuning and λ candidates) used 4.61 GPU-hours and at most 2.65 kWh, an upper bound at the GPU's board power.
- **Provenance:** every `run.json` records the code commit, the GPU and library versions, the configuration hash, and the hash of its prediction file.

## Reproduce

The data and the rendered images are not in Git: about 1.3 GB, shared separately as `data_v1` / `render_v1`. They can also be rebuilt deterministically from the configs.

What is in Git: the code, the configs, the frozen evaluation tables (`outputs/eval/`), the results-freeze manifest (`outputs/results_freeze.json`) and the validation-only selection logs (`outputs/selection/`). What is not in Git, because it is produced by running the pipeline: the frozen run configs (`configs/frozen/`, written by `select_config.py` / `freeze_config.py`), and the run directories, checkpoints and prediction files (`outputs/runs/`, `outputs/predictions/`). The commands below therefore regenerate those artifacts in order. The evaluation and report-asset steps need the run outputs from the training steps; they cannot be run from a fresh clone alone.

```bash
pip install -r requirements.txt
python scripts/build_dataset.py --config configs/data.yaml                      # data_v1
python scripts/build_render_registry.py --styles configs/render_styles.yaml --tar  # render_v1
pytest -q                                                                        # full-data tests run when data/ exists
python scripts/verify_g1g2.py --data-dir data --all-images                       # independent G1/G2 checks

# tuning (validation only), selection, freezing, promotion, seeds: see docs/evaluation.md
python scripts/train.py --config experiments/E1.yaml --set lr=3e-4 weight_decay=0.01 --data-root data/render_v1
python scripts/select_config.py --experiment E1 --runs-dir outputs/runs --out configs/frozen/E1.yaml
python scripts/train.py --config configs/frozen/E1.yaml --seed 1 --data-root data/render_v1
python scripts/run_b0.py --config configs/frozen/B0-momentum.yaml --data-root data/data_v1

# results freeze, evaluation and report assets
python scripts/freeze_results.py --include-b0 --write
python scripts/evaluate.py --data-root data/render_v1 --runs-from outputs/results_freeze.json --split test --out outputs/eval
python scripts/make_report_assets.py --eval-dir outputs/eval --out report/generated --freeze outputs/results_freeze.json
```

To build the report, run `pdflatex main`, then `bibtex main`, then `pdflatex main` twice, inside `report/`. Alternatively, upload `report/` to Overleaf.

## Repository layout

```text
configs/   data, rendering styles, experiment configs
src/       data, rendering, training, models, evaluation, interpretability, baselines
scripts/   dataset and rendering build, training, selection, prediction, evaluation, interpretability, report assets
tests/     data, leakage, rendering, training, evaluation, gates, interpretability, report checks
outputs/   frozen evaluation tables, results-freeze manifest, selection logs
report/    IEEE LaTeX source and generated tables and figures
docs/      evaluation and validation guide, compute environment, training notes, lessons learned, AI-use log
```

## Contributions

As stated in the report's CRediT section:

- **Farzaan Dalwai:** data curation, preprocessing, temporal splitting and leakage controls, counterfactual rendering pipeline, numeric baseline methodology, masking analysis.
- **Kaushik Hariharan:** shared training infrastructure, ResNet-18 methodology and experiments, augmentation and consistency CNN experiments, Grad-CAM analysis.
- **Khushi Donda:** the paired Jensen-Shannon consistency loss, the pretrained DeiT-S classifier, the λ-selection script, technical ownership of E2, E4b and E4, and the visual sample review of the rendering registry.
- **Parth Patel:** evaluation methodology, robustness metrics, calibration, statistical analysis, experiment integration, result synthesis and report integration.

## Limitations

- **Scope:** 50 liquid U.S. assets, one window length and horizon, and a weak predictive signal.
- **Renderings:** one counterfactual per factor, with three factors held out.
- **Training:** three seeds, one compute environment and one recipe. λ = 3 is the top of the tested grid.
- **Interpretability:** seed-0 models and 2,000 test windows.
- No trading or profitability claim is made.
