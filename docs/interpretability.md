# Interpretability, B0 and report assets: how to run them

Plan Sections 12, 13, 15.1, 10.6 and 10.7. Every tool below has been tested on CPU (`pytest -q`). None of them has
been run on production data or on a GPU yet.

## B0 baselines (CPU; owner Farzaan)

B0 has nothing to tune, so its configs can be frozen first and then run with the plan's seeds.

```bash
for k in majority random momentum; do
  python scripts/freeze_config.py --config experiments/B0-$k.yaml --rule "B0: no tuning (plan 5.1)" \
      --out configs/frozen/B0-$k.yaml
done
python scripts/run_b0.py --config configs/frozen/B0-majority.yaml --data-root data/render_v1
for s in 0 1 2; do python scripts/run_b0.py --config configs/frozen/B0-random.yaml --data-root data/render_v1 --seed $s; done
python scripts/run_b0.py --config configs/frozen/B0-momentum.yaml --data-root data/render_v1
# then the owner signs each run, as for every experiment:
python scripts/validate_run.py --run-dir outputs/runs/<run_id> --owner Farzaan --data-root data/render_v1
```

The B0 runs join the results freeze through `scripts/freeze_results.py --include-b0`.

## Interpretability (seed-0 checkpoints of the frozen E1, E2, E3b, E4b, E3, E4)

The D15 sample is 2,000 test windows, stratified by true class. It is drawn by `interpretability_sample` with seed 0,
so every model sees the same windows, and the output JSON records the sample's sha256.

```bash
# Grad-CAM (ResNet-18 runs; owner Kaushik) or attention rollout (DeiT-S runs; owner Khushi): chosen from the run
python scripts/run_attribution.py --run-dir outputs/runs/<seed-0 run> --data-root data/render_v1
# region masking with area-matched controls (all six models; owner Farzaan)
python scripts/run_masking.py --run-dir outputs/runs/<seed-0 run> --data-root data/render_v1
```

- **Outputs:** `outputs/attribution/<run_id>.parquet` and `outputs/masking/<run_id>.parquet`, with the plan 10.6
  columns. Each comes with a `<run_id>.json` that records the sample hash and the settings.
- **Defaults:** R0 only. Pass `--renders R0 R1 ...` to cover more renderings.
- **Refusals:** anything other than a seed-0 run of a frozen configuration is refused. So is any image whose PNG
  differs from the registry or from a fresh render, because the region masks would not line up.
- **Region partition:** candles, volume, grid, axes, plot background, margins. It comes from the renderer's draw
  list, and every pixel belongs to exactly one region (T-M1).
- **Masking:** each region is filled with the rendering's background colour (T-M2). The control is random non-ink
  4 × 4 blocks in the same panel, matched to the region's area within 2% (T-M5).
- **Empty regions:** an empty region, such as the grid in R2, gives Δp = 0 exactly (T-M3).
- **T-M4:** on a model trained on CPU, ink masking beats the other regions. Re-check it on the real seed-0 models
  with `masking_summary`.
- **Attribution-to-flip link:** `src/interpretability/link.py`. The default region-to-rendering map pairs grid with R2,
  candles with R4 and R5, plot background with R1, and margins with R1 and R6. The owner should confirm it before the
  masking job runs.

## Report assets and the compute ledger (owner Kaushik; report integration Parth)

```bash
python scripts/evaluate.py --data-root data/render_v1 --runs-from outputs/results_freeze.json --out outputs/eval
python scripts/make_report_assets.py --eval-dir outputs/eval --out report/generated \
    --masking-dir outputs/masking --attribution-dir outputs/attribution \
    --freeze outputs/results_freeze.json --data-root data/render_v1
python scripts/compute_ledger.py --runs-dir outputs/runs --tex report/generated/tables/table_ledger.tex
```

`report/main.tex` pulls each fragment in with `\gen` or `\genfig` and shows a red placeholder until it exists. The
generator refuses synthetic evaluation outputs; with `--allow-synthetic` it writes a stamped preview outside
`report/`. Each fragment records the sha256 of its sources.

## GPU sessions

`notebooks/kaggle_launch.ipynb` and `notebooks/colab_launch.ipynb` each hold one cell that calls
`scripts/gpu_session.sh`. That script:

1. checks out the frozen commit;
2. installs the requirements;
3. runs `scripts/check_gpu.py`, which stops anything but the D16 GPU (NVIDIA GeForce RTX 5090);
4. verifies and unpacks the shards;
5. hands over to `scripts/run_queue.py`.

Run it with `--check-only` first.
