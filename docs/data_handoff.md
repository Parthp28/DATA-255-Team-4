# data_v1 / render_v1 handoff (G1, G2)

Owner: Farzaan. Consumers: Kaushik (trainer, ResNet, E5 loader), Khushi (DeiT), Parth (evaluation).
Large files never enter Git; they live on the shared Drive folder `DATA255_shared/data/` (D20).

## What exists

| Path (local and on Drive under `DATA255_shared/data/`) | Contents |
|---|---|
| `raw/<TICKER>.csv`, `raw/manifest.json` | yfinance daily adjusted OHLCV, 2005-01-03 to 2024-12-31, sha256 per file |
| `data_v1/windows.parquet` | window registry, schema 10.1, 50,050 rows (train 29,900, val 7,200, test 12,250, purged 700) |
| `data_v1/ohlcv.npy` | (50050, 30, 5) float32, columns O, H, L, C, V; `row_idx` indexes it |
| `data_v1/thresholds.json` | train-only tertiles of z: t1 = -0.278376, t2 = 0.519627 |
| `data_v1/split_report.md` | boundaries, purge derivation, counts, label balance (copy in `docs/split_report.md`) |
| `data_v1/build_manifest.json` | config / raw / output sha256, git commit |
| `render_v1/render_registry.parquet` | schema 10.2, 345,450 rows = 49,350 windows x R0-R6 |
| `render_v1/shards/{split}_{rid}/{window_id}.png` | 21 shard folders, 224 x 224 RGB PNG |
| `render_v1/windows.parquet`, `render_v1/ohlcv.npy` | byte-identical copies, so `render_v1/` alone is a complete `--data-root` |
| `render_v1/region_geometry.json` | per render_id: plot, price/volume panels, margins, axes box, grid rows/cols |
| `render_v1/figure1_sheet.png` | G2 review sheet: sample windows in R0..R6 |
| `render_v1/render_manifest.json` | registry / styles / renderer-code sha256, per-shard counts, tar sha256 |
| `render_v1_tar/{split}_{rid}.tar`, `SHA256SUMS` | the 21 shard folders packed for upload (PNG is already compressed, so tar is uncompressed) |

## How to use it

```bash
# Trainer (Kaushik's scripts/train.py reads windows.parquet, render_registry.parquet, ohlcv.npy, shards/)
python scripts/train.py --config configs/experiments/E1.yaml --data-root data/render_v1
```

- Image models: filter `render_registry.parquet` by `split` and `render_id`; `image_path` is relative to `render_v1/shards/`.
- E5 / B0: `windows.parquet` + `ohlcv.npy` (`row_idx`); use rows with `split` in train/val/test.
- Purged windows (`split == "purged"`) are kept in `windows.parquet` for the T-L4 audit, are never rendered,
  and must never be used for training or evaluation.

## Getting the data onto a GPU lane

Copying about 345k small PNGs through Drive one file at a time is far too slow, so the shards travel as 21 tar files.

Upload (Farzaan, once, from the machine that built the data):

```bash
rclone copy data/raw          drive:DATA255_shared/data/raw
rclone copy data/data_v1      drive:DATA255_shared/data/data_v1
rclone copy data/render_v1    drive:DATA255_shared/data/render_v1 --exclude "shards/**"
rclone copy data/render_v1_tar drive:DATA255_shared/data/render_v1_tar
```

On a Kaggle/Colab lane (before `run_queue.py`):

```bash
rclone copy drive:DATA255_shared/data/render_v1     data/render_v1
rclone copy drive:DATA255_shared/data/render_v1_tar data/render_v1_tar
(cd data/render_v1_tar && sha256sum -c SHA256SUMS)
mkdir -p data/render_v1/shards && for t in data/render_v1_tar/*.tar; do tar -xf "$t" -C data/render_v1/shards; done
python -m pytest -q tests/test_rendering.py tests/test_data.py tests/test_leakage.py   # full-data tests run if data/ is present
```

A Kaggle Dataset built from `render_v1/` + `render_v1_tar/` works the same way and avoids the Drive copy per session.

## Rebuild from scratch (deterministic)

```bash
pip install -r requirements.txt
python scripts/build_dataset.py --config configs/data.yaml            # add --offline to use only data/raw
python scripts/build_render_registry.py --styles configs/render_styles.yaml --tar
pytest -q tests/test_data.py tests/test_leakage.py tests/test_rendering.py
```

## Interpretation choices made inside the frozen spec

- sigma_30 is the sample SD (ddof = 1) of the 29 log returns inside the 30-row input window (input rows only).
- Labels: DOWN if z <= t1, UP if z > t2, else NEUTRAL; t1, t2 = 1/3, 2/3 quantiles of train z.
- Stride 5 is applied per contiguous segment (train, purged, val, purged, test), aligned to the segment's first
  global anchor, so anchor dates line up across tickers (convenient for the date-block bootstrap).
- Dates in both registries are ISO `YYYY-MM-DD` strings, as in the dummy registry.
- `render_role` is `reference` for R0, `seen` for R1/R2/R4, `held_out` for R3/R5/R6 (10.2). Select by `render_id`
  lists, not by role, if you treat R0 as part of the seen set.
- Pixel quantization: body width = round(0.60 x slot) = 4 px (R5: round(0.35 x slot) = 2 px); volume bars keep the
  R0 width in every style, so R5 changes only candle bodies; R6 scales all geometry to the 176 x 176 plot.
- Price panel 156 rows (75% of 208), volume panel 42 rows (20%), 10-row gap; R6: 132 / 35 / 9.
