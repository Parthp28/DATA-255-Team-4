# Evaluation, result validation, and frozen configurations

Owner: Parth Patel. Covers execution plan Sections 10.5, 10.7, 14.4, 15, and 16 (T-E tests).

## Run identifiers

Runs keep the team convention from `scripts/train.py`: `run_id = <experiment>_s<seed>_<sha8>`, where `sha8` is the
first 8 hex digits of `ExperimentConfig.sha256()`. Every tuning candidate, λ candidate, seed, and frozen config gets
its own id, because the hash covers lr, weight decay, λ, seed, phase, and the frozen flag. The id names
`outputs/runs/<run_id>/`, `run.json`, and `outputs/predictions/<run_id>.parquet`.

Nothing parses run ids. Evaluation reads `experiment_id` and `seed` from the prediction files, and the link
between a run and its frozen config is recorded in `run.json` (`frozen_config_path`, `frozen_config_sha256`).

## Result validation (`validation_status` in run.json)

"No result enters evaluation before RESULT_VALIDATED" (Section 14.4).

| Status | Meaning |
|---|---|
| `pending` (or absent) | Written by `train.py`. Not evaluable. |
| `RESULT_VALIDATED` | Signed by the experiment's technical owner. |
| `rejected` | The owner rejected the run. |
| `superseded` | Replaced by a rerun. This also sets `status: superseded`, which `select_lambda.py` skips. |

```bash
python scripts/validate_run.py --run-dir outputs/runs/<run_id> --owner <Kaushik|Khushi|Farzaan> --data-root data/render_v1
python scripts/validate_run.py --run-dir outputs/runs/<run_id> --owner Kaushik --reject "reason"
python scripts/validate_run.py --run-dir outputs/runs/<run_id> --owner Kaushik --supersede "reason"
```

Only the technical owner in Section 5.1 can sign: Kaushik for E1, E3b, and E3; Khushi for E2, E4b, and E4; Farzaan
for E5 and B0.

`RESULT_VALIDATED` requires all of the following:
- the run completed;
- the best epoch and a finite validation Macro F1 are recorded;
- the prediction file passes the team validator with full R0–R6 coverage (every window for E5/B0);
- test predictions come only from a verified frozen config;
- the git tree was clean;
- the run used full data (no `--max-steps` debug cap);
- the data is not synthetic;
- the run record contains no compute-logistics fields.

The sha256 of the prediction file is recorded at signing.

`scripts/evaluate.py` refuses a real prediction file unless three things hold: its `run.json` says
`RESULT_VALIDATED`, the file's sha256 is unchanged since signing, and any test rows trace to a frozen config. The
internal operations queue is never read, and run records never hold executor, lane, or other logistics fields.

## Frozen configurations (no test number from a mutable config)

`scripts/train.py` writes test-split predictions only when `--config` is a verified frozen config: a file in
`configs/frozen/` whose `freeze_record.config_sha256` still matches its content. Every other config, including
tuning, λ candidates, and anything in `experiments/`, produces validation predictions only. An edited frozen file
makes training fail.

```bash
# D11 (E1, E2, E5): freeze the selected seed-0 tuning run
python scripts/freeze_config.py --from-run outputs/runs/<run_id> --rule "D11: highest validation Macro F1" --out configs/frozen/E1.yaml
# D13 (E3, E4): freeze the λ run chosen by scripts/select_lambda.py
python scripts/freeze_config.py --from-run outputs/runs/<run_id> --rule "D13: select_lambda.py log" --out configs/frozen/E3.yaml
# E3b / E4b: the experiment YAML with the frozen parent's lr and weight decay
python scripts/freeze_config.py --config experiments/E3b.yaml --set lr=<E1 lr> weight_decay=<E1 wd> --rule "inherits configs/frozen/E1.yaml" --out configs/frozen/E3b.yaml
```

Frozen files are immutable. Re-freezing with different content is refused.

D13 is read as written in `select_lambda.py`: a candidate is eligible when its validation Macro F1 is at least the
matching E3b/E4b validation Macro F1 minus 0.01, one-sided.

## Evaluation outputs (`outputs/eval/`, Section 10.7)

```bash
python scripts/evaluate.py --data-root data/render_v1 --predictions outputs/predictions --split test --out outputs/eval
python scripts/make_synthetic_predictions.py --out outputs/synthetic_dev      # development fixtures only
python scripts/evaluate.py --data-root outputs/synthetic_dev/data --predictions outputs/synthetic_dev/predictions --split test --allow-synthetic
```

`metrics_long.csv` (`run_id, experiment_id, seed, split, render_set, metric, value`):

| render_set | metrics |
|---|---|
| `R0`–`R6`, `NUM` | `accuracy`, `macro_f1`, `macro_auroc`, `precision_*`, `recall_*`, `f1_*`, `pred_share_*`, `max_pred_share`, `ece`, `brier` |
| `R0`–`R6` (image models only) | `flip_rate`, `prob_dispersion`, `delta_macro_f1`, `class_shift`, each against R0 |
| `seen` | `flip_rate`, `prob_dispersion` (mean over R1, R2, R4) |
| `held_out` | `flip_rate`, `prob_dispersion` (mean over R3, R5, R6) |
| `all` | `any_flip_rate` |

`scripts/select_lambda.py` reads `macro_f1` on R0, R1, R2, and R4 and `flip_rate` on `seen`. This contract is
covered by `tests/test_evaluate.py::test_metrics_long_feeds_select_lambda_d13`.

The other tables:
- `per_render.csv`: FR, PD, ΔF1, and class shift per rendering.
- `seeds.csv`: mean ± SD over seeds, with cross-seed disagreement on R0.
- `comparisons.csv`: ablations A1–A8, with seed-paired deltas and 95% moving-block bootstrap intervals over test anchor
  dates (63-day blocks, 1,000 resamples).

Synthetic fixtures are refused without `--allow-synthetic`, and their outputs are stamped `SYNTHETIC_NOT_RESULTS`.
