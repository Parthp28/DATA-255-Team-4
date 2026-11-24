# Training implementation notes (inside the frozen D10 recipe)

These are choices the code makes where the execution plan does not fix the detail. Each one names its owner. Items
marked **open** must be settled and frozen before the first run of the affected experiment.

## 1. Paired step: one forward over 128 images (E3b, E3, E4b, E4)

Plan 5.1: "Paired runs use 64 windows x 2 renderings = 128 images per step, the same images per step as E1/E2."

`Trainer._batch_loss` concatenates the two views and runs one forward over all 128 images, then splits the logits
back into `logits_a` and `logits_b` (rows 0-63 and 64-127). Two separate 64-image forwards would not be equivalent
for ResNet-18. Its BatchNorm layers would normalise with 64-image batch statistics instead of E1's 128, and would
update their running statistics twice per optimizer step instead of once. DeiT-S uses LayerNorm, so it is unaffected
either way. `tests/test_paired_forward.py` checks four things:

- there is exactly one forward of 2N images;
- there is one running-stat update, and it equals the joint batch mean;
- the a/b split keeps each view aligned with its images;
- unpaired runs are unchanged.

Owner: Kaushik (trainer); Khushi reviews for the DeiT runs.

## 2. Pair sampling is fixed per window across epochs

`PairSampler.draw_pair(available, key=i)` seeds its generator with `(seed, i)`, where `i` is the window's dataset index.
Each training window therefore keeps the same two distinct seen renderings in every epoch and on resume. Shuffling
changes which windows share a batch, not which renderings a window shows. Across the training set the six unordered
pairs of {R0, R1, R2, R4} are drawn with equal probability.

The plan asks for two distinct seen renderings per window per step and does not ask for per-epoch resampling. The
fixed pairing is deterministic, needs no extra state in `last.pt`, and at the same seed is identical for E3b/E3 and E4b/E4, so the
lambda = 0 controls and the consistency runs see the same image pairs. Per-epoch resampling (a key of
`(seed, epoch, i)`) would show each window in more renderings. Adopting it is a recipe change, would need the epoch
passed into the dataset, and must be decided before any paired run. The current code keeps the fixed pairing.

Owner: Kaushik (sampler).

## 3. E5 numeric 1D CNN: architecture and weight decay (decided before any E5 test prediction)

**Architecture, kept as implemented.** Three blocks of Conv1d(k = 3, 64 channels, padding 1), BatchNorm1d and ReLU
(5 -> 64 -> 64 -> 64 channels over the 30 days), then a global average pool over time and Linear(64 -> 3). BatchNorm
has been part of the team's E5 implementation (`src/models/numeric.py`) since it was first committed on 2026-11-22,
before any E5 result existed. The plan's summary ("3 conv layers (k = 3, 64 ch), global average pool, linear head")
does not exclude it, so it stays, and the report states the full architecture.

**Weight decay = 0.01.** The plan's D11 grid tunes only lr for E5 and leaves weight decay open. It was settled by a
small, prospective, validation-only comparison:

- Candidates: weight decay 0.01 and 0.05.
- Held fixed: seed 0, lr 1e-3 (the E5 base value), the same data, preprocessing, architecture and D10 recipe.
- Rule: higher validation Macro F1 wins. A tie at 3 decimals would have gone to 0.05, the `ExperimentConfig` and
  shared-recipe default.

| weight decay | validation Macro F1 (seed 0) |
|---|---|
| 0.01 | 0.3719 |
| 0.05 | 0.3562 |

Weight decay 0.01 was frozen at 2026-11-24 08:25 UTC, before any E5 test prediction existed. D11 then selected
lr = 1e-3 over 3e-3 (validation Macro F1 0.3719 vs 0.3680) at that weight decay, and the config was frozen through
`scripts/select_config.py`. Neither choice is revisited after test evaluation.

Owner: Kaushik (model code), Farzaan (E5 runs).

## 4. E5 fits on CPU

Plan 5.1 runs E5 on CPU (D16 keeps it there).

The estimate below used synthetic numeric windows of the production size (29,900 train and 7,200 val), through the
shared trainer. One epoch took 2.7 s with 4 CPU threads and 6.9 s with 24. The worst case is the 2 D11 tuning runs
plus seeds 1 and 2, each running the full 10 epochs, which comes to under 5 minutes. This is an engineering estimate
on synthetic tensors, not a result, so the run records stay the source of truth.

`tests/test_experiment_configs.py` checks every `experiments/*.yaml` against:

- the plan matrix (5.1);
- the shared recipe D10;
- the D11 grid;
- the lambda grid D12/D13;
- the E3b/E3 <- E1 and E4b/E4 <- E2 inheritance.

It also checks that E5's weight decay is still unset (open decision).
