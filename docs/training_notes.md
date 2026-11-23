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

## 3. E5 numeric 1D CNN: BatchNorm after each convolution (open)

The plan specifies the E5 model as "3 conv layers (k = 3, 64 ch), global average pool, linear head".
`src/models/numeric.py` implements Conv1d -> BatchNorm1d -> ReLU three times, then the global average pool and the
linear head. The plan does not mention BatchNorm. There are two options:

- keep BatchNorm and state it in the report's method section;
- remove it to match the plan's wording literally.

**Open, pre-E5 freeze decision**, together with E5's `weight_decay` (`null` in `experiments/E5.yaml`;
`scripts/train.py` refuses to run until it is set). Whichever option is chosen is frozen with the E5 config through
`scripts/freeze_config.py`, and the report describes the model that was actually run.

Owner: Kaushik (model code), Farzaan (E5 runs).
