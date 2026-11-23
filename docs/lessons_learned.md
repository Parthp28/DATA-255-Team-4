# Lessons Learned Log

Owner: Farzaan Dalwai (compiles Section 14 of the report on Nov 23). Every member adds at least three entries by Nov 23, ideally one per day, written when the problem happens rather than reconstructed later.

What makes a useful entry: a concrete technical problem, how you found the cause, and the practice you would follow next time. Failed approaches count; the rubric asks for them.

| Date | Member | Technical problem | Cause | Diagnosis (how we found it) | Solution | Lesson | Best practice going forward |
|---|---|---|---|---|---|---|---|
| 2026-11-22 | Parth Patel | The merged trainer stack could not be imported: `ModuleNotFoundError: No module named 'src.common'` in the trainer, models, tracking and `scripts/train.py` | The shared `common/` package was committed under `scripts/` while every module imported it as `src.common` | An import check of every module on the integration branch; the test suite had passed because no test imported the trainer stack | Moved the package to `src/common/` as a pure rename (no content change); `scripts/train.py` then ran E1, E3b and E3 on the dummy registry | A green test suite says nothing about modules it never imports | Every PR runs an import check of all `src/` modules and `--help` on every CLI, not only pytest |
| 2026-11-22 | Parth Patel | The queue runner's integrity check passed a job when `git rev-parse HEAD` failed | The signed-commit comparison treated an unreadable HEAD as nothing to compare and skipped the check | Writing a failure-mode test (no git, unsigned commit, unsigned config) instead of only the happy path | The queue now fails closed outside an explicit `--test-mode`; only a verified `configs/frozen/` file may be unsigned | An integrity check that cannot run must count as a failed check | Test integrity guards with the inputs they are meant to reject, including missing tools |
| 2026-11-23 | Parth Patel | Paired runs (E3b/E3) ran two 64-image forwards per step, so ResNet-18's BatchNorm used 64-image batch statistics and updated its running statistics twice per step, unlike E1's single 128-image batch | The paired step called the model once per view | Design review against plan 5.1 ("64 windows x 2 renderings = 128 images per step, the same images per step as E1/E2") | One forward over the concatenated views, then split the logits; regression tests check the single forward and the joint BatchNorm statistics | "Same images per step" also means the same normalization batch for BatchNorm models | When two arms must be comparable, check what each layer sees per step, not only the batch size |
| 2026-11-23 | Parth Patel | Re-running the P00/H00 smoke test into the same output folder failed its resume check | The resume check reused its scratch folder, so the second run resumed from the first run's `last.pt` and skipped the interrupted epoch | Running the CPU smokes for four configs into one folder; the second config was also refused as a configuration mismatch | The resume check now starts from empty scratch folders; a regression test runs the smoke three times into one folder | A check that is meant to be re-run must not depend on leftovers from its last run | Clear or version every scratch path a check writes, and test the second run, not only the first |
| 2026-11-23 | Parth Patel | The results-freeze check would have rejected every E5 run | It required an NVIDIA T4 for every run, while plan 5.1 runs E5 and B0 on CPU; the existing test even asserted the wrong rule | Wiring B0 into the same gate and asking which device B0 was allowed to use | One helper decides which runs have an image backbone; only those need the T4 and pretrained weights. E5 and B0 accept CPU or a T4 | A test can encode a misreading of the plan as easily as code can | When a test asserts a policy, cite the plan line it comes from next to the assertion |
| 2026-11-23 | Parth Patel | Masking a region that does not exist (the grid in R2) gave a tiny non-zero delta_p instead of exactly 0 (T-M3) | The unchanged image was run in the same batch as the masked ones, and batched convolution is not bit-identical across batch positions | T-M3 asserted exact equality and failed at about 1e-8 | A variant whose pixels equal the original reuses the original probabilities, so delta_p is exactly 0 and the forward pass is skipped | Floating-point results can depend on batch layout | Compare inputs before comparing outputs when a test needs an exact zero |

## Themes to watch for (fill only with what actually happened)

- Leakage and temporal splitting
- Rendering equivalence and pixel quantization
- CNN vs Transformer fine-tuning behavior
- Paired training and the consistency loss
- Compute, Colab and Kaggle sessions, checkpointing
- Evaluation: metrics that disagreed, statistics, seeds
- Interpretability: heatmaps vs masking
- Team integration: schemas, handoffs, frozen interfaces
