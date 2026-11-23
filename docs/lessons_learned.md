# Lessons Learned Log

Owner: Farzaan Dalwai (compiles Section 14 of the report on Nov 23). Every member adds at least three entries by Nov 23, ideally one per day, written when the problem happens rather than reconstructed later.

What makes a useful entry: a concrete technical problem, how you found the cause, and the practice you would follow next time. Failed approaches count; the rubric asks for them.

| Date | Member | Technical problem | Cause | Diagnosis (how we found it) | Solution | Lesson | Best practice going forward |
|---|---|---|---|---|---|---|---|
| 2026-11-22 | Parth Patel | The merged trainer stack could not be imported: `ModuleNotFoundError: No module named 'src.common'` in the trainer, models, tracking and `scripts/train.py` | The shared `common/` package was committed under `scripts/` while every module imported it as `src.common` | An import check of every module on the integration branch; the test suite had passed because no test imported the trainer stack | Moved the package to `src/common/` as a pure rename (no content change); `scripts/train.py` then ran E1, E3b and E3 on the dummy registry | A green test suite says nothing about modules it never imports | Every PR runs an import check of all `src/` modules and `--help` on every CLI, not only pytest |
| 2026-11-22 | Parth Patel | The queue runner's integrity check passed a job when `git rev-parse HEAD` failed | The signed-commit comparison treated an unreadable HEAD as nothing to compare and skipped the check | Writing a failure-mode test (no git, unsigned commit, unsigned config) instead of only the happy path | The queue now fails closed outside an explicit `--test-mode`; only a verified `configs/frozen/` file may be unsigned | An integrity check that cannot run must count as a failed check | Test integrity guards with the inputs they are meant to reject, including missing tools |
| 2026-11-23 | Parth Patel | Paired runs (E3b/E3) ran two 64-image forwards per step, so ResNet-18's BatchNorm used 64-image batch statistics and updated its running statistics twice per step, unlike E1's single 128-image batch | The paired step called the model once per view | Design review against plan 5.1 ("64 windows x 2 renderings = 128 images per step, the same images per step as E1/E2") | One forward over the concatenated views, then split the logits; regression tests check the single forward and the joint BatchNorm statistics | "Same images per step" also means the same normalization batch for BatchNorm models | When two arms must be comparable, check what each layer sees per step, not only the batch size |

## Themes to watch for (fill only with what actually happened)

- Leakage and temporal splitting
- Rendering equivalence and pixel quantization
- CNN vs Transformer fine-tuning behavior
- Paired training and the consistency loss
- Compute, Colab and Kaggle sessions, checkpointing
- Evaluation: metrics that disagreed, statistics, seeds
- Interpretability: heatmaps vs masking
- Team integration: schemas, handoffs, frozen interfaces
