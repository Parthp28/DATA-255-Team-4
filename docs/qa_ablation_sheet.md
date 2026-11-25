# Q&A and Ablation Sheet

Owner: Khushi Donda. Every member must be able to answer every item here without notes. Result lines marked "after G6" are filled from the frozen tables on Nov 23 and nowhere else.

## 1. What an ablation is

An ablation changes one component of the experiment while holding everything else fixed, so that any change in a metric can be attributed to that component. Our ablations change one of four things: the training renderings, the loss, the architecture, or the input representation. Everything else (data, labels, splits, test windows, resolution, training recipe, epoch cap, evaluation code) is identical.

**60-second answer to "Name your ablations":** A1 E1 to E3b and A3 E2 to E4b add rendering augmentation. A2 E3b to E3 and A4 E4b to E4 add the consistency loss on top of identical pairs. A5 compares ResNet-18 and DeiT-S under the same recipe. A6 compares chart images with a numeric 1D CNN on the same windows. A7 compares seen and held-out renderings for the same trained model. A8 compares every model with trivial baselines.

## 2. The eight ablations

### A1: E1 to E3b (rendering augmentation, CNN)
- **Changed:** training renderings, from R0 only to pairs drawn from the seen set {R0, R1, R2, R4}.
- **Held constant:** ResNet-18 weights, learning rate, weight decay, epoch cap, data, test windows.
- **Why it matters:** tells us whether simply seeing several styles makes predictions less dependent on style.
- **Metrics:** seen flip rate, held-out flip rate, probability dispersion, Macro F1.
- **Allowed conclusion:** "Training ResNet-18 on the four seen styles changed its seen flip rate by Δ (95% interval)."
- **Not allowed:** "Augmentation makes CNNs rendering-invariant."
- **Result (after G6):** frozen values in the report's planned-comparisons table (Table 5, row A1; `outputs/eval/comparisons.csv`).

### A2: E3b to E3 (consistency loss, CNN)
- **Changed:** the loss gains λ · JS(p_a, p_b).
- **Held constant:** the exact same pairs (same sampler seed), recipe, and epochs as E3b.
- **Why it matters:** this is the only comparison that isolates the consistency objective from data volume.
- **Metrics:** seen and held-out flip rate, dispersion, Macro F1, ECE.
- **Allowed conclusion:** "The consistency term added (or did not add) stability beyond seeing the same pairs."
- **Not allowed:** crediting the whole E1 to E3 change to consistency.
- **Result (after G6):** report Table 5, row A2 (`outputs/eval/comparisons.csv`).

### A3: E2 to E4b and A4: E4b to E4
Same logic as A1 and A2, for DeiT-S.
- **Result A3 (after G6):** report Table 5, row A3.
- **Result A4 (after G6):** report Table 5, row A4.

### A5: E1 vs E2 (and E3b vs E4b, E3 vs E4)
- **Changed:** architecture and its pretrained weights (ResNet-18, 11.18M parameters with the three-class head, 11.7M with the original 1000-class head; DeiT-S, 21.67M and 22.1M).
- **Held constant:** data, recipe, tuning budget (4 runs each), resolution, renderings, evaluation.
- **Why it matters:** our paper presentation (Bai et al., NeurIPS 2021) showed that apparent Transformer robustness can come from the training recipe, so we hold the recipe fixed and report the size difference openly.
- **Allowed conclusion:** "Under our tested renderings and training setup, DeiT-S showed a lower (or higher) flip rate than ResNet-18," read at matched Macro F1.
- **Not allowed:** "Transformers are more robust than CNNs."
- **Result (after G6):** report Table 5, row A5; Macro F1 and flip rates per model in Table 3 (`outputs/eval/seeds.csv`).

### A6: image models vs E5
- **Changed:** input representation (224 × 224 chart vs the raw 30 × 5 matrix) and model family.
- **Held constant:** windows, labels, splits, test windows, metrics.
- **Why it matters:** if numbers alone predict as well, converting to an image adds little predictive value.
- **Not allowed:** "Images contain no information."
- **Result (after G6):** report Table 5, row A6 (E1 minus E5 Macro F1).

### A7: seen vs held-out renderings
- **Changed:** which renderings we evaluate on; the trained model is the same.
- **Why it matters:** R3 is an unseen decoration change, R5 and R6 are unseen geometric changes. If gains appear on seen styles only, the learned invariance does not generalize.
- **Result (after G6):** report Table 5, row A7; per-rendering flip rates in Table 4 (`outputs/eval/per_render.csv`).

### A8: B0 vs everything
- **Changed:** learning vs trivial rules (majority class, stratified random, momentum).
- **Why it matters:** "better than chance" needs a concrete floor.
- **Result (after G6):** report Table 3 (B0 rows, and each model's Macro F1 minus the best B0 with its bootstrap interval).

### The one sentence everyone must know
E3b and E4b train on the same pairs as E3 and E4 with λ = 0. A control that saw one image per step would see half as many images, so any E3 gain could come from data volume instead of the loss. With identical pairs, the only difference is the consistency term.

## 3. Thirty-second technical answers

**Why ResNet-18?** A standard, well-understood CNN that fine-tunes in minutes on a single GPU. Residual connections compute y = F(x) + x, so gradients pass through the identity path and deep networks train reliably. We use ImageNet-1k weights trained with a modern strong-augmentation recipe (timm `resnet18.a1_in1k`) so its pretraining is closer to DeiT's.

**How does a residual block work?** Two 3 × 3 convolutions with batch normalization and ReLU, plus a skip connection that adds the block's input to its output; when the shape changes, a 1 × 1 convolution projects the skip path.

**Why DeiT-S and not ViT-B?** DeiT-S has the ViT-Small shape (12 layers, 384-dimensional tokens, 6 heads, 22M parameters) and was trained data-efficiently on ImageNet-1k. ViT-B is about four times larger and would not fit three seeds of every experiment on T4s in five days (the original compute plan; the reported runs used one RTX 5090, decision D16).

**How does a ViT read a chart?** The 224 × 224 image is cut into 16 × 16 patches (196 patches), each linearly projected to a token, with position embeddings and a class token added. Self-attention lets every patch exchange information with every other patch in every layer, and the class token is classified.

**Why pretrained weights?** About 30,000 training images are far too few to train a Transformer from scratch, and pretraining both models on the same dataset keeps the comparison fair.

**Why 30-day windows and a 5-day target?** Thirty trading days is about six weeks, enough candles to show short-term patterns while keeping each candle about 7 pixels wide at 224 px. Five days is one trading week, a short horizon of the kind used in chart-image work such as Jiang, Kelly, and Xiu (2023).

**Why a chronological split?** A random split puts overlapping windows from the same weeks in train and test and lets the model see the future; markets also change over time. Training on the past and testing on later years matches how such a model would be used.

**Why a 35-day gap?** A sample uses 30 input days and 5 label days, a 35-day span. For no trading day to appear in both a training span and an evaluation span, the last training anchor must be at least 35 days before the first evaluation anchor: 5 days of purge on the training side and 29 days of embargo on the evaluation side.

**Why Macro F1 as primary?** It averages F1 over the three classes with equal weight, so a model that predicts mostly NEUTRAL cannot look good, which accuracy would allow.

**Why AUROC too?** It measures ranking quality without an argmax threshold, so it can show signal even when argmax decisions are dominated by one class.

**Why flip rate?** It counts how often the predicted class changes when only the rendering changes, which is exactly the question. It needs no label, so it measures behavior, not accuracy.

**What does probability dispersion add?** It measures how far the probability vector moves even when the argmax does not flip, so it catches partial gains from consistency training.

**Why ECE and Brier?** ECE checks whether confidence matches accuracy; Brier is a proper scoring rule that does not depend on binning. Consistency training can change confidence, so we check both.

**Why E5?** To test whether drawing the chart adds anything beyond the raw numbers.

**How does the consistency loss work?** For two renderings of the same window, JS(p_a, p_b) = ½ KL(p_a ‖ m) + ½ KL(p_b ‖ m) with m = (p_a + p_b) / 2. It is zero only when the two predictions agree, symmetric, and bounded by ln 2. Gradients flow through both branches, so neither rendering acts as the teacher; λ sets its weight against classification.

**Why seen and held-out renderings?** Seen styles test whether training removed sensitivity to what the model saw; held-out styles test whether that invariance generalizes to changes it never saw.

**Why is cross-seed disagreement only a reference?** It shows how much decisions change from retraining alone. It is a useful comparison for normal training variability, not a formal statistical noise floor.

**Why are heatmaps only supporting evidence?** Grad-CAM and attention rollout show where a method assigns importance, not what causes the decision. Masking a region and comparing with an area-matched random mask is a direct intervention, so it is the stronger evidence.

**What do our key results mean?** (after G6) State each result with its metric, its baseline, the comparison, and the scope "under our tested renderings and training setup."

## 4. Pipeline questions (how the code enforces the plan)

These answers describe the code on `main`, not results. Each one names the file a reviewer can open.

**How do you guarantee that no test number comes from a tuned or unfrozen configuration?**
Five checks enforce this:
- `scripts/train.py` writes test predictions only for a verified `configs/frozen/` file.
- The owner signs `RESULT_VALIDATED` with `scripts/validate_run.py`.
- `scripts/evaluate.py` refuses any file that is unsigned, changed after signing, or from an unfrozen configuration.
- `scripts/freeze_results.py` lists exactly one valid run per experiment and seed.
- `--runs-from` re-checks every prediction hash.

**How is a paired step built?**
One forward pass runs over all 128 images (64 windows × 2 renderings). The logits are then split into the two views. The pass must be joint: ResNet-18's BatchNorm then sees the same 128-image batch statistics as E1 (`docs/training_notes.md`).

**Does a window get new rendering pairs every epoch?**
No. The pair is fixed per window, seeded by (seed, window index). At the same seed, E3b and E3 (and E4b and E4) therefore see identical pairs.

**How do you know a heatmap's "grid" pixels are really the grid?**
The pixel partition comes from the renderer's own draw list (`src/interpretability/regions.py`). It has six regions: candles, volume, grid, axes, plot background and margins. Before any mask is used, each PNG is hash-checked against the registry and against a fresh render.

**What does the area-matched control remove?**
It removes random 4 × 4 blocks of non-ink pixels in the same panel, trimmed to the region's pixel count. It never touches candles or volume bars and matches the area within 2% (T-M5). An image the mask does not change gives Δp = 0 exactly (T-M3).

**What is the momentum baseline?**
z_past = (Close[d] / Close[d−5] − 1) / (σ30 · √5) uses the label's own scaling, run backwards in time. The class thresholds are the training-split tertiles of z_past (`src/baselines.py`).

**Why does Grad-CAM explain the predicted class?**
Masking measures Δp for the original predicted class ĉ, so both methods explain the same decision.

**What exactly is attention rollout here?**
1. Average the attention over heads.
2. Add the residual as 0.5·A + 0.5·I and renormalise the rows.
3. Multiply across the 12 blocks.
4. Take the CLS row over the 196 patches, with each patch's value spread evenly over its 16 × 16 pixels.

**What stops a run on the wrong GPU from entering the results?**
D16 is enforced at both ends:
- At launch: `scripts/train.py` and `scripts/check_gpu.py` refuse any GPU other than the D16 GPU (NVIDIA GeForce RTX 5090).
- At the results freeze: `scripts/freeze_results.py` rejects image-model runs that are not on the D16 GPU.

E5 and B0 are the exception: they run on CPU by design.

**How is resuming after a session dies exact?**
`last.pt` holds the model, optimizer, scheduler, AMP scaler, all RNG states and the counters. The smoke test interrupts a run and resumes it, and the losses match the uninterrupted run. On CPU the difference was 0.0.

**How is the compute footprint measured?**
GPU-hours come from each run record's runtime-measured `gpu_minutes`. Energy is GPU-hours × 575 W (the RTX 5090's board power), an upper bound. CPU runs report wall time only (`scripts/compute_ledger.py`).

**Was anything left undecided before E5?**
Two things were open and both were decided before any E5 test prediction. E5's weight decay is 0.01, chosen by validation Macro F1 at seed 0 (0.372 against 0.356 for 0.05; `outputs/selection/E5_weight_decay.json`). `Numeric1DCNN` keeps its BatchNorm layers as implemented (`docs/training_notes.md`). Both are frozen in the E5 config.
