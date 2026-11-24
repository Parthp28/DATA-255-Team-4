# Compute environment (D16)

## D16, reported image-model compute environment

**Decision:** all reported image-model experiments (E1, E2, E3b, E4b, E3, E4) run on the same NVIDIA GeForce RTX 5090
GPU. Reported image-model runs are never mixed across GPU types. B0 and E5 run on CPU, as the project design
specifies.

**When:** this amends the earlier plan, which named NVIDIA T4 GPUs. The amendment was made on 2026-11-23, after the
G1 and G2 data and rendering gates passed and before any reported GPU run. No reported training run and no
validation or test result existed at that point, so the change could not have been driven by results.

**Why:**
- One identical GPU for every reported image-model run keeps the ResNet-18 and DeiT-S comparisons on the same
  hardware.
- The validated `data_v1` and `render_v1` artifacts are already installed and verified on this machine, so no
  transfer is needed.
- Removing that transfer also makes the schedule more reliable.

**What did not change:**
- the research questions;
- the data, split and leakage controls;
- the rendering definitions;
- the architectures and pretrained weights;
- the losses and the paired and consistency design;
- the D10 recipe, including 128 images per step;
- the D11 and D13 selection rules;
- the seeds, metrics, ablations and interpretation logic.

Hardware is a controlled execution environment, not an experimental variable.

## How the code enforces it

| Where | Behaviour |
|---|---|
| `src/evaluation/gate.py` | `D16_GPU = "NVIDIA GeForce RTX 5090"`. `is_d16_gpu()` compares the runtime device name with vendor prefixes and spacing normalised; any other model, including other RTX 5090 variants, fails. |
| `scripts/train.py` | Refuses to train an image model on any other CUDA GPU, and refuses to run E5 on any GPU. |
| `scripts/smoke_test.py` | Adds a `d16_gpu` check to P00/H00 on CUDA. |
| `scripts/freeze_results.py` | Accepts image-model runs only from the D16 GPU, and E5/B0 runs only from CPU. |

Every run record reads the environment from the runtime (`src/tracking.py`):

- the GPU model, total memory and compute capability;
- the CUDA, cuDNN, torch, timm and Python versions;
- whether AMP is active;
- the physical and effective batch size (128 images per step; 64 windows × 2 for paired runs) and the gradient
  accumulation steps;
- the fit time, training throughput and peak GPU memory.

## Planned report wording (past tense only after the runs)

> All reported image-model experiments were executed on a single NVIDIA GeForce RTX 5090 GPU with fp16 mixed
> precision, so ResNet-18 and DeiT-S were trained and evaluated in the same compute environment.
