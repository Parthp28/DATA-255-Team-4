"""Experiment config (Execution Plan Section 10.4) + deterministic hashing.

One frozen YAML per run. The only things that differ between the comparable
experiments are: architecture, training renderings, the loss, and lambda.
Everything else (recipe D10) is shared and identical, which is what makes the
CNN-vs-ViT comparison fair (Section 5.3).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

import yaml

# Loss modes the shared trainer understands:
#   "ce"          -> E1/E2 single-rendering cross-entropy
#   "paired_ce"   -> E3b/E4b paired CE, lambda = 0 (augmentation-only control)
#   "ce_plus_js"  -> E3/E4 paired CE + lambda * JS consistency (the intervention)
LOSS_MODES = {"ce", "paired_ce", "ce_plus_js"}


@dataclass
class ExperimentConfig:
    experiment_id: str                 # e.g. "E3"
    model: str                         # e.g. "resnet18.a1_in1k", "deit_small_patch16_224.fb_in1k", "numeric1d"
    seed: int = 0
    phase: str = "dev"                 # dev | final
    data_version: str = "data_v1"
    render_version: str = "render_v1"

    # training renderings + pairing (10.4)
    train_renders: list[str] = field(default_factory=lambda: ["R0"])
    pairing: str = "none"              # "none" | "pair_distinct"

    # loss (D12)
    loss: str = "ce"
    consistency_lambda: float = 0.0

    # optimizer / recipe (D10) -- shared across all image experiments
    lr: float = 3e-4
    weight_decay: float = 0.05
    images_per_step: int = 128
    epochs_max: int = 10
    patience: int = 3
    early_stop_metric: str = "val_macro_f1_own_renders"
    amp: bool = True
    grad_clip: float = 1.0
    image_size: int = 224

    frozen: bool = False

    def __post_init__(self):
        if self.loss not in LOSS_MODES:
            raise ValueError(f"loss must be one of {LOSS_MODES}, got {self.loss!r}")
        if self.pairing not in {"none", "pair_distinct"}:
            raise ValueError(f"bad pairing {self.pairing!r}")
        if self.loss in {"paired_ce", "ce_plus_js"} and self.pairing != "pair_distinct":
            raise ValueError("paired losses require pairing='pair_distinct'")
        if self.loss == "ce_plus_js" and self.consistency_lambda <= 0:
            raise ValueError("ce_plus_js requires consistency_lambda > 0")
        if self.loss in {"ce", "paired_ce"} and self.consistency_lambda != 0:
            raise ValueError(f"{self.loss} requires consistency_lambda == 0")

    @property
    def is_numeric(self) -> bool:
        return self.model.startswith("numeric")

    @property
    def windows_per_step(self) -> int:
        """Pairs halve the window count so images_per_step stays 128 (Section 5.1)."""
        return self.images_per_step // 2 if self.pairing == "pair_distinct" else self.images_per_step

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        known = {k: raw[k] for k in raw if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        """Canonical hash of the config, written to run.json as config_sha256 (10.5)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()
