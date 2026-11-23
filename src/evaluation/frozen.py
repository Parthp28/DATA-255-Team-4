"""Frozen experiment configurations: no test-split number from a mutable or unselected config.

Execution plan Section 14.4: "No test number is computed for a config that is not in configs/frozen/."
A frozen config is a YAML file in a directory named ``frozen`` (configs/frozen/<experiment>.yaml) whose
``freeze_record`` stores the team's own config hash (``ExperimentConfig.sha256()``, computed at seed 0)
plus provenance: source run, selection rule, git commit, data and render versions, selected
hyperparameters. ``ExperimentConfig.from_yaml`` ignores the extra ``freeze_record`` key, so a frozen file
is a normal training config for scripts/train.py.

Written by scripts/freeze_config.py. Verified by scripts/train.py (test predictions only for a verified
frozen config), scripts/validate_run.py and scripts/evaluate.py.
"""
from __future__ import annotations

import dataclasses
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.common.config import ExperimentConfig

FROZEN_DIR_NAME = "frozen"
SELECTED_KEYS = ("lr", "weight_decay", "consistency_lambda")


class FrozenConfigError(RuntimeError):
    """A frozen-only action met a configuration that is not a verified frozen config."""


def config_hash_at_seed0(cfg: ExperimentConfig) -> str:
    """The team's ExperimentConfig.sha256(), normalized to seed 0 so all seeds of a frozen config match."""
    return dataclasses.replace(cfg, seed=0).sha256()


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def build_frozen(cfg: ExperimentConfig, provenance: dict) -> dict:
    """Frozen YAML content: every ExperimentConfig field (frozen, final phase, seed 0) + freeze_record."""
    frozen = dataclasses.replace(cfg, frozen=True, phase="final", seed=0)
    out = frozen.to_dict()
    out["freeze_record"] = {
        "config_sha256": frozen.sha256(),
        "experiment_id": frozen.experiment_id,
        "data_version": frozen.data_version,
        "render_version": frozen.render_version,
        "selected_hyperparameters": {k: getattr(frozen, k) for k in SELECTED_KEYS},
        "git_commit": _git_commit(),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **provenance,
    }
    return out


def write_frozen(content: dict, path) -> Path:
    """Write once. Re-freezing identical content is a no-op; different content is refused."""
    p = Path(path)
    if p.parent.name != FROZEN_DIR_NAME:
        raise FrozenConfigError(f"frozen configs live in a '{FROZEN_DIR_NAME}/' directory, not {p.parent}")
    if p.exists():
        old = (yaml.safe_load(p.read_text()) or {}).get("freeze_record", {})
        if old.get("config_sha256") == content["freeze_record"]["config_sha256"]:
            return p
        raise FrozenConfigError(
            f"{p} is already frozen with a different configuration. Frozen configs are immutable: if a selection "
            f"was rejected at G4/G5, move the old file to {p.parent / 'superseded'}/ and record the deviation "
            f"before freezing again.")
    p.parent.mkdir(parents=True, exist_ok=True)
    header = ("# FROZEN configuration (scripts/freeze_config.py). Do not edit: freeze_record.config_sha256 is\n"
              "# re-verified before any test-split inference; an edited file is refused.\n")
    p.write_text(header + yaml.safe_dump(content, sort_keys=False))
    return p


def verify_frozen(path) -> dict:
    """Return the freeze_record of a genuine frozen config, else raise FrozenConfigError."""
    p = Path(path)
    if p.parent.name != FROZEN_DIR_NAME:
        raise FrozenConfigError(f"{p} is not in a '{FROZEN_DIR_NAME}/' directory")
    raw = yaml.safe_load(p.read_text()) or {}
    rec = raw.get("freeze_record")
    if raw.get("frozen") is not True or not isinstance(rec, dict) or "config_sha256" not in rec:
        raise FrozenConfigError(f"{p} is not a frozen config (needs frozen: true and a freeze_record)")
    cfg = ExperimentConfig.from_yaml(str(p))
    got = config_hash_at_seed0(cfg)
    if got != rec["config_sha256"]:
        raise FrozenConfigError(f"{p} was edited after freezing (config hash {got[:8]} != recorded "
                                f"{rec['config_sha256'][:8]})")
    for k in ("experiment_id", "data_version", "render_version"):
        if rec.get(k) != getattr(cfg, k):
            raise FrozenConfigError(f"{p}: freeze_record.{k} does not match the config")
    return rec


def frozen_config_info(config_path) -> dict | None:
    """Provenance fields for run.json when `config_path` is a verified frozen config, else None.

    A file inside a frozen/ directory that fails verification raises (fail closed); a config elsewhere
    (experiments/, tuning, lambda candidates) returns None, so only validation predictions are produced.
    """
    p = Path(config_path)
    if p.parent.name != FROZEN_DIR_NAME:
        return None
    rec = verify_frozen(p)
    return {"frozen_config_path": str(p).replace("\\", "/"), "frozen_config_sha256": rec["config_sha256"]}


def run_matches_frozen(run_record: dict) -> bool:
    """True when a run's recorded configuration equals its frozen config (any seed)."""
    sha = run_record.get("frozen_config_sha256")
    if not sha:
        return False
    fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
    cfg = ExperimentConfig(**{k: run_record[k] for k in fields if k in run_record})
    return config_hash_at_seed0(cfg) == sha
