"""Run tracking (Execution Plan 10.5): outputs/runs/<run_id>/run.json + index.csv.

No MLflow/W&B -- a JSON per run plus one CSV line, so lanes can run unattended and
every reported number traces to a frozen config_sha256. Owner: Kaushik.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess

from .common.config import ExperimentConfig


def git_info() -> dict:
    def _run(args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""
    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain"]))
    return {"git_commit": commit, "git_dirty": dirty}


def environment_info(device) -> dict:
    """Compute environment for run.json (plan 10.5: GPU model, torch and timm versions). Read from the
    runtime, never hardcoded; CUDA is queried only when the run's device is CUDA. No logistics fields."""
    import platform

    import timm
    import torch

    dev = torch.device(device)
    info = {"device": dev.type, "gpu_model": "cpu", "torch_version": torch.__version__,
            "timm_version": timm.__version__, "python_version": platform.python_version(),
            "cuda_version": torch.version.cuda, "cudnn_version": None}
    if dev.type == "cuda":
        info["gpu_model"] = torch.cuda.get_device_name(dev)
        info["cudnn_version"] = torch.backends.cudnn.version()
    return info


def write_run_json(output_dir: str, run_id: str, config: ExperimentConfig,
                   status: str, metrics: dict, timings: dict, extra: dict | None = None):
    os.makedirs(output_dir, exist_ok=True)
    record = {
        "run_id": run_id,
        **config.to_dict(),
        "config_sha256": config.sha256(),
        **git_info(),
        "status": status,
        **timings,
        **metrics,
    }
    if extra:
        record.update(extra)
    with open(os.path.join(output_dir, "run.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)
    return record


def append_index_csv(index_path: str, record: dict):
    """Append one flat line per run to outputs/runs/index.csv."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    cols = ["run_id", "experiment_id", "seed", "status", "config_sha256",
            "best_val_f1", "best_epoch", "epochs_run", "gpu_minutes", "technical_owner"]
    exists = os.path.exists(index_path)
    with open(index_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(record)
