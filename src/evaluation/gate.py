"""Scientific result validation and the evaluation gate (execution plan Sections 14.4, 16.1).

"No result enters evaluation before RESULT_VALIDATED." The status lives in the reproducible run record
outputs/runs/<run_id>/run.json (field ``validation_status``), written by the run's technical owner with
scripts/validate_run.py. It never depends on the internal operations queue, and a run record never holds
compute logistics (who ran a job, lanes, executors).

validation_status: pending (absent = pending) -> RESULT_VALIDATED | rejected; any -> superseded.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.common.schema import sha256_file
from src.evaluation.frozen import run_matches_frozen

VALIDATION_STATUSES = ("pending", "RESULT_VALIDATED", "rejected", "superseded")
EVALUABLE_RUN_STATUSES = ("complete", "promoted")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")   # same rule as scripts/select_lambda.py

# Technical owner per experiment (execution plan Section 5.1 / 9). Running a job is not ownership.
EXPERIMENT_OWNERS = {"E1": "Kaushik", "E3b": "Kaushik", "E3": "Kaushik",
                     "E2": "Khushi", "E4b": "Khushi", "E4": "Khushi",
                     "E5": "Farzaan", "B0": "Farzaan"}

# Compute logistics stay in the internal queue (Section 10.8 firewall) and never in a run record.
LOGISTICS_KEYS = ("gpu_executor", "backup_executor", "executor", "lane", "queue_id", "launched_by")


class EvaluationRefused(RuntimeError):
    """A prediction file may not enter evaluation."""


def owner_of(experiment_id: str) -> str:
    base = str(experiment_id).split("-")[0]
    if base not in EXPERIMENT_OWNERS:
        raise KeyError(f"no technical owner recorded for experiment {experiment_id!r}")
    return EXPERIMENT_OWNERS[base]


# Experiments without an image backbone: plan 5.1 runs them on CPU (E5 may move to a T4 if CPU is too slow).
CPU_TRACK_EXPERIMENTS = ("E5", "B0")


def has_image_backbone(record: dict) -> bool:
    """ResNet-18 / DeiT-S runs: they need pretrained weights (D9) and the D16 GPU. E5 and B0 do not."""
    return str(record.get("experiment_id", "")).split("-")[0] not in CPU_TRACK_EXPERIMENTS


def logistics_fields(record: dict) -> list:
    return [k for k in record if k in LOGISTICS_KEYS or "executor" in str(k).lower()]


def read_run_record(runs_dir, run_id: str) -> dict:
    if not RUN_ID_PATTERN.match(run_id or "") or ".." in run_id:
        raise EvaluationRefused(f"run_id {run_id!r} is not a valid run identifier")
    p = Path(runs_dir) / run_id / "run.json"
    if not p.is_file():
        raise EvaluationRefused(f"{run_id}: no run record at {p}")
    return json.loads(p.read_text())


def check_evaluable(record: dict, prediction_path, has_test_rows: bool) -> None:
    """Raise EvaluationRefused unless this run's prediction file may enter evaluation."""
    rid = record.get("run_id", "?")
    status = record.get("validation_status", "pending")
    if status != "RESULT_VALIDATED":
        raise EvaluationRefused(f"{rid}: validation_status is {status!r}, not RESULT_VALIDATED")
    if record.get("status") not in EVALUABLE_RUN_STATUSES:
        raise EvaluationRefused(f"{rid}: run status {record.get('status')!r} is not evaluable")
    bad = logistics_fields(record)
    if bad:
        raise EvaluationRefused(f"{rid}: run record holds compute-logistics fields {bad}")
    if sha256_file(str(prediction_path)) != (record.get("validation") or {}).get("prediction_sha256"):
        raise EvaluationRefused(f"{rid}: prediction file changed after RESULT_VALIDATED (sha256 mismatch)")
    if has_test_rows and not run_matches_frozen(record):
        raise EvaluationRefused(f"{rid}: test-split predictions without a matching frozen configuration")
