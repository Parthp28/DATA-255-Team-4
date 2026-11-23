"""Scientific validation of a run by its technical owner: RESULT_VALIDATED / rejected / superseded.

    python scripts/validate_run.py --run-dir outputs/runs/E1_s1_3da7f438 --owner Kaushik --data-root data/render_v1
    python scripts/validate_run.py --run-dir ... --owner Kaushik --reject "loss diverged at epoch 2"
    python scripts/validate_run.py --run-dir ... --owner Kaushik --supersede "E1 selection rejected at G4"

Writes ``validation_status`` and a ``validation`` block into the run's run.json (execution plan 10.5), the
reproducible record that scripts/evaluate.py checks. Nothing here reads or writes the internal operations
queue. Only the experiment's technical owner (Section 5.1) can sign. RESULT_VALIDATED checks (Section 14.4):
run complete; best epoch and a finite validation Macro F1 recorded; the prediction file passes the team
validator with full R0-R6 coverage (or every window for E5/B0); test predictions only from a verified frozen
config; no compute-logistics fields; and, outside --test-mode, a clean git tree, full data (no --max-steps
debug cap) and non-synthetic data.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from src.common.schema import sha256_file  # noqa: E402
from src.evaluation.frozen import FrozenConfigError, run_matches_frozen, verify_frozen  # noqa: E402
from src.evaluation.gate import has_image_backbone, logistics_fields, owner_of  # noqa: E402

import evaluate as ev  # noqa: E402


class ValidationRefused(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_checks(rec: dict, data_root, test_mode: bool = False) -> tuple[dict, str]:
    checks = {"status_complete": rec.get("status") in ("complete", "promoted"),
              "best_epoch_recorded": isinstance(rec.get("best_epoch"), int) and rec["best_epoch"] >= 0,
              "val_macro_f1_finite": isinstance(rec.get("best_val_f1"), (int, float))
              and math.isfinite(rec["best_val_f1"]),
              "no_logistics_fields": not logistics_fields(rec)}
    if not test_mode:
        checks["clean_git_tree"] = rec.get("git_dirty") is False
        checks["full_data_no_step_cap"] = rec.get("max_steps") in (None, 0)
        checks["not_synthetic"] = not str(rec.get("data_version", "")).startswith("synthetic")
        checks["pretrained_backbone"] = rec.get("pretrained") is True or not has_image_backbone(rec)
    paths = rec.get("prediction_paths") or []
    sha = ""
    if not paths or not Path(paths[0]).exists():
        checks["prediction_file_exists"] = False
        return checks, sha
    pred = pd.read_parquet(paths[0])
    windows = pd.read_parquet(Path(data_root) / "windows.parquet")
    reg_path = Path(data_root) / "render_registry.parquet"
    registry = pd.read_parquet(reg_path) if reg_path.exists() else None
    try:
        ev.check_prediction_file(pred, windows, registry)
        checks["prediction_schema_and_full_coverage"] = True
    except Exception as exc:  # noqa: BLE001 - reported as a failed check
        checks["prediction_schema_and_full_coverage"] = f"FAILED: {exc}"
    checks["prediction_run_id_matches"] = set(pred["run_id"]) == {rec["run_id"]}
    if (pred["split"] == "test").any():
        ok = run_matches_frozen(rec)
        if ok:
            try:
                ok = verify_frozen(rec["frozen_config_path"])["config_sha256"] == rec["frozen_config_sha256"]
            except (FrozenConfigError, OSError, KeyError):
                ok = False
        checks["test_predictions_from_frozen_config"] = ok
    sha = sha256_file(paths[0])
    return checks, sha


def validate(run_dir, owner, data_root=None, reject=None, supersede=None, test_mode=False) -> dict:
    run_dir = Path(run_dir)
    rec = json.loads((run_dir / "run.json").read_text())
    expected = owner_of(rec["experiment_id"])
    if owner.strip().lower() != expected.lower():
        raise ValidationRefused(f"only the technical owner of {rec['experiment_id']} ({expected}) can sign "
                                f"{rec['run_id']}")
    block = {"technical_owner": expected, "validated_at": _now()}
    if supersede:
        rec.update(validation_status="superseded", status="superseded", validation={**block, "reason": supersede})
    elif reject:
        rec.update(validation_status="rejected", validation={**block, "reason": reject})
    else:
        if data_root is None:
            raise ValidationRefused("--data-root is required for RESULT_VALIDATED")
        checks, sha = run_checks(rec, data_root, test_mode)
        bad = {k: v for k, v in checks.items() if v is not True}
        if bad:
            raise ValidationRefused(f"{rec['run_id']}: RESULT_VALIDATED refused, failed checks: {bad}")
        rec.update(validation_status="RESULT_VALIDATED",
                   validation={**block, "prediction_sha256": sha, "checks": checks, "test_mode": bool(test_mode)})
    tmp = run_dir / "run.json.tmp"
    tmp.write_text(json.dumps(rec, indent=2, default=str))
    tmp.replace(run_dir / "run.json")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--owner", required=True, help="technical owner of the experiment (Section 5.1)")
    ap.add_argument("--data-root", help="directory with windows.parquet and render_registry.parquet")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--reject", metavar="REASON")
    g.add_argument("--supersede", metavar="REASON")
    ap.add_argument("--test-mode", action="store_true", help="unit tests only: synthetic/debug runs allowed")
    a = ap.parse_args(argv)
    try:
        rec = validate(a.run_dir, a.owner, a.data_root, a.reject, a.supersede, a.test_mode)
    except ValidationRefused as exc:
        raise SystemExit(str(exc))
    print(f"{rec['run_id']}: {rec['validation_status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
