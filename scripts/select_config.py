"""Selection step D11 (SEL-E1, SEL-E2, E5): freeze the best seed-0 tuning run. Plan owner: Kaushik.

    python scripts/select_config.py --experiment E1 --runs-dir outputs/runs --out configs/frozen/E1.yaml
    python scripts/select_config.py --experiment E5 --runs-dir outputs/runs --out configs/frozen/E5.yaml   # 2 lr runs

D11: among the seed-0 tuning runs of one experiment (E1, E2: 2 lr x 2 weight decay; E5: 2 lr), select the
highest validation Macro F1 (best_val_f1 in run.json, the early-stopping metric over the experiment's own
renderings). Ties: smallest run_id. Validation metrics only; no test-split number exists for these runs.
The selected configuration is frozen into configs/frozen/<experiment>.yaml (scripts/freeze_config.py
format); scripts/predict.py then promotes the selected seed-0 run with test predictions.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import ExperimentConfig  # noqa: E402
from src.evaluation.frozen import build_frozen, science_sha256, write_frozen  # noqa: E402

EXPECTED = {"E1": 4, "E2": 4, "E5": 2}                 # D11 tuning budgets
TUNED = ("lr", "weight_decay")


class SelectionError(RuntimeError):
    pass


def run_config(rec: dict) -> ExperimentConfig:
    fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
    return ExperimentConfig(**{k: rec[k] for k in fields if k in rec})


def ineligible(rec: dict, experiment: str, test_mode: bool) -> str:
    if rec.get("experiment_id") != experiment or int(rec.get("seed", -1)) != 0:
        return "other experiment or seed"
    if rec.get("status") not in ("complete", "promoted"):
        return f"status {rec.get('status')}"
    if rec.get("validation_status") in ("rejected", "superseded"):
        return f"validation_status {rec.get('validation_status')}"
    if rec.get("frozen") or rec.get("frozen_config_path"):
        return "already a frozen-config run"
    if not isinstance(rec.get("best_val_f1"), (int, float)) or not math.isfinite(rec["best_val_f1"]):
        return "no finite validation Macro F1"
    if not test_mode:
        if rec.get("max_steps"):
            return "debug run (--max-steps)"
        if rec.get("git_dirty") is not False:
            return "dirty git tree"
        if str(rec.get("data_version", "")).startswith("synthetic"):
            return "synthetic data"
    return ""


def select(runs_dir, experiment: str, expected: int | None = None, test_mode: bool = False):
    recs, skipped = [], {}
    for p in sorted(Path(runs_dir).glob("*/run.json")):
        rec = json.loads(p.read_text())
        why = ineligible(rec, experiment, test_mode)
        if why:
            if rec.get("experiment_id") == experiment:
                skipped[rec.get("run_id", p.parent.name)] = why
            continue
        recs.append(rec)
    expected = EXPECTED.get(experiment) if expected is None else expected
    if expected and len(recs) != expected:
        raise SelectionError(f"{experiment}: expected {expected} eligible seed-0 tuning runs, found {len(recs)} "
                             f"(skipped: {skipped})")
    if not recs:
        raise SelectionError(f"{experiment}: no eligible tuning runs (skipped: {skipped})")
    base = {science_sha256(dataclasses.replace(run_config(r), lr=0.0, weight_decay=0.0)) for r in recs}
    if len(base) != 1:
        raise SelectionError(f"{experiment}: candidates differ in more than {TUNED}; refusing to select")
    best = sorted(recs, key=lambda r: (-float(r["best_val_f1"]), r["run_id"]))[0]
    table = [{"run_id": r["run_id"], "lr": r["lr"], "weight_decay": r["weight_decay"],
              "val_macro_f1_own_renders": r["best_val_f1"], "selected": r is best}
             for r in sorted(recs, key=lambda r: r["run_id"])]
    return best, table, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", required=True, choices=sorted(EXPECTED))
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--out", required=True, help="configs/frozen/<experiment>.yaml")
    ap.add_argument("--expected", type=int, help="override the D11 run count (tests only)")
    ap.add_argument("--log-dir", default="outputs/selection")
    ap.add_argument("--test-mode", action="store_true", help="unit tests only: allow debug / dirty / synthetic runs")
    a = ap.parse_args(argv)
    try:
        best, table, skipped = select(a.runs_dir, a.experiment, a.expected, a.test_mode)
    except SelectionError as exc:
        raise SystemExit(str(exc))
    prov = {"rule": "D11: highest validation Macro F1 (own renderings), seed 0", "source_run_id": best["run_id"],
            "source_config_sha256": best.get("config_sha256"), "candidates": table, "test_mode": bool(a.test_mode)}
    path = write_frozen(build_frozen(run_config(best), prov), a.out)
    log = {"experiment_id": a.experiment, "rule": prov["rule"], "selected_run_id": best["run_id"],
           "frozen_config": str(path).replace("\\", "/"), "candidates": table, "skipped": skipped,
           "metrics_used": "validation only", "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(a.log_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.log_dir) / f"SEL-{a.experiment}.json").write_text(json.dumps(log, indent=2))
    print(f"{a.experiment}: selected {best['run_id']} (val macro F1 {best['best_val_f1']:.4f}) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
