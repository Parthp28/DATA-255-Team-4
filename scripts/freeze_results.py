"""Results freeze (plan Section 22; gate G6). Owner: Parth.

    python scripts/freeze_results.py --check-complete        # 2:00 pm completeness check
    python scripts/freeze_results.py --write                 # 5:00 pm: outputs/results_freeze.json, then tag results-v1

For every final experiment (E1, E2, E3b, E4b, E3, E4, E5) and seed 0-2 exactly one run must be:
RESULT_VALIDATED; complete or promoted; trained with the configuration frozen in configs/frozen/<exp>.yaml
(verified file; same scientific config for all three seeds); carrying test predictions whose sha256 equals
the validated hash; full data (no --max-steps), clean git tree, non-synthetic, pretrained backbone for image
models, and on the GPU type fixed by D16 (NVIDIA T4).

The manifest records scientific provenance only (run ids, prediction hashes, frozen-config hashes, commit,
data and render versions, technical owner, GPU model); never who ran a job. scripts/evaluate.py
--runs-from reads it and re-verifies every prediction hash. B0 baselines join with --include-b0 once they
have run records.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.schema import sha256_file  # noqa: E402
from src.evaluation.frozen import FrozenConfigError, run_matches_frozen, verify_frozen  # noqa: E402
from src.evaluation.gate import logistics_fields  # noqa: E402

FINAL_EXPERIMENTS = ("E1", "E2", "E3b", "E4b", "E3", "E4", "E5")
SEEDS = (0, 1, 2)
D16_GPU = "T4"        # decision D16 (unchanged): every reported run on an NVIDIA T4


def problems(rec: dict, frozen_path: Path, frozen_sha: str, test_mode: bool) -> list:
    p = []
    if rec.get("validation_status") != "RESULT_VALIDATED":
        p.append(f"validation_status {rec.get('validation_status')}")
    if rec.get("status") not in ("complete", "promoted"):
        p.append(f"status {rec.get('status')}")
    if Path(str(rec.get("frozen_config_path", ""))).resolve() != frozen_path.resolve() \
            or rec.get("frozen_config_sha256") != frozen_sha or not run_matches_frozen(rec):
        p.append("not trained with the frozen config")
    if "test" not in (rec.get("prediction_splits") or []):
        p.append("no test predictions")
    paths = rec.get("prediction_paths") or []
    if not paths or not Path(paths[0]).exists():
        p.append("prediction file missing")
    elif sha256_file(paths[0]) != (rec.get("validation") or {}).get("prediction_sha256"):
        p.append("prediction file changed after validation")
    if logistics_fields(rec):
        p.append("compute-logistics fields in run record")
    if not test_mode:
        if rec.get("git_dirty") is not False:
            p.append("dirty git tree")
        if rec.get("max_steps"):
            p.append("debug run (--max-steps)")
        if str(rec.get("data_version", "")).startswith("synthetic"):
            p.append("synthetic data")
        if D16_GPU not in str(rec.get("gpu_model", "")):
            p.append(f"gpu_model {rec.get('gpu_model')!r} is not an NVIDIA {D16_GPU} (D16)")
        if not str(rec.get("model", "")).startswith("numeric") and rec.get("pretrained") is not True:
            p.append("image model without pretrained weights (D9)")
    return p


def build_manifest(runs_dir, frozen_dir, experiments=FINAL_EXPERIMENTS, test_mode=False):
    recs = [json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("*/run.json"))]
    entries, missing = [], []
    for exp in experiments:
        fpath = Path(frozen_dir) / f"{exp}.yaml"
        try:
            frec = verify_frozen(fpath)
        except (FrozenConfigError, OSError) as exc:
            missing.append(f"{exp}: no verified frozen config ({exc})")
            continue
        if frec.get("test_mode") and not test_mode:
            missing.append(f"{exp}: frozen in test mode")
            continue
        for seed in SEEDS:
            cands = [r for r in recs if r.get("experiment_id") == exp and int(r.get("seed", -1)) == seed
                     and r.get("status") != "superseded" and r.get("validation_status") not in ("rejected", "superseded")]
            why = {r["run_id"]: problems(r, fpath, frec["config_sha256"], test_mode) for r in cands}
            ok = [r for r in cands if not why[r["run_id"]]]
            if len(ok) != 1:
                missing.append(f"{exp} s{seed}: {len(ok)} eligible runs (need exactly 1); candidates {why}")
                continue
            r = ok[0]
            entries.append({"experiment_id": exp, "seed": seed, "run_id": r["run_id"], "status": r["status"],
                            "prediction_path": r["prediction_paths"][0],
                            "prediction_sha256": sha256_file(r["prediction_paths"][0]),
                            "frozen_config_path": str(fpath).replace("\\", "/"),
                            "frozen_config_sha256": frec["config_sha256"], "config_sha256": r.get("config_sha256"),
                            "git_commit": r.get("git_commit"), "data_version": r.get("data_version"),
                            "render_version": r.get("render_version"), "technical_owner": r.get("technical_owner"),
                            "gpu_model": r.get("gpu_model"), "validated_at": r["validation"].get("validated_at")})
    return entries, missing


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--frozen-dir", default="configs/frozen")
    ap.add_argument("--out", default="outputs/results_freeze.json")
    ap.add_argument("--experiments", nargs="+", default=list(FINAL_EXPERIMENTS))
    ap.add_argument("--check-complete", action="store_true", help="report only; write nothing")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--test-mode", action="store_true", help="unit tests only: synthetic data, any device")
    a = ap.parse_args(argv)
    entries, missing = build_manifest(a.runs_dir, a.frozen_dir, a.experiments, a.test_mode)
    if missing:
        print("NOT COMPLETE:\n  " + "\n  ".join(missing))
        return 1
    print(f"complete: {len(entries)} runs ({len(a.experiments)} experiments x {len(SEEDS)} seeds)")
    if a.write and not a.check_complete:
        man = {"schema": "results_freeze_v1", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "git_commit": _git_commit(), "required_gpu": D16_GPU, "test_mode": bool(a.test_mode),
               "data_versions": sorted({e["data_version"] for e in entries}),
               "render_versions": sorted({e["render_version"] for e in entries}), "runs": entries}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(man, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
