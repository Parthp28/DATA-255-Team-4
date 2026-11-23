"""Freeze a selected experiment configuration into configs/frozen/<experiment>.yaml (immutable).

    # D11 (E1, E2, E5): freeze the selected tuning run's configuration
    python scripts/freeze_config.py --from-run outputs/runs/E1_s0_3da7f438 --rule "D11: highest validation Macro F1" \
        --out configs/frozen/E1.yaml
    # D13 (E3, E4): freeze the lambda run that scripts/select_lambda.py selected
    python scripts/freeze_config.py --from-run outputs/runs/E3_s0_c5aa882d --rule "D13 (select_lambda.py log)" \
        --out configs/frozen/E3.yaml
    # E3b / E4b (no selection of their own): the experiment YAML with the frozen E1 / E2 lr and weight decay
    python scripts/freeze_config.py --config experiments/E3b.yaml --inherit configs/frozen/E1.yaml \
        --rule "inherits configs/frozen/E1.yaml" --out configs/frozen/E3b.yaml

Only a verified frozen config makes scripts/train.py produce test-split predictions. Selection itself is
done from validation metrics (D11, D13); this script reads no metric and never touches the test split.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.common.config import ExperimentConfig  # noqa: E402
from src.evaluation.frozen import build_frozen, verify_frozen, write_frozen  # noqa: E402

TUNABLE = ("lr", "weight_decay", "consistency_lambda")


def config_from_run(run_dir: Path) -> tuple[ExperimentConfig, dict]:
    rec = json.loads((run_dir / "run.json").read_text())
    if rec.get("status") not in ("complete", "promoted"):
        raise SystemExit(f"{rec.get('run_id')}: status {rec.get('status')!r}; only a completed run can be frozen")
    if rec.get("validation_status") in ("rejected", "superseded"):
        raise SystemExit(f"{rec.get('run_id')}: validation_status {rec['validation_status']!r}")
    if int(rec.get("seed", -1)) != 0:
        raise SystemExit("selection uses seed-0 runs only")
    fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
    cfg = ExperimentConfig(**{k: rec[k] for k in fields if k in rec})
    return cfg, {"source_run_id": rec["run_id"], "source_config_sha256": rec.get("config_sha256"),
                 "source_best_val_f1": rec.get("best_val_f1")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-run", help="outputs/runs/<run_id> of the selected seed-0 run")
    src.add_argument("--config", help="experiment YAML (E3b/E4b, which have no selection of their own)")
    ap.add_argument("--inherit", help="verified frozen parent (E3b/E3 <- configs/frozen/E1.yaml, E4b/E4 <- E2.yaml)")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help=f"only {TUNABLE}")
    ap.add_argument("--rule", required=True, help="selection rule or provenance note recorded in freeze_record")
    ap.add_argument("--out", required=True, help="configs/frozen/<experiment>.yaml")
    a = ap.parse_args(argv)
    if a.from_run:
        cfg, prov = config_from_run(Path(a.from_run))
    else:
        cfg, prov = ExperimentConfig.from_yaml(a.config), {"source_config": a.config.replace("\\", "/")}
    if a.inherit:
        rec = verify_frozen(a.inherit)
        parent = ExperimentConfig.from_yaml(a.inherit)
        cfg = dataclasses.replace(cfg, lr=parent.lr, weight_decay=parent.weight_decay)
        prov.update(inherit_path=a.inherit.replace("\\", "/"), inherit_config_sha256=rec["config_sha256"])
    for item in a.set:
        k, _, v = item.partition("=")
        if k not in TUNABLE:
            raise SystemExit(f"--set may only change {TUNABLE}, not {k!r}")
        cfg = dataclasses.replace(cfg, **{k: float(yaml.safe_load(v))})
    path = write_frozen(build_frozen(cfg, {"rule": a.rule, **prov}), a.out)
    print(f"{cfg.experiment_id} frozen -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
