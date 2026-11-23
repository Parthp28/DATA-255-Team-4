"""Evaluate prediction files into outputs/eval/ (execution plan Sections 10.7, 15). Owner: Parth.

    python scripts/evaluate.py --data-root data/render_v1 --predictions outputs/predictions --split test
    python scripts/evaluate.py --data-root data/render_v1 --runs-from outputs/results_freeze.json --split test
    python scripts/evaluate.py --data-root outputs/synthetic_dev/data --predictions outputs/synthetic_dev/predictions \
        --split test --allow-synthetic                           # development fixtures only

--data-root holds windows.parquet and (for image models) render_registry.parquet, the same layout
scripts/train.py reads.

Evaluation gate (Sections 14.4, 16.1): a real prediction file is evaluated only if its run record
outputs/runs/<run_id>/run.json has validation_status RESULT_VALIDATED (set by the technical owner with
scripts/validate_run.py), its sha256 equals the validated hash, and any test rows come from a verified
frozen configuration. Synthetic fixtures are refused unless --allow-synthetic; that test-only path cannot
admit real data (it is keyed on the synthetic data_version), and its outputs are stamped SYNTHETIC.

metrics_long.csv rows (run_id, experiment_id, seed, split, render_set, metric, value):
  render_set R0..R6 (or NUM): accuracy, macro_f1, macro_auroc, precision_/recall_/f1_<down|neutral|up>,
      pred_share_<class>, max_pred_share, ece, brier; image models also flip_rate, prob_dispersion,
      delta_macro_f1, class_shift against R0
  render_set seen (mean over R1, R2, R4) and held_out (mean over R3, R5, R6): flip_rate, prob_dispersion
  render_set all: any_flip_rate
scripts/select_lambda.py (D13) reads macro_f1 on R0, R1, R2, R4 and flip_rate on seen from this table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.schema import (HELD_OUT_RENDERS, NUMERIC_RENDER_ID, PREDICTION_COLUMNS, RENDER_IDS,  # noqa: E402
                               sha256_file, validate_prediction_frame)
from src.evaluation.bootstrap import block_bootstrap  # noqa: E402
from src.evaluation.calibration import brier, ece  # noqa: E402
from src.evaluation.gate import EvaluationRefused, check_evaluable, read_run_record  # noqa: E402
from src.evaluation.metrics import macro_auroc_ovr, macro_f1, predictive_metrics  # noqa: E402
from src.evaluation.robustness import (FR_SEEN_RENDERS, any_flip_rate, class_shift, flip_rate,  # noqa: E402
                                       pivot_predictions, prob_dispersion)
from src.evaluation.seeds import cross_seed_disagreement, mean_sd, sign_agreement  # noqa: E402

METRICS_LONG_COLUMNS = ("run_id", "experiment_id", "seed", "split", "render_set", "metric", "value")
PER_RENDER_COLUMNS = ("experiment_id", "seed", "render_id", "FR", "PD", "dF1", "class_shift")
SEEDS_COLUMNS = ("experiment_id", "metric", "mean", "sd", "s0", "s1", "s2", "cross_seed_disagreement")
COMPARISONS_COLUMNS = ("comparison_id", "metric", "exp_a", "exp_b", "delta_mean", "delta_sd", "delta_s0",
                       "delta_s1", "delta_s2", "ci_low", "ci_high", "n_boot", "block_len", "sign_agree")
PROB_COLS = ["p_down", "p_neutral", "p_up"]
IMAGE_EXPS = ("E1", "E2", "E3b", "E4b", "E3", "E4")

# comparison_id -> list of (exp_a, exp_b, metrics); delta = metric(exp_b) - metric(exp_a) (Section 6).
ROBUST = ("FR_seen", "FR_held", "PD_seen", "PD_held", "macro_f1")
COMPARISONS = {
    "A1": [("E1", "E3b", ROBUST)],
    "A2": [("E3b", "E3", ROBUST + ("ece",))],
    "A3": [("E2", "E4b", ROBUST)],
    "A4": [("E4b", "E4", ROBUST + ("ece",))],
    "A5": [("E1", "E2", ROBUST), ("E3b", "E4b", ROBUST), ("E3", "E4", ROBUST)],
    "A6": [("E5", "E1", ("macro_f1", "macro_auroc")), ("E5", "E2", ("macro_f1", "macro_auroc"))],
    "A7": [(e, e, ("FR_held_minus_seen",)) for e in IMAGE_EXPS],
    "A8": [("B0*", e, ("macro_f1", "macro_auroc")) for e in IMAGE_EXPS + ("E5",)],
}


# --------------------------------------------------------------------------- input checks
def check_prediction_file(df: pd.DataFrame, windows: pd.DataFrame, registry: pd.DataFrame | None) -> None:
    """Team validator (columns, softmax sums, argmax, R0-R6 coverage) + checks evaluation relies on."""
    missing = sorted(set(PREDICTION_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"missing prediction columns {missing}")
    for col in ("run_id", "experiment_id", "seed"):
        if df[col].nunique() != 1:
            raise ValueError(f"one prediction file holds exactly one {col}")
    if not df["split"].isin(["val", "test"]).all():
        raise ValueError("prediction split must be val or test")
    p = df[PROB_COLS].to_numpy(np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities outside [0, 1]")
    if df.duplicated(["split", "window_id", "render_id"]).any():
        raise ValueError("duplicate (split, window_id, render_id) rows")
    numeric = (df["render_id"] == NUMERIC_RENDER_ID).all()
    for split, part in df.groupby("split"):
        if numeric:
            validate_prediction_frame(part, image_model=False)
            want = windows.loc[windows["split"] == split, ["window_id", "label"]]
            if set(part["window_id"]) != set(want["window_id"]):
                raise ValueError(f"incomplete coverage: {split} does not cover every {split} window")
            lab = want.set_index("window_id")["label"]
        else:
            if registry is None:
                raise ValueError("image-model predictions need render_registry.parquet in --data-root")
            want = registry.loc[registry["split"] == split, ["window_id", "render_id", "label"]]
            validate_prediction_frame(part, expected_pairs=list(zip(want["window_id"], want["render_id"])),
                                      image_model=True)
            lab = want.drop_duplicates("window_id").set_index("window_id")["label"]
        if not (part["true_label"].to_numpy() == lab.loc[part["window_id"]].to_numpy()).all():
            raise ValueError(f"true_label differs from the registry in {split}")


class Run:
    """One run's predictions on one split, pivoted to a common window order."""

    def __init__(self, df: pd.DataFrame, wids: np.ndarray):
        self.run_id, self.exp, self.seed = df["run_id"].iloc[0], df["experiment_id"].iloc[0], int(df["seed"].iloc[0])
        self.numeric = bool((df["render_id"] == NUMERIC_RENDER_ID).all())
        self.renders = [NUMERIC_RENDER_ID] if self.numeric else list(RENDER_IDS)
        ids, self.y, self.pred, self.probs = pivot_predictions(df, self.renders)
        if not np.array_equal(ids, wids):
            raise ValueError(f"{self.run_id}: windows differ from the evaluation window set")

    def col(self, r):
        return 0 if self.numeric else self.renders.index(r)

    def metric(self, name, idx=None):
        idx = np.arange(len(self.y)) if idx is None else idx
        y, P, p = self.y[idx], self.pred[idx], self.probs[idx]
        j0 = self.col("R0")
        if name == "macro_f1":
            return macro_f1(y, P[:, j0])
        if name == "macro_auroc":
            return macro_auroc_ovr(y, p[:, j0])
        if name == "ece":
            return ece(p[:, j0], y)
        if name == "brier":
            return brier(p[:, j0], y)
        if self.numeric:
            return float("nan")
        fr = {r: flip_rate(P[:, self.col(r)], P[:, j0]) for r in RENDER_IDS}
        pdv = {r: prob_dispersion(p[:, self.col(r)], p[:, j0]) for r in RENDER_IDS}
        if name == "FR_seen":
            return float(np.mean([fr[r] for r in FR_SEEN_RENDERS]))
        if name == "FR_held":
            return float(np.mean([fr[r] for r in HELD_OUT_RENDERS]))
        if name == "PD_seen":
            return float(np.mean([pdv[r] for r in FR_SEEN_RENDERS]))
        if name == "PD_held":
            return float(np.mean([pdv[r] for r in HELD_OUT_RENDERS]))
        if name == "FR_held_minus_seen":
            return float(np.mean([fr[r] for r in HELD_OUT_RENDERS]) - np.mean([fr[r] for r in FR_SEEN_RENDERS]))
        if name == "any_flip":
            return any_flip_rate(P)
        raise KeyError(name)


def load_runs(pred_files, windows, registry, split, runs_dir, allow_synthetic):
    synthetic = bool(windows["data_version"].astype(str).str.startswith("synthetic").any())
    wids = np.array(sorted(windows.loc[windows["split"] == split, "window_id"]))
    runs = []
    for f in pred_files:
        df = pd.read_parquet(f)
        check_prediction_file(df, windows, registry)
        rid = df["run_id"].iloc[0]
        if synthetic:
            if not allow_synthetic:
                raise SystemExit(f"{rid}: synthetic data refused (pass --allow-synthetic for development fixtures)")
        else:
            try:
                check_evaluable(read_run_record(runs_dir, rid), f, bool((df["split"] == "test").any()))
            except EvaluationRefused as exc:
                raise SystemExit(f"evaluation refused: {exc}")
        part = df[df["split"] == split]
        if len(part):
            runs.append(Run(part, wids))
    return runs, wids, synthetic


# --------------------------------------------------------------------------- tables
def metrics_long(runs, split):
    rows = []
    for r in runs:
        j0 = r.col("R0")
        for rd in r.renders:
            j = r.col(rd)
            m = predictive_metrics(r.y, r.probs[:, j])
            m["ece"], m["brier"] = ece(r.probs[:, j], r.y), brier(r.probs[:, j], r.y)
            if not r.numeric:
                m["flip_rate"] = flip_rate(r.pred[:, j], r.pred[:, j0])
                m["prob_dispersion"] = prob_dispersion(r.probs[:, j], r.probs[:, j0])
                m["delta_macro_f1"] = macro_f1(r.y, r.pred[:, j]) - macro_f1(r.y, r.pred[:, j0])
                m["class_shift"] = class_shift(r.pred[:, j], r.pred[:, j0])
            rows += [(r.run_id, r.exp, r.seed, split, rd, k, v) for k, v in m.items()]
        if not r.numeric:
            rows += [(r.run_id, r.exp, r.seed, split, "seen", "flip_rate", r.metric("FR_seen")),
                     (r.run_id, r.exp, r.seed, split, "seen", "prob_dispersion", r.metric("PD_seen")),
                     (r.run_id, r.exp, r.seed, split, "held_out", "flip_rate", r.metric("FR_held")),
                     (r.run_id, r.exp, r.seed, split, "held_out", "prob_dispersion", r.metric("PD_held")),
                     (r.run_id, r.exp, r.seed, split, "all", "any_flip_rate", r.metric("any_flip"))]
    return pd.DataFrame(rows, columns=list(METRICS_LONG_COLUMNS))


def per_render(runs):
    rows = []
    for r in runs:
        if r.numeric:
            continue
        j0 = r.col("R0")
        f0 = macro_f1(r.y, r.pred[:, j0])
        for rd in RENDER_IDS:
            j = r.col(rd)
            rows.append((r.exp, r.seed, rd, flip_rate(r.pred[:, j], r.pred[:, j0]),
                         prob_dispersion(r.probs[:, j], r.probs[:, j0]), macro_f1(r.y, r.pred[:, j]) - f0,
                         class_shift(r.pred[:, j], r.pred[:, j0])))
    return pd.DataFrame(rows, columns=list(PER_RENDER_COLUMNS))


SEED_METRICS = ("macro_f1", "macro_auroc", "ece", "brier", "FR_seen", "FR_held", "PD_seen", "PD_held", "any_flip")


def by_exp(runs):
    out = {}
    for r in runs:
        out.setdefault(r.exp, []).append(r)
    return {k: sorted(v, key=lambda r: r.seed) for k, v in out.items()}


def seeds_table(runs):
    rows = []
    for exp, rs in by_exp(runs).items():
        csd = cross_seed_disagreement([r.pred[:, r.col("R0")] for r in rs])
        for m in SEED_METRICS:
            vals = [r.metric(m) for r in rs]
            if not np.isfinite(vals).any():
                continue
            mean, sd = mean_sd(vals)
            s = (vals + [np.nan] * 3)[:3]
            rows.append((exp, m, mean, sd, s[0], s[1], s[2], csd))
    return pd.DataFrame(rows, columns=list(SEEDS_COLUMNS))


def best_b0(groups):
    b0 = {k: v for k, v in groups.items() if k.startswith("B0")}
    if not b0:
        return None
    return max(b0, key=lambda k: np.mean([r.metric("macro_f1") for r in b0[k]]))


def comparisons(runs, anchor_dates, n_boot, block_len, seed=0):
    groups = by_exp(runs)
    b0 = best_b0(groups)
    rows = []
    for cid, items in COMPARISONS.items():
        for ea, eb, metrics in items:
            ea = b0 if ea == "B0*" else ea
            if ea not in groups or eb not in groups:
                continue
            A, B = groups[ea], groups[eb]
            for m in metrics:
                if ea == eb:     # A7: within-model contrast
                    deltas = [b.metric(m) for b in B]

                    def stat(idx, B=B, m=m):
                        return float(np.mean([b.metric(m, idx) for b in B]))
                else:
                    a_mean = np.mean([a.metric(m) for a in A])
                    paired = len(A) == len(B)
                    deltas = [b.metric(m) - (a.metric(m) if paired else a_mean) for a, b in zip(A if paired else B, B)]

                    def stat(idx, A=A, B=B, m=m):
                        return float(np.mean([b.metric(m, idx) for b in B]) - np.mean([a.metric(m, idx) for a in A]))
                bs = block_bootstrap(stat, anchor_dates, n_boot=n_boot, block_len=block_len, seed=seed)
                mean, sd = mean_sd(deltas)
                d = (list(deltas) + [np.nan] * 3)[:3]
                rows.append((cid, m, ea, eb, mean, sd, d[0], d[1], d[2], bs["ci_low"], bs["ci_high"], n_boot,
                             block_len, sign_agreement(deltas)))
    return pd.DataFrame(rows, columns=list(COMPARISONS_COLUMNS))


def evaluate(pred_files, data_root, split, out_dir, runs_dir="outputs/runs", allow_synthetic=False,
             n_boot=1000, block_len=63):
    windows = pd.read_parquet(Path(data_root) / "windows.parquet")
    reg_path = Path(data_root) / "render_registry.parquet"
    registry = pd.read_parquet(reg_path) if reg_path.exists() else None
    runs, wids, synthetic = load_runs(pred_files, windows, registry, split, runs_dir, allow_synthetic)
    anchor = windows.set_index("window_id").loc[wids, "anchor_date"].to_numpy()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"metrics_long.csv": metrics_long(runs, split), "per_render.csv": per_render(runs),
              "seeds.csv": seeds_table(runs), "comparisons.csv": comparisons(runs, anchor, n_boot, block_len)}
    for name, t in tables.items():
        t.to_csv(out / name, index=False)
    if synthetic:
        (out / "SYNTHETIC_NOT_RESULTS.txt").write_text("Computed from synthetic fixtures. Not results.\n")
    return tables


def files_from_freeze(manifest_path) -> list:
    """Prediction files listed in a results-freeze manifest {"runs": [{run_id, prediction_path,
    prediction_sha256}, ...]}, each verified against its frozen sha256."""
    man = json.loads(Path(manifest_path).read_text())
    files = []
    for r in man["runs"]:
        p = Path(r["prediction_path"])
        if sha256_file(str(p)) != r["prediction_sha256"]:
            raise SystemExit(f"{r['run_id']}: prediction file differs from {manifest_path}; evaluation refused")
        files.append(p)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True, help="directory with windows.parquet (+ render_registry.parquet)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--predictions", help="directory of <run_id>.parquet prediction files")
    src.add_argument("--runs-from", help="results-freeze manifest (frozen run set with sha256 checks)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--allow-synthetic", action="store_true", help="development fixtures only")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--block-len", type=int, default=63)
    a = ap.parse_args(argv)
    files = files_from_freeze(a.runs_from) if a.runs_from else sorted(Path(a.predictions).glob("*.parquet"))
    if not files:
        raise SystemExit("no prediction files found")
    tables = evaluate(files, a.data_root, a.split, a.out, a.runs_dir, a.allow_synthetic, a.n_boot, a.block_len)
    for k, t in tables.items():
        print(f"{k}: {len(t)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
