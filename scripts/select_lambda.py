"""Decision D13: select the consistency weight lambda for E3 and E4 (CPU only, standard library only).

Rule (execution plan v2, D13; not modified here)
    Among the candidates lambda in {0.3, 1, 3}, keep those whose VALIDATION Macro F1 is within 0.01 of
    the matching augmentation-only control (E3 -> E3b, E4 -> E4b). Select the one with the LOWEST
    VALIDATION seen-rendering flip rate (FR_seen). If no candidate is within 0.01, select the highest
    validation Macro F1. Seed 0 and validation data only; E3 and E4 are selected independently.

How the two quantities are measured
    * Validation Macro F1: the MEAN of the per-rendering validation Macro F1 over the training renderings
      R0, R1, R2, R4, for the candidates and for the matching control alike; all four must be present.
      This is the team's agreed interpretation of Section 5.1 ("validation Macro F1 averaged over each
      experiment's own training renderings"), which defines the early-stopping metric; D13 itself says
      only "validation Macro F1". It is a mean of four per-render values, not an F1 over pooled predictions.
    * FR_seen: the validation flip rate on render_set "seen", defined over R1, R2, R4 (Section 15.1: mean
      flip rate over {R1, R2, R4}; R0 is excluded because its flip rate is 0 by definition). This script
      trusts the aggregate ``seen`` row of the metrics table. PARTH MUST VERIFY that the ``seen`` row of
      metrics_long.csv is the mean over R1, R2, R4 (and not over R0, R1, R2, R4) before the real selection.
    * "Within 0.01" is one-sided with a 1e-9 float tolerance: F1(candidate) >= F1(control) - 0.01.
      A candidate that is better than its control is never penalized.
    * Ties (deterministic, independent of input order):
        eligible candidates : lowest FR_seen, then highest F1, then smallest lambda
        fallback            : highest F1, then lowest FR_seen, then smallest lambda
      Values equal within 1e-9 count as equal.

Structure
    * :func:`select_lambda` is the pure rule: numbers in, a :class:`Selection` out. No files, no CSV.
    * :func:`read_metrics_csv`, :func:`extract_inputs` and the run resolvers read the inputs and hand plain
      numbers to the rule. They REFUSE test-split, held-out-rendering and non-seed-0 rows, so this script can
      never be run on data the rule is not allowed to see.
    * :func:`main` is the command line.

Inputs
    1. The metrics table: exactly the planned ``outputs/eval/metrics_long.csv`` columns (Section 10.7),
       ``run_id, experiment_id, seed, split, render_set, metric, value``. No column is added to that schema.
    2. Where lambda comes from. metrics_long.csv has no lambda and a run_id (for example
       ``E3_s1_final_a1b2c3d``) does not encode it, so lambda is read from the run configuration:
         --runs-dir DIR     DIR/<run_id>/run.json, the run record of Section 10.5 (config fields of
                            Section 10.4 plus status). Fields used: ``consistency_lambda`` (required for
                            E3/E4, absent or 0 for E3b/E4b), ``experiment_id`` and ``seed`` (cross-checked
                            against the table), ``status`` (REQUIRED: "superseded" runs are skipped, anything
                            other than "complete" or "promoted" is refused) and ``git_dirty`` (REQUIRED and
                            must be false: Section 14.4 asks for a clean tree). ASSUMPTION: flat top-level
                            keys; no such file exists in the repo yet, so the layout is unverified.
         --lambda-map CSV   TEMPORARY adapter for use until run.json files are available: a CSV with columns
                            ``run_id, consistency_lambda``. It carries no run metadata, so status and git_dirty
                            are NOT checked, and it needs --allow-unchecked-run-metadata. Not for the real
                            selection. Remove it once --runs-dir works on real runs.

Blocking prerequisites for a REAL selection (not checkable by this script; open until integration)
    The documented run record (Section 10.5) does not identify debug runs: Section 14.4 says 20%-subset debug
    runs are never promoted, but no field marks one, and the key name of the recorded GPU model is not given.
    RESULT_VALIDATED is recorded in the internal run queue, which this script must not read; the Eval gate
    (Section 16) has evaluate.py enforce it before any metrics_long.csv row exists. So, before the G5
    selection, the team must confirm that:
      1. the metrics table was produced by evaluate.py, and every run in it is RESULT_VALIDATED by its owner;
      2. every run was trained on the FULL training data (no 20% debug subset) on the D16 GPU;
      3. Parth has verified the aggregate ``seen`` row (above), and the real run.json layout matches the
         assumption above.
    Until then the operator must say so explicitly with --attest-validated-full-data (items 1 and 2); the
    attestation is printed in the log. Without it the script refuses to select.

Output
    A text log on stdout (G5 asks for this log to be committed; --log writes it to a file) and, with --out, a
    CSV with one row per candidate (:data:`OUT_COLUMNS`; a proposal, not a frozen schema). Exit code 0 on
    success, 2 if the inputs are invalid or the attestation is missing (nothing is selected).
"""
import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

CANDIDATE_LAMBDAS = (0.3, 1.0, 3.0)  # D12
F1_MARGIN = 0.01  # D13
EPS = 1e-9  # float tolerance for the 0.01 boundary and for ties; far below any meaningful metric difference
BASELINE_OF = {"E3": "E3b", "E4": "E4b"}  # Section 5.1
SELECTION_SEED = 0
SELECTION_SPLIT = "val"
TRAIN_RENDERS = ("R0", "R1", "R2", "R4")  # training renderings of E3b/E3/E4b/E4 (Section 5.1, D8 seen set)
FR_RENDER_SET = "seen"  # FR_seen
SEEN_SIDE_RENDER_SETS = TRAIN_RENDERS + (FR_RENDER_SET,)  # anything else (R3, R5, R6, held_out, ...) is refused
REQUIRED_COLUMNS = ("run_id", "experiment_id", "seed", "split", "render_set", "metric", "value")
USABLE_RUN_STATUSES = ("complete", "promoted")
SUPERSEDED_STATUS = "superseded"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")  # also keeps run_id from escaping --runs-dir
OUT_COLUMNS = (
    "experiment_id", "baseline_experiment_id", "run_id", "baseline_run_id", "lambda", "val_macro_f1",
    "baseline_val_macro_f1", "f1_delta", "val_fr_seen", "eligible", "selected", "rule", "f1_renders", "f1_margin",
    "run_metadata_checked",
)
ATTESTATION = (
    "operator attests, NOT verified by this script: every run is RESULT_VALIDATED by its technical owner, was trained "
    "on the full training data (not a 20% debug subset) on the D16 GPU, and its metrics come from evaluate.py"
)


class SelectionInputError(ValueError):
    """The inputs are invalid or forbidden for selection; nothing is selected."""


# --------------------------------------------------------------------------------------------------
# Pure selection rule
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    lam: float
    val_macro_f1: float  # mean over R0, R1, R2, R4
    val_fr_seen: float
    run_id: str = field(default="", compare=False)  # for the audit log only


@dataclass(frozen=True)
class CandidateResult:
    lam: float
    val_macro_f1: float
    val_fr_seen: float
    f1_delta: float  # candidate minus control
    eligible: bool
    run_id: str = field(default="", compare=False)


@dataclass(frozen=True)
class Selection:
    experiment_id: str
    baseline_experiment_id: str
    baseline_val_macro_f1: float
    selected_lambda: float
    rule: str  # "eligible" (lowest FR among F1-eligible) or "fallback" (highest F1)
    reason: str
    results: Tuple[CandidateResult, ...]  # sorted by lambda
    margin: float
    baseline_run_id: str = field(default="", compare=False)


def _is_real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_metric(name: str, value) -> float:
    if not _is_real(value) or not math.isfinite(value):
        raise SelectionInputError(f"{name} must be a finite number, got {value!r}")
    if not 0.0 <= value <= 1.0:
        raise SelectionInputError(f"{name} must be in [0, 1], got {value!r}")
    return float(value)


def _canonical_lambda(lam, lambdas: Sequence[float]) -> float:
    if not _is_real(lam) or not math.isfinite(lam) or lam <= 0:
        raise SelectionInputError(f"lambda must be a finite number > 0, got {lam!r}")
    for ref in lambdas:
        if math.isclose(lam, ref, rel_tol=1e-9, abs_tol=0.0):
            return float(ref)
    raise SelectionInputError(f"lambda {lam!r} is not a candidate; the candidates are {list(lambdas)}")


def select_lambda(
    experiment_id: str,
    baseline_val_macro_f1: float,
    candidates: Iterable[Candidate],
    *,
    margin: float = F1_MARGIN,
    lambdas: Sequence[float] = CANDIDATE_LAMBDAS,
    baseline_run_id: str = "",
) -> Selection:
    """Apply D13 to validation seed-0 numbers. Pure: no I/O, independent of the order of ``candidates``."""
    if experiment_id not in BASELINE_OF:
        raise SelectionInputError(f"experiment_id must be one of {sorted(BASELINE_OF)}, got {experiment_id!r}")
    base = _check_metric("baseline validation Macro F1", baseline_val_macro_f1)
    if not _is_real(margin) or not math.isfinite(margin) or margin < 0:
        raise SelectionInputError(f"margin must be a finite number >= 0, got {margin!r}")

    by_lambda: Dict[float, Candidate] = {}
    for c in candidates:
        lam = _canonical_lambda(c.lam, lambdas)
        if lam in by_lambda:
            raise SelectionInputError(f"{experiment_id}: duplicate candidate for lambda={lam:g}")
        by_lambda[lam] = Candidate(
            lam, _check_metric(f"{experiment_id} lambda={lam:g} validation Macro F1", c.val_macro_f1),
            _check_metric(f"{experiment_id} lambda={lam:g} validation FR_seen", c.val_fr_seen), c.run_id,
        )
    missing = [lam for lam in lambdas if lam not in by_lambda]
    if missing:
        raise SelectionInputError(
            f"{experiment_id}: missing candidate(s) for lambda={[f'{m:g}' for m in missing]}; all of {[f'{m:g}' for m in lambdas]} are required"
        )

    ordered = [by_lambda[lam] for lam in sorted(by_lambda)]  # ascending lambda: final ties resolve to the smaller lambda
    results = []
    for c in ordered:
        delta = c.val_macro_f1 - base
        results.append(CandidateResult(c.lam, c.val_macro_f1, c.val_fr_seen, delta, delta >= -margin - EPS, c.run_id))

    pool = [r for r in results if r.eligible]
    base_name = BASELINE_OF[experiment_id]
    if pool:
        best = pool[0]
        for r in pool[1:]:  # lowest FR_seen, then highest F1, then (by iteration order) smallest lambda
            if r.val_fr_seen < best.val_fr_seen - EPS or (
                abs(r.val_fr_seen - best.val_fr_seen) <= EPS and r.val_macro_f1 > best.val_macro_f1 + EPS
            ):
                best = r
        rule = "eligible"
        reason = (f"lowest validation seen flip rate ({best.val_fr_seen:.6f}) among {len(pool)} of {len(results)} candidates "
                  f"with validation Macro F1 within {margin:g} of {base_name} ({base:.6f})")
    else:
        best = results[0]
        for r in results[1:]:  # highest F1, then lowest FR_seen, then smallest lambda
            if r.val_macro_f1 > best.val_macro_f1 + EPS or (
                abs(r.val_macro_f1 - best.val_macro_f1) <= EPS and r.val_fr_seen < best.val_fr_seen - EPS
            ):
                best = r
        rule = "fallback"
        reason = (f"no candidate has validation Macro F1 within {margin:g} of {base_name} ({base:.6f}); "
                  f"selected the highest validation Macro F1 ({best.val_macro_f1:.6f})")
    return Selection(experiment_id, base_name, base, best.lam, rule, reason, tuple(results), float(margin), baseline_run_id)


# --------------------------------------------------------------------------------------------------
# Run resolvers: run_id -> lambda and run metadata
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RunInfo:
    lam: Optional[float]  # None when the run record has no consistency_lambda
    status: Optional[str] = None
    experiment_id: Optional[str] = None
    seed: Optional[int] = None
    git_dirty: Optional[bool] = None  # Section 10.5; None when the record does not say


Resolver = Callable[[str], RunInfo]


def _check_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.match(run_id or "") or ".." in run_id:
        raise SelectionInputError(f"run_id {run_id!r} is not a valid run identifier")


def runs_dir_resolver(runs_dir) -> Resolver:
    """Read ``<runs_dir>/<run_id>/run.json`` (Section 10.5; flat top-level keys assumed)."""
    base = Path(runs_dir)

    def resolve(run_id: str) -> RunInfo:
        _check_run_id(run_id)
        path = base / run_id / "run.json"
        if not path.is_file():
            raise SelectionInputError(f"no run record for run_id {run_id!r}: {path} does not exist")
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as err:
            raise SelectionInputError(f"{path}: cannot read run record ({err})") from None
        if not isinstance(rec, dict):
            raise SelectionInputError(f"{path}: run record must be a JSON object")
        lam = rec.get("consistency_lambda")
        if lam is not None and (not _is_real(lam) or not math.isfinite(lam)):
            raise SelectionInputError(f"{path}: consistency_lambda must be a finite number, got {lam!r}")
        seed = rec.get("seed")
        if seed is not None and (not _is_real(seed) or int(seed) != seed):
            raise SelectionInputError(f"{path}: seed must be an integer, got {seed!r}")
        dirty = rec.get("git_dirty")
        if dirty is not None and not isinstance(dirty, bool):
            raise SelectionInputError(f"{path}: git_dirty must be true or false, got {dirty!r}")
        return RunInfo(lam=lam, status=rec.get("status"), experiment_id=rec.get("experiment_id"),
                       seed=None if seed is None else int(seed), git_dirty=dirty)

    return resolve


def lambda_map_resolver(path) -> Resolver:
    """TEMPORARY adapter, used only until run.json files exist: CSV with ``run_id, consistency_lambda``."""
    table: Dict[str, float] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in ("run_id", "consistency_lambda") if c not in (reader.fieldnames or [])]
        if missing:
            raise SelectionInputError(f"{path}: missing column(s) {missing}")
        for i, row in enumerate(reader, start=2):
            rid, text = (row["run_id"] or "").strip(), (row["consistency_lambda"] or "").strip()
            _check_run_id(rid)
            if rid in table:
                raise SelectionInputError(f"{path} line {i}: duplicate run_id {rid!r}")
            try:
                val = float(text) if text else 0.0
            except ValueError:
                raise SelectionInputError(f"{path} line {i}: consistency_lambda {text!r} is not a number") from None
            if not math.isfinite(val):
                raise SelectionInputError(f"{path} line {i}: consistency_lambda {text!r} is not finite")
            table[rid] = val

    def resolve(run_id: str) -> RunInfo:
        _check_run_id(run_id)
        if run_id not in table:
            raise SelectionInputError(f"run_id {run_id!r} is not in the lambda map {path}")
        return RunInfo(lam=table[run_id])

    return resolve


# --------------------------------------------------------------------------------------------------
# Metrics table input (planned metrics_long.csv columns)
# --------------------------------------------------------------------------------------------------
def read_metrics_csv(path) -> List[dict]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SelectionInputError(f"{path}: missing column(s) {missing}; expected the metrics_long.csv columns {list(REQUIRED_COLUMNS)}")
        return [dict(r, _line=i) for i, r in enumerate(reader, start=2)]


def _describe(row: dict) -> str:
    return (f"line {row['_line']} (run_id={row['run_id']!r}, experiment={row['experiment_id']!r}, seed={row['seed']!r}, "
            f"split={row['split']!r}, render_set={row['render_set']!r}, metric={row['metric']!r})")


def _forbidden_reason(row: dict) -> Optional[str]:
    try:
        seed_ok = float(row["seed"]) == SELECTION_SEED
    except (TypeError, ValueError):
        seed_ok = False
    if row["split"] != SELECTION_SPLIT:
        return f"split {row['split']!r} is not {SELECTION_SPLIT!r}"
    if row["render_set"] not in SEEN_SIDE_RENDER_SETS:
        return f"render_set {row['render_set']!r} is not a seen-side rendering {list(SEEN_SIDE_RENDER_SETS)}"
    if not seed_ok:
        return f"seed {row['seed']!r} is not {SELECTION_SEED}"
    return None


def _parse_value(row: dict) -> float:
    try:
        v = float(row["value"])
    except (TypeError, ValueError):
        raise SelectionInputError(f"{_describe(row)}: value {row['value']!r} is not a number") from None
    if not math.isfinite(v):
        raise SelectionInputError(f"{_describe(row)}: value {row['value']!r} is not finite")
    return v


@dataclass(frozen=True)
class ExperimentInput:
    baseline_val_macro_f1: float
    baseline_run_id: str
    candidates: Tuple[Candidate, ...]


def extract_inputs(
    rows: Sequence[dict],
    experiments: Sequence[str],
    resolve_run: Resolver,
    *,
    f1_metric: str = "macro_f1",
    fr_metric: str = "flip_rate",
    filter_input: bool = False,
    require_run_metadata: bool = True,
) -> Tuple[Dict[str, ExperimentInput], Dict[str, int]]:
    """Turn metrics rows plus run records into ``{experiment: ExperimentInput}``.

    Rows of the experiments in play that are test-split, held-out/other-rendering or not seed 0 make this
    raise, unless ``filter_input`` is set, in which case they are dropped and counted (and never read).
    Rows of other experiments (E1, E2, ...) are ignored. With ``require_run_metadata`` (the default) every
    non-superseded run record must state ``status`` (complete or promoted) and ``git_dirty`` = false;
    only the temporary lambda-map adapter, which has no run records, switches this off.
    Returns the inputs and a row-count summary.
    """
    for e in experiments:
        if e not in BASELINE_OF:
            raise SelectionInputError(f"experiment must be one of {sorted(BASELINE_OF)}, got {e!r}")
    in_play = set(experiments) | {BASELINE_OF[e] for e in experiments}
    counts = {"rows": len(rows), "other_experiments": 0, "excluded": 0, "used": 0, "superseded_runs": 0}
    allowed, refused = [], []
    for row in rows:
        if row["experiment_id"] not in in_play:
            counts["other_experiments"] += 1
            continue
        why = _forbidden_reason(row)
        (refused if why else allowed).append((row, why) if why else row)
    if refused and not filter_input:
        shown = "; ".join(f"{_describe(r)}: {w}" for r, w in refused[:3])
        raise SelectionInputError(
            f"refusing input: {len(refused)} row(s) are outside validation / seed 0 / training renderings and seen (first: {shown}). "
            f"Pass a validation-only extract, or use --filter-input to drop them explicitly."
        )
    counts["excluded"] = len(refused)

    # Group the usable rows by run: per-render F1 values and the seen flip rate.
    runs: Dict[str, dict] = {}
    for row in allowed:
        is_f1 = row["metric"] == f1_metric and row["render_set"] in TRAIN_RENDERS
        is_fr = row["metric"] == fr_metric and row["render_set"] == FR_RENDER_SET
        if not (is_f1 or is_fr):
            continue
        rid = row["run_id"]
        _check_run_id(rid)
        run = runs.setdefault(rid, {"experiment": row["experiment_id"], "f1": {}, "fr": None})
        if run["experiment"] != row["experiment_id"]:
            raise SelectionInputError(f"run_id {rid!r} appears under two experiments: {run['experiment']!r} and {row['experiment_id']!r}")
        if is_f1:
            if row["render_set"] in run["f1"]:
                raise SelectionInputError(f"duplicate {f1_metric} for run {rid!r} on {row['render_set']}: {_describe(row)}")
            run["f1"][row["render_set"]] = _parse_value(row)
        else:
            if run["fr"] is not None:
                raise SelectionInputError(f"duplicate {fr_metric} (seen) for run {rid!r}: {_describe(row)}")
            run["fr"] = _parse_value(row)
        counts["used"] += 1

    controls: Dict[str, List[Tuple[str, float]]] = {}
    cands: Dict[str, Dict[float, List[Candidate]]] = {}
    for rid, run in runs.items():
        exp = run["experiment"]
        info = resolve_run(rid)
        if info.experiment_id is not None and info.experiment_id != exp:
            raise SelectionInputError(f"run {rid!r}: the table says experiment {exp!r} but its run record says {info.experiment_id!r}")
        if info.seed is not None and info.seed != SELECTION_SEED:
            raise SelectionInputError(f"run {rid!r}: its run record has seed {info.seed}, but selection uses seed {SELECTION_SEED} only")
        if info.status == SUPERSEDED_STATUS:
            counts["superseded_runs"] += 1
            continue
        if info.status is not None and info.status not in USABLE_RUN_STATUSES:
            raise SelectionInputError(f"run {rid!r}: status {info.status!r} is not usable; expected one of {list(USABLE_RUN_STATUSES)} (or {SUPERSEDED_STATUS!r} to skip)")
        if require_run_metadata:
            if info.status is None:
                raise SelectionInputError(f"run {rid!r}: its run record has no status, so it cannot be established that the run finished (Section 10.5)")
            if info.git_dirty is None:
                raise SelectionInputError(f"run {rid!r}: its run record has no git_dirty, so a clean tree cannot be established (Section 14.4)")
            if info.git_dirty:
                raise SelectionInputError(f"run {rid!r}: it was run on a dirty git tree (git_dirty is true); selection needs a clean tree (Section 14.4)")
        missing = [r for r in TRAIN_RENDERS if r not in run["f1"]]
        if missing:
            raise SelectionInputError(f"run {rid!r} ({exp}): validation {f1_metric} is missing for render(s) {missing}; the mean over {list(TRAIN_RENDERS)} needs all of them")
        f1 = sum(run["f1"][r] for r in TRAIN_RENDERS) / len(TRAIN_RENDERS)
        if exp in BASELINE_OF.values():
            if info.lam not in (None, 0):
                raise SelectionInputError(f"run {rid!r}: the control {exp} must have lambda 0, got {info.lam!r}")
            controls.setdefault(exp, []).append((rid, f1))
        else:
            if info.lam is None:
                raise SelectionInputError(f"run {rid!r} ({exp}): no consistency_lambda in its run record")
            if run["fr"] is None:
                raise SelectionInputError(f"run {rid!r} ({exp}): validation {fr_metric} on render_set {FR_RENDER_SET!r} is missing")
            cands.setdefault(exp, {}).setdefault(info.lam, []).append(Candidate(info.lam, f1, run["fr"], rid))

    out: Dict[str, ExperimentInput] = {}
    for e in experiments:
        b = BASELINE_OF[e]
        if not controls.get(b):
            raise SelectionInputError(f"missing control: no usable validation seed-0 run with {f1_metric} on {list(TRAIN_RENDERS)} for {b}")
        if len(controls[b]) > 1:
            raise SelectionInputError(f"duplicate control runs for {b}: {[r for r, _ in controls[b]]}; resolve reruns (mark the old ones superseded)")
        if not cands.get(e):
            raise SelectionInputError(f"missing candidates: no usable validation seed-0 runs for {e}")
        flat: List[Candidate] = []
        for lam, group in sorted(cands[e].items()):
            if len(group) > 1:
                raise SelectionInputError(f"duplicate candidate runs for {e} lambda={lam:g}: {[c.run_id for c in group]}; resolve reruns (mark the old ones superseded)")
            flat.append(group[0])
        out[e] = ExperimentInput(controls[b][0][1], controls[b][0][0], tuple(flat))
    return out, counts


# --------------------------------------------------------------------------------------------------
# Output and CLI
# --------------------------------------------------------------------------------------------------
def selection_rows(sel: Selection, run_metadata_checked: bool = True) -> List[dict]:
    return [
        {
            "experiment_id": sel.experiment_id, "baseline_experiment_id": sel.baseline_experiment_id, "run_id": r.run_id,
            "baseline_run_id": sel.baseline_run_id, "lambda": f"{r.lam:g}", "val_macro_f1": f"{r.val_macro_f1:.6f}",
            "baseline_val_macro_f1": f"{sel.baseline_val_macro_f1:.6f}", "f1_delta": f"{r.f1_delta:+.6f}",
            "val_fr_seen": f"{r.val_fr_seen:.6f}", "eligible": str(r.eligible).lower(),
            "selected": str(r.lam == sel.selected_lambda).lower(), "rule": sel.rule,
            "f1_renders": "+".join(TRAIN_RENDERS), "f1_margin": f"{sel.margin:g}",
            "run_metadata_checked": str(run_metadata_checked).lower(),
        }
        for r in sel.results
    ]


def format_log(sel: Selection) -> str:
    lines = [f"{sel.experiment_id} (control {sel.baseline_experiment_id}, run {sel.baseline_run_id or '?'}, "
             f"validation Macro F1 {sel.baseline_val_macro_f1:.6f}):"]
    for r in sel.results:
        mark = "  <== selected" if r.lam == sel.selected_lambda else ""
        lines.append(f"  lambda={r.lam:<4g} run {r.run_id or '?'}: val F1={r.val_macro_f1:.6f} (delta {r.f1_delta:+.6f}) "
                     f"val FR_seen={r.val_fr_seen:.6f} {'eligible' if r.eligible else 'not eligible'}{mark}")
    lines.append(f"  -> lambda = {sel.selected_lambda:g}: {sel.reason}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="D13 lambda selection for E3 and E4 (validation, seed 0, seen renderings only).")
    ap.add_argument("--metrics", required=True, help="metrics_long.csv (columns: run_id, experiment_id, seed, split, render_set, metric, value)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs-dir", help="directory of <run_id>/run.json run records (Section 10.5); the source of consistency_lambda")
    src.add_argument("--lambda-map", help="TEMPORARY adapter: CSV with columns run_id, consistency_lambda (use until run.json files exist)")
    ap.add_argument("--f1-metric", default="macro_f1", help="metric name of validation Macro F1 in the table")
    ap.add_argument("--fr-metric", default="flip_rate", help="metric name of the flip rate in the table")
    ap.add_argument("--experiments", nargs="+", choices=sorted(BASELINE_OF), default=sorted(BASELINE_OF))
    ap.add_argument("--filter-input", action="store_true", help="drop (and count) rows that are not validation / seed 0 / seen instead of refusing the file")
    ap.add_argument("--attest-validated-full-data", action="store_true",
                    help="REQUIRED: you attest every run is RESULT_VALIDATED, trained on the full data (no 20%% debug subset) on the D16 GPU, "
                         "and measured by evaluate.py. The script cannot verify this; the attestation is printed in the log")
    ap.add_argument("--allow-unchecked-run-metadata", action="store_true",
                    help="required with --lambda-map: acknowledge that status and git_dirty are NOT checked (not valid for the real selection)")
    ap.add_argument("--out", help="write the per-candidate selection table as CSV")
    ap.add_argument("--log", help="also write the text log to this file")
    args = ap.parse_args(argv)
    if not args.attest_validated_full_data:
        print("select_lambda: refusing to select without --attest-validated-full-data. This script cannot tell debug-subset or "
              "unvalidated runs from final ones; you must confirm that every run is RESULT_VALIDATED, trained on the full data on the D16 GPU, "
              "and measured by evaluate.py (see the module docstring).", file=sys.stderr)
        return 2
    if args.lambda_map and not args.allow_unchecked_run_metadata:
        print("select_lambda: --lambda-map carries no run metadata, so status and git_dirty cannot be checked; add "
              "--allow-unchecked-run-metadata to acknowledge this (not valid for the real selection), or use --runs-dir.", file=sys.stderr)
        return 2
    try:
        rows = read_metrics_csv(args.metrics)
        resolver = runs_dir_resolver(args.runs_dir) if args.runs_dir else lambda_map_resolver(args.lambda_map)
        inputs, counts = extract_inputs(rows, args.experiments, resolver, f1_metric=args.f1_metric, fr_metric=args.fr_metric,
                                        filter_input=args.filter_input, require_run_metadata=bool(args.runs_dir))
        selections = [select_lambda(e, x.baseline_val_macro_f1, x.candidates, baseline_run_id=x.baseline_run_id) for e, x in inputs.items()]
    except SelectionInputError as err:
        print(f"select_lambda: {err}", file=sys.stderr)
        return 2
    checked = bool(args.runs_dir)
    log = [
        "D13 lambda selection: validation, seed 0, seen renderings only",
        f"ATTESTATION: {ATTESTATION}",
        ("run metadata: status and git_dirty checked from the run records (status complete/promoted, git_dirty false)" if checked else
         "WARNING: run metadata UNCHECKED (--lambda-map adapter): status and git_dirty were not verified; this is NOT valid for the real selection"),
        f"input rows: {counts['rows']} read, {counts['used']} used, {counts['excluded']} excluded (not val / seed 0 / seen), "
        f"{counts['other_experiments']} belong to other experiments, {counts['superseded_runs']} superseded run(s) skipped",
        f"Macro F1 = mean over {'+'.join(TRAIN_RENDERS)}; FR on {FR_RENDER_SET!r}; margin {F1_MARGIN:g} (one-sided); "
        f"candidates {[f'{x:g}' for x in CANDIDATE_LAMBDAS]}; lambda from {'run records in ' + args.runs_dir if args.runs_dir else 'TEMPORARY lambda map ' + args.lambda_map}",
        "",
    ] + [format_log(s) for s in selections]
    text = "\n".join(log)
    print(text)
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log).write_text(text + "\n")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
            w.writeheader()
            for s in selections:
                w.writerows(selection_rows(s, checked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
