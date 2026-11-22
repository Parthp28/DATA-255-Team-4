"""Unit tests for scripts/select_lambda.py (decision D13). Standard library plus pytest only."""
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import select_lambda as sl  # noqa: E402
from select_lambda import Candidate, RunInfo, SelectionInputError, select_lambda  # noqa: E402


def cands(spec):
    """spec: {lambda: (val_macro_f1, val_fr_seen)} -> list of Candidate."""
    return [Candidate(lam, f1, fr) for lam, (f1, fr) in spec.items()]


# ======================================================================== pure rule: eligibility
def test_picks_lowest_flip_rate_among_f1_eligible_candidates():
    sel = select_lambda("E3", 0.40, cands({0.3: (0.395, 0.20), 1: (0.392, 0.12), 3: (0.391, 0.15)}))
    assert (sel.selected_lambda, sel.rule) == (1.0, "eligible")
    assert [r.eligible for r in sel.results] == [True, True, True]


def test_ineligible_candidate_with_lowest_flip_rate_is_never_chosen():
    sel = select_lambda("E3", 0.40, cands({0.3: (0.399, 0.20), 1: (0.395, 0.15), 3: (0.350, 0.01)}))
    assert sel.selected_lambda == 1.0 and sel.rule == "eligible"
    assert [r.eligible for r in sel.results] == [True, True, False]


def test_candidate_better_than_control_is_eligible_however_large_the_gain():
    sel = select_lambda("E3", 0.40, cands({0.3: (0.90, 0.01), 1: (0.401, 0.20), 3: (0.39, 0.25)}))
    assert [r.eligible for r in sel.results] == [True, True, True]
    assert sel.selected_lambda == 0.3  # one-sided: a big gain is not penalized


def test_two_sided_option_is_gone():
    with pytest.raises(TypeError):
        select_lambda("E3", 0.40, cands({0.3: (0.4, 0.1), 1: (0.4, 0.1), 3: (0.4, 0.1)}), two_sided=True)
    assert not hasattr(sl, "two_sided")
    with pytest.raises(SystemExit):
        sl.main(["--metrics", "x.csv", "--lambda-map", "y.csv", "--two-sided"])


# ======================================================================== exact 0.01 boundary
@pytest.mark.parametrize("base", [round(0.30 + 0.01 * i, 2) for i in range(0, 61)])
def test_exact_boundary_is_eligible_despite_float_rounding(base):
    # base - 0.01 is not exactly representable (for example 0.6 - 0.01 = 0.5900000000000001); it must still qualify.
    boundary = round(base - 0.01, 2)
    sel = select_lambda("E3", base, cands({0.3: (boundary, 0.10), 1: (0.0, 0.20), 3: (0.0, 0.30)}))
    assert sel.results[0].eligible, f"base={base} boundary={boundary} delta={boundary - base!r}"


@pytest.mark.parametrize("base", [0.60, 0.7, 0.35, 0.5])
def test_just_beyond_the_boundary_is_not_eligible(base):
    sel = select_lambda("E3", base, cands({0.3: (base - 0.0101, 0.01), 1: (0.0, 0.02), 3: (0.0, 0.03)}))
    assert not sel.results[0].eligible


# ======================================================================== no eligible candidates, ties, determinism
def test_no_eligible_candidates_selects_highest_macro_f1():
    sel = select_lambda("E4", 0.60, cands({0.3: (0.40, 0.01), 1: (0.45, 0.50), 3: (0.30, 0.02)}))
    assert (sel.selected_lambda, sel.rule) == (1.0, "fallback")
    assert not any(r.eligible for r in sel.results) and "no candidate" in sel.reason


def test_eligible_tie_break_flip_rate_then_f1_then_smallest_lambda():
    # lowest FR_seen wins even with lower F1
    assert select_lambda("E3", 0.40, cands({0.3: (0.399, 0.20), 1: (0.391, 0.10), 3: (0.395, 0.15)})).selected_lambda == 1.0
    # equal FR_seen -> higher F1
    assert select_lambda("E3", 0.40, cands({0.3: (0.395, 0.10), 1: (0.399, 0.10), 3: (0.390, 0.30)})).selected_lambda == 1.0
    # equal FR_seen and F1 -> smallest lambda
    assert select_lambda("E3", 0.40, cands({0.3: (0.395, 0.10), 1: (0.395, 0.10), 3: (0.395, 0.10)})).selected_lambda == 0.3
    assert select_lambda("E3", 0.40, cands({0.3: (0.390, 0.30), 1: (0.395, 0.10), 3: (0.395, 0.10)})).selected_lambda == 1.0


def test_fallback_tie_break_f1_then_flip_rate_then_smallest_lambda():
    # highest F1 wins even with higher FR_seen
    assert select_lambda("E3", 0.90, cands({0.3: (0.50, 0.01), 1: (0.55, 0.40), 3: (0.40, 0.02)})).selected_lambda == 1.0
    # equal F1 -> lower FR_seen
    assert select_lambda("E3", 0.90, cands({0.3: (0.50, 0.20), 1: (0.50, 0.10), 3: (0.40, 0.01)})).selected_lambda == 1.0
    # equal F1 and FR_seen -> smallest lambda
    sel = select_lambda("E3", 0.90, cands({0.3: (0.50, 0.10), 1: (0.50, 0.10), 3: (0.40, 0.01)}))
    assert (sel.selected_lambda, sel.rule) == (0.3, "fallback")


def test_differences_below_float_tolerance_count_as_ties():
    sel = select_lambda("E3", 0.40, cands({0.3: (0.395, 0.10 + 1e-12), 1: (0.395, 0.10), 3: (0.395, 0.20)}))
    assert sel.selected_lambda == 0.3  # 1e-12 is a tie, so the smaller lambda wins
    sel = select_lambda("E3", 0.90, cands({0.3: (0.50 - 1e-12, 0.10), 1: (0.50, 0.10), 3: (0.1, 0.5)}))
    assert sel.selected_lambda == 0.3


@pytest.mark.parametrize("order", list(itertools.permutations([0.3, 1, 3])))
def test_result_does_not_depend_on_candidate_order(order):
    spec = {0.3: (0.395, 0.10), 1: (0.399, 0.10), 3: (0.300, 0.01)}
    sel = select_lambda("E3", 0.40, cands({lam: spec[lam] for lam in order}))
    assert sel.selected_lambda == 1.0 and [r.lam for r in sel.results] == [0.3, 1.0, 3.0]


def test_selection_is_repeatable():
    spec = cands({0.3: (0.395, 0.12), 1: (0.395, 0.12), 3: (0.39, 0.11)})
    assert select_lambda("E3", 0.40, spec) == select_lambda("E3", 0.40, list(reversed(spec)))


# ======================================================================== pure rule: invalid input
def test_missing_candidate_is_an_error_not_a_smaller_selection():
    with pytest.raises(SelectionInputError, match="missing candidate"):
        select_lambda("E3", 0.40, cands({0.3: (0.4, 0.1), 1: (0.4, 0.1)}))
    with pytest.raises(SelectionInputError, match="missing candidate"):
        select_lambda("E3", 0.40, [])


@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.3, 2, 0.5, 10, float("nan"), float("inf"), None, True, "1"])
def test_invalid_lambda_values_are_rejected(bad):
    with pytest.raises(SelectionInputError):
        select_lambda("E3", 0.40, cands({0.3: (0.4, 0.1), 1: (0.4, 0.1)}) + [Candidate(bad, 0.4, 0.1)])


def test_duplicate_lambda_is_rejected():
    spec = cands({0.3: (0.4, 0.1), 1: (0.4, 0.1), 3: (0.4, 0.1)}) + [Candidate(1.0, 0.41, 0.05)]
    with pytest.raises(SelectionInputError, match="duplicate"):
        select_lambda("E3", 0.40, spec)


def test_lambda_zero_is_not_a_candidate_and_controls_are_not_experiments():
    with pytest.raises(SelectionInputError):
        select_lambda("E3", 0.40, cands({0: (0.4, 0.1), 0.3: (0.4, 0.1), 1: (0.4, 0.1)}))
    for bad in ("E3b", "E4b", "E1", "E5", ""):
        with pytest.raises(SelectionInputError):
            select_lambda(bad, 0.4, cands({0.3: (0.4, 0.1), 1: (0.4, 0.1), 3: (0.4, 0.1)}))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "0.4", -0.1, 1.5])
def test_bad_metric_values_are_rejected(bad):
    good = {0.3: (0.4, 0.1), 1: (0.4, 0.1), 3: (0.4, 0.1)}
    for position in (0, 1):  # F1 and flip rate
        spec = dict(good)
        spec[1] = tuple(bad if i == position else v for i, v in enumerate(spec[1]))
        with pytest.raises(SelectionInputError):
            select_lambda("E3", 0.40, cands(spec))
    with pytest.raises(SelectionInputError):
        select_lambda("E3", bad, cands(good))  # control F1


# ======================================================================== table + run-record fixtures
RENDERS = ("R0", "R1", "R2", "R4")


def f1_by_render(f1):
    """A scalar means the same F1 on all four training renderings; a 4-tuple is per render (R0, R1, R2, R4)."""
    return dict(zip(RENDERS, f1 if isinstance(f1, tuple) else (f1,) * 4))


def build(tmp_path, e3=None, e4=None, extra_rows=(), run_overrides=None, name="w"):
    """Write metrics_long.csv (planned columns only) and run.json records. e3/e4: (control_f1, {lambda: (f1, fr)})."""
    runs = tmp_path / name / "runs"
    runs.mkdir(parents=True)
    rows, overrides = [], run_overrides or {}

    def add_run(run_id, exp, f1, fr, lam):
        for render, v in f1_by_render(f1).items():
            rows.append([run_id, exp, 0, "val", render, "macro_f1", v])
        if fr is not None:
            rows.append([run_id, exp, 0, "val", "seen", "flip_rate", fr])
        rec = {"experiment_id": exp, "seed": 0, "status": "complete", "git_dirty": False}
        if lam is not None:
            rec["consistency_lambda"] = lam
        rec.update(overrides.get(run_id, {}))
        (runs / run_id).mkdir()
        (runs / run_id / "run.json").write_text(json.dumps(rec))

    for exp, spec in (("E3", e3), ("E4", e4)):
        if spec is None:
            continue
        base_f1, by_lam = spec
        add_run(f"{exp}b_s0_final_c0", f"{exp}b", base_f1, 0.2, 0.0)  # the control's flip rate is never used
        for lam, (f1, fr) in by_lam.items():
            add_run(f"{exp}_s0_l{lam}_c1", exp, f1, fr, lam)
    rows.extend(extra_rows)
    path = tmp_path / name / "metrics_long.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(sl.REQUIRED_COLUMNS)
        w.writerows(rows)
    return path, runs


E3 = (0.40, {0.3: (0.395, 0.20), 1: (0.392, 0.12), 3: (0.30, 0.01)})
E4 = (0.45, {0.3: (0.449, 0.05), 1: (0.447, 0.09), 3: (0.446, 0.04)})


def extract(tmp_path, experiments=None, *, filter_input=False, f1_metric="macro_f1", fr_metric="flip_rate", **build_kw):
    """Select for the experiments whose data was built, unless ``experiments`` is given explicitly."""
    if experiments is None:
        experiments = tuple(e for e, spec in (("E3", build_kw.get("e3")), ("E4", build_kw.get("e4"))) if spec)
    path, runs = build(tmp_path, **build_kw)
    return sl.extract_inputs(sl.read_metrics_csv(path), list(experiments), sl.runs_dir_resolver(runs),
                             f1_metric=f1_metric, fr_metric=fr_metric, filter_input=filter_input)


def pick(inputs, exp):
    x = inputs[exp]
    return select_lambda(exp, x.baseline_val_macro_f1, x.candidates, baseline_run_id=x.baseline_run_id)


# ======================================================================== Macro F1 = mean over R0, R1, R2, R4
def test_extract_and_select_from_table_and_run_records(tmp_path):
    inputs, counts = extract(tmp_path, e3=E3, e4=E4)
    # Per run: 4 F1 rows (+1 flip rate for candidates); controls have no flip-rate row used.
    assert counts["rows"] == 2 * (4 + 1) + 6 * 5 and counts["used"] == counts["rows"] and counts["excluded"] == 0
    assert pick(inputs, "E3").selected_lambda == 1.0 and pick(inputs, "E4").selected_lambda == 3.0
    assert [c.lam for c in inputs["E3"].candidates] == [0.3, 1.0, 3.0]
    assert inputs["E3"].baseline_run_id == "E3b_s0_final_c0" and inputs["E3"].candidates[1].run_id == "E3_s0_l1_c1"


def test_macro_f1_is_the_mean_over_the_four_training_renderings(tmp_path):
    # Candidate: R0 alone is far below the control (0.50 vs 0.60) but the mean over R0,R1,R2,R4 is 0.595 -> eligible.
    spec = (0.60, {0.3: ((0.50, 0.65, 0.66, 0.57), 0.10), 1: (0.0, 0.2), 3: (0.0, 0.3)})
    inputs, _ = extract(tmp_path, e3=spec)
    assert inputs["E3"].candidates[0].val_macro_f1 == pytest.approx(0.595)
    assert pick(inputs, "E3").results[0].eligible and pick(inputs, "E3").selected_lambda == 0.3
    # And the other way: R0 equals the control but the mean falls more than 0.01 short -> not eligible.
    spec = (0.60, {0.3: ((0.60, 0.55, 0.55, 0.60), 0.01), 1: (0.60, 0.20), 3: (0.0, 0.3)})
    inputs, _ = extract(tmp_path, e3=spec, name="w2")
    assert not pick(inputs, "E3").results[0].eligible and pick(inputs, "E3").selected_lambda == 1.0


def test_the_control_is_averaged_the_same_way(tmp_path):
    spec = ((0.50, 0.70, 0.60, 0.60), {0.3: (0.605, 0.10), 1: (0.585, 0.05), 3: (0.50, 0.01)})  # control mean = 0.60
    inputs, _ = extract(tmp_path, e3=spec)
    assert inputs["E3"].baseline_val_macro_f1 == pytest.approx(0.60)
    assert [r.eligible for r in pick(inputs, "E3").results] == [True, False, False]


def test_exact_boundary_holds_after_averaging(tmp_path):
    # mean of the four candidate F1 values is exactly 0.01 below the control's mean (up to float rounding).
    spec = (0.60, {0.3: ((0.58, 0.60, 0.58, 0.60), 0.10), 1: (0.0, 0.2), 3: (0.0, 0.3)})
    inputs, _ = extract(tmp_path, e3=spec)
    assert pick(inputs, "E3").results[0].eligible


@pytest.mark.parametrize("missing", RENDERS)
def test_a_missing_training_rendering_is_an_error_for_candidates_and_controls(tmp_path, missing):
    path, runs = build(tmp_path, e3=E3)
    for run_id in ("E3_s0_l1_c1", "E3b_s0_final_c0"):
        rows = [r for r in sl.read_metrics_csv(path) if not (r["run_id"] == run_id and r["render_set"] == missing and r["metric"] == "macro_f1")]
        with pytest.raises(SelectionInputError, match=rf"missing for render\(s\) \['{missing}'\]"):
            sl.extract_inputs(rows, ["E3"], sl.runs_dir_resolver(runs))


def test_only_seen_flip_rate_counts_and_seen_aggregate_f1_is_not_used(tmp_path):
    extra = [["E3_s0_l1_c1", "E3", 0, "val", "R1", "flip_rate", 0.99],  # per-render flip rate: ignored
             ["E3_s0_l1_c1", "E3", 0, "val", "seen", "macro_f1", 0.0]]  # a 'seen' aggregate F1: ignored (mean of the four renders is used)
    inputs, _ = extract(tmp_path, e3=E3, extra_rows=extra)
    c = inputs["E3"].candidates[1]
    assert (c.val_macro_f1, c.val_fr_seen) == (pytest.approx(0.392), 0.12)


def test_missing_flip_rate_for_a_candidate_is_an_error(tmp_path):
    path, runs = build(tmp_path, e3=E3)
    rows = [r for r in sl.read_metrics_csv(path) if not (r["run_id"] == "E3_s0_l1_c1" and r["metric"] == "flip_rate")]
    with pytest.raises(SelectionInputError, match="flip_rate"):
        sl.extract_inputs(rows, ["E3"], sl.runs_dir_resolver(runs))


# ======================================================================== lambda from run records, not from the table
def test_table_has_exactly_the_planned_metrics_long_columns(tmp_path):
    path, _ = build(tmp_path, e3=E3)
    assert tuple(next(csv.reader(open(path)))) == ("run_id", "experiment_id", "seed", "split", "render_set", "metric", "value")
    assert "consistency_lambda" not in sl.REQUIRED_COLUMNS


def test_a_stray_lambda_column_in_the_table_is_ignored(tmp_path):
    path, runs = build(tmp_path, e3=E3)
    text = path.read_text().splitlines()
    path.write_text("\n".join([text[0] + ",consistency_lambda"] + [line + ",99" for line in text[1:]]) + "\n")
    inputs, _ = sl.extract_inputs(sl.read_metrics_csv(path), ["E3"], sl.runs_dir_resolver(runs))
    assert [c.lam for c in inputs["E3"].candidates] == [0.3, 1.0, 3.0]


def test_missing_run_record_is_an_error(tmp_path):
    path, runs = build(tmp_path, e3=E3)
    (runs / "E3_s0_l1_c1" / "run.json").unlink()
    with pytest.raises(SelectionInputError, match="no run record"):
        sl.extract_inputs(sl.read_metrics_csv(path), ["E3"], sl.runs_dir_resolver(runs))


def test_run_record_must_agree_with_the_table(tmp_path):
    with pytest.raises(SelectionInputError, match="run record says"):
        extract(tmp_path, experiments=("E3",), e3=E3, run_overrides={"E3_s0_l1_c1": {"experiment_id": "E4"}})
    with pytest.raises(SelectionInputError, match="seed"):
        extract(tmp_path, experiments=("E3",), e3=E3, run_overrides={"E3_s0_l1_c1": {"seed": 1}}, name="w2")


def test_candidate_without_lambda_in_its_run_record_is_an_error(tmp_path):
    path, runs = build(tmp_path, e3=E3)
    rec = json.loads((runs / "E3_s0_l1_c1" / "run.json").read_text())
    del rec["consistency_lambda"]
    (runs / "E3_s0_l1_c1" / "run.json").write_text(json.dumps(rec))
    with pytest.raises(SelectionInputError, match="no consistency_lambda"):
        sl.extract_inputs(sl.read_metrics_csv(path), ["E3"], sl.runs_dir_resolver(runs))


@pytest.mark.parametrize("bad", [0, -1, 2, 0.5, "abc", "1", None, [1], float("nan"), float("inf")])
def test_invalid_lambda_in_a_run_record_is_an_error(tmp_path, bad):
    with pytest.raises(SelectionInputError):
        inputs, _ = extract(tmp_path, experiments=("E3",), e3=E3, run_overrides={"E3_s0_l1_c1": {"consistency_lambda": bad}})
        pick(inputs, "E3")


def test_control_must_have_lambda_zero_or_none(tmp_path):
    with pytest.raises(SelectionInputError, match="lambda 0"):
        extract(tmp_path, experiments=("E3",), e3=E3, run_overrides={"E3b_s0_final_c0": {"consistency_lambda": 0.3}})
    path, runs = build(tmp_path, e3=E3, name="w2")
    rec = json.loads((runs / "E3b_s0_final_c0" / "run.json").read_text())
    rec.pop("consistency_lambda")  # a control record without lambda is fine
    (runs / "E3b_s0_final_c0" / "run.json").write_text(json.dumps(rec))
    inputs, _ = sl.extract_inputs(sl.read_metrics_csv(path), ["E3"], sl.runs_dir_resolver(runs))
    assert inputs["E3"].baseline_val_macro_f1 == pytest.approx(0.40)


def test_unsafe_or_malformed_run_ids_are_rejected(tmp_path):
    for rid in ("../escape", "a/b", "..", "", ".hidden", "a b", "a\\b"):
        with pytest.raises(SelectionInputError):
            sl.runs_dir_resolver(tmp_path)(rid)


def test_malformed_run_record_is_an_error(tmp_path):
    path, runs = build(tmp_path, e3=E3)
    (runs / "E3_s0_l1_c1" / "run.json").write_text("{not json")
    with pytest.raises(SelectionInputError, match="cannot read"):
        sl.extract_inputs(sl.read_metrics_csv(path), ["E3"], sl.runs_dir_resolver(runs))
    (runs / "E3_s0_l1_c1" / "run.json").write_text("[1, 2]")
    with pytest.raises(SelectionInputError, match="JSON object"):
        sl.extract_inputs(sl.read_metrics_csv(path), ["E3"], sl.runs_dir_resolver(runs))


# ======================================================================== reruns: superseded / duplicates / unusable status
def add_rerun(path, runs, old_run, new_run, f1, fr, lam, exp="E3", old_status="superseded"):
    rec = json.loads((runs / old_run / "run.json").read_text())
    rec["status"] = old_status
    (runs / old_run / "run.json").write_text(json.dumps(rec))
    (runs / new_run).mkdir()
    (runs / new_run / "run.json").write_text(json.dumps({"experiment_id": exp, "seed": 0, "status": "complete", "git_dirty": False, "consistency_lambda": lam}))
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        for r in RENDERS:
            w.writerow([new_run, exp, 0, "val", r, "macro_f1", f1])
        w.writerow([new_run, exp, 0, "val", "seen", "flip_rate", fr])


def test_superseded_run_is_skipped_and_its_rerun_is_used(tmp_path):
    path, runs = build(tmp_path, e3=E3)
    add_rerun(path, runs, "E3_s0_l1_c1", "E3_s0_l1_rerun", 0.399, 0.02, 1.0)
    inputs, counts = sl.extract_inputs(sl.read_metrics_csv(path), ["E3"], sl.runs_dir_resolver(runs))
    assert counts["superseded_runs"] == 1
    assert inputs["E3"].candidates[1].run_id == "E3_s0_l1_rerun" and inputs["E3"].candidates[1].val_fr_seen == 0.02


def test_two_usable_runs_for_the_same_lambda_or_control_are_duplicates(tmp_path):
    path, runs = build(tmp_path, e3=E3)
    add_rerun(path, runs, "E3_s0_l1_c1", "E3_s0_l1_rerun", 0.399, 0.02, 1.0, old_status="complete")
    with pytest.raises(SelectionInputError, match="duplicate candidate runs"):
        sl.extract_inputs(sl.read_metrics_csv(path), ["E3"], sl.runs_dir_resolver(runs))
    path, runs = build(tmp_path, e3=E3, name="w2")
    add_rerun(path, runs, "E3b_s0_final_c0", "E3b_rerun", 0.5, 0.0, 0.0, exp="E3b", old_status="complete")
    with pytest.raises(SelectionInputError, match="duplicate control runs"):
        sl.extract_inputs(sl.read_metrics_csv(path), ["E3"], sl.runs_dir_resolver(runs))


def test_equal_lambda_spellings_in_run_records_are_duplicates(tmp_path):
    with pytest.raises(SelectionInputError, match="duplicate candidate runs"):
        extract(tmp_path, experiments=("E3",), e3=E3, run_overrides={"E3_s0_l0.3_c1": {"consistency_lambda": 1}})  # 0.3's run now claims lambda 1


@pytest.mark.parametrize("status", ["running", "failed", "weird", ""])
def test_unfinished_or_failed_runs_are_refused(tmp_path, status):
    with pytest.raises(SelectionInputError, match="not usable"):
        extract(tmp_path, experiments=("E3",), e3=E3, run_overrides={"E3_s0_l1_c1": {"status": status}})


@pytest.mark.parametrize("status", ["complete", "promoted"])
def test_complete_and_promoted_runs_are_usable(tmp_path, status):
    inputs, _ = extract(tmp_path, experiments=("E3",), e3=E3, run_overrides={"E3_s0_l1_c1": {"status": status}})
    assert pick(inputs, "E3").selected_lambda == 1.0


def test_run_id_under_two_experiments_is_an_error(tmp_path):
    path, runs = build(tmp_path, e3=E3, e4=E4)
    rows = sl.read_metrics_csv(path)
    rows.append(dict(rows[0], experiment_id="E4b", _line=999))  # same run_id as an E3b row, other experiment
    with pytest.raises(SelectionInputError, match="two experiments"):
        sl.extract_inputs(rows, ["E3", "E4"], sl.runs_dir_resolver(runs))


# ======================================================================== unvalidated / unclean runs cannot be used silently
def strip_key(runs, run_id, key):
    rec = json.loads((runs / run_id / "run.json").read_text())
    rec.pop(key, None)
    (runs / run_id / "run.json").write_text(json.dumps(rec))


@pytest.mark.parametrize("run_id", ["E3_s0_l1_c1", "E3b_s0_final_c0"])
def test_a_run_record_without_status_is_refused(tmp_path, run_id):
    path, runs = build(tmp_path, e3=E3)
    strip_key(runs, run_id, "status")
    with pytest.raises(SelectionInputError, match="no status"):
        sl.extract_inputs(sl.read_metrics_csv(path), ["E3"], sl.runs_dir_resolver(runs))


@pytest.mark.parametrize("run_id", ["E3_s0_l1_c1", "E3b_s0_final_c0"])
def test_a_run_record_without_git_dirty_is_refused(tmp_path, run_id):
    path, runs = build(tmp_path, e3=E3)
    strip_key(runs, run_id, "git_dirty")
    with pytest.raises(SelectionInputError, match="no git_dirty"):
        sl.extract_inputs(sl.read_metrics_csv(path), ["E3"], sl.runs_dir_resolver(runs))


@pytest.mark.parametrize("run_id", ["E3_s0_l1_c1", "E3b_s0_final_c0"])
def test_a_run_on_a_dirty_tree_is_refused(tmp_path, run_id):
    with pytest.raises(SelectionInputError, match="dirty git tree"):
        extract(tmp_path, experiments=("E3",), e3=E3, run_overrides={run_id: {"git_dirty": True}})


@pytest.mark.parametrize("bad", ["false", "False", 0, 1, "", [False]])
def test_git_dirty_must_be_a_real_boolean(tmp_path, bad):
    with pytest.raises(SelectionInputError, match="git_dirty must be true or false"):
        extract(tmp_path, experiments=("E3",), e3=E3, run_overrides={"E3_s0_l1_c1": {"git_dirty": bad}})


def test_a_superseded_run_is_skipped_even_if_it_is_dirty_or_unfinished(tmp_path):
    path, runs = build(tmp_path, e3=E3)
    add_rerun(path, runs, "E3_s0_l1_c1", "E3_s0_l1_rerun", 0.399, 0.02, 1.0)
    old = json.loads((runs / "E3_s0_l1_c1" / "run.json").read_text())
    old["git_dirty"] = True
    (runs / "E3_s0_l1_c1" / "run.json").write_text(json.dumps(old))
    inputs, counts = sl.extract_inputs(sl.read_metrics_csv(path), ["E3"], sl.runs_dir_resolver(runs))
    assert counts["superseded_runs"] == 1 and inputs["E3"].candidates[1].run_id == "E3_s0_l1_rerun"


def test_run_metadata_checks_are_off_only_when_explicitly_disabled(tmp_path):
    path, runs = build(tmp_path, e3=E3)
    strip_key(runs, "E3_s0_l1_c1", "status")
    strip_key(runs, "E3_s0_l1_c1", "git_dirty")
    rows = sl.read_metrics_csv(path)
    with pytest.raises(SelectionInputError):
        sl.extract_inputs(rows, ["E3"], sl.runs_dir_resolver(runs))  # default is strict
    inputs, _ = sl.extract_inputs(rows, ["E3"], sl.runs_dir_resolver(runs), require_run_metadata=False)
    assert pick(inputs, "E3").selected_lambda == 1.0
    # A status that is present and bad is refused even when the metadata requirement is off.
    strip_key(runs, "E3_s0_l1_c1", "status")
    rec = json.loads((runs / "E3_s0_l1_c1" / "run.json").read_text())
    rec["status"] = "failed"
    (runs / "E3_s0_l1_c1" / "run.json").write_text(json.dumps(rec))
    with pytest.raises(SelectionInputError, match="not usable"):
        sl.extract_inputs(rows, ["E3"], sl.runs_dir_resolver(runs), require_run_metadata=False)


def test_cli_refuses_to_select_without_the_attestation(tmp_path, capsys):
    path, runs = build(tmp_path, e3=E3)
    out = tmp_path / "unattested.csv"
    assert sl.main(["--metrics", str(path), "--runs-dir", str(runs), "--experiments", "E3", "--out", str(out)]) == 2
    captured = capsys.readouterr()
    assert "--attest-validated-full-data" in captured.err and "debug-subset" in captured.err
    assert captured.out == "" and not out.exists()  # nothing is selected or written


def test_cli_prints_the_attestation_and_marks_checked_metadata(tmp_path, capsys):
    path, runs = build(tmp_path, e3=E3)
    out = tmp_path / "attested.csv"
    assert sl.main(cli_args(path, runs, "--experiments", "E3", "--out", str(out))) == 0
    text = capsys.readouterr().out
    assert sl.ATTESTATION in text and "RESULT_VALIDATED" in text and "20% debug subset" in text
    assert "status and git_dirty checked" in text and "UNCHECKED" not in text
    assert {r["run_metadata_checked"] for r in csv.DictReader(open(out))} == {"true"}


def test_cli_rejects_a_dirty_run_and_writes_nothing(tmp_path, capsys):
    path, runs = build(tmp_path, e3=E3, run_overrides={"E3_s0_l1_c1": {"git_dirty": True}})
    out = tmp_path / "dirty.csv"
    assert sl.main(cli_args(path, runs, "--experiments", "E3", "--out", str(out))) == 2
    assert "dirty git tree" in capsys.readouterr().err and not out.exists()


def test_the_script_documents_its_interpretations_and_blocking_prerequisites():
    doc = " ".join(sl.__doc__.split())  # normalize line wrapping
    for phrase in ("MEAN of the per-rendering validation Macro F1", "R0, R1, R2, R4", "not an F1 over pooled predictions",
                   "R1, R2, R4", "PARTH MUST VERIFY", "``seen`` row", "Blocking prerequisites", "RESULT_VALIDATED",
                   "20% debug subset", "--attest-validated-full-data", "never promoted", "TEMPORARY"):
        assert phrase in doc, phrase


# ======================================================================== E3 and E4 are independent
def test_e3_and_e4_are_selected_independently(tmp_path):
    inputs, _ = extract(tmp_path, e3=E3, e4=E4)
    s3, s4 = pick(inputs, "E3"), pick(inputs, "E4")
    assert (s3.selected_lambda, s4.selected_lambda) == (1.0, 3.0)
    other4 = (0.80, {0.3: (0.2, 0.9), 1: (0.2, 0.8), 3: (0.8, 0.7)})
    inputs2, _ = extract(tmp_path, e3=E3, e4=other4, name="w2")
    assert pick(inputs2, "E3") == s3  # changing every E4 number cannot change E3
    only3, _ = extract(tmp_path, experiments=("E3",), e3=E3, name="w3")
    assert list(only3) == ["E3"] and pick(only3, "E3") == s3
    with pytest.raises(SelectionInputError, match="missing control"):
        extract(tmp_path, experiments=("E3", "E4"), e3=E3, name="w4")


def test_never_mixes_control_from_the_other_family(tmp_path):
    inputs, _ = extract(tmp_path, e3=E3, e4=(0.99, {0.3: (0.98, 0.1), 1: (0.98, 0.1), 3: (0.98, 0.1)}))
    assert inputs["E3"].baseline_val_macro_f1 == pytest.approx(0.40) and inputs["E4"].baseline_val_macro_f1 == pytest.approx(0.99)


# ======================================================================== test / held-out / non-seed-0 data is refused
@pytest.mark.parametrize(
    "bad_row",
    [
        ["x", "E3", 0, "test", "R0", "macro_f1", 0.9],  # test split
        ["x", "E3", 0, "test", "seen", "flip_rate", 0.0],  # test split, seen renderings
        ["x", "E3", 0, "val", "held_out", "flip_rate", 0.0],  # held-out aggregate
        ["x", "E3", 0, "val", "R3", "flip_rate", 0.0],  # held-out renderings
        ["x", "E3", 0, "val", "R5", "macro_f1", 0.9],
        ["x", "E3b", 0, "val", "R6", "macro_f1", 0.9],
        ["x", "E3", 1, "val", "R0", "macro_f1", 0.9],  # not seed 0
        ["x", "E4", 2, "val", "seen", "flip_rate", 0.1],
        ["x", "E3", 0, "test", "R0", "auroc", 0.5],  # any metric on forbidden data
    ],
)
def test_test_heldout_and_non_seed0_rows_are_refused(tmp_path, bad_row):
    with pytest.raises(SelectionInputError, match="refusing input"):
        extract(tmp_path, e3=E3, e4=E4, extra_rows=[bad_row])


def test_unrelated_experiments_are_ignored(tmp_path):
    inputs, counts = extract(tmp_path, e3=E3, e4=E4, extra_rows=[["x", "E1", 0, "test", "R3", "macro_f1", 0.5]])  # E1 is not part of D13
    assert counts["other_experiments"] == 1 and set(inputs) == {"E3", "E4"}


def test_filter_input_drops_forbidden_rows_and_cannot_change_the_result(tmp_path):
    clean, _ = extract(tmp_path, e3=E3, e4=E4)
    poison = [
        ["E3_s0_l1_c1", "E3", 0, "test", "R0", "macro_f1", 0.99],
        ["E3_s0_l1_c1", "E3", 0, "test", "seen", "flip_rate", 0.0],
        ["E3_s0_l1_c1", "E3", 0, "val", "held_out", "flip_rate", 0.0],
        ["E3_s0_l1_c1", "E3", 0, "val", "R3", "macro_f1", 0.0],
        ["E4_s0_l0.3_c1", "E4", 1, "val", "R0", "macro_f1", 0.99],
        ["E3b_s0_final_c0", "E3b", 0, "test", "R0", "macro_f1", 0.0],
    ]
    filtered, counts = extract(tmp_path, e3=E3, e4=E4, extra_rows=poison, filter_input=True, name="w2")
    assert counts["excluded"] == 6 and filtered == clean
    assert pick(filtered, "E3") == pick(clean, "E3") and pick(filtered, "E4") == pick(clean, "E4")


def test_held_out_renderings_cannot_be_requested_as_f1_renders():
    assert sl.TRAIN_RENDERS == ("R0", "R1", "R2", "R4") and set(sl.TRAIN_RENDERS).isdisjoint({"R3", "R5", "R6"})
    assert set(sl.SEEN_SIDE_RENDER_SETS).isdisjoint({"R3", "R5", "R6", "held_out", "test"})


# ======================================================================== other table problems
def test_missing_control_candidates_or_metric_rows_are_errors(tmp_path):
    path, runs = build(tmp_path, e3=E3)
    resolver = sl.runs_dir_resolver(runs)
    rows = sl.read_metrics_csv(path)
    without = lambda pred: [r for r in rows if not pred(r)]  # noqa: E731
    with pytest.raises(SelectionInputError, match="missing control"):
        sl.extract_inputs(without(lambda r: r["experiment_id"] == "E3b"), ["E3"], resolver)
    with pytest.raises(SelectionInputError, match="missing candidate"):  # lambda=3's run is absent entirely
        x = sl.extract_inputs(without(lambda r: r["run_id"] == "E3_s0_l3_c1"), ["E3"], resolver)[0]
        pick(x, "E3")
    with pytest.raises(SelectionInputError, match="missing candidates"):
        sl.extract_inputs(without(lambda r: r["experiment_id"] == "E3"), ["E3"], resolver)
    with pytest.raises(SelectionInputError, match="f1_macro is missing for render"):  # wrong metric name for F1
        sl.extract_inputs(rows, ["E3"], resolver, f1_metric="f1_macro")
    with pytest.raises(SelectionInputError, match="validation fr on render_set 'seen' is missing"):  # wrong flip-rate metric name
        sl.extract_inputs(rows, ["E3"], resolver, fr_metric="fr")


def test_metric_names_are_configurable(tmp_path):
    path, runs = build(tmp_path, e3=E3)
    rows = [dict(r, metric={"macro_f1": "val_f1", "flip_rate": "FR"}.get(r["metric"], r["metric"])) for r in sl.read_metrics_csv(path)]
    inputs, _ = sl.extract_inputs(rows, ["E3"], sl.runs_dir_resolver(runs), f1_metric="val_f1", fr_metric="FR")
    assert pick(inputs, "E3").selected_lambda == 1.0


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "abc", ""])
def test_nonfinite_or_non_numeric_values_in_the_table_are_errors(tmp_path, bad):
    path, runs = build(tmp_path, e3=E3)
    rows = sl.read_metrics_csv(path)
    rows[5]["value"] = bad
    with pytest.raises(SelectionInputError):
        sl.extract_inputs(rows, ["E3"], sl.runs_dir_resolver(runs))


def test_out_of_range_metric_values_are_errors(tmp_path):
    with pytest.raises(SelectionInputError, match=r"\[0, 1\]"):
        inputs, _ = extract(tmp_path, experiments=("E3",), e3=(0.40, {0.3: (1.5, 0.1), 1: (0.4, 0.1), 3: (0.4, 0.1)}))
        pick(inputs, "E3")


def test_duplicate_metric_rows_in_a_run_are_errors(tmp_path):
    with pytest.raises(SelectionInputError, match="duplicate macro_f1"):
        extract(tmp_path, experiments=("E3",), e3=E3, extra_rows=[["E3_s0_l1_c1", "E3", 0, "val", "R1", "macro_f1", 0.1]])
    with pytest.raises(SelectionInputError, match="duplicate flip_rate"):
        extract(tmp_path, experiments=("E3",), e3=E3, extra_rows=[["E3_s0_l1_c1", "E3", 0, "val", "seen", "flip_rate", 0.1]], name="w2")


def test_missing_required_columns_are_an_error(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("experiment_id,seed,split,render_set,metric,value\n")  # no run_id
    with pytest.raises(SelectionInputError, match="run_id"):
        sl.read_metrics_csv(p)


def test_row_order_does_not_matter(tmp_path):
    path, runs = build(tmp_path, e3=E3, e4=E4)
    rows = sl.read_metrics_csv(path)
    a, _ = sl.extract_inputs(rows, ["E3", "E4"], sl.runs_dir_resolver(runs))
    b, _ = sl.extract_inputs(list(reversed(rows)), ["E3", "E4"], sl.runs_dir_resolver(runs))
    assert a == b


# ======================================================================== temporary lambda-map adapter
def write_map(tmp_path, runs_table, name="map.csv"):
    p = tmp_path / name
    p.write_text("run_id,consistency_lambda\n" + "".join(f"{r},{lam}\n" for r, lam in runs_table))
    return p


def test_lambda_map_adapter_gives_the_same_selection_as_run_records(tmp_path):
    path, runs = build(tmp_path, e3=E3, e4=E4)
    pairs = [(d.name, json.loads((d / "run.json").read_text()).get("consistency_lambda", "")) for d in sorted(runs.iterdir())]
    via_map, _ = sl.extract_inputs(sl.read_metrics_csv(path), ["E3", "E4"], sl.lambda_map_resolver(write_map(tmp_path, pairs)),
                                   require_run_metadata=False)  # the adapter has no run metadata to check
    via_runs, _ = sl.extract_inputs(sl.read_metrics_csv(path), ["E3", "E4"], sl.runs_dir_resolver(runs))
    assert via_map == via_runs


def test_lambda_map_errors(tmp_path):
    with pytest.raises(SelectionInputError, match="duplicate run_id"):
        sl.lambda_map_resolver(write_map(tmp_path, [("a", 1), ("a", 3)]))
    with pytest.raises(SelectionInputError, match="not a number"):
        sl.lambda_map_resolver(write_map(tmp_path, [("a", "abc")], name="m2.csv"))
    with pytest.raises(SelectionInputError, match="not finite"):
        sl.lambda_map_resolver(write_map(tmp_path, [("a", "nan")], name="m3.csv"))
    with pytest.raises(SelectionInputError, match="not in the lambda map"):
        sl.lambda_map_resolver(write_map(tmp_path, [("a", 1)], name="m4.csv"))("b")
    with pytest.raises(SelectionInputError, match="not a valid run identifier"):
        sl.lambda_map_resolver(write_map(tmp_path, [("../x", 1)], name="m5.csv"))
    bad = tmp_path / "m6.csv"
    bad.write_text("run_id,lam\na,1\n")
    with pytest.raises(SelectionInputError, match="missing column"):
        sl.lambda_map_resolver(bad)


# ======================================================================== command line
def cli_args(path, runs, *more):
    return ["--metrics", str(path), "--runs-dir", str(runs), "--attest-validated-full-data", *more]


def test_cli_writes_log_and_csv(tmp_path, capsys):
    path, runs = build(tmp_path, e3=E3, e4=E4)
    out, log = tmp_path / "sel" / "lambda_selection.csv", tmp_path / "sel" / "select.log"
    assert sl.main(cli_args(path, runs, "--out", str(out), "--log", str(log))) == 0
    text = capsys.readouterr().out
    assert "lambda = 1" in text and "lambda = 3" in text and log.read_text().strip() == text.strip()
    assert "mean over R0+R1+R2+R4" in text and "one-sided" in text
    table = list(csv.DictReader(open(out)))
    assert tuple(table[0].keys()) == sl.OUT_COLUMNS and len(table) == 6
    assert {(r["experiment_id"], r["lambda"]) for r in table if r["selected"] == "true"} == {("E3", "1"), ("E4", "3")}
    assert {r["baseline_experiment_id"] for r in table if r["experiment_id"] == "E4"} == {"E4b"}
    assert {r["f1_renders"] for r in table} == {"R0+R1+R2+R4"} and all(r["run_id"] and r["baseline_run_id"] for r in table)


def test_cli_with_the_temporary_lambda_map(tmp_path, capsys):
    path, runs = build(tmp_path, e3=E3)
    pairs = [(d.name, json.loads((d / "run.json").read_text()).get("consistency_lambda", "")) for d in sorted(runs.iterdir())]
    args = ["--metrics", str(path), "--lambda-map", str(write_map(tmp_path, pairs)), "--experiments", "E3", "--attest-validated-full-data"]
    assert sl.main(args) == 2  # the adapter alone is refused: it cannot check run metadata
    assert "--allow-unchecked-run-metadata" in capsys.readouterr().err
    out = tmp_path / "adapter.csv"
    assert sl.main(args + ["--allow-unchecked-run-metadata", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "TEMPORARY lambda map" in text and "WARNING: run metadata UNCHECKED" in text and "NOT valid for the real selection" in text
    assert {r["run_metadata_checked"] for r in csv.DictReader(open(out))} == {"false"}


def test_cli_requires_exactly_one_lambda_source(tmp_path):
    path, runs = build(tmp_path, e3=E3)
    with pytest.raises(SystemExit):
        sl.main(["--metrics", str(path)])
    with pytest.raises(SystemExit):
        sl.main(["--metrics", str(path), "--runs-dir", str(runs), "--lambda-map", "m.csv"])


def test_cli_single_experiment_and_metric_name_options(tmp_path, capsys):
    path, runs = build(tmp_path, e3=E3)
    assert sl.main(cli_args(path, runs, "--experiments", "E3")) == 0
    assert "E3 (control E3b" in capsys.readouterr().out
    assert sl.main(cli_args(path, runs, "--experiments", "E3", "--f1-metric", "nope")) == 2


def test_cli_refuses_test_data_with_exit_code_2_and_writes_nothing(tmp_path, capsys):
    path, runs = build(tmp_path, e3=E3, e4=E4, extra_rows=[["x", "E3", 0, "test", "R0", "macro_f1", 0.9]])
    out = tmp_path / "out.csv"
    assert sl.main(cli_args(path, runs, "--out", str(out))) == 2
    assert "refusing input" in capsys.readouterr().err and not out.exists()
    assert sl.main(cli_args(path, runs, "--out", str(out), "--filter-input")) == 0 and out.exists()


def test_cli_reports_invalid_input_with_exit_code_2(tmp_path, capsys):
    path, runs = build(tmp_path, e3=E3, run_overrides={"E3_s0_l1_c1": {"consistency_lambda": 2}})
    assert sl.main(cli_args(path, runs, "--experiments", "E3")) == 2
    assert "not a candidate" in capsys.readouterr().err


def test_selection_constants_match_the_plan():
    assert sl.CANDIDATE_LAMBDAS == (0.3, 1.0, 3.0) and sl.F1_MARGIN == 0.01 and sl.SELECTION_SEED == 0
    assert sl.BASELINE_OF == {"E3": "E3b", "E4": "E4b"} and math.isclose(sl.EPS, 1e-9)
    assert sl.FR_RENDER_SET == "seen" and sl.SELECTION_SPLIT == "val"
