"""T-E1 to T-E8 and T-E10 (owner Parth): metric implementations on hand-built and synthetic cases.

Synthetic fixtures (scripts/make_synthetic_predictions.py) are test inputs, not results."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from make_synthetic_predictions import fixture_dataset, planted_run  # noqa: E402
from src.evaluation.calibration import brier, ece, reliability_table  # noqa: E402
from src.evaluation.metrics import (accuracy, confusion_matrix, macro_auroc_ovr, macro_f1,  # noqa: E402
                                    per_class_prf, predictive_metrics)
from src.evaluation.robustness import (any_flip_rate, class_shift, flip_rate,  # noqa: E402
                                       fr_by_confidence_decile, prob_dispersion, robustness_summary)
from src.evaluation.seeds import cross_seed_disagreement, mean_sd, sign_agreement  # noqa: E402

WINDOWS = fixture_dataset()[0]


def test_e1_flip_rate_r0_vs_r0_is_exactly_zero():
    df = planted_run(WINDOWS, "E1", 0, flips={r: 0.3 for r in ("R1", "R2", "R3", "R4", "R5", "R6")})
    test = df[df["split"] == "test"]
    r0 = test[test["render_id"] == "R0"].sort_values("window_id")["pred_class"].to_numpy()
    assert flip_rate(r0, r0) == 0.0
    assert robustness_summary(test)["per_render"]["R0"]["FR"] == 0.0


def test_e2_identical_probability_vectors_give_zero_dispersion():
    p = np.random.default_rng(0).dirichlet([1, 1, 1], 50)
    assert prob_dispersion(p, p) == 0.0
    assert prob_dispersion(p, p[:, [1, 2, 0]]) > 0
    assert prob_dispersion(np.array([[1, 0, 0.]]), np.array([[0, 1, 0.]])) == 1.0   # bounded by 1


def test_e3_planted_20pct_flips_return_exactly_020():
    flips = {"R1": 0.2, "R2": 0.2, "R3": 0.2, "R4": 0.2, "R5": 0.2, "R6": 0.2}
    df = planted_run(WINDOWS, "E1", 0, splits=("test",), flips=flips)
    s = robustness_summary(df)
    n = df["window_id"].nunique()
    expect = round(0.2 * n) / n
    for r in flips:
        assert s["per_render"][r]["FR"] == pytest.approx(expect, abs=1e-12)
    assert s["FR_seen"] == pytest.approx(expect) and s["FR_held"] == pytest.approx(expect)


def test_e3b_planted_flips_on_larger_set():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 3, 1000)
    other = base.copy()
    idx = rng.choice(1000, 200, replace=False)
    other[idx] = (other[idx] + 1) % 3
    assert flip_rate(other, base) == 0.20


def test_e4_macro_f1_hand_worked_and_sklearn():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 2, 0])
    # class 0: TP1 FP1 FN1 -> 0.5; class 1: TP2 FP1 FN0 -> 0.8; class 2: TP1 FP0 FN1 -> 2/3
    assert macro_f1(y, p) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    rng = np.random.default_rng(2)
    for _ in range(20):
        y, p = rng.integers(0, 3, 300), rng.integers(0, 3, 300)
        assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0))
    # collapse to NEUTRAL: accuracy can look fine, macro F1 exposes it
    y = np.array([1] * 80 + [0] * 10 + [2] * 10)
    p = np.ones(100, int)
    assert accuracy(y, p) == 0.8 and macro_f1(y, p) == pytest.approx((0 + 2 * 80 / 180 + 0) / 3)


def test_e5_auroc_hand_worked_and_sklearn():
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1, 0], [0.4, 0.6, 0], [0.35, 0.65, 0], [0.8, 0.2, 0]])
    # class 0 vs rest: scores pos {0.9, 0.4}, neg {0.35, 0.8} -> pairs won: (0.9>0.35, 0.9>0.8, 0.4>0.35) = 3/4
    assert macro_auroc_ovr(y, probs) == pytest.approx(0.75)   # class 1 is the mirror; class 2 absent -> skipped
    rng = np.random.default_rng(3)
    for _ in range(10):
        y = rng.integers(0, 3, 400)
        pr = rng.dirichlet([1, 1, 1], 400)
        pr[np.arange(400), y] += 0.2
        pr /= pr.sum(1, keepdims=True)
        assert macro_auroc_ovr(y, pr) == pytest.approx(roc_auc_score(y, pr, multi_class="ovr", average="macro"))


def test_per_class_and_confusion():
    y = np.array([0, 0, 1, 2])
    p = np.array([0, 2, 1, 2])
    prf = per_class_prf(y, p)
    np.testing.assert_allclose(prf["precision"], [1, 1, 0.5])
    np.testing.assert_allclose(prf["recall"], [0.5, 1, 1])
    cm = confusion_matrix(y, p, normalize=True)
    np.testing.assert_allclose(cm.sum(1), 1)
    assert cm[0, 2] == 0.5
    m = predictive_metrics(y, np.eye(3)[p] * 0.9 + 0.1 / 3)
    assert m["pred_share_up"] == 0.5 and m["max_pred_share"] == 0.5


def test_e6_shuffled_predictions_give_fr_near_1_minus_sum_q2():
    rng = np.random.default_rng(4)
    q = np.array([0.5, 0.3, 0.2])
    base = rng.choice(3, 200_000, p=q)
    shuffled = rng.permutation(base)
    assert flip_rate(shuffled, base) == pytest.approx(1 - (q ** 2).sum(), abs=0.005)


def test_e7_ece_zero_when_calibrated():
    # every prediction has confidence 0.6 and exactly 60% are correct
    n = 1000
    probs = np.tile([0.6, 0.3, 0.1], (n, 1))
    y = np.zeros(n, int)
    y[600:] = 1
    assert ece(probs, y) == pytest.approx(0.0, abs=1e-12)


def test_e8_ece_is_03_for_confidence_09_at_60pct_accuracy():
    n = 1000
    probs = np.tile([0.9, 0.05, 0.05], (n, 1))
    y = np.zeros(n, int)
    y[600:] = 2
    assert ece(probs, y) == pytest.approx(0.3)
    t = reliability_table(probs, y)
    assert t["n"].sum() == n and t.loc[t["n"] > 0, "bin"].tolist() == [13]


def test_brier_hand_worked():
    probs = np.array([[1.0, 0, 0], [0.5, 0.5, 0]])
    y = np.array([0, 1])
    assert brier(probs, y) == pytest.approx((0 + (0.25 + 0.25)) / 2)


def test_e10_identical_seeds_give_zero_cross_seed_disagreement():
    p = np.random.default_rng(5).integers(0, 3, 300)
    assert cross_seed_disagreement([p, p, p]) == 0.0
    q = p.copy()
    q[:30] = (q[:30] + 1) % 3
    assert cross_seed_disagreement([p, q]) == pytest.approx(0.1)
    assert cross_seed_disagreement([p, p, q]) == pytest.approx((0 + 0.1 + 0.1) / 3)


def test_supporting_robustness_metrics():
    ref = np.array([0, 1, 2, 1])
    r = np.array([0, 1, 1, 1])
    assert class_shift(ref, ref) == 0.0
    assert class_shift(r, ref) == pytest.approx(0.5 * (0 + 0.25 + 0.25))
    assert any_flip_rate(np.stack([ref, ref, r], axis=1)) == 0.25
    assert any_flip_rate(np.stack([ref, ref], axis=1)) == 0.0
    assert mean_sd([1, 2, 3]) == (2.0, 1.0)
    assert sign_agreement([0.1, 0.2, 0.05]) and not sign_agreement([0.1, -0.2, 0.05]) and not sign_agreement([0, 1, 1])


def test_dF1_and_confidence_deciles():
    df = planted_run(WINDOWS, "E2", 1, splits=("test",),
                     flips={"R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0.5, "R6": 0})
    s = robustness_summary(df)
    assert s["per_render"]["R1"]["dF1"] == 0.0 and s["per_render"]["R1"]["PD"] == 0.0
    assert s["per_render"]["R5"]["FR"] > 0.45
    dec = fr_by_confidence_decile(df, "R5")
    assert dec["n"].sum() == df["window_id"].nunique()
    assert isinstance(dec, pd.DataFrame)
