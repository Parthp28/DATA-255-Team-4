"""Predictive metrics (Section 15.1). Owner: Parth.

Implemented from definitions (not wrappers) so they are auditable; T-E4/E5 check them against
hand-worked cases and scikit-learn.
"""
from __future__ import annotations

import numpy as np

from src.common.schema import N_CLASSES


def confusion_matrix(y, pred, k: int = N_CLASSES, normalize: bool = False) -> np.ndarray:
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    cm = np.zeros((k, k), dtype=np.float64)
    np.add.at(cm, (y, pred), 1)
    if normalize:   # row-normalized (true class rows), as reported on R0
        rs = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0)
    return cm


def per_class_prf(y, pred, k: int = N_CLASSES) -> dict:
    cm = confusion_matrix(y, pred, k)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(y, pred, k: int = N_CLASSES) -> float:
    """Mean over DOWN, NEUTRAL, UP of F1_c = 2TP_c / (2TP_c + FP_c + FN_c) (0 when undefined)."""
    return float(per_class_prf(y, pred, k)["f1"].mean())


def accuracy(y, pred) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def _rank_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc_binary(score, positive) -> float:
    """Mann-Whitney AUROC with average ranks for ties; nan if one class is absent."""
    score, positive = np.asarray(score, np.float64), np.asarray(positive, bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rank_average(score)
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_auroc_ovr(y, probs) -> float:
    """Mean over classes of AUROC(p_c, 1[y = c]); classes absent from y are skipped."""
    y, probs = np.asarray(y, int), np.asarray(probs, np.float64)
    vals = [auroc_binary(probs[:, c], y == c) for c in range(probs.shape[1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def predictive_metrics(y, probs) -> dict:
    """All predictive metrics for one (run, split, rendering)."""
    probs = np.asarray(probs, np.float64)
    pred = probs.argmax(axis=1)
    prf = per_class_prf(y, pred)
    out = {"accuracy": accuracy(y, pred), "macro_f1": macro_f1(y, pred), "macro_auroc": macro_auroc_ovr(y, probs)}
    for c, name in enumerate(("down", "neutral", "up")):
        for m in ("precision", "recall", "f1"):
            out[f"{m}_{name}"] = float(prf[m][c])
    q = np.bincount(pred, minlength=probs.shape[1]) / len(pred)
    for c, name in enumerate(("down", "neutral", "up")):
        out[f"pred_share_{name}"] = float(q[c])
    out["max_pred_share"] = float(q.max())   # > 0.8 triggers interpretation case E (Section 15.3)
    return out
