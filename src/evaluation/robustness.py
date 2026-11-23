"""Rendering-robustness metrics (Section 15.1). Owner: Parth.

W = windows of the split; p(w, r) probabilities for window w in rendering r; yhat its argmax.
FR(r) = mean_w 1[yhat(w, r) != yhat(w, R0)]; PD(r) = mean_w 1/2 ||p(w, r) - p(w, R0)||_1;
CS(r) = 1/2 sum_c |q_c(r) - q_c(R0)|; dF1(r) = MacroF1(r) - MacroF1(R0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.schema import HELD_OUT_RENDERS, N_CLASSES, RENDER_IDS, SEEN_RENDERS
from src.evaluation.metrics import macro_f1

REFERENCE_RENDER = "R0"
# FR_seen averages the seen renderings other than the reference (Section 15.1: mean over R1, R2, R4);
# R0 against itself is always 0 and is excluded. FR_held averages R3, R5, R6.
FR_SEEN_RENDERS = tuple(r for r in SEEN_RENDERS if r != REFERENCE_RENDER)
PROB_COLS = ("p_down", "p_neutral", "p_up")


def pivot_predictions(df: pd.DataFrame, render_ids=RENDER_IDS):
    """Align one run/split to arrays. Returns window_ids, y (W), pred (W x R), probs (W x R x 3)."""
    d = df[df["render_id"].isin(list(render_ids))]
    wids = np.array(sorted(d["window_id"].unique()))
    pos = {w: i for i, w in enumerate(wids)}
    rpos = {r: j for j, r in enumerate(render_ids)}
    W, R = len(wids), len(render_ids)
    probs = np.full((W, R, N_CLASSES), np.nan)
    wi = d["window_id"].map(pos).to_numpy()
    ri = d["render_id"].map(rpos).to_numpy()
    probs[wi, ri] = d[list(PROB_COLS)].to_numpy(np.float64)
    if np.isnan(probs).any():
        raise ValueError("predictions do not cover every window x rendering")
    y = d.drop_duplicates("window_id").set_index("window_id").loc[wids, "true_label"].to_numpy()
    return wids, y, probs.argmax(axis=2), probs


def flip_rate(pred_r, pred_ref) -> float:
    return float(np.mean(np.asarray(pred_r) != np.asarray(pred_ref)))


def prob_dispersion(p_r, p_ref) -> float:
    return float(np.mean(0.5 * np.abs(np.asarray(p_r) - np.asarray(p_ref)).sum(axis=-1)))


def class_shift(pred_r, pred_ref, k: int = N_CLASSES) -> float:
    q_r = np.bincount(pred_r, minlength=k) / len(pred_r)
    q_0 = np.bincount(pred_ref, minlength=k) / len(pred_ref)
    return float(0.5 * np.abs(q_r - q_0).sum())


def any_flip_rate(pred_matrix) -> float:
    """Share of windows whose predictions are not identical across all renderings."""
    p = np.asarray(pred_matrix)
    return float(np.mean((p != p[:, :1]).any(axis=1)))


def robustness_summary(df: pd.DataFrame, render_ids=RENDER_IDS) -> dict:
    """Per-rendering FR, PD, dF1, CS plus FR_seen, FR_held, PD_seen, PD_held, any-flip."""
    render_ids = list(render_ids)
    _, y, pred, probs = pivot_predictions(df, render_ids)
    j0 = render_ids.index(REFERENCE_RENDER)
    f1_ref = macro_f1(y, pred[:, j0])
    per = {}
    for j, r in enumerate(render_ids):
        per[r] = {"FR": flip_rate(pred[:, j], pred[:, j0]), "PD": prob_dispersion(probs[:, j], probs[:, j0]),
                  "dF1": macro_f1(y, pred[:, j]) - f1_ref, "class_shift": class_shift(pred[:, j], pred[:, j0])}
    seen = [r for r in FR_SEEN_RENDERS if r in per]
    held = [r for r in HELD_OUT_RENDERS if r in per]
    out = {"per_render": per, "any_flip": any_flip_rate(pred)}
    out["FR_seen"] = float(np.mean([per[r]["FR"] for r in seen])) if seen else float("nan")
    out["FR_held"] = float(np.mean([per[r]["FR"] for r in held])) if held else float("nan")
    out["PD_seen"] = float(np.mean([per[r]["PD"] for r in seen])) if seen else float("nan")
    out["PD_held"] = float(np.mean([per[r]["PD"] for r in held])) if held else float("nan")
    return out


def fr_by_confidence_decile(df: pd.DataFrame, render_id: str, n_bins: int = 10) -> pd.DataFrame:
    """FR(render_id) within deciles of the R0 margin (top-1 minus top-2 probability)."""
    ids = [REFERENCE_RENDER, render_id]
    _, _, pred, probs = pivot_predictions(df, ids)
    s = np.sort(probs[:, 0], axis=1)
    margin = s[:, -1] - s[:, -2]
    edges = np.quantile(margin, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, margin, side="right") - 1, 0, n_bins - 1)
    rows = [{"decile": k, "n": int((b == k).sum()),
             "FR": float(np.mean(pred[b == k, 1] != pred[b == k, 0])) if (b == k).any() else float("nan")}
            for k in range(n_bins)]
    return pd.DataFrame(rows)
