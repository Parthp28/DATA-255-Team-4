"""Attribution-to-flip link (plan 15.1). Owner: Farzaan; Parth checks it with the evaluation tables.

    AUROC of the R0 attribution mass on region g for predicting a flip under the rendering that changes g

A flip for window w under rendering r means pred_class(w, r) != pred_class(w, R0), from the run's prediction
file. The AUROC is undefined (NaN) when every sampled window flips or none does; that case is reported, not
dropped.

REGION_FACTORS is the default region-to-rendering map: each region paired with the renderings that change it
(plan 5.2). R3 adds axes that R0 does not draw, so R0 has no axes mass to link; the map leaves it out. This map is a
proposal for the owner to confirm before H09.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

REGION_FACTORS = {"grid": ("R2",), "candles": ("R4", "R5"), "plot_background": ("R1",), "margins": ("R1", "R6")}
LINK_COLUMNS = ["run_id", "method", "region", "render_id", "n_windows", "n_flips", "auroc"]


def flips_vs_r0(predictions: pd.DataFrame, render_id: str) -> pd.Series:
    """bool Series indexed by window_id: prediction under `render_id` differs from R0."""
    p = predictions.pivot_table(index="window_id", columns="render_id", values="pred_class", aggfunc="first")
    if "R0" not in p or render_id not in p:
        raise ValueError(f"predictions lack R0 or {render_id}")
    return (p[render_id] != p["R0"]).rename("flip")


def attribution_flip_auroc(attribution: pd.DataFrame, predictions: pd.DataFrame,
                           region_factors=REGION_FACTORS) -> pd.DataFrame:
    """One row per (run, method, region, rendering) from R0 attribution rows and the run's prediction file."""
    rows = []
    a0 = attribution[attribution["render_id"] == "R0"]
    for (run_id, method), grp in a0.groupby(["run_id", "method"]):
        for region, renders in region_factors.items():
            mass = grp[grp["region"] == region].set_index("window_id")["attribution_mass"]
            for r in renders:
                f = flips_vs_r0(predictions[predictions["run_id"] == run_id], r)
                j = pd.concat([mass, f], axis=1, join="inner").dropna()
                k = int(j["flip"].sum())
                auc = float(roc_auc_score(j["flip"], j["attribution_mass"])) if 0 < k < len(j) else float("nan")
                rows.append({"run_id": run_id, "method": method, "region": region, "render_id": r,
                             "n_windows": int(len(j)), "n_flips": k, "auroc": auc})
    return pd.DataFrame(rows, columns=LINK_COLUMNS)


def masking_summary(masking: pd.DataFrame) -> pd.DataFrame:
    """Mean |delta_p| and flip rate per (run, rendering, region, fill mode) for Table 3; a region beats its
    control when its mean |delta_p| exceeds the area-matched control's."""
    m = masking.assign(abs_delta_p=masking["delta_p"].abs())
    s = (m.groupby(["run_id", "render_id", "region", "fill_mode"])
         .agg(n=("window_id", "size"), mean_abs_delta_p=("abs_delta_p", "mean"), flip_rate=("flipped", "mean"),
              area_frac=("area_frac", "mean")).reset_index())
    wide = s.pivot_table(index=["run_id", "render_id", "region"], columns="fill_mode",
                         values="mean_abs_delta_p").reset_index()
    if {"background", "random_area_matched"} <= set(wide.columns):
        wide["region_minus_control"] = wide["background"] - wide["random_area_matched"]
        s = s.merge(wide[["run_id", "render_id", "region", "region_minus_control"]],
                    on=["run_id", "render_id", "region"], how="left")
    return s
