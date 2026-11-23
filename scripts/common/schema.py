"""Frozen data contracts (Execution Plan Section 10).

The trainer READS the window registry (10.1) + rendering registry (10.2) and
WRITES prediction files (10.3). These column specs are the single source of
truth so every producer/consumer agrees on the schema string. A schema change
needs sign-off and a version bump (spec-v1).

Owner: Kaushik (consumes Farzaan's data_v1/render_v1; produces predictions).
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd

# ---- Labels (D5): 3-class future return. Logit/softmax column order is fixed. ----
LABELS = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}
CLASS_ORDER = ["down", "neutral", "up"]  # index i == label i
N_CLASSES = 3

# ---- Rendering factors (D7 / Section 5.2). Seen are trainable; held-out never tuned. ----
RENDER_IDS = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
SEEN_RENDERS = ["R0", "R1", "R2", "R4"]      # D8: appearance changes seen in training
HELD_OUT_RENDERS = ["R3", "R5", "R6"]        # D8: one decoration + two geometric, held out
NUMERIC_RENDER_ID = "NUM"                     # E5 / B0 use this sentinel (10.3)

# ---- 10.1 Window registry columns (data_v1/windows.parquet) ----
WINDOW_COLUMNS = [
    "window_id", "row_idx", "ticker", "asset_type",
    "input_start_date", "anchor_date", "label_end_date",
    "split", "fwd_return", "sigma_30", "label_score",
    "label", "ohlcv_hash", "data_version",
]

# ---- 10.2 Rendering registry columns (render_v1/render_registry.parquet) ----
RENDER_COLUMNS = [
    "window_id", "render_id", "split", "label", "ohlcv_hash",
    "render_role", "image_path", "image_hash", "render_version",
]

# ---- 10.3 Prediction file columns (outputs/predictions/<run_id>.parquet) ----
PREDICTION_COLUMNS = [
    "run_id", "experiment_id", "seed", "split", "window_id", "render_id",
    "true_label",
    "logit_down", "logit_neutral", "logit_up",
    "p_down", "p_neutral", "p_up",
    "pred_class", "checkpoint",
]

SPLITS = ["train", "val", "test", "purged"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def validate_prediction_frame(
    df: pd.DataFrame,
    expected_pairs: Iterable[tuple[str, str]] | None = None,
    image_model: bool = True,
) -> None:
    """Reject incomplete / malformed prediction files (10.3 validator contract).

    - columns must match the frozen schema exactly
    - softmax rows must sum to 1 within 1e-5
    - pred_class must equal argmax of the logits
    - for image models, every (window_id, render_id) in `expected_pairs` must be present
    """
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"prediction frame missing columns: {missing}")

    p = df[["p_down", "p_neutral", "p_up"]].to_numpy()
    sums = p.sum(axis=1)
    if ((sums < 1 - 1e-5) | (sums > 1 + 1e-5)).any():
        raise ValueError("softmax rows do not sum to 1 within 1e-5")

    logits = df[["logit_down", "logit_neutral", "logit_up"]].to_numpy()
    if (logits.argmax(axis=1) != df["pred_class"].to_numpy()).any():
        raise ValueError("pred_class disagrees with argmax(logits)")

    if image_model and expected_pairs is not None:
        have = set(zip(df["window_id"], df["render_id"]))
        want = set(expected_pairs)
        if have != want:
            raise ValueError(
                f"incomplete coverage: missing {len(want - have)} pairs, "
                f"extra {len(have - want)} pairs"
            )
