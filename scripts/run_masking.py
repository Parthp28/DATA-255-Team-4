"""Region masking with area-matched random controls over the D15 sample (plan 12, 10.6; H09). Owner: Farzaan.

    python scripts/run_masking.py --run-dir outputs/runs/<seed-0 run of E1/E2/E3b/E4b/E3/E4> --data-root data/render_v1

For each sampled image, the script masks candles, volume, grid and axes by filling them with the background
colour, and masks an area-matched random set of non-ink pixels in the same panel as a control. It writes
outputs/masking/<run_id>.parquet (plan 10.6 columns) and <run_id>.json (sample and provenance). Only seed-0
checkpoints of frozen configurations are accepted (D15).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.interpretability.masking import MASKING_COLUMNS, masking_rows  # noqa: E402
from src.interpretability.regions import interpretability_sample  # noqa: E402
from src.interpretability.runs import load_run, sample_items, sample_sha256  # noqa: E402


def run(run_dir, data_root, out_dir="outputs/masking", renders=("R0",), n=2000, batch=64, device="cpu",
        test_mode=False, seed=0):
    rec, _cfg, model = load_run(run_dir, device, test_mode)
    windows = pd.read_parquet(Path(data_root) / "windows.parquet")
    ids = interpretability_sample(windows, n=n, seed=seed)
    rows = masking_rows(model, rec["run_id"], sample_items(data_root, ids, list(renders)), device, seed, batch)
    df = pd.DataFrame(rows, columns=MASKING_COLUMNS)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{rec['run_id']}.parquet", index=False)
    meta = {"run_id": rec["run_id"], "experiment_id": rec["experiment_id"], "render_ids": list(renders),
            "n_windows": len(ids), "sample_seed": seed, "sample_sha256": sample_sha256(ids),
            "fill": "background colour of the rendering", "control": "random non-ink 4x4 blocks, same panel, same area",
            "test_mode": bool(test_mode)}
    (out / f"{rec['run_id']}.json").write_text(json.dumps(meta, indent=2))
    return df, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs/masking")
    ap.add_argument("--renders", nargs="+", default=["R0"])
    ap.add_argument("--n", type=int, default=2000, help="D15 sample size")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--test-mode", action="store_true", help="unit tests only: any run, any seed")
    a = ap.parse_args(argv)
    df, meta = run(a.run_dir, a.data_root, a.out, a.renders, a.n, a.batch, torch.device(a.device), a.test_mode)
    print(f"[{meta['run_id']}] masking: {meta['n_windows']} windows x {len(a.renders)} renderings -> {len(df)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
