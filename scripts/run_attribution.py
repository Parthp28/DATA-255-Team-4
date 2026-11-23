"""Grad-CAM (ResNet-18 runs) or attention rollout (DeiT-S runs) over the D15 sample (plan 12, 10.6).

    python scripts/run_attribution.py --run-dir outputs/runs/<E1 seed-0 run> --data-root data/render_v1
    python scripts/run_attribution.py --run-dir outputs/runs/<E2 seed-0 run> --data-root data/render_v1 --renders R0 R2

Owners: Kaushik (Grad-CAM: E1, E3b, E3), Khushi (rollout: E2, E4b, E4). The method follows the run's architecture.
Writes outputs/attribution/<run_id>.parquet (plan 10.6 columns) and <run_id>.json, which records the sample and
its provenance. Only seed-0 checkpoints of frozen configurations are accepted (D15), and the D15 sample is the
same 2,000 class-stratified test windows for every model.
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

from src.interpretability.gradcam import gradcam  # noqa: E402
from src.interpretability.masking import to_tensor  # noqa: E402
from src.interpretability.regions import ATTRIBUTION_COLUMNS, attribution_rows, interpretability_sample  # noqa: E402
from src.interpretability.rollout import attention_rollout  # noqa: E402
from src.interpretability.runs import load_run, method_for, sample_items, sample_sha256  # noqa: E402


def run(run_dir, data_root, out_dir="outputs/attribution", renders=("R0",), n=2000, batch=32, device="cpu",
        test_mode=False, seed=0):
    rec, cfg, model = load_run(run_dir, device, test_mode)
    method = method_for(cfg)
    windows = pd.read_parquet(Path(data_root) / "windows.parquet")
    ids = interpretability_sample(windows, n=n, seed=seed)
    rows, chunk = [], []

    def flush():
        x = to_tensor([c[2] for c in chunk]).to(device)
        heat = gradcam(model, x)[0] if method == "gradcam" else attention_rollout(model, x)[0]
        for (wid, rid, _img, masks, _style), h in zip(chunk, heat):
            rows.extend(attribution_rows(rec["run_id"], wid, rid, method, h, masks))
        chunk.clear()

    for item in sample_items(data_root, ids, list(renders)):
        chunk.append(item)
        if len(chunk) == batch:
            flush()
    if chunk:
        flush()
    df = pd.DataFrame(rows, columns=ATTRIBUTION_COLUMNS)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{rec['run_id']}.parquet", index=False)
    meta = {"run_id": rec["run_id"], "experiment_id": rec["experiment_id"], "method": method,
            "render_ids": list(renders), "n_windows": len(ids), "sample_seed": seed, "sample_sha256": sample_sha256(ids),
            "no_signal_images": int(df.groupby(["window_id", "render_id"])["attribution_mass"]
                                    .apply(lambda s: s.isna().all()).sum()),
            "rollout_residual": 0.5 if method == "rollout" else None, "test_mode": bool(test_mode)}
    (out / f"{rec['run_id']}.json").write_text(json.dumps(meta, indent=2))
    return df, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs/attribution")
    ap.add_argument("--renders", nargs="+", default=["R0"])
    ap.add_argument("--n", type=int, default=2000, help="D15 sample size")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--test-mode", action="store_true", help="unit tests only: any run, any seed")
    a = ap.parse_args(argv)
    df, meta = run(a.run_dir, a.data_root, a.out, a.renders, a.n, a.batch, torch.device(a.device), a.test_mode)
    print(f"[{meta['run_id']}] {meta['method']}: {meta['n_windows']} windows x {len(a.renders)} renderings -> "
          f"{len(df)} rows; images without signal: {meta['no_signal_images']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
