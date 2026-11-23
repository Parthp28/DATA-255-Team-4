"""Independent G1/G2 verification of data_v1 and render_v1 (owner: Parth; plan Sections 9.4, 16).

    python scripts/verify_g1g2.py --data-dir data                    # G1 + G2 (2,000-image sample)
    python scripts/verify_g1g2.py --data-dir data --all-images       # G2: decode and hash every image
    python scripts/verify_g1g2.py --data-dir data --skip-render      # G1 only

Re-derives every property from the artifacts and the raw cache with pandas/numpy/PIL only; it does not import
the data-pipeline code under review. Farzaan's renderer and Kaushik's loader are imported only to show that a
fresh process re-renders identical images and that the production loader reads the shards. Writes a JSON
report (default outputs/gates/g1g2_report.json) and exits non-zero if any check fails.

G1: manifest hashes, schema, 50 tickers, ohlcv_hash for every row, T-L3 disjoint splits, T-L1/T-L2 span order,
    strict purge (no shared day; 34 daily anchors -> 7 strided windows per ticker per boundary),
    T-L5 train-only thresholds that reproduce and differ when val/test are included, labels, D2 recomputed
    from raw for 200 windows, class balance.
G2: SHA256SUMS of the shard tars, registry completeness (7 x non-purged windows, no purged), equal
    hash/label/split per window, roles, every (or sampled) PNG is 224 x 224 x 3 uint8 with the registry hash,
    fresh-process re-render, 200-window wick spot check against the plan geometry, production loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

RENDER_IDS = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
ROLE = {"R0": "reference", "R1": "seen", "R2": "seen", "R4": "seen", "R3": "held_out", "R5": "held_out",
        "R6": "held_out"}
WINDOW_COLUMNS = ["window_id", "row_idx", "ticker", "asset_type", "input_start_date", "anchor_date",
                  "label_end_date", "split", "fwd_return", "sigma_30", "label_score", "label", "ohlcv_hash",
                  "data_version"]
RENDER_COLUMNS = ["window_id", "render_id", "split", "label", "ohlcv_hash", "render_role", "image_path",
                  "image_hash", "render_version"]
L, H, STRIDE = 30, 5, 5


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Report:
    def __init__(self):
        self.checks = []

    def add(self, gate, name, ok, detail=""):
        self.checks.append({"gate": gate, "check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        print(f"[{gate}] {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)

    def skip(self, gate, name, why):
        self.checks.append({"gate": gate, "check": name, "status": "SKIPPED", "detail": why})
        print(f"[{gate}] SKIP  {name}  ({why})", flush=True)

    def failed(self):
        return [c for c in self.checks if c["status"] == "FAIL"]


# =========================================================================== G1
def raw_frames(raw_dir: Path, tickers) -> dict:
    out = {}
    for t in tickers:
        df = pd.read_csv(raw_dir / f"{t}.csv")
        df["Date"] = df["Date"].astype(str).str.slice(0, 10)
        out[t] = df.drop_duplicates("Date", keep="last").sort_values("Date").set_index("Date")
    return out


def verify_g1(data: Path, cfg: dict, rep: Report) -> dict:
    d1 = data / cfg["data_version"]
    man = json.loads((d1 / "build_manifest.json").read_text())
    rep.add("G1", "windows.parquet sha256 = build_manifest", sha_file(d1 / "windows.parquet") == man["windows_parquet_sha256"])
    rep.add("G1", "ohlcv.npy sha256 = build_manifest", sha_file(d1 / "ohlcv.npy") == man["ohlcv_npy_sha256"])
    rep.add("G1", "raw manifest sha256 = build_manifest", sha_file(data / "raw" / "manifest.json") == man["raw_manifest_sha256"])
    w = pd.read_parquet(d1 / "windows.parquet")
    ohlcv = np.load(d1 / "ohlcv.npy", mmap_mode="r")
    tickers = cfg["tickers"]["stock"] + cfg["tickers"]["etf"]
    rep.add("G1", "schema 10.1 columns", list(w.columns) == WINDOW_COLUMNS, str(list(w.columns)))
    rep.add("G1", "ohlcv.npy is N x 30 x 5 float32", ohlcv.shape == (len(w), L, 5) and ohlcv.dtype == np.float32,
            f"{ohlcv.shape} {ohlcv.dtype}")
    rep.add("G1", "ticker set = configs/data.yaml D1 universe (50 in production)", sorted(w["ticker"].unique()) == sorted(tickers), f"{w['ticker'].nunique()}")
    rep.add("G1", "window_id unique and = ticker_anchor",
            not w["window_id"].duplicated().any() and (w["window_id"] == w["ticker"] + "_" + w["anchor_date"].astype(str)).all())
    rep.add("G1", "single data_version", set(w["data_version"]) == {cfg["data_version"]})
    rep.add("G1", "T-D4 ohlcv_hash recomputes for every row",
            all(hashlib.sha256(np.ascontiguousarray(ohlcv[i], np.float32).tobytes()).hexdigest() == h
                for i, h in zip(w["row_idx"], w["ohlcv_hash"])))
    live = w[w["split"] != "purged"]
    ids = {s: set(w.loc[w["split"] == s, "window_id"]) for s in ("train", "val", "test", "purged")}
    rep.add("G1", "T-L3 split window_id sets disjoint (independent)",
            all(not (ids[a] & ids[b]) for a in ids for b in ids if a < b))
    a = {s: live.loc[live["split"] == s] for s in ("train", "val", "test")}
    rep.add("G1", "T-L2 train anchors < val anchors < test anchors",
            a["train"]["anchor_date"].max() < a["val"]["anchor_date"].min()
            and a["val"]["anchor_date"].max() < a["test"]["anchor_date"].min())
    ok = all(a[e].groupby("ticker")["label_end_date"].max().lt(a[l].groupby("ticker")["input_start_date"].min()).all()
             for e, l in (("train", "val"), ("val", "test"), ("train", "test")))
    rep.add("G1", "T-L1 no trading day shared by earlier label span and later input span (per ticker)", ok)

    # strict purge against the global calendar built from the raw cache (plan 5.4)
    raw = raw_frames(data / "raw", tickers)
    counts = pd.concat([pd.Series(f.index) for f in raw.values()]).value_counts()
    share = float(cfg.get("calendar_min_ticker_share", 0.5))
    cal = np.array(sorted(counts[counts >= share * len(raw)].index))
    pos = {d: i for i, d in enumerate(cal)}
    detail, ok_all = {}, True
    for name, start in (("val", cfg["splits"]["val"][0]), ("test", cfg["splits"]["test"][0])):
        b = int(np.searchsorted(cal, start))
        lo, hi = b - H, b + L - 2                       # purged daily anchors B-5 .. B+28 (34)
        anc = w["anchor_date"].map(pos)
        in_band = w[(anc >= lo) & (anc <= hi)]
        per = in_band.groupby("ticker")["split"].agg(lambda s: (s == "purged").all())
        n_purged = w[(w["split"] == "purged") & (anc >= lo) & (anc <= hi)].groupby("ticker").size()
        prev = {"val": "train", "test": "val"}[name]
        train_ok = (live.loc[live["split"] == prev, "label_end_date"].map(pos) <= b - 1).all()
        eval_ok = (live.loc[live["split"] == name, "input_start_date"].map(pos) >= b).all()
        ok = (hi - lo + 1 == L + H - 1) and per.all() and (n_purged == math.ceil((L + H - 1) / STRIDE)).all() \
            and len(n_purged) == len(tickers) and train_ok and eval_ok
        ok_all &= bool(ok)
        detail[name] = {"B": str(cal[b]), "purged_daily_anchors": hi - lo + 1,
                        "purged_windows_per_ticker": sorted(set(n_purged.tolist())), "train_side_ok": bool(train_ok),
                        "eval_side_ok": bool(eval_ok)}
    rep.add("G1", "T-L4 strict purge: 34 daily anchors, 7 strided windows per ticker per boundary", ok_all, json.dumps(detail))

    th = json.loads((d1 / "thresholds.json").read_text())
    z_tr = w.loc[w["split"] == "train", "label_score"].to_numpy(np.float64)
    t1, t2 = np.quantile(z_tr, [1 / 3, 2 / 3])
    rep.add("G1", "T-L5 thresholds reproduce from train windows only (independent)",
            abs(t1 - th["t1"]) < 1e-12 and abs(t2 - th["t2"]) < 1e-12, f"t1={t1:.6f} t2={t2:.6f}")
    a1, a2 = np.quantile(live["label_score"].to_numpy(np.float64), [1 / 3, 2 / 3])
    rep.add("G1", "T-L5 thresholds differ when val/test are included", not (abs(a1 - t1) < 1e-9 and abs(a2 - t2) < 1e-9))
    z = w["label_score"].to_numpy(np.float64)
    lab = np.where(z <= th["t1"], 0, np.where(z > th["t2"], 2, 1))
    rep.add("G1", "labels = DOWN/NEUTRAL/UP at the train tertiles", (lab == w["label"].to_numpy()).all())

    rng = np.random.default_rng(0)
    sample = live.iloc[rng.choice(len(live), min(200, len(live)), replace=False)]
    bad = 0
    for r in sample.itertuples():
        f = raw[r.ticker]
        i = f.index.get_loc(r.anchor_date)
        close = f["Close"].to_numpy(np.float64)
        rr = close[i + H] / close[i] - 1
        sig = np.std(np.diff(np.log(close[i - L + 1: i + 1])), ddof=1)
        win = ohlcv[r.row_idx]
        same_oc = np.allclose(win[:, [0, 3, 4]], f[["Open", "Close", "Volume"]].iloc[i - L + 1: i + 1].to_numpy(np.float32))
        bad += not (abs(rr - r.fwd_return) < 1e-9 and abs(sig - r.sigma_30) < 1e-9 and same_oc
                    and f.index[i + H] == r.label_end_date and f.index[i - L + 1] == r.input_start_date)
    rep.add("G1", "T-D1/T-D2/T-D3 recomputed from raw for 200 windows (r, sigma_30, O/C/V, 30+5 rows)", bad == 0, f"{bad} mismatches")
    balance = {s: g["label"].value_counts(normalize=True).sort_index().round(3).tolist() for s, g in live.groupby("split")}
    rep.add("G1", "train classes balanced (tertiles)", max(abs(x - 1 / 3) for x in balance["train"]) < 0.01, json.dumps(balance))
    counts = live["split"].value_counts().to_dict()
    print(f"[G1] windows per split {counts}; purged {int((w['split'] == 'purged').sum())}")
    return {"windows": w, "ohlcv": ohlcv, "counts": counts}


# =========================================================================== G2
def price_row(p, lo, hi, top, height):
    return top + (height - 1) // 2 if hi <= lo else top + int(math.floor((hi - p) / (hi - lo) * (height - 1) + 0.5))


def verify_g2(data: Path, styles_cfg: dict, g1: dict, rep: Report, all_images: bool, sample: int, seed: int = 0):
    rv = styles_cfg["render_version"]
    r1 = data / rv
    man = json.loads((r1 / "render_manifest.json").read_text())
    rep.add("G2", "render_registry.parquet sha256 = render_manifest", sha_file(r1 / "render_registry.parquet") == man["render_registry_sha256"])
    tar_dir = data / f"{rv}_tar"
    if (tar_dir / "SHA256SUMS").exists():
        sums = [ln.split() for ln in (tar_dir / "SHA256SUMS").read_text().splitlines() if ln.strip()]
        ok = len(sums) == 21 and all(sha_file(tar_dir / n.lstrip("*")) == h for h, n in sums)
        rep.add("G2", "SHA256SUMS of the 21 shard tars", ok, f"{len(sums)} entries")
    else:
        rep.skip("G2", "SHA256SUMS of the shard tars", f"{tar_dir} not present")
    w, ohlcv = g1["windows"], g1["ohlcv"]
    reg = pd.read_parquet(r1 / "render_registry.parquet")
    live = w[w["split"] != "purged"]
    rep.add("G2", "schema 10.2 columns", list(reg.columns) == RENDER_COLUMNS)
    rep.add("G2", f"registry rows = 7 x non-purged windows ({7 * len(live):,})",
            len(reg) == 7 * len(live) and not reg.duplicated(["window_id", "render_id"]).any(), f"{len(reg):,} rows")
    rep.add("G2", "no purged window rendered; every live window has R0-R6",
            set(reg["window_id"]) == set(live["window_id"]) and (reg.groupby("window_id")["render_id"].nunique() == 7).all())
    m = reg.merge(w[["window_id", "split", "label", "ohlcv_hash"]], on="window_id", suffixes=("", "_w"))
    rep.add("G2", "T-R1..R3 same ohlcv_hash, label, split for every rendering",
            all((m[c] == m[f"{c}_w"]).all() for c in ("split", "label", "ohlcv_hash")))
    rep.add("G2", "render roles (D8) and single render_version",
            (reg["render_role"] == reg["render_id"].map(ROLE)).all() and set(reg["render_version"]) == {rv})
    shards = r1 / "shards"
    check = reg if all_images else reg.sample(min(sample, len(reg)), random_state=seed)
    bad = []
    for path, h in zip(check["image_path"], check["image_hash"]):
        with Image.open(shards / path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if arr.shape != (224, 224, 3) or hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest() != h:
            bad.append(path)
    rep.add("G2", f"T-R4 PNGs are 224x224x3 uint8 with the registry hash ({'all' if all_images else 'sample'} "
                  f"{len(check):,})", not bad, f"{len(bad)} bad: {bad[:3]}")
    try:
        from src.rendering.rasterizer import render
        from src.rendering.styles import load_styles
        styles = load_styles(str(ROOT / "configs" / "render_styles.yaml"))[0]
        rows = live.sample(min(50, len(live)), random_state=seed + 1)
        regi = reg.set_index(["window_id", "render_id"])["image_hash"]
        mism = sum(hashlib.sha256(np.ascontiguousarray(render(ohlcv[r.row_idx], styles[rid])).tobytes()).hexdigest()
                   != regi[(r.window_id, rid)] for r in rows.itertuples() for rid in RENDER_IDS)
        rep.add("G2", "T-R7 fresh-process re-render reproduces registry hashes (50 windows x 7)", mism == 0, f"{mism} mismatches")
    except ImportError as exc:
        rep.skip("G2", "T-R7 fresh-process re-render", f"renderer not importable ({exc})")
    base = styles_cfg["base"]
    mg, cv = int(base["margin"]), int(styles_cfg["canvas"])
    pw = cv - 2 * mg
    ph = int(math.floor(base["price_frac"] * pw + 0.5))
    colors = np.array([base["up_color"], base["down_color"]], dtype=np.uint8)
    slot = pw / L
    bad_spot = 0
    rows = live.sample(min(200, len(live)), random_state=seed + 2)
    for r in rows.itertuples():
        win = np.asarray(ohlcv[r.row_idx], np.float64)
        lo, hi = win[:, 2].min(), win[:, 1].max()
        path = reg.loc[(reg["window_id"] == r.window_id) & (reg["render_id"] == "R0"), "image_path"].iloc[0]
        with Image.open(shards / path) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.uint8)
        ink = (img[..., None, :] == colors[None, None]).all(-1).any(-1)
        for i in range(L):
            c0, c1 = mg + int(math.floor(i * slot)), mg + int(math.floor((i + 1) * slot))
            rows_i = np.where(ink[mg:mg + ph, c0:c1].any(axis=1))[0] + mg
            if not len(rows_i) or rows_i.min() != price_row(win[i, 1], lo, hi, mg, ph) \
                    or rows_i.max() != price_row(win[i, 2], lo, hi, mg, ph):
                bad_spot += 1
    rep.add("G2", "T-R6 200-window pixel spot check: wick extremes = plan price-to-pixel map (R0)", bad_spot == 0,
            f"{bad_spot} candles off")
    try:
        from src.training.datasets import ChartImageDataset
        ok = True
        for s in ("train", "val", "test"):
            ds = ChartImageDataset(reg, str(shards), s, RENDER_IDS, 224)
            for i in np.linspace(0, len(ds) - 1, 5).astype(int):
                x = ds[int(i)]["image"]
                ok &= tuple(x.shape) == (3, 224, 224) and bool(np.isfinite(x.numpy()).all())
        rep.add("G2", "production loader (ChartImageDataset) reads every split's shards", ok)
    except ImportError as exc:
        rep.skip("G2", "production loader", f"loader not importable ({exc})")
    sheet = r1 / "figure1_sheet.png"
    rep.add("G2", "Figure 1 review sheet present (team review required)", sheet.exists(), str(sheet))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--styles", default=str(ROOT / "configs" / "render_styles.yaml"))
    ap.add_argument("--all-images", action="store_true")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--report", default=str(ROOT / "outputs" / "gates" / "g1g2_report.json"))
    a = ap.parse_args(argv)
    rep = Report()
    data = Path(a.data_dir)
    g1 = verify_g1(data, yaml.safe_load(Path(a.config).read_text()), rep)
    if not a.skip_render:
        verify_g2(data, yaml.safe_load(Path(a.styles).read_text()), g1, rep, a.all_images, a.sample)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps({"checks": rep.checks, "split_counts": g1["counts"]}, indent=2))
    failed = rep.failed()
    print(f"\n{'ALL CHECKS PASSED' if not failed else f'{len(failed)} CHECK(S) FAILED'} -> {a.report}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
