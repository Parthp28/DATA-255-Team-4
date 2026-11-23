"""Raw OHLCV acquisition and cache (Execution Plan D1, D2; owner Farzaan).

Each ticker is downloaded once from yfinance (daily, auto-adjusted) into
``<raw_dir>/<TICKER>.csv`` and recorded in ``<raw_dir>/manifest.json`` with its
sha256. The dataset build reads only the cache, so data_v1 is reproducible from
the cached files even after Yahoo revises its adjustments.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import time

import pandas as pd

RAW_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
MANIFEST = "manifest.json"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(raw_dir: str) -> dict:
    path = os.path.join(raw_dir, MANIFEST)
    if not os.path.exists(path):
        return {"files": {}}
    with open(path) as fh:
        return json.load(fh)


def save_manifest(raw_dir: str, manifest: dict) -> None:
    path = os.path.join(raw_dir, MANIFEST)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _fetch_yfinance(ticker: str, start: str, end_inclusive: str) -> pd.DataFrame:
    import yfinance as yf  # imported lazily: only the download step needs network

    end_excl = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end_excl, auto_adjust=True, actions=False,
                     progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: empty download")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df[RAW_COLUMNS]


def download_all(tickers: list[str], raw_dir: str, start: str, end: str,
                 retries: int = 4, pause: float = 1.0, force: bool = False,
                 fetch=_fetch_yfinance) -> dict:
    """Download every missing ticker into the cache; return the updated manifest.

    A cached file whose sha256 matches the manifest is never re-downloaded
    (unless ``force``), so a second run is a no-op.
    """
    os.makedirs(raw_dir, exist_ok=True)
    manifest = load_manifest(raw_dir)
    files = manifest.setdefault("files", {})
    for t in tickers:
        path = os.path.join(raw_dir, f"{t}.csv")
        entry = files.get(t)
        if not force and entry and os.path.exists(path) and sha256_file(path) == entry["sha256"]:
            continue
        last_err = None
        for attempt in range(retries):
            try:
                df = fetch(t, start, end)
                break
            except Exception as err:  # network / rate limit: back off and retry
                last_err = err
                time.sleep(pause * (2 ** attempt))
        else:
            raise RuntimeError(f"download failed for {t}: {last_err}")
        df.to_csv(path, index=False)
        files[t] = {
            "sha256": sha256_file(path),
            "rows": int(len(df)),
            "first_date": str(df["Date"].iloc[0]),
            "last_date": str(df["Date"].iloc[-1]),
            "downloaded_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }
        save_manifest(raw_dir, manifest)
        time.sleep(pause)
    manifest["source"] = {"provider": "yfinance", "auto_adjust": True, "start": start, "end": end}
    try:
        import yfinance as yf
        manifest["source"]["yfinance_version"] = yf.__version__
    except Exception:
        pass
    save_manifest(raw_dir, manifest)
    return manifest


def verify_cache(tickers: list[str], raw_dir: str) -> None:
    """Raise if any ticker is missing from the cache or its hash does not match."""
    manifest = load_manifest(raw_dir)
    bad = []
    for t in tickers:
        path = os.path.join(raw_dir, f"{t}.csv")
        entry = manifest.get("files", {}).get(t)
        if entry is None or not os.path.exists(path):
            bad.append(f"{t}: missing")
        elif sha256_file(path) != entry["sha256"]:
            bad.append(f"{t}: sha256 mismatch")
    if bad:
        raise RuntimeError("raw cache check failed: " + "; ".join(bad))


def load_raw(raw_dir: str, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(raw_dir, f"{ticker}.csv"), dtype={"Date": str})
    return df[RAW_COLUMNS]
