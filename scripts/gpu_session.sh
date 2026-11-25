#!/usr/bin/env bash
# One GPU session on Kaggle or Colab (plan 14.5). Owner: Kaushik (tooling). Nothing here names a person or account.
#
#   bash scripts/gpu_session.sh --lane <lane> --data-src <dir> [--commit <sha>] [--queue ops/run_queue.csv]
#                               [--check-only] [--dry-run] [--no-install]
#
# --data-src is a directory that holds render_v1/ (registry, windows.parquet, ohlcv.npy, manifests) and render_v1_tar/
# (the 21 shard tars and SHA256SUMS), for example a mounted shared folder or an attached Kaggle Dataset.
#
# Steps, each of which stops the session on failure:
#   1. optional checkout of the frozen commit (detached), refusing a dirty tree
#   2. pip install -r requirements.txt (skip with --no-install)
#   3. D16: scripts/check_gpu.py must report the D16 GPU
#   4. data: copy render_v1/, verify SHA256SUMS of the tars, extract the 21 shard folders, count the PNGs
#   5. --check-only stops here; otherwise hand over to scripts/run_queue.py for the given lane
#      (--dry-run passes through, so the queue lists what it would run and runs nothing)
set -euo pipefail

LANE="" DATA_SRC="" COMMIT="" QUEUE="ops/run_queue.csv" CHECK_ONLY=0 DRY_RUN=0 INSTALL=1
usage() { sed -n '2,17p' "$0"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --lane) LANE="$2"; shift 2 ;;
    --data-src) DATA_SRC="$2"; shift 2 ;;
    --commit) COMMIT="$2"; shift 2 ;;
    --queue) QUEUE="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-install) INSTALL=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[ -n "$LANE" ] || { echo "--lane is required" >&2; exit 2; }
[ -n "$DATA_SRC" ] || { echo "--data-src is required" >&2; exit 2; }

cd "$(dirname "$0")/.."
if [ -n "$COMMIT" ]; then
  [ -z "$(git status --porcelain)" ] || { echo "dirty tree; refusing to check out $COMMIT" >&2; exit 5; }
  git fetch -q origin && git checkout -q --detach "$COMMIT"
fi
echo "commit: $(git rev-parse HEAD)"
[ "$INSTALL" = 1 ] && python -m pip install -q -r requirements.txt

python scripts/check_gpu.py            # exits non-zero on anything but the D16 GPU

mkdir -p data
if [ ! -f data/render_v1/render_registry.parquet ]; then
  mkdir -p data/render_v1
  (cd "$DATA_SRC/render_v1" && tar --exclude='./shards' -cf - .) | (cd data/render_v1 && tar -xf -)
fi
if [ ! -d data/render_v1/shards ] || [ "$(find data/render_v1/shards -mindepth 1 -maxdepth 1 -type d | wc -l)" -ne 21 ]; then
  (cd "$DATA_SRC/render_v1_tar" && sha256sum -c --quiet SHA256SUMS)
  mkdir -p data/render_v1/shards
  for t in "$DATA_SRC"/render_v1_tar/*.tar; do tar -xf "$t" -C data/render_v1/shards; done
fi
n_png=$(find data/render_v1/shards -name '*.png' | wc -l)
n_reg=$(python -c "import pandas as pd; print(len(pd.read_parquet('data/render_v1/render_registry.parquet')))")
echo "shards: $n_png PNG files; registry rows: $n_reg"
[ "$n_png" -eq "$n_reg" ] || { echo "PNG count differs from the registry" >&2; exit 6; }

if [ "$CHECK_ONLY" = 1 ]; then echo "pre-flight passed (check only)"; exit 0; fi
args=(--lane "$LANE" --queue "$QUEUE")
[ "$DRY_RUN" = 1 ] && args+=(--dry-run)
exec python scripts/run_queue.py "${args[@]}"
