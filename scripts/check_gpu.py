"""D16 pre-flight: refuse a GPU session that is not on the D16 GPU (docs/compute_environment.md). Owner: Kaushik.

    python scripts/check_gpu.py            # exit 0 on the D16 GPU; 3 on any other GPU; 4 when no CUDA GPU is visible

Every reported image-model run uses the same GPU (D16: NVIDIA GeForce RTX 5090). train.py refuses any other GPU and
freeze_results.py rejects image-model runs from one; this check stops a session before it spends any time.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.gate import D16_GPU, is_d16_gpu  # noqa: E402

REQUIRED = D16_GPU


def gpu_names() -> list[str]:
    import torch
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def check(names: list[str]) -> tuple[int, str]:
    if not names:
        return 4, "no CUDA GPU visible: this session cannot run GPU jobs"
    bad = [n for n in names if not is_d16_gpu(n)]
    if bad:
        return 3, f"GPU {bad} is not the D16 GPU ({REQUIRED}). Stop this session."
    return 0, f"OK: {names} (D16)"


def main(argv=None, names=None) -> int:
    code, msg = check(gpu_names() if names is None else names)
    print(msg)
    return code


if __name__ == "__main__":
    sys.exit(main())
