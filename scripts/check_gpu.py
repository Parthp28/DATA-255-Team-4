"""D16 pre-flight: refuse a GPU session that is not on an NVIDIA T4 (plan 14.5). Owner: Kaushik (tooling).

    python scripts/check_gpu.py            # exit 0 on a T4; 3 on any other GPU; 4 when no CUDA GPU is visible

Every reported run uses the same GPU type (D16, unchanged). The queue runner and train.py record the runtime GPU
model, and freeze_results.py rejects image-model runs that are not on a T4. This check stops a session before it
spends any time on a wrong device.
"""
from __future__ import annotations

import sys

REQUIRED = "T4"


def gpu_names() -> list[str]:
    import torch
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def check(names: list[str]) -> tuple[int, str]:
    if not names:
        return 4, "no CUDA GPU visible: this session cannot run GPU jobs"
    bad = [n for n in names if REQUIRED not in n]
    if bad:
        return 3, f"GPU {bad} is not an NVIDIA {REQUIRED}; D16 allows reported runs on T4 only. Stop this session."
    return 0, f"OK: {names} (D16 {REQUIRED})"


def main(argv=None, names=None) -> int:
    code, msg = check(gpu_names() if names is None else names)
    print(msg)
    return code


if __name__ == "__main__":
    sys.exit(main())
