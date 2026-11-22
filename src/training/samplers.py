"""PairSampler (Execution Plan 9.2).

Draws two DISTINCT renderings of the same window for paired training. Draws are
reproducible: the same (seed, window key) always yields the same pair, so a run
is deterministic and re-runnable from a checkpoint.
"""
from __future__ import annotations

import numpy as np


class PairSampler:
    def __init__(self, render_ids: list[str], seed: int = 0):
        if len(render_ids) < 2:
            raise ValueError("pair_distinct needs >= 2 training renderings")
        self.render_ids = list(render_ids)
        self.seed = seed

    def draw_pair(self, available: list[str], key: int) -> tuple[str, str]:
        """Return two distinct render_ids from `available`, deterministically in `key`."""
        avail = [r for r in self.render_ids if r in available]
        if len(avail) < 2:
            raise ValueError(f"window exposes < 2 of the training renderings: {available}")
        rng = np.random.default_rng((self.seed, key))
        a, b = rng.choice(len(avail), size=2, replace=False)
        return avail[a], avail[b]
