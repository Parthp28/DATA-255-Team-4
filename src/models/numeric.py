"""1D numeric CNN for E5 (D-row: 3 conv layers k=3, 64 ch, global avg pool, linear head).

Owner: Kaushik writes the code; Farzaan runs E5 on CPU. This is the numeric
control (A6): does turning OHLCV into an image add anything over the raw numbers?
Input is a (batch, 5, 30) tensor = 5 OHLCV channels x 30 trading days.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..common.schema import N_CLASSES


class Numeric1DCNN(nn.Module):
    def __init__(self, in_channels: int = 5, width: int = 64, num_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=3, padding=1),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 5, 30)
        h = self.features(x)
        h = h.mean(dim=-1)            # global average pool over time
        return self.head(h)
