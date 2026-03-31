from __future__ import annotations

import torch.nn as nn


class TEPNet(nn.Module):
    """1D CNN for TEP fault classification (original architecture)."""

    def __init__(self, in_channels: int = 52, num_classes: int = 21):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze(-1)
        return self.classifier(h)
