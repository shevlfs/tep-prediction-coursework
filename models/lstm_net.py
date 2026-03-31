from __future__ import annotations

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """Bidirectional LSTM for TEP fault classification."""

    def __init__(self, in_channels: int = 52, num_classes: int = 21,
                 hidden: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C) for LSTM
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        # Take last timestep
        h = out[:, -1, :]
        return self.classifier(h)
