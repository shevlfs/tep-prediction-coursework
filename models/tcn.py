from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.relu2 = nn.ReLU()
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            ) 
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.relu1(out)
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.relu2(out)
        return out + self.downsample(x)


class TCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 52,
        num_classes: int = 21,
        hidden: int = 64,
        kernel_size: int = 3,
        num_levels: int = 4
    ):
        super().__init__()
        layers = []
        dilations = [2 ** i for i in range(num_levels)]
        for i, d in enumerate(dilations):
            in_chs = in_channels if i == 0 else hidden
            layers.append(TemporalBlock(in_chs, hidden, kernel_size, d))
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.network(x)
        h = self.pool(h)
        h = h.squeeze(-1)
        return self.classifier(h)
