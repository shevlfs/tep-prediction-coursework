from __future__ import annotations

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """Patch-based Time Series Transformer for TEP fault classification.
    Inspired by the PatchTST architecture — segments time series into
    non-overlapping patches before transformer encoding."""

    def __init__(self, in_channels: int = 52, num_classes: int = 21,
                 seq_len: int = 32, patch_len: int = 8,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.patch_len = patch_len
        n_patches = seq_len // patch_len
        patch_dim = in_channels * patch_len

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, T = x.shape
        x = x.reshape(B, C, T // self.patch_len, self.patch_len)
        x = x.permute(0, 2, 1, 3).reshape(B, T // self.patch_len, C * self.patch_len)
        x = self.patch_embed(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)
