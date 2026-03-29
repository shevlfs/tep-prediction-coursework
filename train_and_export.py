from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

DATA_DIR = Path("small_tep")
ONNX_DIR = Path("onnx")
ONNX_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 32
NUM_FEATURES = 52

df = pd.read_csv(DATA_DIR / "df.csv")
target = pd.read_csv(DATA_DIR / "target.csv")
train_mask = pd.read_csv(DATA_DIR / "train_mask.csv")

feature_cols = [c for c in df.columns if c not in ("run_id", "sample")]
assert len(feature_cols) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(feature_cols)}"

df["target"] = target["target"].values
df["train_mask"] = train_mask["train_mask"].values


def make_windows(group: pd.DataFrame, window: int):
    X_raw = group[feature_cols].values.astype(np.float32)
    y_raw = group["target"].values.astype(np.int64)
    xs, ys = [], []
    for i in range(len(X_raw) - window + 1):
        xs.append(X_raw[i : i + window].T)
        ys.append(y_raw[i + window - 1])
    return np.array(xs), np.array(ys)

train_df = df[df["train_mask"] == 1]
test_df = df[df["train_mask"] == 0]

print("Building train windows...")
train_xs, train_ys = [], []
for _, g in train_df.groupby("run_id"):
    x, y = make_windows(g, WINDOW_SIZE)
    if len(x):
        train_xs.append(x); train_ys.append(y)
X_train = np.concatenate(train_xs)
y_train = np.concatenate(train_ys)

print("Building test windows...")
test_xs, test_ys = [], []
for _, g in test_df.groupby("run_id"):
    x, y = make_windows(g, WINDOW_SIZE)
    if len(x):
        test_xs.append(x); test_ys.append(y)
X_test = np.concatenate(test_xs)
y_test = np.concatenate(test_ys)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train positive rate: {y_train.mean():.3f}")

mean = X_train.mean(axis=(0, 2), keepdims=True)
std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

np.save(ONNX_DIR / "norm_mean.npy", mean.squeeze())
np.save(ONNX_DIR / "norm_std.npy", std.squeeze())


class TEPNet(nn.Module):
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

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
model = TEPNet().to(device)

from torch.utils.data import TensorDataset, DataLoader

X_tr_t = torch.from_numpy(X_train)
y_tr_t = torch.from_numpy(y_train)
X_te_t = torch.from_numpy(X_test)
y_te_t = torch.from_numpy(y_test)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)

    model.eval()
    with torch.no_grad():
        logits = model(X_te_t.to(device))
        preds = logits.argmax(dim=1).cpu()
        acc = (preds == y_te_t).float().mean().item()
    print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(X_train):.4f}  test_acc={acc:.4f}")

model = model.cpu()
model.eval()
dummy = torch.randn(1, NUM_FEATURES, WINDOW_SIZE)
onnx_path = ONNX_DIR / "model.onnx"
torch.onnx.export(
    model, dummy,
    str(onnx_path),
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
)
print(f"\nONNX model saved to {onnx_path}")

torch.save(model.state_dict(), ONNX_DIR / "model.pt")
print("Done!")
