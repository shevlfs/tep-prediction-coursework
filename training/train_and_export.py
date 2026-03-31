from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from models import get_model, list_models

DATA_DIR = Path("small_tep")
WINDOW_SIZE = 32
NUM_FEATURES = 52
NUM_CLASSES = 21


def parse_args():
    parser = argparse.ArgumentParser(description="Train and export TEP fault classification model")
    parser.add_argument("--model", choices=list_models(), default="tepnet",
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("onnx"),
                        help="Base output directory (model saved to {output-dir}/{model_name}/)")
    return parser.parse_args()


def load_data():
    df = pd.read_csv(DATA_DIR / "df.csv")
    target = pd.read_csv(DATA_DIR / "target.csv")
    train_mask = pd.read_csv(DATA_DIR / "train_mask.csv")

    feature_cols = [c for c in df.columns if c not in ("run_id", "sample")]
    assert len(feature_cols) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(feature_cols)}"

    df["target"] = target["target"].values
    df["train_mask"] = train_mask["train_mask"].values
    return df, feature_cols


def make_windows(group: pd.DataFrame, feature_cols: list[str], window: int):
    X_raw = group[feature_cols].values.astype(np.float32)
    y_raw = group["target"].values.astype(np.int64)
    xs, ys = [], []
    for i in range(len(X_raw) - window + 1):
        xs.append(X_raw[i : i + window].T)
        ys.append(y_raw[i + window - 1])
    return np.array(xs), np.array(ys)


def build_windows(df: pd.DataFrame, feature_cols: list[str], mask_val: int):
    subset = df[df["train_mask"] == mask_val]
    all_x, all_y = [], []
    for _, g in subset.groupby("run_id"):
        x, y = make_windows(g, feature_cols, WINDOW_SIZE)
        if len(x):
            all_x.append(x)
            all_y.append(y)
    return np.concatenate(all_x), np.concatenate(all_y)


def main():
    args = parse_args()
    model_name = args.model

    out_dir = args.output_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Training {model_name} ===")

    df, feature_cols = load_data()

    print("Building train windows...")
    X_train, y_train = build_windows(df, feature_cols, mask_val=1)
    print("Building test windows...")
    X_test, y_test = build_windows(df, feature_cols, mask_val=0)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.3f}")

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    np.save(out_dir / "norm_mean.npy", mean.squeeze())
    np.save(out_dir / "norm_std.npy", std.squeeze())

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = get_model(model_name, in_channels=NUM_FEATURES, num_classes=NUM_CLASSES).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    X_tr_t = torch.from_numpy(X_train)
    y_tr_t = torch.from_numpy(y_train)
    X_te_t = torch.from_numpy(X_test)
    y_te_t = torch.from_numpy(y_test)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
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
        print(f"Epoch {epoch+1}/{args.epochs}  loss={total_loss/len(X_train):.4f}  test_acc={acc:.4f}")

    model = model.cpu()
    model.eval()
    dummy = torch.randn(1, NUM_FEATURES, WINDOW_SIZE)
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        model, dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    print(f"\nONNX model saved to {onnx_path}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"PyTorch weights saved to {out_dir / 'model.pt'}")
    print("Done!")


if __name__ == "__main__":
    main()
