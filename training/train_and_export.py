from __future__ import annotations

import argparse
import logging
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

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and export TEP fault classification model")
    parser.add_argument("--model", choices=list_models(), default="tepnet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("onnx"),
        help="Base output directory (model saved to {output-dir}/{model_name}/)",
    )
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


def normalize(X_train: np.ndarray, X_test: np.ndarray, out_dir: Path):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    np.save(out_dir / "norm_mean.npy", mean.squeeze())
    np.save(out_dir / "norm_std.npy", std.squeeze())
    return (X_train - mean) / std, (X_test - mean) / std


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(X.to(device)).argmax(dim=1).cpu()
    return (preds == y).float().mean().item()


def train(
    model: nn.Module,
    train_dl: DataLoader,
    X_te: torch.Tensor,
    y_te: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        acc = evaluate(model, X_te, y_te, device)
        log.info("Epoch %d/%d  loss=%.4f  test_acc=%.4f", epoch + 1, epochs, avg_loss, acc)


def export(model: nn.Module, out_dir: Path):
    model.cpu().eval()
    dummy = torch.randn(1, NUM_FEATURES, WINDOW_SIZE)
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    log.info("ONNX model saved to %s", onnx_path)

    pt_path = out_dir / "model.pt"
    torch.save(model.state_dict(), pt_path)
    log.info("PyTorch weights saved to %s", pt_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    out_dir = args.output_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Training %s", args.model)

    df, feature_cols = load_data()
    X_train, y_train = build_windows(df, feature_cols, mask_val=1)
    X_test, y_test = build_windows(df, feature_cols, mask_val=0)
    log.info("Train: %s; Test: %s", X_train.shape, X_test.shape)

    X_train, X_test = normalize(X_train, X_test, out_dir)

    device = get_device()
    log.info("Using device: %s", device)

    model = get_model(args.model, in_channels=NUM_FEATURES, num_classes=NUM_CLASSES).to(device)
    log.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    X_te_t = torch.from_numpy(X_test)
    y_te_t = torch.from_numpy(y_test)

    train(model, train_dl, X_te_t, y_te_t, device, args.epochs, args.lr)
    export(model, out_dir)
    log.info("Training and ONNX export complete")


if __name__ == "__main__":
    main()
