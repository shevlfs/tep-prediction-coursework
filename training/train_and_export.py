from __future__ import annotations

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--window-stride",
        type=int,
        default=4,
        help="Stride for sliding window"
    )
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


def make_windows(group: pd.DataFrame, feature_cols: list[str], window: int, stride: int = 1):
    X_raw = group[feature_cols].values.astype(np.float32)
    y_raw = group["target"].values.astype(np.int64)
    xs, ys = [], []
    for i in range(0, len(X_raw) - window + 1, stride):
        xs.append(X_raw[i : i + window].T)
        ys.append(y_raw[i + window - 1])
    return np.array(xs), np.array(ys)


def build_windows(df: pd.DataFrame, feature_cols: list[str], mask_val: int, stride: int = 1):
    subset = df[df["train_mask"] == mask_val]
    all_x, all_y = [], []
    for _, g in subset.groupby("run_id"):
        x, y = make_windows(g, feature_cols, WINDOW_SIZE, stride=stride)
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
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += len(xb)
    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(xb)
    return total_loss / total, correct / total


def plot_metrics(history: dict, out_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss curves")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train acc")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="Val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy curves")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Training curves saved to %s", path)


def save_metrics_csv(history: dict, out_dir: Path):
    df = pd.DataFrame(history)
    df.index = df.index + 1
    df.index.name = "epoch"
    path = out_dir / "training_metrics.csv"
    df.to_csv(path)
    log.info("Metrics saved to %s", path)


def train(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    device: torch.device,
    out_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    best_metric = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_dl, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_metric = val_loss
        improved = current_metric < (best_metric - early_stopping_min_delta)

        if improved or best_state_dict is None:
            best_metric = current_metric
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1

        gap = train_acc - val_acc
        log.info(
            "Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  train_acc=%.4f  val_acc=%.4f  gap=%.4f%s",
            epoch + 1, epochs, train_loss, val_loss, train_acc, val_acc, gap,
            "  [best]" if improved else "",
        )

        if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
            log.info(
                "Early stopping at epoch %d (best epoch=%d, best val_loss=%.4f)",
                epoch + 1,
                best_epoch,
                best_metric,
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        best_path = out_dir / "best_model.pt"
        torch.save(best_state_dict, best_path)
        log.info("Best model state_dict saved to %s", best_path)
        log.info(
            "Restored best checkpoint from epoch %d (val_loss=%.4f)",
            best_epoch,
            best_metric,
        )

    save_metrics_csv(history, out_dir)
    plot_metrics(history, out_dir)
    return history


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
    X_train, y_train = build_windows(df, feature_cols, mask_val=1, stride=args.window_stride)
    X_test, y_test = build_windows(df, feature_cols, mask_val=0, stride=args.window_stride)
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
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    train(
        model,
        train_dl,
        val_dl,
        device,
        out_dir,
        args.epochs,
        args.lr,
    )
    export(model, out_dir)
    log.info("Training complete")


if __name__ == "__main__":
    main()
