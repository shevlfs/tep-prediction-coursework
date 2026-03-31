#!/usr/bin/env python3
"""Evaluate accuracy of RKNN models vs PyTorch baseline.

Usage (on RK3568):
    python3 accuracy_eval.py --rknn-dir ./rknn --data-dir ./small_tep --onnx-dir ./onnx

Compares per-class F1, overall accuracy, and confusion matrices.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from rknnlite.api import RKNNLite


WINDOW_SIZE = 32
NUM_FEATURES = 52

FAULT_LABELS = {
    0: "Normal", 1: "Fault 1", 2: "Fault 2", 3: "Fault 3",
    4: "Fault 4", 5: "Fault 5", 6: "Fault 6", 7: "Fault 7",
    8: "Fault 8", 9: "Fault 9", 10: "Fault 10", 11: "Fault 11",
    12: "Fault 12", 13: "Fault 13", 14: "Fault 14", 15: "Fault 15",
    16: "Fault 16", 17: "Fault 17", 18: "Fault 18", 19: "Fault 19",
    20: "Fault 20",
}


def load_test_data(data_dir: Path):
    df = pd.read_csv(data_dir / "df.csv")
    target = pd.read_csv(data_dir / "target.csv")
    train_mask = pd.read_csv(data_dir / "train_mask.csv")

    feature_cols = [c for c in df.columns if c not in ("run_id", "sample")]
    df["target"] = target["target"].values
    df["train_mask"] = train_mask["train_mask"].values

    test_df = df[df["train_mask"] == 0]

    all_x, all_y = [], []
    for _, g in test_df.groupby("run_id"):
        X_raw = g[feature_cols].values.astype(np.float32)
        y_raw = g["target"].values.astype(np.int64)
        for i in range(len(X_raw) - WINDOW_SIZE + 1):
            all_x.append(X_raw[i : i + WINDOW_SIZE].T)
            all_y.append(y_raw[i + WINDOW_SIZE - 1])

    return np.array(all_x), np.array(all_y)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 21):
    accuracy = float((y_true == y_pred).mean())

    per_class = {}
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int((y_true == c).sum())

        per_class[FAULT_LABELS.get(c, f"Class {c}")] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    return {
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_rknn(model_path: str, X_test: np.ndarray, y_test: np.ndarray,
                  mean: np.ndarray, std: np.ndarray) -> dict:
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        return {"error": f"Failed to load: {ret}"}

    ret = rknn.init_runtime()
    if ret != 0:
        rknn.release()
        return {"error": f"Failed to init: {ret}"}

    preds = []
    for i in range(len(X_test)):
        x = X_test[i:i+1].copy()
        x = (x - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]
        outputs = rknn.inference(inputs=[x.astype(np.float32)])
        preds.append(int(np.argmax(outputs[0][0])))

    rknn.release()

    y_pred = np.array(preds)
    return compute_metrics(y_test, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RKNN model accuracy")
    parser.add_argument("--rknn-dir", type=Path, default=Path("rknn"))
    parser.add_argument("--data-dir", type=Path, default=Path("small_tep"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("onnx"))
    parser.add_argument("--output", type=Path, default=Path("accuracy_results.json"))
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit test samples for faster evaluation")
    args = parser.parse_args()

    print("Loading test data...")
    X_test, y_test = load_test_data(args.data_dir)
    print(f"Test set: {X_test.shape}, {len(np.unique(y_test))} classes")

    if args.max_samples and args.max_samples < len(X_test):
        idx = np.random.RandomState(42).choice(len(X_test), args.max_samples, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]
        print(f"Subsampled to {len(X_test)} samples")

    results = {}

    for model_dir in sorted(args.rknn_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        norm_dir = args.onnx_dir / model_name

        mean_path = norm_dir / "norm_mean.npy"
        std_path = norm_dir / "norm_std.npy"
        if not mean_path.exists() or not std_path.exists():
            print(f"[SKIP] {model_name}: normalization params not found in {norm_dir}")
            continue

        mean = np.load(mean_path)
        std = np.load(std_path)

        for rknn_file in sorted(model_dir.glob("*.rknn")):
            quant = rknn_file.stem.replace("model_", "")
            key = f"{model_name}_{quant}"
            print(f"\nEvaluating: {key} ({rknn_file})")

            result = evaluate_rknn(str(rknn_file), X_test, y_test, mean, std)
            results[key] = {
                "model": model_name,
                "quantization": quant,
                "path": str(rknn_file),
                **result,
            }

            if "accuracy" in result:
                print(f"  Accuracy: {result['accuracy']:.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print("\n## Accuracy Summary\n")
    print("| Model | Quantization | Accuracy | Macro F1 |")
    print("|-------|-------------|----------|----------|")
    for key, r in results.items():
        if "error" in r:
            print(f"| {r['model']} | {r['quantization']} | ERROR | ERROR |")
            continue
        f1s = [v["f1"] for v in r["per_class"].values() if v["support"] > 0]
        macro_f1 = np.mean(f1s) if f1s else 0
        print(f"| {r['model']} | {r['quantization']} | {r['accuracy']:.4f} | {macro_f1:.4f} |")


if __name__ == "__main__":
    main()
