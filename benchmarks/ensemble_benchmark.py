#!/usr/bin/env python3
"""Benchmark ensemble inference strategies on RK3568 NPU.

Tests multiple ensemble configurations:
1. Soft voting (average probabilities) across model combinations
2. Weighted voting (weight by historical accuracy)
3. Cascade: fast model first, escalate to accurate model if confidence low

Usage (on RK3568):
    python3 ensemble_benchmark.py --rknn-dir ./rknn --data-dir ./small_tep --onnx-dir ./onnx
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from rknnlite.api import RKNNLite

WINDOW_SIZE = 32
NUM_FEATURES = 52
NUM_CLASSES = 21

FAULT_LABELS = {i: f"Fault {i}" if i > 0 else "Normal" for i in range(NUM_CLASSES)}

# Historical accuracy weights (from previous benchmarks, FP16)
MODEL_ACCURACY_WEIGHTS = {
    "tepnet": 0.7242,
    "tcn": 0.7818,
    "lstm": 0.7045,
    "transformer": 0.7758,
    "patchtst": 0.8299,
}

# Ensemble configurations to test
ENSEMBLE_CONFIGS = {
    "tepnet+tcn+transformer": ["tepnet", "tcn", "transformer"],
    "tcn+transformer": ["tcn", "transformer"],
    "tepnet+tcn": ["tepnet", "tcn"],
    "tepnet+tcn+transformer+patchtst": ["tepnet", "tcn", "transformer", "patchtst"],
    "all_no_lstm": ["tepnet", "tcn", "transformer", "patchtst"],
    "all_5": ["tepnet", "tcn", "lstm", "transformer", "patchtst"],
}

CASCADE_CONFIGS = {
    "tepnet->patchtst_t0.5": {"fast": "tepnet", "accurate": "patchtst", "threshold": 0.5},
    "tepnet->patchtst_t0.6": {"fast": "tepnet", "accurate": "patchtst", "threshold": 0.6},
    "tepnet->patchtst_t0.7": {"fast": "tepnet", "accurate": "patchtst", "threshold": 0.7},
    "tepnet->patchtst_t0.8": {"fast": "tepnet", "accurate": "patchtst", "threshold": 0.8},
    "tcn->patchtst_t0.6": {"fast": "tcn", "accurate": "patchtst", "threshold": 0.6},
    "tcn->patchtst_t0.7": {"fast": "tcn", "accurate": "patchtst", "threshold": 0.7},
    "tepnet->transformer_t0.7": {"fast": "tepnet", "accurate": "transformer", "threshold": 0.7},
}


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


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


def compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES):
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
        per_class[FAULT_LABELS[c]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    f1s = [v["f1"] for v in per_class.values() if v["support"] > 0]
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4), "per_class": per_class}


class ModelPool:
    """Load and manage multiple RKNN models."""

    def __init__(self, rknn_dir: Path, onnx_dir: Path, quant: str = "fp16"):
        self.models = {}
        self.norms = {}
        self.rknn_dir = rknn_dir
        self.onnx_dir = onnx_dir
        self.quant = quant

    def load(self, model_name: str):
        if model_name in self.models:
            return
        model_path = self.rknn_dir / model_name / f"model_{self.quant}.rknn"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        rknn = RKNNLite()
        ret = rknn.load_rknn(str(model_path))
        assert ret == 0, f"Failed to load {model_path}: {ret}"
        ret = rknn.init_runtime()
        assert ret == 0, f"Failed to init runtime for {model_name}: {ret}"
        self.models[model_name] = rknn

        norm_dir = self.onnx_dir / model_name
        mean = np.load(norm_dir / "norm_mean.npy")
        std = np.load(norm_dir / "norm_std.npy")
        self.norms[model_name] = (mean, std)

        # warmup
        dummy = np.random.randn(1, NUM_FEATURES, WINDOW_SIZE).astype(np.float32)
        for _ in range(50):
            rknn.inference(inputs=[dummy])

        print(f"  Loaded {model_name} ({self.quant})")

    def infer(self, model_name: str, x_raw: np.ndarray):
        """Run inference, returns raw logits and latency_ms."""
        mean, std = self.norms[model_name]
        x = x_raw.copy()
        x = (x - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]
        x = x.astype(np.float32)

        t0 = time.perf_counter()
        outputs = self.models[model_name].inference(inputs=[x])
        lat = (time.perf_counter() - t0) * 1000

        return outputs[0][0], lat

    def release_all(self):
        for name, rknn in self.models.items():
            rknn.release()
        self.models.clear()


def benchmark_ensemble_soft_voting(pool, model_names, X_test, y_test, weighted=False):
    """Soft voting: average (or weighted average) of softmax probabilities."""
    all_preds = []
    total_latencies = []

    for i in range(len(X_test)):
        x = X_test[i:i+1]
        combined_probs = np.zeros(NUM_CLASSES)
        total_lat = 0.0

        for name in model_names:
            logits, lat = pool.infer(name, x)
            probs = softmax(logits)
            weight = MODEL_ACCURACY_WEIGHTS.get(name, 1.0) if weighted else 1.0
            combined_probs += probs * weight
            total_lat += lat

        all_preds.append(int(np.argmax(combined_probs)))
        total_latencies.append(total_lat)

    y_pred = np.array(all_preds)
    lat_arr = np.array(total_latencies)
    metrics = compute_metrics(y_test, y_pred)
    metrics["latency"] = {
        "mean_ms": round(float(lat_arr.mean()), 3),
        "p50_ms": round(float(np.percentile(lat_arr, 50)), 3),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 3),
        "p99_ms": round(float(np.percentile(lat_arr, 99)), 3),
        "min_ms": round(float(lat_arr.min()), 3),
        "max_ms": round(float(lat_arr.max()), 3),
    }
    return metrics


def benchmark_cascade(pool, fast_name, accurate_name, threshold, X_test, y_test):
    """Cascade: use fast model, escalate to accurate model if confidence < threshold."""
    all_preds = []
    total_latencies = []
    escalation_count = 0

    for i in range(len(X_test)):
        x = X_test[i:i+1]

        logits_fast, lat_fast = pool.infer(fast_name, x)
        probs_fast = softmax(logits_fast)
        confidence = float(np.max(probs_fast))

        if confidence >= threshold:
            all_preds.append(int(np.argmax(probs_fast)))
            total_latencies.append(lat_fast)
        else:
            logits_acc, lat_acc = pool.infer(accurate_name, x)
            probs_acc = softmax(logits_acc)
            all_preds.append(int(np.argmax(probs_acc)))
            total_latencies.append(lat_fast + lat_acc)
            escalation_count += 1

    y_pred = np.array(all_preds)
    lat_arr = np.array(total_latencies)
    metrics = compute_metrics(y_test, y_pred)
    metrics["latency"] = {
        "mean_ms": round(float(lat_arr.mean()), 3),
        "p50_ms": round(float(np.percentile(lat_arr, 50)), 3),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 3),
        "p99_ms": round(float(np.percentile(lat_arr, 99)), 3),
        "min_ms": round(float(lat_arr.min()), 3),
        "max_ms": round(float(lat_arr.max()), 3),
    }
    metrics["escalation_rate"] = round(escalation_count / len(X_test), 4)
    metrics["escalation_count"] = escalation_count
    metrics["total_samples"] = len(X_test)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Ensemble benchmark on RK3568 NPU")
    parser.add_argument("--rknn-dir", type=Path, default=Path("rknn"))
    parser.add_argument("--data-dir", type=Path, default=Path("small_tep"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("onnx"))
    parser.add_argument("--output", type=Path, default=Path("results/ensemble_benchmark_results.json"))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    print("Loading test data...")
    X_test, y_test = load_test_data(args.data_dir)
    print(f"Test set: {X_test.shape[0]} samples, {len(np.unique(y_test))} classes")

    if args.max_samples and args.max_samples < len(X_test):
        idx = np.random.RandomState(42).choice(len(X_test), args.max_samples, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]
        print(f"Subsampled to {len(X_test)} samples")

    # Collect all needed models
    all_model_names = set()
    for names in ENSEMBLE_CONFIGS.values():
        all_model_names.update(names)
    for cfg in CASCADE_CONFIGS.values():
        all_model_names.add(cfg["fast"])
        all_model_names.add(cfg["accurate"])

    print(f"\nLoading {len(all_model_names)} models...")
    pool = ModelPool(args.rknn_dir, args.onnx_dir, quant="fp16")
    for name in sorted(all_model_names):
        try:
            pool.load(name)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    results = {"ensemble_soft_voting": {}, "ensemble_weighted_voting": {}, "cascade": {}, "single_model_baselines": {}}

    # Single model baselines
    print("\n" + "="*60)
    print("SINGLE MODEL BASELINES")
    print("="*60)
    for name in sorted(pool.models.keys()):
        print(f"\n  Evaluating: {name}")
        preds, lats = [], []
        for i in range(len(X_test)):
            logits, lat = pool.infer(name, X_test[i:i+1])
            preds.append(int(np.argmax(softmax(logits))))
            lats.append(lat)
        y_pred = np.array(preds)
        lat_arr = np.array(lats)
        metrics = compute_metrics(y_test, y_pred)
        metrics["latency"] = {
            "mean_ms": round(float(lat_arr.mean()), 3),
            "p50_ms": round(float(np.percentile(lat_arr, 50)), 3),
            "p95_ms": round(float(np.percentile(lat_arr, 95)), 3),
        }
        results["single_model_baselines"][name] = metrics
        print(f"    Accuracy: {metrics['accuracy']:.4f}, Latency: {metrics['latency']['mean_ms']:.1f}ms")

    # Ensemble soft voting
    print("\n" + "="*60)
    print("ENSEMBLE SOFT VOTING")
    print("="*60)
    for config_name, model_names in ENSEMBLE_CONFIGS.items():
        if not all(n in pool.models for n in model_names):
            print(f"\n  [SKIP] {config_name}: missing models")
            continue
        print(f"\n  Config: {config_name}")
        metrics = benchmark_ensemble_soft_voting(pool, model_names, X_test, y_test, weighted=False)
        results["ensemble_soft_voting"][config_name] = metrics
        print(f"    Accuracy: {metrics['accuracy']:.4f}, Latency: {metrics['latency']['mean_ms']:.1f}ms")

    # Ensemble weighted voting
    print("\n" + "="*60)
    print("ENSEMBLE WEIGHTED VOTING")
    print("="*60)
    for config_name, model_names in ENSEMBLE_CONFIGS.items():
        if not all(n in pool.models for n in model_names):
            continue
        print(f"\n  Config: {config_name} (weighted)")
        metrics = benchmark_ensemble_soft_voting(pool, model_names, X_test, y_test, weighted=True)
        results["ensemble_weighted_voting"][config_name] = metrics
        print(f"    Accuracy: {metrics['accuracy']:.4f}, Latency: {metrics['latency']['mean_ms']:.1f}ms")

    # Cascade benchmarks
    print("\n" + "="*60)
    print("CASCADE INFERENCE")
    print("="*60)
    for config_name, cfg in CASCADE_CONFIGS.items():
        if cfg["fast"] not in pool.models or cfg["accurate"] not in pool.models:
            continue
        print(f"\n  Config: {config_name}")
        metrics = benchmark_cascade(pool, cfg["fast"], cfg["accurate"], cfg["threshold"], X_test, y_test)
        results["cascade"][config_name] = metrics
        print(f"    Accuracy: {metrics['accuracy']:.4f}, Latency: {metrics['latency']['mean_ms']:.1f}ms, "
              f"Escalation: {metrics['escalation_rate']:.1%}")

    pool.release_all()

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Strategy':<45} {'Accuracy':>10} {'Macro F1':>10} {'Latency (ms)':>14} {'Escalation':>12}")
    print("-" * 95)
    for name, m in results["single_model_baselines"].items():
        print(f"  baseline/{name:<39} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['latency']['mean_ms']:>14.1f}")
    for name, m in results["ensemble_soft_voting"].items():
        print(f"  soft_vote/{name:<38} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['latency']['mean_ms']:>14.1f}")
    for name, m in results["ensemble_weighted_voting"].items():
        print(f"  weighted/{name:<39} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['latency']['mean_ms']:>14.1f}")
    for name, m in results["cascade"].items():
        print(f"  cascade/{name:<40} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['latency']['mean_ms']:>14.1f} {m['escalation_rate']:>11.1%}")


if __name__ == "__main__":
    main()
