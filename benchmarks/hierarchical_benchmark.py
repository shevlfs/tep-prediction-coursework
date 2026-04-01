#!/usr/bin/env python3
"""Benchmark hierarchical edge-cloud inference pipeline.

Simulates the full pipeline:
1. Edge (RK3568 NPU) runs fast model
2. If confidence < threshold, sends data to cloud (x86 GPU) for ensemble
3. Measures accuracy, latency, and escalation rates

Usage (on RK3568):
    python3 hierarchical_benchmark.py \
        --rknn-dir ./rknn --data-dir ./small_tep --onnx-dir ./onnx \
        --cloud-url http://192.168.88.243:8080 \
        --output hierarchical_benchmark_results.json
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from rknnlite.api import RKNNLite

WINDOW_SIZE = 32
NUM_FEATURES = 52
NUM_CLASSES = 21
FAULT_LABELS = {i: f"Fault {i}" if i > 0 else "Normal" for i in range(NUM_CLASSES)}


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


def compute_metrics(y_true, y_pred):
    accuracy = float((y_true == y_pred).mean())
    per_class = {}
    for c in range(NUM_CLASSES):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int((y_true == c).sum())
        per_class[FAULT_LABELS[c]] = {"precision": round(precision, 4), "recall": round(recall, 4),
                                       "f1": round(f1, 4), "support": support}
    f1s = [v["f1"] for v in per_class.values() if v["support"] > 0]
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4), "per_class": per_class}


def cloud_infer(cloud_url: str, x_raw: np.ndarray):
    """Send input to cloud server for ensemble inference."""
    payload = json.dumps({"input": x_raw.tolist()}).encode()
    req = urllib.request.Request(
        f"{cloud_url}/infer",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
    network_lat = (time.perf_counter() - t0) * 1000
    result["network_latency_ms"] = round(network_lat, 3)
    return result


def benchmark_hierarchical(rknn_dir, onnx_dir, edge_model, cloud_url, threshold,
                           X_test, y_test):
    """Run hierarchical benchmark: edge NPU + cloud fallback."""
    model_path = rknn_dir / edge_model / "model_fp16.rknn"

    rknn = RKNNLite()
    assert rknn.load_rknn(str(model_path)) == 0
    assert rknn.init_runtime() == 0

    mean = np.load(onnx_dir / edge_model / "norm_mean.npy")
    std = np.load(onnx_dir / edge_model / "norm_std.npy")

    # Warmup
    dummy = np.random.randn(1, NUM_FEATURES, WINDOW_SIZE).astype(np.float32)
    for _ in range(50):
        rknn.inference(inputs=[dummy])

    preds = []
    latencies_edge = []
    latencies_cloud = []
    latencies_total = []
    escalated = []

    for i in range(len(X_test)):
        x = X_test[i:i+1].copy()
        x_norm = (x - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]
        x_norm = x_norm.astype(np.float32)

        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[x_norm])
        edge_lat = (time.perf_counter() - t0) * 1000

        logits = outputs[0][0]
        probs = softmax(logits)
        confidence = float(np.max(probs))

        if confidence >= threshold:
            preds.append(int(np.argmax(probs)))
            latencies_edge.append(edge_lat)
            latencies_cloud.append(0.0)
            latencies_total.append(edge_lat)
            escalated.append(False)
        else:
            try:
                cloud_result = cloud_infer(cloud_url, X_test[i:i+1])
                preds.append(cloud_result["prediction"])
                cloud_lat = cloud_result["network_latency_ms"]
                latencies_cloud.append(cloud_lat)
                latencies_edge.append(edge_lat)
                latencies_total.append(edge_lat + cloud_lat)
                escalated.append(True)
            except Exception as e:
                # Cloud unavailable, use edge prediction
                preds.append(int(np.argmax(probs)))
                latencies_edge.append(edge_lat)
                latencies_cloud.append(0.0)
                latencies_total.append(edge_lat)
                escalated.append(False)

    rknn.release()

    y_pred = np.array(preds)
    metrics = compute_metrics(y_test, y_pred)

    lat_total = np.array(latencies_total)
    lat_edge = np.array(latencies_edge)
    lat_cloud = np.array([l for l, e in zip(latencies_cloud, escalated) if e])

    esc_count = sum(escalated)
    metrics["escalation_rate"] = round(esc_count / len(X_test), 4)
    metrics["escalation_count"] = esc_count
    metrics["total_samples"] = len(X_test)

    metrics["latency_total"] = {
        "mean_ms": round(float(lat_total.mean()), 3),
        "p50_ms": round(float(np.percentile(lat_total, 50)), 3),
        "p95_ms": round(float(np.percentile(lat_total, 95)), 3),
        "p99_ms": round(float(np.percentile(lat_total, 99)), 3),
    }
    metrics["latency_edge_only"] = {
        "mean_ms": round(float(lat_edge.mean()), 3),
        "p50_ms": round(float(np.percentile(lat_edge, 50)), 3),
    }
    if len(lat_cloud) > 0:
        metrics["latency_cloud_calls"] = {
            "mean_ms": round(float(lat_cloud.mean()), 3),
            "p50_ms": round(float(np.percentile(lat_cloud, 50)), 3),
            "p95_ms": round(float(np.percentile(lat_cloud, 95)), 3),
            "count": len(lat_cloud),
        }

    return metrics


def benchmark_cloud_only(cloud_url, onnx_dir, X_test, y_test):
    """Benchmark sending everything to cloud (baseline for comparison)."""
    preds = []
    latencies = []

    for i in range(len(X_test)):
        try:
            result = cloud_infer(cloud_url, X_test[i:i+1])
            preds.append(result["prediction"])
            latencies.append(result["network_latency_ms"])
        except Exception as e:
            preds.append(0)
            latencies.append(0)

    y_pred = np.array(preds)
    metrics = compute_metrics(y_test, y_pred)
    lat = np.array(latencies)
    metrics["latency"] = {
        "mean_ms": round(float(lat.mean()), 3),
        "p50_ms": round(float(np.percentile(lat, 50)), 3),
        "p95_ms": round(float(np.percentile(lat, 95)), 3),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Hierarchical inference benchmark")
    parser.add_argument("--rknn-dir", type=Path, default=Path("rknn"))
    parser.add_argument("--data-dir", type=Path, default=Path("small_tep"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("onnx"))
    parser.add_argument("--cloud-url", type=str, default="http://192.168.88.243:8080")
    parser.add_argument("--output", type=Path, default=Path("hierarchical_benchmark_results.json"))
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    print("Loading test data...")
    X_test, y_test = load_test_data(args.data_dir)
    print(f"Test set: {X_test.shape[0]} samples")

    if args.max_samples and args.max_samples < len(X_test):
        idx = np.random.RandomState(42).choice(len(X_test), args.max_samples, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]
        print(f"Subsampled to {len(X_test)} samples")

    results = {}

    # Test cloud connectivity
    print(f"\nTesting cloud at {args.cloud_url}...")
    try:
        req = urllib.request.Request(f"{args.cloud_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            health = json.loads(resp.read())
        print(f"  Cloud OK: {health}")
        cloud_available = True
    except Exception as e:
        print(f"  Cloud unavailable: {e}")
        print("  Skipping hierarchical and cloud-only benchmarks")
        cloud_available = False

    if cloud_available:
        # Cloud-only baseline
        print("\n" + "="*60)
        print("CLOUD-ONLY BASELINE (all samples sent to x86)")
        print("="*60)
        cloud_metrics = benchmark_cloud_only(args.cloud_url, args.onnx_dir, X_test, y_test)
        results["cloud_only"] = cloud_metrics
        print(f"  Accuracy: {cloud_metrics['accuracy']:.4f}, Latency: {cloud_metrics['latency']['mean_ms']:.1f}ms")

        # Hierarchical configurations
        configs = [
            ("tepnet", 0.5), ("tepnet", 0.6), ("tepnet", 0.7), ("tepnet", 0.8), ("tepnet", 0.9),
            ("tcn", 0.5), ("tcn", 0.6), ("tcn", 0.7), ("tcn", 0.8),
        ]

        print("\n" + "="*60)
        print("HIERARCHICAL INFERENCE (edge NPU + cloud fallback)")
        print("="*60)
        for edge_model, threshold in configs:
            key = f"{edge_model}_t{threshold}"
            print(f"\n  {key}: edge={edge_model}, threshold={threshold}")
            metrics = benchmark_hierarchical(
                args.rknn_dir, args.onnx_dir, edge_model,
                args.cloud_url, threshold, X_test, y_test
            )
            results[key] = metrics
            print(f"    Accuracy: {metrics['accuracy']:.4f}, "
                  f"Mean latency: {metrics['latency_total']['mean_ms']:.1f}ms, "
                  f"Escalation: {metrics['escalation_rate']:.1%}")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    if results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\n{'Config':<30} {'Accuracy':>10} {'Latency':>12} {'Escalation':>12}")
        print("-" * 68)
        for name, m in results.items():
            lat = m.get("latency_total", m.get("latency", {}))
            esc = m.get("escalation_rate", "N/A")
            esc_str = f"{esc:.1%}" if isinstance(esc, float) else esc
            print(f"  {name:<28} {m['accuracy']:>10.4f} {lat.get('mean_ms', 0):>10.1f}ms {esc_str:>12}")


if __name__ == "__main__":
    main()
