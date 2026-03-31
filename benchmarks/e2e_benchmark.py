#!/usr/bin/env python3
"""End-to-end benchmark: OPC UA read → normalize → RKNN inference → prediction.
Measures the complete pipeline latency for each model.
"""
import sys
import time
import json
import numpy as np
from pathlib import Path
from rknnlite.api import RKNNLite
from opcua import Client as OPCUAClient

WINDOW_SIZE = 32
NUM_FEATURES = 52
NAMESPACE_URI = "urn:my-csv-server"

XMEAS_TAGS = [f"xmeas_{i}" for i in range(1, 42)]
XMV_TAGS = [f"xmv_{i}" for i in range(1, 12)]
ALL_TAGS = XMEAS_TAGS + XMV_TAGS

FAULT_LABELS = {
    0: "Normal", 1: "Fault 1", 2: "Fault 2", 3: "Fault 3",
    4: "Fault 4", 5: "Fault 5", 6: "Fault 6", 7: "Fault 7",
    8: "Fault 8", 9: "Fault 9", 10: "Fault 10", 11: "Fault 11",
    12: "Fault 12", 13: "Fault 13", 14: "Fault 14", 15: "Fault 15",
    16: "Fault 16", 17: "Fault 17", 18: "Fault 18", 19: "Fault 19",
    20: "Fault 20",
}

MODELS = {
    "tepnet_fp16": {"rknn": "rknn/tepnet/model_fp16.rknn", "norm_dir": "onnx/tepnet"},
    "tcn_fp16": {"rknn": "rknn/tcn/model_fp16.rknn", "norm_dir": "onnx/tcn"},
    "lstm_fp16": {"rknn": "rknn/lstm/model_fp16.rknn", "norm_dir": "onnx/lstm"},
    "transformer_fp16": {"rknn": "rknn/transformer/model_fp16.rknn", "norm_dir": "onnx/transformer"},
    "patchtst_fp16": {"rknn": "rknn/patchtst/model_fp16.rknn", "norm_dir": "onnx/patchtst"},
}


def read_all_tags(client, ns_idx):
    row = []
    for tag in ALL_TAGS:
        node = client.get_node(f"ns={ns_idx};s={tag}")
        val = node.get_value()
        row.append(float(val))
    return row


def benchmark_model(model_name, model_cfg, opcua_client, ns_idx, num_cycles=50):
    """Run num_cycles of full pipeline and measure latency."""
    rknn_path = Path.home() / model_cfg["rknn"]
    norm_dir = Path.home() / model_cfg["norm_dir"]

    if not rknn_path.exists():
        return {"error": f"Model not found: {rknn_path}"}

    mean = np.load(norm_dir / "norm_mean.npy")
    std = np.load(norm_dir / "norm_std.npy")

    rknn = RKNNLite()
    ret = rknn.load_rknn(str(rknn_path))
    if ret != 0:
        return {"error": f"Failed to load: {ret}"}
    ret = rknn.init_runtime()
    if ret != 0:
        rknn.release()
        return {"error": f"Failed to init: {ret}"}

    print(f"  [{model_name}] Filling buffer ({WINDOW_SIZE} samples)...")
    buffer = []
    for i in range(WINDOW_SIZE):
        row = read_all_tags(opcua_client, ns_idx)
        buffer.append(row)
        if i < WINDOW_SIZE - 1:
            time.sleep(0.1)

    print(f"  [{model_name}] Running {num_cycles} inference cycles...")
    latencies = []
    predictions = []
    confidences = []

    for cycle in range(num_cycles):
        t0 = time.perf_counter()

        row = read_all_tags(opcua_client, ns_idx)

        buffer.pop(0)
        buffer.append(row)

        window = np.array(buffer, dtype=np.float32).T
        normalized = (window - mean[:, np.newaxis]) / std[:, np.newaxis]
        input_data = np.expand_dims(normalized, axis=0).astype(np.float32)

        outputs = rknn.inference(inputs=[input_data])
        logits = outputs[0][0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        predictions.append(pred)
        confidences.append(conf)

    rknn.release()

    lat = np.array(latencies)
    return {
        "mean_ms": round(float(lat.mean()), 2),
        "std_ms": round(float(lat.std()), 2),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "min_ms": round(float(lat.min()), 2),
        "max_ms": round(float(lat.max()), 2),
        "cycles": num_cycles,
        "avg_confidence": round(float(np.mean(confidences)), 4),
        "most_common_prediction": int(np.bincount(predictions).argmax()),
        "most_common_label": FAULT_LABELS.get(int(np.bincount(predictions).argmax()), "Unknown"),
    }


def main():
    server_url = sys.argv[1] if len(sys.argv) > 1 else "opc.tcp://localhost:4840"
    num_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    print(f"Connecting to OPC UA server: {server_url}")
    client = OPCUAClient(server_url)
    client.connect()
    ns_idx = client.get_namespace_index(NAMESPACE_URI)
    print(f"Connected (ns={ns_idx})")

    results = {}

    for model_name, model_cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"E2E Benchmark: {model_name}")
        print(f"{'='*60}")

        result = benchmark_model(model_name, model_cfg, client, ns_idx, num_cycles)
        results[model_name] = result

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Mean: {result['mean_ms']}ms | P50: {result['p50_ms']}ms | P95: {result['p95_ms']}ms")
            print(f"  Prediction: {result['most_common_label']} | Avg confidence: {result['avg_confidence']:.2%}")

    client.disconnect()

    out_path = Path.home() / "e2e_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n## End-to-End Pipeline Benchmark\n")
    print("| Model | E2E Mean (ms) | E2E P50 (ms) | E2E P95 (ms) | Prediction | Confidence |")
    print("|-------|--------------|-------------|-------------|------------|------------|")
    for name, r in results.items():
        if "error" in r:
            print(f"| {name} | ERROR | | | | |")
        else:
            print(f"| {name} | {r['mean_ms']} | {r['p50_ms']} | {r['p95_ms']} | {r['most_common_label']} | {r['avg_confidence']:.2%} |")


if __name__ == "__main__":
    main()
