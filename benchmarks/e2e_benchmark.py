from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from opcua import Client as OPCUAClient
from rknnlite.api import RKNNLite

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

log = logging.getLogger(__name__)


def read_all_tags(client: OPCUAClient, ns_idx: int) -> list[float]:
    row = []
    for tag in ALL_TAGS:
        node = client.get_node(f"ns={ns_idx};s={tag}")
        row.append(float(node.get_value()))
    return row


def benchmark_model(
    model_name: str,
    model_cfg: dict,
    opcua_client: OPCUAClient,
    ns_idx: int,
    num_cycles: int = 50,
) -> dict:
    rknn_path = Path.home() / model_cfg["rknn"]
    norm_dir = Path.home() / model_cfg["norm_dir"]

    if not rknn_path.exists():
        return {"error": f"model not found: {rknn_path}"}

    mean = np.load(norm_dir / "norm_mean.npy")
    std = np.load(norm_dir / "norm_std.npy")

    latencies: list[float] = []
    predictions: list[int] = []
    confidences: list[float] = []

    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed with code {ret}")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed with code {ret}")

        log.info("[%s] filling buffer (%d samples)", model_name, WINDOW_SIZE)
        buffer: list[list[float]] = []
        for i in range(WINDOW_SIZE):
            buffer.append(read_all_tags(opcua_client, ns_idx))
            if i < WINDOW_SIZE - 1:
                time.sleep(0.1)

        log.info("[%s] running %d inference cycles", model_name, num_cycles)
        for _ in range(num_cycles):
            t0 = time.perf_counter()

            buffer.pop(0)
            buffer.append(read_all_tags(opcua_client, ns_idx))

            window = np.array(buffer, dtype=np.float32).T
            normalized = (window - mean[:, np.newaxis]) / std[:, np.newaxis]
            input_data = np.expand_dims(normalized, axis=0).astype(np.float32)

            logits = rknn.inference(inputs=[input_data])[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            pred = int(np.argmax(probs))

            latencies.append((time.perf_counter() - t0) * 1000)
            predictions.append(pred)
            confidences.append(float(probs[pred]))

    except Exception:
        log.exception("[%s] benchmark failed", model_name)
        return {"error": "benchmark failed"}
    finally:
        rknn.release()

    lat = np.array(latencies)
    most_common = int(np.bincount(predictions).argmax())
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
        "most_common_prediction": most_common,
        "most_common_label": FAULT_LABELS.get(most_common, "Unknown"),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="E2E pipeline benchmark via OPC UA")
    parser.add_argument("server_url", nargs="?", default="opc.tcp://localhost:4840")
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path.home() / "e2e_benchmark_results.json")
    args = parser.parse_args()

    log.info("Connecting to OPC UA server: %s", args.server_url)
    client = OPCUAClient(args.server_url)
    client.connect()
    ns_idx = client.get_namespace_index(NAMESPACE_URI)
    log.info("Connected (ns=%d)", ns_idx)

    results = {}
    try:
        for model_name, model_cfg in MODELS.items():
            log.info("E2E benchmark: %s", model_name)
            result = benchmark_model(model_name, model_cfg, client, ns_idx, args.cycles)
            results[model_name] = result

            if "error" in result:
                log.warning("%s: %s", model_name, result["error"])
            else:
                log.info(
                    "%s: mean=%.2f ms  p50=%.2f ms  p95=%.2f ms  pred=%s  conf=%.2f%%",
                    model_name, result["mean_ms"], result["p50_ms"], result["p95_ms"],
                    result["most_common_label"], result["avg_confidence"] * 100,
                )
    finally:
        client.disconnect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", args.output)

    log.info("%-25s %12s %12s %12s %-20s %10s",
             "Model", "Mean (ms)", "P50 (ms)", "P95 (ms)", "Prediction", "Confidence")
    for name, r in results.items():
        if "error" in r:
            log.warning("%-25s ERROR: %s", name, r["error"])
        else:
            log.info("%-25s %12.2f %12.2f %12.2f %-20s %9.2f%%",
                     name, r["mean_ms"], r["p50_ms"], r["p95_ms"],
                     r["most_common_label"], r["avg_confidence"] * 100)


if __name__ == "__main__":
    main()
