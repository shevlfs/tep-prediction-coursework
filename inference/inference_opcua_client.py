from __future__ import annotations

import argparse
import sys
import time
import signal
import os
import numpy as np
from collections import deque
from pathlib import Path
from rknnlite.api import RKNNLite
from opcua import Client as OPCUAClient

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

AVAILABLE_MODELS = {
    "tepnet":      {"rknn_fp16": "tepnet/model_fp16.rknn",      "rknn_int8": "tepnet/model_int8.rknn",      "norm": "tepnet"},
    "tcn":         {"rknn_fp16": "tcn/model_fp16.rknn",         "rknn_int8": "tcn/model_int8.rknn",         "norm": "tcn"},
    "lstm":        {"rknn_fp16": "lstm/model_fp16.rknn",        "rknn_int8": "lstm/model_int8.rknn",        "norm": "lstm"},
    "transformer": {"rknn_fp16": "transformer/model_fp16.rknn", "rknn_int8": "transformer/model_int8.rknn", "norm": "transformer"},
    "patchtst":    {"rknn_fp16": "patchtst/model_fp16.rknn",    "rknn_int8": "patchtst/model_int8.rknn",    "norm": "patchtst"},
}

WINDOW_SIZE = 32
NUM_FEATURES = 52
POLL_INTERVAL_S = float(os.environ.get("POLL_INTERVAL", "5.0"))
NAMESPACE_URI = "urn:my-csv-server"
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8000"))

XMEAS_TAGS = [f"xmeas_{i}" for i in range(1, 42)]
XMV_TAGS = [f"xmv_{i}" for i in range(1, 12)]
ALL_TAGS = XMEAS_TAGS + XMV_TAGS

FAULT_LABELS = {
    0: "Normal",
    1: "Fault 1: A/C feed ratio step",
    2: "Fault 2: B composition step",
    3: "Fault 3: D feed temp step",
    4: "Fault 4: Reactor cooling water inlet temp step",
    5: "Fault 5: Condenser cooling water inlet temp step",
    6: "Fault 6: A feed loss step",
    7: "Fault 7: C header pressure loss step",
    8: "Fault 8: A,B,C feed composition random",
    9: "Fault 9: D feed temp random",
    10: "Fault 10: C feed temp random",
    11: "Fault 11: Reactor cooling water inlet temp random",
    12: "Fault 12: Condenser cooling water inlet temp random",
    13: "Fault 13: Reaction kinetics slow drift",
    14: "Fault 14: Reactor cooling water valve sticking",
    15: "Fault 15: Condenser cooling water valve sticking",
    16: "Fault 16: Unknown",
    17: "Fault 17: Unknown",
    18: "Fault 18: Unknown",
    19: "Fault 19: Unknown",
    20: "Fault 20: Unknown",
}

running = True

if HAS_PROMETHEUS:
    INFERENCE_TOTAL = Counter("tep_inference_total", "Total inference count", ["predicted_class"])
    INFERENCE_LATENCY = Histogram("tep_inference_latency_seconds", "Inference latency in seconds",
                                  buckets=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])
    PREDICTION_CONFIDENCE = Gauge("tep_prediction_confidence", "Confidence of the latest prediction")
    CURRENT_PREDICTION = Gauge("tep_current_prediction", "Current predicted fault class")
    BUFFER_SIZE = Gauge("tep_buffer_size", "Current sliding window buffer size")
    OPCUA_READ_ERRORS = Counter("tep_opcua_read_errors_total", "Total OPC UA read errors")
    MODEL_INFO = Gauge("tep_model_info", "Currently loaded model", ["model_name", "quantization"])


def stop_handler(signum, frame):
    global running
    running = False


def parse_args():
    parser = argparse.ArgumentParser(description="TEP Inference OPC UA Client")
    parser.add_argument("server_url", nargs="?", default="opc.tcp://192.168.88.243:4840",
                        help="OPC UA server URL")
    parser.add_argument("--model", choices=list(AVAILABLE_MODELS.keys()),
                        default=os.environ.get("MODEL", "patchtst"),
                        help="Model architecture (default: patchtst, env: MODEL)")
    parser.add_argument("--quantization", choices=["fp16", "int8"],
                        default=os.environ.get("QUANTIZATION", "fp16"),
                        help="Quantization type (default: fp16, env: QUANTIZATION)")
    parser.add_argument("--rknn-dir", type=Path,
                        default=Path(os.environ.get("RKNN_DIR", "/app/rknn")),
                        help="Directory containing model subdirectories")
    parser.add_argument("--norm-dir", type=Path,
                        default=Path(os.environ.get("NORM_DIR", "/app/onnx")),
                        help="Directory containing normalization param subdirectories")
    parser.add_argument("--poll-interval", type=float, default=POLL_INTERVAL_S,
                        help="Polling interval in seconds")
    return parser.parse_args()


def resolve_model_paths(args):
    model_cfg = AVAILABLE_MODELS[args.model]
    quant_key = f"rknn_{args.quantization}"

    rknn_path = args.rknn_dir / model_cfg[quant_key]
    norm_mean = args.norm_dir / model_cfg["norm"] / "norm_mean.npy"
    norm_std = args.norm_dir / model_cfg["norm"] / "norm_std.npy"

    if not rknn_path.exists():
        legacy = Path("/app/model_fp.rknn")
        if legacy.exists():
            print(f"[WARN] Model {rknn_path} not found, falling back to {legacy}")
            rknn_path = legacy
    if not norm_mean.exists():
        legacy_mean = Path("/app/norm_mean.npy")
        if legacy_mean.exists():
            norm_mean = legacy_mean
            norm_std = Path("/app/norm_std.npy")

    return rknn_path, norm_mean, norm_std


def init_rknn(model_path: str) -> RKNNLite:
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"Failed to load RKNN model: {ret}")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
    print(f"[RKNN] Model loaded: {model_path}")
    return rknn


def run_inference(rknn: RKNNLite, window: np.ndarray,
                  mean: np.ndarray, std: np.ndarray) -> tuple[int, np.ndarray]:
    normalized = (window - mean[:, np.newaxis]) / std[:, np.newaxis]
    input_data = np.expand_dims(normalized, axis=0).astype(np.float32)
    outputs = rknn.inference(inputs=[input_data])
    logits = outputs[0][0]
    return int(np.argmax(logits)), logits


def read_all_tags(opcua_client: OPCUAClient, ns_idx: int) -> list[float]:
    row = []
    for tag in ALL_TAGS:
        node = opcua_client.get_node(f"ns={ns_idx};s={tag}")
        val = node.get_value()
        row.append(float(val))
    return row


def main():
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    args = parse_args()

    if HAS_PROMETHEUS:
        start_http_server(METRICS_PORT)
        print(f"[METRICS] Prometheus metrics available on :{METRICS_PORT}/metrics")

    rknn_path, norm_mean_path, norm_std_path = resolve_model_paths(args)

    print(f"[CONFIG] Model: {args.model} ({args.quantization})")
    print(f"[CONFIG] RKNN:  {rknn_path}")
    print(f"[CONFIG] Norm:  {norm_mean_path.parent}")

    print("[INIT] Loading normalization params...")
    mean = np.load(str(norm_mean_path))
    std = np.load(str(norm_std_path))
    print(f"[INIT] mean shape={mean.shape}, std shape={std.shape}")

    print("[INIT] Loading RKNN model...")
    rknn = init_rknn(str(rknn_path))

    if HAS_PROMETHEUS:
        MODEL_INFO.labels(model_name=args.model, quantization=args.quantization).set(1)

    print(f"[INIT] Connecting to OPC UA at {args.server_url}...")
    opcua_client = OPCUAClient(args.server_url)
    opcua_client.connect()
    ns_idx = opcua_client.get_namespace_index(NAMESPACE_URI)
    print(f"[OPC UA] Connected to {args.server_url} (ns={ns_idx})")
    print(f"[OPC UA] Reading {len(ALL_TAGS)} tags every {args.poll_interval}s")
    print(f"[RKNN]   Window size: {WINDOW_SIZE}, will start inference after {WINDOW_SIZE} samples")
    print("=" * 70)

    buffer = deque(maxlen=WINDOW_SIZE)
    cycle = 0

    try:
        while running:
            t0 = time.time()

            try:
                row = read_all_tags(opcua_client, ns_idx)
            except Exception as e:
                print(f"[WARN] OPC UA read error: {e}")
                if HAS_PROMETHEUS:
                    OPCUA_READ_ERRORS.inc()
                time.sleep(args.poll_interval)
                continue

            buffer.append(row)
            cycle += 1

            if HAS_PROMETHEUS:
                BUFFER_SIZE.set(len(buffer))

            if len(buffer) < WINDOW_SIZE:
                print(f"[{time.strftime('%H:%M:%S')}] Buffering {len(buffer)}/{WINDOW_SIZE} "
                      f"| xmeas_1={row[0]:.4f}")
            else:
                window = np.array(buffer, dtype=np.float32).T

                inf_start = time.time()
                pred, logits = run_inference(rknn, window, mean, std)
                inf_duration = time.time() - inf_start

                label = FAULT_LABELS.get(pred, f"Class {pred}")
                confidence = np.exp(logits[pred]) / np.sum(np.exp(logits))
                elapsed_ms = (time.time() - t0) * 1000

                if HAS_PROMETHEUS:
                    INFERENCE_TOTAL.labels(predicted_class=str(pred)).inc()
                    INFERENCE_LATENCY.observe(inf_duration)
                    PREDICTION_CONFIDENCE.set(float(confidence))
                    CURRENT_PREDICTION.set(pred)

                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"Cycle {cycle:5d} | "
                      f"Pred: {pred:2d} ({label}) | "
                      f"Conf: {confidence:.2%} | "
                      f"xmeas_1={row[0]:.3f} xmeas_2={row[1]:.1f} | "
                      f"{elapsed_ms:.0f}ms")

            elapsed = time.time() - t0
            sleep_time = max(0, args.poll_interval - elapsed)
            if sleep_time > 0 and running:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        opcua_client.disconnect()
        rknn.release()
        print("Done.")


if __name__ == "__main__":
    main()
