from __future__ import annotations

import sys
import time
import signal
import os
import numpy as np
from collections import deque
from rknnlite.api import RKNNLite
from opcua import Client as OPCUAClient

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
RKNN_MODEL_PATH = "model_fp.rknn"
NORM_MEAN_PATH = "norm_mean.npy"
NORM_STD_PATH = "norm_std.npy"

WINDOW_SIZE = 32
NUM_FEATURES = 52
POLL_INTERVAL_S = 5.0
NAMESPACE_URI = "urn:my-csv-server"

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


def stop_handler(signum, frame):
    global running
    running = False


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

    server_url = sys.argv[1] if len(sys.argv) > 1 else "opc.tcp://192.168.88.243:4840"

    print("[INIT] Loading normalization params...")
    mean = np.load(NORM_MEAN_PATH)
    std = np.load(NORM_STD_PATH)
    print(f"[INIT] mean shape={mean.shape}, std shape={std.shape}")

    print("[INIT] Loading RKNN model...")
    rknn = init_rknn(RKNN_MODEL_PATH)

    print(f"[INIT] Connecting to OPC UA at {server_url}...")
    opcua_client = OPCUAClient(server_url)
    opcua_client.connect()
    ns_idx = opcua_client.get_namespace_index(NAMESPACE_URI)
    print(f"[OPC UA] Connected to {server_url} (ns={ns_idx})")
    print(f"[OPC UA] Reading {len(ALL_TAGS)} tags every {POLL_INTERVAL_S}s")
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
                time.sleep(POLL_INTERVAL_S)
                continue

            buffer.append(row)
            cycle += 1

            if len(buffer) < WINDOW_SIZE:
                print(f"[{time.strftime('%H:%M:%S')}] Buffering {len(buffer)}/{WINDOW_SIZE} "
                      f"| xmeas_1={row[0]:.4f}")
            else:
                window = np.array(buffer, dtype=np.float32).T
                pred, logits = run_inference(rknn, window, mean, std)
                label = FAULT_LABELS.get(pred, f"Class {pred}")

                confidence = np.exp(logits[pred]) / np.sum(np.exp(logits))
                elapsed_ms = (time.time() - t0) * 1000

                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"Cycle {cycle:5d} | "
                      f"Pred: {pred:2d} ({label}) | "
                      f"Conf: {confidence:.2%} | "
                      f"xmeas_1={row[0]:.3f} xmeas_2={row[1]:.1f} | "
                      f"{elapsed_ms:.0f}ms")

            elapsed = time.time() - t0
            sleep_time = max(0, POLL_INTERVAL_S - elapsed)
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
