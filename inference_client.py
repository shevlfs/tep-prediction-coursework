from __future__ import annotations

import sys
import time
import numpy as np
from collections import deque
from rknnlite.api import RKNNLite

RKNN_MODEL_PATH = "model.rknn"
NORM_MEAN_PATH = "norm_mean.npy"
NORM_STD_PATH = "norm_std.npy"

WINDOW_SIZE = 32
NUM_FEATURES = 52
POLL_INTERVAL_S = 5.0

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

XMEAS_TAGS = [f"xmeas_{i}" for i in range(1, 42)]
XMV_TAGS = [f"xmv_{i}" for i in range(1, 12)]
ALL_TAGS = XMEAS_TAGS + XMV_TAGS


def init_rknn(model_path: str) -> RKNNLite:
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"Failed to load RKNN model: {ret}")

    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"Failed to init RKNN runtime: {ret}")

    print(f"RKNN model loaded: {model_path}")
    return rknn


def run_inference(rknn: RKNNLite, window: np.ndarray, mean: np.ndarray, std: np.ndarray) -> int:
    normalized = (window - mean[:, np.newaxis]) / std[:, np.newaxis]
    input_data = np.expand_dims(normalized, axis=0).astype(np.float32)
    outputs = rknn.inference(inputs=[input_data])
    logits = outputs[0][0]
    return int(np.argmax(logits))


def run_with_opcua(rknn: RKNNLite, mean: np.ndarray, std: np.ndarray,
                   server_url: str):
    from opcua import Client

    client = Client(server_url)
    client.connect()
    print(f"Connected to OPC UA server: {server_url}")

    try:
        ns_idx = client.get_namespace_index("urn:my-csv-server")
        buffer = deque(maxlen=WINDOW_SIZE)

        while True:
            row = []
            for tag in ALL_TAGS:
                node = client.get_node(f"ns={ns_idx};s={tag}")
                val = node.get_value()
                row.append(float(val))

            buffer.append(row)

            if len(buffer) == WINDOW_SIZE:
                window = np.array(buffer, dtype=np.float32).T
                pred = run_inference(rknn, window, mean, std)
                label = FAULT_LABELS.get(pred, f"Unknown ({pred})")
                print(f"[{time.strftime('%H:%M:%S')}] Prediction: {pred} - {label}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Buffering... {len(buffer)}/{WINDOW_SIZE}")

            time.sleep(POLL_INTERVAL_S)
    finally:
        client.disconnect()


def run_with_csv(rknn: RKNNLite, mean: np.ndarray, std: np.ndarray,
                 csv_path: str):
    import pandas as pd

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("run_id", "sample", "target", "train_mask")]
    data = df[feature_cols].values.astype(np.float32)

    print(f"Running inference on {csv_path} ({len(data)} rows)")
    correct = 0
    total = 0

    target = None
    try:
        target_df = pd.read_csv(csv_path.replace("df.csv", "target.csv"))
        target = target_df["target"].values
    except Exception:
        pass

    for i in range(WINDOW_SIZE - 1, min(len(data), 5000)):
        window = data[i - WINDOW_SIZE + 1: i + 1].T
        pred = run_inference(rknn, window, mean, std)

        if target is not None:
            true_label = target[i]
            if pred == true_label:
                correct += 1
            total += 1

        if i % 100 == 0:
            label = FAULT_LABELS.get(pred, f"Unknown ({pred})")
            acc_str = f" acc={correct/total:.3f}" if total > 0 else ""
            print(f"  Row {i}: pred={pred} ({label}){acc_str}")

    if total > 0:
        print(f"\nAccuracy: {correct}/{total} = {correct/total:.4f}")


def main():
    mean = np.load(NORM_MEAN_PATH)
    std = np.load(NORM_STD_PATH)

    rknn = init_rknn(RKNN_MODEL_PATH)

    if len(sys.argv) > 1 and sys.argv[1] == "--csv":
        csv_path = sys.argv[2] if len(sys.argv) > 2 else "small_tep/df.csv"
        run_with_csv(rknn, mean, std, csv_path)
    else:
        server_url = sys.argv[1] if len(sys.argv) > 1 else "opc.tcp://192.168.88.243:4840"
        run_with_opcua(rknn, mean, std, server_url)

    rknn.release()


if __name__ == "__main__":
    main()
