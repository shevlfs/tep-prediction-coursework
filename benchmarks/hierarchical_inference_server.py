#!/usr/bin/env python3
"""Cloud fallback inference server for hierarchical edge-cloud pipeline.

Runs on x86 GPU server. Accepts uncertain predictions from RK3568 boards
and runs full PyTorch ensemble inference.

Usage (on x86):
    python3 hierarchical_inference_server.py --model-dir ./onnx --port 8080
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

WINDOW_SIZE = 32
NUM_FEATURES = 52
NUM_CLASSES = 21

FAULT_LABELS = {i: f"Fault {i}" if i > 0 else "Normal" for i in range(NUM_CLASSES)}


def load_pytorch_model(model_name: str, onnx_dir: Path):
    """Load a model from ONNX using onnxruntime for x86 inference."""
    import onnxruntime as ort

    onnx_path = onnx_dir / model_name / "model.onnx"
    if not onnx_path.exists():
        return None, None, None

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    mean = np.load(onnx_dir / model_name / "norm_mean.npy")
    std = np.load(onnx_dir / model_name / "norm_std.npy")

    return session, mean, std


def run_onnx_inference(session, x_norm: np.ndarray):
    """Run ONNX inference and return logits."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x_norm.astype(np.float32)})
    return outputs[0][0]


class InferencePool:
    """Manages multiple ONNX models for ensemble inference."""

    def __init__(self, onnx_dir: Path, model_names: list[str]):
        self.models = {}
        self.norms = {}
        for name in model_names:
            session, mean, std = load_pytorch_model(name, onnx_dir)
            if session is not None:
                self.models[name] = session
                self.norms[name] = (mean, std)
                print(f"  Loaded: {name}")
            else:
                print(f"  [SKIP] {name}: ONNX not found")

    def infer_single(self, model_name: str, x_raw: np.ndarray):
        """Run single model inference. Returns logits and latency_ms."""
        mean, std = self.norms[model_name]
        x_norm = (x_raw - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]
        t0 = time.perf_counter()
        logits = run_onnx_inference(self.models[model_name], x_norm)
        lat = (time.perf_counter() - t0) * 1000
        return logits, lat

    def infer_ensemble(self, x_raw: np.ndarray, weights: dict[str, float] | None = None):
        """Run all models and return weighted average prediction."""
        all_probs = []
        all_weights = []
        total_lat = 0.0
        model_results = {}

        for name, session in self.models.items():
            logits, lat = self.infer_single(name, x_raw)
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            w = weights.get(name, 1.0) if weights else 1.0

            all_probs.append(probs * w)
            all_weights.append(w)
            total_lat += lat

            model_results[name] = {
                "prediction": int(np.argmax(probs)),
                "confidence": float(np.max(probs)),
                "latency_ms": round(lat, 3),
            }

        combined = np.sum(all_probs, axis=0) / sum(all_weights)
        pred = int(np.argmax(combined))
        conf = float(np.max(combined))

        return {
            "prediction": pred,
            "label": FAULT_LABELS[pred],
            "confidence": round(conf, 4),
            "total_latency_ms": round(total_lat, 3),
            "models_used": len(self.models),
            "model_results": model_results,
        }


# Global pool reference
_pool: InferencePool | None = None

MODEL_WEIGHTS = {
    "tepnet": 0.7242,
    "tcn": 0.7818,
    "lstm": 0.7045,
    "transformer": 0.7758,
    "patchtst": 0.8299,
}


class InferenceHandler(BaseHTTPRequestHandler):
    """HTTP handler for inference requests from edge devices."""

    def do_POST(self):
        if self.path == "/infer":
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body)
                x_raw = np.array(data["input"], dtype=np.float32)

                if x_raw.shape != (1, NUM_FEATURES, WINDOW_SIZE):
                    x_raw = x_raw.reshape(1, NUM_FEATURES, WINDOW_SIZE)

                model_name = data.get("model", None)
                if model_name and model_name in _pool.models:
                    logits, lat = _pool.infer_single(model_name, x_raw)
                    probs = np.exp(logits - np.max(logits))
                    probs = probs / probs.sum()
                    result = {
                        "prediction": int(np.argmax(probs)),
                        "label": FAULT_LABELS[int(np.argmax(probs))],
                        "confidence": round(float(np.max(probs)), 4),
                        "latency_ms": round(lat, 3),
                        "model": model_name,
                    }
                else:
                    result = _pool.infer_ensemble(x_raw, weights=MODEL_WEIGHTS)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif self.path == "/health":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            result = {"status": "ok", "models": list(_pool.models.keys())}
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            result = {"status": "ok", "models": list(_pool.models.keys())}
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default logging


def main():
    global _pool

    parser = argparse.ArgumentParser(description="Cloud inference server")
    parser.add_argument("--onnx-dir", type=Path, default=Path("onnx"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--models", nargs="+",
                        default=["tepnet", "tcn", "lstm", "transformer", "patchtst"])
    args = parser.parse_args()

    print(f"Loading models from {args.onnx_dir}...")
    _pool = InferencePool(args.onnx_dir, args.models)
    print(f"\nLoaded {len(_pool.models)} models. Starting server on :{args.port}")

    server = HTTPServer(("0.0.0.0", args.port), InferenceHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
