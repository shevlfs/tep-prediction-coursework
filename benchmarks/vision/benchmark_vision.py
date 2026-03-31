#!/usr/bin/env python3
"""Benchmark vision RKNN models (MobileNetV2, ResNet18) on RK3568.

Usage (on RK3568):
    python3 benchmark_vision.py --rknn-dir ./vision_rknn --output vision_benchmark_results.json
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
from pathlib import Path
from rknnlite.api import RKNNLite


WARMUP = 100
ITERATIONS = 1000
INPUT_SHAPE = (1, 3, 224, 224)


def benchmark_model(model_path: str, core_mask: int, warmup: int, iterations: int) -> dict:
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        return {"error": f"Failed to load model: {ret}"}

    ret = rknn.init_runtime()
    if ret != 0:
        rknn.release()
        return {"error": f"Failed to init runtime: {ret}"}

    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)

    for _ in range(warmup):
        rknn.inference(inputs=[dummy])

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        rknn.inference(inputs=[dummy])
        latencies.append((time.perf_counter() - t0) * 1000)

    rknn.release()

    lat = np.array(latencies)
    throughput = 1000.0 / lat.mean() if lat.mean() > 0 else 0
    return {
        "mean_ms": round(float(lat.mean()), 3),
        "std_ms": round(float(lat.std()), 3),
        "p50_ms": round(float(np.percentile(lat, 50)), 3),
        "p95_ms": round(float(np.percentile(lat, 95)), 3),
        "p99_ms": round(float(np.percentile(lat, 99)), 3),
        "min_ms": round(float(lat.min()), 3),
        "max_ms": round(float(lat.max()), 3),
        "throughput_fps": round(throughput, 2),
        "iterations": iterations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rknn-dir", type=Path, default=Path("vision_rknn"))
    parser.add_argument("--output", type=Path, default=Path("vision_benchmark_results.json"))
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()

    results = {}

    for model_dir in sorted(args.rknn_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "calibration":
            continue

        for rknn_file in sorted(model_dir.glob("*.rknn")):
            model_name = model_dir.name
            quant = rknn_file.stem.replace("model_", "")
            key = f"{model_name}_{quant}"

            print(f"\n{'='*60}")
            print(f"Benchmarking: {model_name} ({quant})")
            print(f"{'='*60}")

            print(f"  [NPU] Running...")
            npu = benchmark_model(str(rknn_file), RKNNLite.NPU_CORE_0, args.warmup, args.iterations)
            if "mean_ms" in npu:
                print(f"  [NPU] mean={npu['mean_ms']}ms, throughput={npu['throughput_fps']} FPS")

            results[key] = {
                "model": model_name,
                "quantization": quant,
                "npu": npu,
            }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print("\n## Vision Model Benchmark Results\n")
    print("| Model | Quant | Mean (ms) | P50 (ms) | P95 (ms) | Throughput (FPS) |")
    print("|-------|-------|-----------|----------|----------|-----------------|")
    for key, r in results.items():
        npu = r.get("npu", {})
        print(f"| {r['model']} | {r['quantization']} | "
              f"{npu.get('mean_ms', 'N/A')} | {npu.get('p50_ms', 'N/A')} | "
              f"{npu.get('p95_ms', 'N/A')} | {npu.get('throughput_fps', 'N/A')} |")


if __name__ == "__main__":
    main()
