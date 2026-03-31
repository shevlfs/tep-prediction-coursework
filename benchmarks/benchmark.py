#!/usr/bin/env python3
"""Benchmark all RKNN models on RK3568 NPU and CPU.

Usage (on RK3568):
    python3 benchmark.py --rknn-dir ./rknn --output benchmark_results.json

Measures mean/p50/p95/p99 latency over 1000 iterations after 100 warmup runs.
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
from pathlib import Path
from rknnlite.api import RKNNLite


WINDOW_SIZE = 32
NUM_FEATURES = 52
WARMUP = 100
ITERATIONS = 1000


def benchmark_model(model_path: str, core_mask: int, warmup: int, iterations: int) -> dict:
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        return {"error": f"Failed to load model: {ret}"}

    ret = rknn.init_runtime()
    if ret != 0:
        rknn.release()
        return {"error": f"Failed to init runtime: {ret}"}

    dummy_input = np.random.randn(1, NUM_FEATURES, WINDOW_SIZE).astype(np.float32)

    for _ in range(warmup):
        rknn.inference(inputs=[dummy_input])

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        rknn.inference(inputs=[dummy_input])
        latencies.append((time.perf_counter() - t0) * 1000)

    rknn.release()

    lat = np.array(latencies)
    return {
        "mean_ms": round(float(lat.mean()), 3),
        "std_ms": round(float(lat.std()), 3),
        "p50_ms": round(float(np.percentile(lat, 50)), 3),
        "p95_ms": round(float(np.percentile(lat, 95)), 3),
        "p99_ms": round(float(np.percentile(lat, 99)), 3),
        "min_ms": round(float(lat.min()), 3),
        "max_ms": round(float(lat.max()), 3),
        "iterations": iterations,
        "warmup": warmup,
    }


def find_models(rknn_dir: Path) -> list[tuple[str, str, str]]:
    """Returns list of (model_name, quant_type, path)."""
    results = []
    for model_dir in sorted(rknn_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for rknn_file in sorted(model_dir.glob("*.rknn")):
            name = model_dir.name
            quant = rknn_file.stem.replace("model_", "")
            results.append((name, quant, str(rknn_file)))
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark RKNN models on RK3568")
    parser.add_argument("--rknn-dir", type=Path, default=Path("rknn"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--cpu-only", action="store_true", help="Only benchmark on CPU (no NPU)")
    args = parser.parse_args()

    models = find_models(args.rknn_dir)
    if not models:
        print(f"No RKNN models found in {args.rknn_dir}")
        return

    results = {}

    for model_name, quant, path in models:
        key = f"{model_name}_{quant}"
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name} ({quant}) — {path}")
        print(f"{'='*60}")

        if not args.cpu_only:
            print(f"  [NPU] {args.warmup} warmup + {args.iterations} iterations...")
            npu_result = benchmark_model(path, RKNNLite.NPU_CORE_0, args.warmup, args.iterations)
            print(f"  [NPU] mean={npu_result.get('mean_ms', 'N/A')}ms "
                  f"p50={npu_result.get('p50_ms', 'N/A')}ms "
                  f"p95={npu_result.get('p95_ms', 'N/A')}ms")
        else:
            npu_result = None

        print(f"  [CPU] {args.warmup} warmup + {args.iterations} iterations...")
        cpu_result = benchmark_model(path, RKNNLite.NPU_CORE_AUTO, args.warmup, args.iterations)

        results[key] = {
            "model": model_name,
            "quantization": quant,
            "path": path,
            "npu": npu_result,
            "cpu": cpu_result,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print("\n## Benchmark Results\n")
    print("| Model | Quantization | NPU Mean (ms) | NPU P50 (ms) | NPU P95 (ms) | NPU P99 (ms) |")
    print("|-------|-------------|---------------|--------------|--------------|--------------|")
    for key, r in results.items():
        npu = r.get("npu") or {}
        print(f"| {r['model']} | {r['quantization']} | "
              f"{npu.get('mean_ms', 'N/A')} | {npu.get('p50_ms', 'N/A')} | "
              f"{npu.get('p95_ms', 'N/A')} | {npu.get('p99_ms', 'N/A')} |")


if __name__ == "__main__":
    main()
