#!/usr/bin/env python3
"""Memory and thermal profiling for RKNN models on RK3568.

Measures:
1. RSS memory before/after loading each model
2. Peak memory during inference
3. CPU/SoC temperature before/during/after sustained inference
4. Memory footprint of each .rknn file on disk vs in-memory

Usage (on RK3568):
    sudo strace -f -e trace=none python3 memory_thermal_benchmark.py --rknn-dir ./rknn 2>/dev/null
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
from pathlib import Path

WINDOW_SIZE = 32
NUM_FEATURES = 52
SUSTAINED_DURATION_SEC = 30
THERMAL_SAMPLE_INTERVAL_SEC = 1


def get_rss_mb():
    """Get current process RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def get_temperatures():
    """Read all available thermal zones."""
    temps = {}
    thermal_base = Path("/sys/class/thermal")
    if not thermal_base.exists():
        return temps
    for tz in sorted(thermal_base.glob("thermal_zone*")):
        try:
            name = (tz / "type").read_text().strip()
            temp_mc = int((tz / "temp").read_text().strip())
            temps[name] = round(temp_mc / 1000.0, 1)
        except Exception:
            continue
    return temps


def get_cpu_freqs():
    """Read current CPU frequencies."""
    freqs = {}
    cpu_base = Path("/sys/devices/system/cpu")
    for i in range(4):
        try:
            freq = int((cpu_base / f"cpu{i}/cpufreq/scaling_cur_freq").read_text().strip())
            freqs[f"cpu{i}_mhz"] = freq / 1000
        except Exception:
            continue
    return freqs


def profile_model(model_path: str, sustained_sec: int = SUSTAINED_DURATION_SEC):
    """Profile memory and thermal behavior for one model."""
    from rknnlite.api import RKNNLite

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Baseline measurements
    rss_before = get_rss_mb()
    temps_before = get_temperatures()

    # Load model
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        return {"error": f"Failed to load: {ret}"}

    rss_after_load = get_rss_mb()

    ret = rknn.init_runtime()
    if ret != 0:
        rknn.release()
        return {"error": f"Failed to init runtime: {ret}"}

    rss_after_init = get_rss_mb()

    dummy_input = np.random.randn(1, NUM_FEATURES, WINDOW_SIZE).astype(np.float32)

    # Warmup
    for _ in range(100):
        rknn.inference(inputs=[dummy_input])

    rss_after_warmup = get_rss_mb()
    temps_after_warmup = get_temperatures()

    # Sustained inference with thermal monitoring
    thermal_log = []
    latencies = []
    start_time = time.time()
    iteration = 0

    last_sample_time = start_time
    while time.time() - start_time < sustained_sec:
        t0 = time.perf_counter()
        rknn.inference(inputs=[dummy_input])
        latencies.append((time.perf_counter() - t0) * 1000)
        iteration += 1

        now = time.time()
        if now - last_sample_time >= THERMAL_SAMPLE_INTERVAL_SEC:
            thermal_log.append({
                "elapsed_sec": round(now - start_time, 1),
                "temperatures": get_temperatures(),
                "cpu_freqs": get_cpu_freqs(),
                "rss_mb": round(get_rss_mb(), 1),
            })
            last_sample_time = now

    rss_peak = get_rss_mb()
    temps_after_sustained = get_temperatures()

    rknn.release()
    rss_after_release = get_rss_mb()

    lat = np.array(latencies)

    return {
        "file_size_mb": round(file_size_mb, 2),
        "memory": {
            "rss_before_mb": round(rss_before, 1),
            "rss_after_load_mb": round(rss_after_load, 1),
            "rss_after_init_mb": round(rss_after_init, 1),
            "rss_after_warmup_mb": round(rss_after_warmup, 1),
            "rss_peak_mb": round(rss_peak, 1),
            "rss_after_release_mb": round(rss_after_release, 1),
            "model_memory_mb": round(rss_after_init - rss_before, 1),
            "runtime_overhead_mb": round(rss_peak - rss_after_init, 1),
        },
        "thermal": {
            "before": temps_before,
            "after_warmup": temps_after_warmup,
            "after_sustained": temps_after_sustained,
            "log": thermal_log,
        },
        "sustained_inference": {
            "duration_sec": sustained_sec,
            "total_iterations": iteration,
            "throughput_fps": round(iteration / sustained_sec, 1),
            "mean_ms": round(float(lat.mean()), 3),
            "std_ms": round(float(lat.std()), 3),
            "p50_ms": round(float(np.percentile(lat, 50)), 3),
            "p95_ms": round(float(np.percentile(lat, 95)), 3),
            "p99_ms": round(float(np.percentile(lat, 99)), 3),
            "min_ms": round(float(lat.min()), 3),
            "max_ms": round(float(lat.max()), 3),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Memory and thermal profiling")
    parser.add_argument("--rknn-dir", type=Path, default=Path("rknn"))
    parser.add_argument("--output", type=Path, default=Path("results/memory_thermal_results.json"))
    parser.add_argument("--sustained-sec", type=int, default=SUSTAINED_DURATION_SEC)
    args = parser.parse_args()

    results = {}

    for model_dir in sorted(args.rknn_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for rknn_file in sorted(model_dir.glob("*.rknn")):
            model_name = model_dir.name
            quant = rknn_file.stem.replace("model_", "")
            key = f"{model_name}_{quant}"

            print(f"\n{'='*60}")
            print(f"Profiling: {key} ({rknn_file})")
            print(f"  File size: {os.path.getsize(rknn_file) / 1024 / 1024:.2f} MB")
            print(f"  Sustained inference: {args.sustained_sec}s")
            print(f"{'='*60}")

            result = profile_model(str(rknn_file), args.sustained_sec)
            results[key] = {"model": model_name, "quantization": quant, "path": str(rknn_file), **result}

            if "error" not in result:
                mem = result["memory"]
                print(f"  Memory: model={mem['model_memory_mb']:.1f}MB, peak_rss={mem['rss_peak_mb']:.1f}MB")
                th = result["thermal"]
                if th["after_sustained"]:
                    max_temp = max(th["after_sustained"].values())
                    print(f"  Max temp after sustained: {max_temp}C")
                si = result["sustained_inference"]
                print(f"  Throughput: {si['throughput_fps']} FPS, mean={si['mean_ms']:.1f}ms")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n## Memory & Thermal Summary\n")
    print(f"{'Model':<25} {'File MB':>8} {'Mem MB':>8} {'Peak RSS':>10} {'Throughput':>12} {'Max Temp':>10}")
    print("-" * 78)
    for key, r in results.items():
        if "error" in r:
            print(f"{key:<25} ERROR")
            continue
        max_temp = max(r["thermal"]["after_sustained"].values()) if r["thermal"]["after_sustained"] else 0
        print(f"{key:<25} {r['file_size_mb']:>8.2f} {r['memory']['model_memory_mb']:>8.1f} "
              f"{r['memory']['rss_peak_mb']:>10.1f} {r['sustained_inference']['throughput_fps']:>10.1f} fps "
              f"{max_temp:>8.1f}C")


if __name__ == "__main__":
    main()
