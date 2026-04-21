"""Memory and thermal profiling for RKNN models on RK3568.

Measures RSS memory before/after model load, peak RSS during sustained inference,
and CPU/SoC temperatures over time.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

WINDOW_SIZE = 32
NUM_FEATURES = 52
SUSTAINED_DURATION_SEC = 30
THERMAL_SAMPLE_INTERVAL_SEC = 1

log = logging.getLogger(__name__)


def get_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def get_temperatures() -> dict:
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


def get_cpu_freqs() -> dict:
    freqs = {}
    cpu_base = Path("/sys/devices/system/cpu")
    for i in range(4):
        try:
            freq = int((cpu_base / f"cpu{i}/cpufreq/scaling_cur_freq").read_text().strip())
            freqs[f"cpu{i}_mhz"] = freq / 1000
        except Exception:
            continue
    return freqs


def profile_model(model_path: str, sustained_sec: int = SUSTAINED_DURATION_SEC) -> dict:
    from rknnlite.api import RKNNLite

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    rss_before = get_rss_mb()
    temps_before = get_temperatures()

    rss_after_load = rss_after_init = rss_after_warmup = rss_peak = 0.0
    temps_after_warmup = temps_after_sustained = {}
    thermal_log: list = []
    latencies: list = []
    iteration = 0
    rss_after_release = 0.0

    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed with code {ret}")
        rss_after_load = get_rss_mb()

        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed with code {ret}")
        rss_after_init = get_rss_mb()

        dummy_input = np.random.randn(1, NUM_FEATURES, WINDOW_SIZE).astype(np.float32)
        for _ in range(100):
            rknn.inference(inputs=[dummy_input])

        rss_after_warmup = get_rss_mb()
        temps_after_warmup = get_temperatures()

        start_time = time.time()
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

    except Exception:
        log.exception("Profiling failed for %s", model_path)
        return {"error": "profiling failed"}
    finally:
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Memory and thermal profiling for RKNN models")
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

            log.info(
                "Profiling %s (%s): %.2f MB on disk, %ds sustained",
                key, rknn_file,
                os.path.getsize(rknn_file) / 1024 / 1024,
                args.sustained_sec,
            )

            result = profile_model(str(rknn_file), args.sustained_sec)
            results[key] = {"model": model_name, "quantization": quant, "path": str(rknn_file), **result}

            if "error" not in result:
                mem = result["memory"]
                th = result["thermal"]
                si = result["sustained_inference"]
                max_temp = max(th["after_sustained"].values()) if th["after_sustained"] else float("nan")
                log.info(
                    "%s: model_mem=%.1f MB  peak_rss=%.1f MB  throughput=%.1f fps  mean=%.1f ms  max_temp=%.1f C",
                    key, mem["model_memory_mb"], mem["rss_peak_mb"],
                    si["throughput_fps"], si["mean_ms"], max_temp,
                )
            else:
                log.warning("%s: %s", key, result["error"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", args.output)

    log.info("%-25s %8s %8s %10s %12s %10s",
             "Model", "File MB", "Mem MB", "Peak RSS", "Throughput", "Max Temp")
    for key, r in results.items():
        if "error" in r:
            log.warning("%-25s ERROR", key)
            continue
        max_temp = max(r["thermal"]["after_sustained"].values()) if r["thermal"]["after_sustained"] else float("nan")
        log.info(
            "%-25s %8.2f %8.1f %10.1f %10.1f fps %8.1f C",
            key, r["file_size_mb"], r["memory"]["model_memory_mb"],
            r["memory"]["rss_peak_mb"], r["sustained_inference"]["throughput_fps"], max_temp,
        )


if __name__ == "__main__":
    main()
