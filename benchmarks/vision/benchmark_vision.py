from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite

WARMUP = 100
ITERATIONS = 1000
INPUT_SHAPE = (1, 3, 224, 224)

log = logging.getLogger(__name__)


def benchmark_model(model_path: str, warmup: int, iterations: int) -> dict:
    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed with code {ret}")

        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed with code {ret}")

        dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)

        for _ in range(warmup):
            rknn.inference(inputs=[dummy])

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            rknn.inference(inputs=[dummy])
            latencies.append((time.perf_counter() - t0) * 1000)

    except Exception:
        log.exception("Benchmark failed for %s", model_path)
        return {"error": "benchmark failed"}
    finally:
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
        "throughput_fps": round(1000.0 / lat.mean(), 2),
        "iterations": iterations,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Benchmark vision RKNN models on RK3568")
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

            log.info("Benchmarking %s (%s): %s", model_name, quant, rknn_file)
            npu = benchmark_model(str(rknn_file), args.warmup, args.iterations)

            if "error" not in npu:
                log.info("%s: mean=%.3f ms  throughput=%.1f fps", key, npu["mean_ms"], npu["throughput_fps"])
            else:
                log.warning("%s: %s", key, npu["error"])

            results[key] = {"model": model_name, "quantization": quant, "npu": npu}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", args.output)

    log.info("%-25s %-8s %12s %12s %12s %16s",
             "Model", "Quant", "Mean (ms)", "P50 (ms)", "P95 (ms)", "Throughput (FPS)")
    for key, r in results.items():
        npu = r.get("npu", {})
        if "error" in npu:
            log.warning("%-25s %-8s ERROR", r["model"], r["quantization"])
        else:
            log.info("%-25s %-8s %12s %12s %12s %16s",
                     r["model"], r["quantization"],
                     npu.get("mean_ms", "N/A"), npu.get("p50_ms", "N/A"),
                     npu.get("p95_ms", "N/A"), npu.get("throughput_fps", "N/A"))


if __name__ == "__main__":
    main()
