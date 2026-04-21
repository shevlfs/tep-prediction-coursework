from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite

WINDOW_SIZE = 32
NUM_FEATURES = 52
WARMUP = 100
ITERATIONS = 1000

log = logging.getLogger(__name__)


def benchmark_model(model_path: str, core_mask: int, warmup: int, iterations: int) -> dict:
    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed with code {ret}")

        ret = rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed with code {ret}")

        dummy_input = np.random.randn(1, NUM_FEATURES, WINDOW_SIZE).astype(np.float32)

        for _ in range(warmup):
            rknn.inference(inputs=[dummy_input])

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            rknn.inference(inputs=[dummy_input])
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
        "iterations": iterations,
        "warmup": warmup,
    }


def find_models(rknn_dir: Path) -> list[tuple[str, str, str]]:
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Benchmark RKNN models on RK3568")
    parser.add_argument("--rknn-dir", type=Path, default=Path("rknn"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    models = find_models(args.rknn_dir)
    if not models:
        log.error("No RKNN models found in %s", args.rknn_dir)
        return

    results = {}

    for model_name, quant, path in models:
        key = f"{model_name}_{quant}"
        log.info("Benchmarking %s (%s): %s", model_name, quant, path)

        if not args.cpu_only:
            log.info("NPU core 0: warmup=%d iterations=%d", args.warmup, args.iterations)
            npu_result = benchmark_model(path, RKNNLite.NPU_CORE_0, args.warmup, args.iterations)
            if "error" not in npu_result:
                log.info(
                    "NPU core 0: mean=%.3f ms  p50=%.3f ms  p95=%.3f ms",
                    npu_result["mean_ms"], npu_result["p50_ms"], npu_result["p95_ms"],
                )
            else:
                log.warning("NPU core 0 benchmark failed: %s", npu_result["error"])
        else:
            npu_result = None

        log.info("NPU auto: warmup=%d iterations=%d", args.warmup, args.iterations)
        auto_result = benchmark_model(path, RKNNLite.NPU_CORE_AUTO, args.warmup, args.iterations)
        if "error" not in auto_result:
            log.info(
                "NPU auto: mean=%.3f ms  p50=%.3f ms  p95=%.3f ms",
                auto_result["mean_ms"], auto_result["p50_ms"], auto_result["p95_ms"],
            )
        else:
            log.warning("NPU auto benchmark failed: %s", auto_result["error"])

        results[key] = {
            "model": model_name,
            "quantization": quant,
            "path": path,
            "npu_core0": npu_result,
            "npu_auto": auto_result,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", args.output)

    log.info("%-15s %-8s %12s %12s %12s %12s", "Model", "Quant", "Mean (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)")
    for r in results.values():
        npu = r.get("npu_core0") or {}
        log.info(
            "%-15s %-8s %12s %12s %12s %12s",
            r["model"], r["quantization"],
            npu.get("mean_ms", "N/A"), npu.get("p50_ms", "N/A"),
            npu.get("p95_ms", "N/A"), npu.get("p99_ms", "N/A"),
        )


if __name__ == "__main__":
    main()
