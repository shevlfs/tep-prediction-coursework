---
name: x86 server system configuration
description: Hardware specs, driver versions, conda environments, and key directories on 192.168.88.243
type: reference
---

# x86 CUDA Server (192.168.88.243)

## Hardware
- CPU: AMD Ryzen 9800X3D
- RAM: 64GB
- GPU: NVIDIA GeForce RTX 5070 Ti (16GB VRAM, sm_120 / Blackwell)
- OS: Ubuntu

## NVIDIA Driver / CUDA
- Driver: 590.48.01
- CUDA: 13.1 (system)
- GPU compute capability: sm_120 — requires PyTorch 2.6+ with cu128

## Conda Environments
- `base`: Python 3.13.12, no torch
- `rknn`: Python 3.11.15, torch 2.4.0+cu121 (does NOT support sm_120 for training), rknn-toolkit2 2.3.2, onnx 1.16.2
- `train_env`: Python 3.11, torch 2.11.0+cu128 (sm_120 compatible), onnx 1.21.0, onnxscript 0.6.2, pandas, scikit-learn

## Key Directories
- `~/onnx/{model}/` — TEP ONNX models (model.onnx, model.pt, norm_mean.npy, norm_std.npy)
- `~/rknn/{model}/` — TEP RKNN models (model_int8.rknn, model_fp16.rknn)
- `~/vision_onnx/{model}/` — Vision ONNX models
- `~/vision_rknn/{model}/` — Vision RKNN models
- `~/small_tep/` — TEP dataset (df.csv, target.csv, train_mask.csv)
- `~/small_tep/calibration/` — RKNN calibration .npy files + dataset.txt
- `~/models/` — TEP PyTorch model definitions
- `~/vision_benchmarks/` — Vision export + conversion scripts

## Conda binary path
`~/miniconda3/bin/conda` (not in default PATH)

## Cross-compilation toolchain (installed 2026-03-31)
- `gcc-aarch64-linux-gnu` / `g++-aarch64-linux-gnu` (GCC 15.2.0) installed via apt
- Sysroot at `/usr/aarch64-linux-gnu`

## Monitoring Stack (deployed 2026-03-31)
- Docker 28.2.2, Docker Compose v2.37.1 (installed system-wide via apt)
- Stack file: `~/coursework/docker-compose.monitoring.yml`
- Config files: `~/coursework/monitoring/prometheus/`, `~/coursework/monitoring/grafana/`
- Services: `prom/prometheus:v2.51.0` on :9090, `grafana/grafana:10.4.0` on :3000
- Grafana: http://192.168.88.243:3000 (admin/admin), database ok
- Prometheus: scrapes RK3568 at 192.168.88.244:8000 (inference-client) and :9100 (node-exporter)
- To manage: `cd ~/coursework && sudo docker compose -f docker-compose.monitoring.yml <up -d|down|ps>`

## RK3568 (192.168.88.244) NPU Access
- OS: Ubuntu 24.04.4 LTS, kernel 6.8.0-106-generic (aarch64)
- rknpu driver 0.9.8 — DRM device at /dev/dri/card2 + renderD129 (minor 2), no /dev/rknpu char device
- rknn-toolkit-lite2 2.3.2 in: bench_env, rknn_env, rknpu/venv
- librknnrt.so 2.3.2 at /usr/lib/librknnrt.so
- NPU init race condition: init_runtime fails without ptrace overhead — must run under `sudo strace -f -e trace=none python3 ... 2>/dev/null`
- Benchmark script: ~/benchmark.py (expects ~/rknn/{model}/model_{fp16|int8}.rknn)

## llama.cpp cross-compiled for RK3568 (aarch64)
- Source: `~/llama-cpp-cross/` (github.com/ggerganov/llama.cpp, build 8604 / 6307ec07d)
- Build dir: `~/llama-cpp-cross/build-arm/bin/` — contains `llama-cli` + all `.so` libs
- GGUF model: `~/models/qwen2-0_5b-instruct-q4_0.gguf` (337MB, Qwen2-0.5B-Instruct Q4_0)
- Deployed to RK3568 at `~/llama.cpp/build/bin/` (binary + all shared libs)
- To run on RK3568: `LD_LIBRARY_PATH=~/llama.cpp/build/bin ~/llama.cpp/build/bin/llama-cli ...`
- Model on RK3568: `~/models/qwen2-0_5b-instruct-q4_0.gguf`
