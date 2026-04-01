# TEP Fault Diagnosis on RK3568 NPU

Fault diagnosis for the Tennessee Eastman Process running on a Rockchip RK3568 board with NPU. Five model architectures trained on an x86 GPU server, converted to RKNN, deployed to the board. Sensor data comes in through OPC UA, predictions go out through Prometheus metrics, Grafana shows what's happening.

## Project Structure

```
coursework/
├── models/           # 5 architectures: TEPNet, TCN, LSTM, Transformer, PatchTST
├── training/         # train_and_export.py and helpers
├── conversion/       # ONNX to RKNN conversion scripts
├── inference/        # OPC UA client that does the actual inference
├── benchmarks/       # latency, accuracy, end-to-end, vision model benchmarks
├── results/          # JSON files with benchmark numbers
├── opcua/            # OPC UA server (C) and diagnostic client (C)
├── llm_experiment/   # llama.cpp setup and benchmark on ARM
├── docker/           # Dockerfiles
├── monitoring/       # Prometheus and Grafana configs
├── data/             # TEP dataset (not in git)
├── onnx/             # exported ONNX models (not in git)
└── rknn/             # converted RKNN models (not in git)
```

## Hardware

Four machines:

- 2 x86 servers with RTX 5070 Ti -- training, conversion
- 2 RK3568 boards, 4x Cortex-A55, 0.8 TOPS NPU, 4GB RAM -- inference

## How to run

### 1. Train models (on x86)

```bash
cd training
python train_and_export.py --model tepnet --epochs 15 --output-dir ../onnx
python train_and_export.py --model tcn --epochs 15 --output-dir ../onnx
python train_and_export.py --model lstm --epochs 15 --output-dir ../onnx
python train_and_export.py --model transformer --epochs 15 --output-dir ../onnx
python train_and_export.py --model patchtst --epochs 15 --output-dir ../onnx
```

### 2. Convert to RKNN (on x86, needs rknn-toolkit2)

```bash
cd conversion
chmod +x convert_all_models.sh
./convert_all_models.sh
```

Makes INT8 and FP16 RKNN files for each model.

### 3. Deploy on the board

```bash
docker compose up -d --build
```

Starts three containers:
- OPC UA server -- reads TEP CSV, publishes 52 sensor tags
- Inference client -- reads tags, runs RKNN inference on NPU
- Node exporter -- system metrics for Prometheus

### 4. Pick a model

By default it runs TEPNet FP16. Change it with env vars:

```bash
# Transformer
MODEL=transformer QUANTIZATION=fp16 docker compose up -d

# PatchTST -- best accuracy at 83%
MODEL=patchtst QUANTIZATION=fp16 docker compose up -d

# TCN with INT8 (faster but worse accuracy for TEP data)
MODEL=tcn QUANTIZATION=int8 docker compose up -d
```

Models: `tepnet`, `tcn`, `lstm`, `transformer`, `patchtst`
Quantizations: `fp16`, `int8`

INT8 is much worse for TEP models (accuracy drops to near random). Use FP16.

Without Docker:

```bash
python inference/inference_opcua_client.py opc.tcp://localhost:4840 --model patchtst --quantization fp16
```

### 5. Monitoring (on x86)

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

Grafana at http://192.168.88.243:3000 (login: admin/admin). Prometheus at :9090. Both scrape metrics from the board over the network.

### 6. Benchmarks (on the board)

```bash
# NPU latency -- 1000 iterations per model
python benchmarks/benchmark.py --rknn-dir ./rknn --output results/benchmark_results.json

# Accuracy -- compares FP16 and INT8 against PyTorch FP32
python benchmarks/accuracy_eval.py --rknn-dir ./rknn --data-dir ./data/small_tep --output results/accuracy_results.json

# Full pipeline through OPC UA
python benchmarks/e2e_benchmark.py opc.tcp://localhost:4840 50

# Vision models (MobileNetV2, ResNet18) -- latency only
python benchmarks/vision/benchmark_vision.py --rknn-dir ./vision_rknn --output results/vision_benchmark_results.json
```

### 7. LLM experiment (on the board)

```bash
cd llm_experiment
./setup_llama_cpp.sh
./run_llm_benchmark.sh
```

Cross-compiling on x86 is faster than building natively on the board.

## Results

### TEP models on NPU (FP16)

| Model | Latency (ms) | Accuracy |
|-------|-------------|----------|
| TEPNet | 1.50 | 70.35% |
| TCN | 1.54 | 75.40% |
| PatchTST | 5.21 | 82.99% |
| Transformer | 3.06 | 78.00% |
| BiLSTM | 7.90 | 70.45% |

### Vision models on NPU (latency only, random input)

| Model | INT8 (ms) | FP16 (ms) | FPS (INT8) |
|-------|-----------|-----------|------------|
| MobileNetV2 | 25.0 | 35.5 | 40 |
| ResNet18 | 24.7 | 47.6 | 41 |

### LLM on ARM CPU (no NPU)

| Model | tok/s |
|-------|-------|
| Qwen2-0.5B | 2.7 |
| Gemma 3 1B | 0.23-0.62 |

The NPU cannot run LLMs. It only supports static CNN-style graphs.

## What's used

- PyTorch for training, ONNX for export, RKNN Toolkit2 for NPU conversion
- OPC UA: server in C (open62541), client in Python (python-opcua)
- RKNNLite for inference on the board
- Docker Compose
- Prometheus + Grafana
- llama.cpp for the LLM experiment
