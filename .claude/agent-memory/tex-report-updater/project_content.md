---
name: Report content scope and implemented features
description: What has been implemented in the project and documented in the report as of 2026-03-31
type: project
---

The report covers an NPU-accelerated fault diagnosis system for Tennessee Eastman Process (TEP) on Rockchip RK3568 SoC.

## Models implemented
- TEPNet (1D CNN baseline)
- TCN: dilated Conv1d, dilation 1/2/4/8, kernel=3, 64ch, 4 levels, residual connections
- BiLSTM: hidden=64, 2 layers, bidirectional, linear head
- Transformer: 2-layer TransformerEncoder, d_model=64, nhead=4, positional encoding, global avg pool
- PatchTST: patch-based Transformer, channel-independent tokenization, 128,277 params, 82.99% test accuracy

All accept input (B, 52, 32). Unified registry in `models/__init__.py`. Training via `train_and_export.py --model {tepnet,tcn,lstm,transformer}`.

## Export and conversion
- ONNX export: input [1,52,32], saved to `onnx/{model_name}/`
- RKNN conversion: `convert_all_models.sh`, INT8 (200-sample calibration) + FP16
- Saved to `rknn/{model_name}/{model_name}_{int8,fp16}.rknn`

## Benchmarking
- `benchmark.py`: 100 warmup + 1000 iterations, mean/p50/p95/p99 latency
- `accuracy_eval.py`: accuracy + per-class F1 for PyTorch FP32, RKNN FP16, RKNN INT8
- Vision models: MobileNetV2, ResNet18 from torchvision, input [1,3,224,224], INT8/FP16

## LLM experiment
- Qwen2-0.5B Q4_0 (337MB, 494M params): 8.1 t/s prefill, 2.7 t/s decode, ~800MB RAM, ~5s load
- Gemma 3 1B Q4_K_M (769MB, 999M params): 2.0–3.5 t/s prefill, 0.23–0.62 t/s decode, ~1.9GB RAM
- All inference on ARM CPU (Cortex-A55); NPU not used — RK3568 NPU lacks dynamic shapes / KV-cache / autoregressive decode support
- llama.cpp cross-compiled with aarch64-linux-gnu-gcc (GCC 15.2.0)

## Observability stack
Prometheus metrics: tep_inference_total (Counter, label:predicted_class), tep_inference_latency_seconds (Histogram), tep_prediction_confidence (Gauge), tep_current_prediction (Gauge), tep_buffer_size (Gauge), tep_opcua_read_errors_total (Counter)

Grafana panels: current prediction (stat), confidence (gauge), latency over time, latency histogram, fault class distribution (pie), inference rate, OPC UA errors, CPU/memory from node-exporter.

Docker Compose: two-file split. docker-compose.yml (on RK3568 board): opcua-server, inference-client, node-exporter. docker-compose.monitoring.yml (on separate x86 machine): prometheus (retention 7d/500MB), grafana (auto-provisioning). All have healthchecks; depends_on with conditions.

## REST API
FastAPI server: GET /health, /prediction, /models

## OPC UA implementation (added to report 2026-03-31)
- server.c (~450 lines, C, open62541 v1.4.6, amalgamation build)
- 52 UA_Double nodes: xmeas_1..41 + xmv_1..11, namespace urn:my-csv-server
- Update interval 5000ms, --loop flag for CSV replay, port 4840
- client inference_opcua_client.py (~160 lines, python-opcua, synchronous polling), deque maxlen=32, RKNN inference
- Diagnostic client.c (~100 lines), same open62541

## Bibliography keys added (2026-03-31)
tep_downs, prometheus_docs, grafana_docs, llama_cpp, qwen2, open62541, iec62541, python_opcua, rk3568_npu_spec

## Benchmark results added to report (2026-03-31)
Inserted four new subsections into Section 6 (Experimental Methodology and Results), immediately before the Conclusion:
- **Model Training Results** — Table `table:training_results` (5 models including PatchTST, params, accuracy, architecture notes)
- **RKNN Conversion** — prose description of FP16/INT8 pipeline, 256-sample calibration, opset-12 BiLSTM note
- **NPU Latency Benchmark Results** — Table `table:npu_latency` (10 rows: 5 models × 2 quant, mean/p50/p95/p99 ms) + analysis prose. PatchTST INT8 (6.96ms) is slower than FP16 (5.21ms) due to LayerNorm CPU fallback.
- **Technical Issues Encountered and Resolutions** — 5 numbered items covering sm_120, OOM, opset-18 LSTM, core_mask, cross-compilation

Label added: `\label{sec:npu_bench_method}` on the NPU Benchmarking Methodology subsection (was unlabelled before).

## LLM and vision benchmark results added to report (2026-03-31)
Inserted two subsections after "Technical Issues" in Section 6, before Conclusion:

**Vision Model Benchmark Results** (`\label{sec:vision_results}`)
- Methodology prose: same protocol as TEP models, FPS = 1000/mean_ms
- Table `table:vision_latency`: MobileNetV2 INT8 (mean 24.962ms, P50 25.020, P95 25.199, P99 26.163, 40.06 FPS), MobileNetV2 FP16 (mean 35.483ms, P50 35.761, P95 36.052, P99 37.304, 28.18 FPS), ResNet18 INT8 (mean 24.673ms, P50 24.635, P95 24.836, P99 25.871, 40.53 FPS), ResNet18 FP16 (mean 47.614ms, P50 47.434, P95 48.401, P99 49.365, 21.00 FPS). All P50/P95/P99 now filled in.
- Analysis prose with actual speedup figures (ResNet18 1.93× INT8/FP16, MobileNetV2 1.42×)

**LLM Experiment Results** (`\label{sec:llm_results}`)
- Overview prose explaining NPU not used, CPU-only, RK3568 vs RK3588 distinction
- Table `table:llm_qwen`: Qwen2-0.5B metrics (2-col lc)
- Table `table:llm_gemma_fc`: Gemma 3 1B function calling test (2-col lc)
- Table `table:llm_gemma_text`: Gemma 3 1B text generation test (2-col lc)
- Table `table:llm_comparison`: side-by-side comparison (6-col: lcccp{2.5cm}c)
- Analysis: 5-item enumerate covering NPU inapplicability, performance contrast (1.26ms NPU vs 285s LLM ≈ 5 orders of magnitude), usability, RAM pressure, output correctness

## End-to-End Pipeline Benchmark Results added (2026-03-31)
Inserted `\subsection{End-to-End Pipeline Benchmark Results}` (`\label{sec:e2e_bench}`) immediately after the NPU Latency section's closing `\end{enumerate}` (before Technical Issues, previously line 978).

- Methodology prose: Docker OPC UA server, 52 sequential get_value() calls, deque(maxlen=32), 50 cycles per model, FP16 only, wall-clock from read-start to class assignment
- Table `table:e2e_latency` (5-col: lcccc): Mean/P50/P95 (ms) + Confidence (%) for 5 FP16 models
- Table `table:e2e_decomp` (4-col: lccc): NPU-only Mean / E2E Mean / OPC UA Overhead (%) decomposition
- Analysis: 5-item enumerate — OPC UA dominates (95--99%), all models satisfy 5s polling interval (20× margin), variability tracks OPC UA not model complexity, confidence 75--93%, primary optimisation = batch read

## TEP fault class table (added 2026-03-31)
Inserted into `\subsection{Dataset: Tennessee Eastman Process}` (after the training split paragraph, before `\subsection{Training Configuration}`).
- Introductory paragraph about Downs & Vogel (1993) origin, 52 variables (XMEAS 1--41, XMV 1--11), FDD benchmark status
- Itemize explaining the 4 disturbance types: Step, Random variation, Slow drift, Sticking
- Note that Faults 16--20 are deliberately undescribed by the authors
- Table `table:tep_faults` (3-col: `c p{8.0cm} l`): Class (0--20), Description, Type — 22 data rows
- Label: `\label{table:tep_faults}`

## Accuracy evaluation section rewritten (2026-03-31)
Replaced the Accuracy Evaluation subsection (was 4 models with old numbers + two tables + enumerate) with updated content:
- Single table `table:rknn_accuracy_comparison` (lccc): all 5 models (TEPNet, TCN, BiLSTM, Transformer, PatchTST) with corrected FP32/FP16/INT8 accuracy
- Updated numbers: TEPNet 72.42/72.42/3.03, TCN 78.18/78.18/18.48, BiLSTM 70.00/70.00/4.85, Transformer 77.58/77.58/6.06, PatchTST 82.99/82.99/15.15
- Removed old `table:rknn_accuracy` (per-quant with Macro F1) — consolidated into single comparison table
- Prose analysis: FP16 preserves full accuracy, INT8 catastrophic failure due to narrow TEP sensor ranges + 256-sample calibration, PatchTST INT8 slower than FP16 (LayerNorm CPU fallback), FP16 recommended for deployment
- Also fixed calibration sample count in Accuracy Evaluation Methodology (200 → 256) to match RKNN Conversion section

## Page count history
- Before 2026-03-31 additions: 26 pages
- After 2026-03-31 additions (OPC UA section, hardware platform, arch details, Docker details): 38 pages
- After benchmark results addition (2026-03-31): 41 pages
- After LLM + vision results addition (2026-03-31): 45 pages
- After PatchTST addition + vision data fill-in (2026-03-31): 46 pages
- After E2E pipeline benchmark addition (2026-03-31): 47 pages
- After TEP fault class table addition (2026-03-31): 48 pages
