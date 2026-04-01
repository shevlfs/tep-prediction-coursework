---
name: Docker monitoring stack deployment on RK3568
description: Full Docker stack (OPC UA server + inference client + node-exporter + Prometheus + Grafana) deployed on 192.168.88.244
type: project
---

Docker monitoring stack is deployed and running on RK3568 (192.168.88.244).

**Architecture:** Two separate docker-compose files:
- `~/coursework/docker-compose.yml` — board services: opcua-server (port 4840), opcua-inference-client (port 8000), node-exporter (port 9100)
- `~/coursework/docker-compose.monitoring.yml` — monitoring: prometheus (port 9090), grafana (port 3000)

**Why:** The compose files are intentionally split — the original design ran Prometheus/Grafana on x86 (192.168.88.243), but were deployed to the board for the full-stack requirement.

**How to apply:** When restarting or managing the stack, use both compose files separately. To bring up the full stack:
```bash
cd ~/coursework
sudo docker compose up -d
sudo docker compose -f docker-compose.monitoring.yml up -d
```

**Key facts:**
- librknnrt.so is at /usr/lib/librknnrt.so (system install, correct volume mount path)
- Images are pre-built and cached — rebuilds only needed if Dockerfiles change
- Inference client buffers 32 OPC UA samples before running RKNN model
- RKNN model: ~/coursework/rknn/model_fp.rknn (fp16)
- Prometheus scrapes 192.168.88.244:8000 (inference) and :9100 (node-exporter)
- Grafana: admin/admin, anonymous viewer access enabled
- open62541 v1.4.6 compiled from source in server container (takes ~6 min fresh build)

**Container names:**
- coursework-opcua-server-1
- coursework-opcua-inference-client-1
- coursework-node-exporter-1
- coursework-prometheus-1
- coursework-grafana-1

**RKNN access note:** RKNNLite requires sudo on this board — running as non-root yields "failed to open rknpu module" even though rknpu is loaded and shevlfs is in the render group. Always run RKNN inference scripts with `sudo -S -E`.

**E2E benchmark results (2026-03-31, 50 cycles, OPC UA → normalize → RKNN → softmax):**
- patchtst_fp16: mean 175.71ms, P50 166.12ms, P95 226.82ms, P99 244.41ms, std 21.22ms
- tcn_fp16:      mean 178.89ms, P50 174.07ms, P95 217.53ms, P99 267.22ms, std 22.89ms
- tepnet_fp16:   mean 180.12ms, P50 165.16ms, P95 250.53ms, P99 267.13ms, std 30.19ms
- lstm_fp16:     mean 191.98ms, P50 185.46ms, P95 230.47ms, P99 243.29ms, std 20.57ms
- transformer:   mean 229.47ms, P50 218.0ms,  P95 345.52ms, P99 370.22ms, std 54.64ms
- Benchmark script: ~/e2e_benchmark.py; results JSON: ~/e2e_benchmark_results.json
- opcua package (0.98.13) installed in ~/bench_env
