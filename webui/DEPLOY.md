# Deployment Guide

This document describes the procedure for deploying the TEP fault diagnosis stack. The system is composed of three physically distinct nodes that must not be collocated, as the web UI assumes independent network reachability between them.

## System Topology

The deployment consists of the following nodes:

| Node | Role |
|---|---|
| Operator workstation | Hosts the Flask-based web UI and initiates SSH sessions to the microcontroller. |
| Monitoring host (x86, Linux) | Hosts the OPC UA server (containerised), Prometheus, and Grafana. |
| Microcontroller (Rockchip RK3568) | Executes the inference client natively on the NPU and exposes Prometheus metrics on TCP port 8000. |

The inference client is executed natively rather than inside a container, as the RKNN runtime exhibits compatibility issues when invoked from within a containerised environment.

## Data Flow

```
  [Operator workstation]
        │
        │  HTTP  :12345     (web UI)
        │  SSH              (lifecycle management of the inference client)
        │                   ──────────────▶  [Microcontroller]
        │                                           │
        │                                           │  OPC UA :4840
        │                                           ▼
        │                                    [Monitoring host]
        │                                           │
        │                                      scrape :8000  (inference metrics)
        │                                      scrape :9100  (node-exporter)
        │                                           │
        └── HTTP ──────────────────────▶ Grafana :3000 / Prometheus :9090
```

## Prerequisites

### Monitoring host

- Docker Engine and the Compose plugin.
- Network reachability from the microcontroller on TCP ports 4840, 9090, and 3000.

### Microcontroller

- A Debian- or Ubuntu-based distribution with the RKNN runtime and corresponding NPU kernel driver installed and functional.
- A Python 3.10+ virtual environment with the packages `rknn-toolkit-lite2`, `prometheus_client`, and `asyncua`.
- The compiled RKNN artefacts (both FP16 and INT8 quantisations) for each model: `tepnet`, `tcn`, `lstm`, `transformer`, `patchtst`.
- The per-model normalisation parameters (mean and standard deviation) stored as JSON files.
- The inference client script `inference_opcua_client.py`, as provided in the `inference/` directory of this repository.
- A passwordless SSH key installed on behalf of a user with sudo privileges, accessible from the operator workstation.
- Optionally, a running instance of `node-exporter` bound to port 9100, required for CPU, memory, and thermal metrics in Grafana.

### Operator workstation

- Python 3.10 or later.
- Network reachability to both the monitoring host and the microcontroller.

## Deployment Procedure

### Monitoring host

Clone the repository and bring the monitoring stack online:

```bash
git clone <repository> coursework
cd coursework
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

The following services will be provisioned:

- `opcua-server`, publishing the 52 TEP tags (`xmeas_1..41`, `xmv_1..11`) on `opc.tcp://<host>:4840`.
- `prometheus`, scraping the inference client and node-exporter endpoints on the microcontroller.
- `grafana`, accessible at `http://<host>:3000` with default credentials `admin` / `admin`.

The Prometheus scrape configuration must be updated in `monitoring/prometheus/prometheus.yml` to reference the microcontroller. A fixed `instance` label is recommended to preserve dashboard continuity in the event of address changes:

```yaml
scrape_configs:
  - job_name: "inference-client"
    static_configs:
      - targets: ["<microcontroller>:8000"]
        labels: { instance: "rk3568" }
  - job_name: "node-exporter"
    static_configs:
      - targets: ["<microcontroller>:9100"]
        labels: { instance: "rk3568" }
```

The provisioned Grafana dashboard (`tep-inference.json`) requires that every panel specify an explicit datasource reference of the form `datasource: { type: prometheus, uid: <uid> }`. From Grafana 10 onwards, panels with a null datasource reference no longer resolve to the default datasource, and will therefore render as empty despite the Prometheus targets being healthy.

### Microcontroller

The following artefacts must be present on the microcontroller prior to deployment:

- The Python virtual environment.
- The inference client script.
- A directory containing the compiled RKNN models for each model and quantisation combination.
- A directory containing the normalisation parameters for each model.

Prior to integration with the web UI, the inference client should be verified in isolation:

```bash
sudo -E strace -f -e trace=none -o /dev/null \
  ~/bench_env/bin/python3 ~/inference_opcua_client.py \
  opc.tcp://<monitoring host>:4840 \
  --model patchtst --quantization fp16 \
  --rknn-dir ~/rknn --norm-dir ~/onnx
```

The `strace` wrapper is required as a workaround for a known initialisation defect in the RKNN driver; omitting it results in intermittent failures during NPU context creation. A successful launch is confirmed by the presence of the metric `tep_model_info{model_name="patchtst",quantization="fp16"} 1` at `http://localhost:8000/metrics`. The process should be terminated prior to proceeding, as the web UI assumes exclusive control over the client lifecycle.

Optionally, `node-exporter` may be deployed as follows:

```bash
docker run -d --name node-exporter --restart=unless-stopped \
  -p 9100:9100 --pid=host -v /:/host:ro,rslave \
  prom/node-exporter:v1.7.0 --path.rootfs=/host
```

### Operator workstation

```bash
cd webui
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Required environment variables
export RK3568_HOSTS="<microcontroller>"
export X86_HOST="<monitoring host>"
export BOARD_SUDO_PASS="<sudo password on the microcontroller>"

# Optional environment variables; defaults are appropriate for the layout above
export BOARD_VENV_PYTHON="/home/<user>/bench_env/bin/python3"
export BOARD_CLIENT_SCRIPT="/home/<user>/inference_opcua_client.py"
export BOARD_RKNN_DIR="/home/<user>/rknn"
export BOARD_NORM_DIR="/home/<user>/onnx"

python app.py
```

The web UI will be available at `http://localhost:12345`.

## Model Deployment Workflow

On selection of a model and invocation of the deployment action, the following sequence is executed over SSH:

1. A mutual-exclusion lock is acquired to prevent concurrent deployments.
2. Any running instance of the inference client is terminated via `pkill -f inference_opcua_client.py`.
3. The system waits until TCP port 8000 is released, with a timeout of approximately ten seconds. Failure to release the port aborts the deployment, as `prometheus_client.start_http_server` would otherwise fail with an "address in use" error.
4. The new client instance is launched in a detached session by means of `nohup setsid bash -c "<command>" > ~/inference_client.log 2>&1 & disown`, ensuring independence from the parent SSH session.
5. The `/metrics` endpoint is polled for up to sixty seconds. The deployment is considered successful only upon observation of `tep_model_info{model_name=<new>, quantization=<new>}`.

In the event of failure, the log file `~/inference_client.log` on the microcontroller should be consulted. The predominant failure modes are RKNN runtime initialisation and OPC UA session establishment.

## Security Considerations

The following considerations apply when deploying the system outside of a controlled laboratory environment:

- The environment variable `BOARD_SUDO_PASS` must not be committed to the repository. It should be set via a shell initialisation file or a systemd `EnvironmentFile` directive.
- The SSH helper disables host key verification (`StrictHostKeyChecking=no`). This is acceptable within a trusted local network but must be replaced with conventional `known_hosts` verification for any other deployment scenario.
- The Flask application binds to `0.0.0.0:12345` by default. In non-isolated environments, the bind address should be restricted to `127.0.0.1`, or the service placed behind a reverse proxy with appropriate authentication.

## Troubleshooting

| Symptom | Recommended action |
|---|---|
| Web UI reports "Offline" | Verify SSH reachability to the microcontroller and the validity of the installed key. |
| "Port :8000 still busy after kill" | Identify the process retaining the port: `ssh <microcontroller> 'sudo ss -lptn sport = :8000'`. |
| Prometheus targets in DOWN state | Inspect `docker compose logs prometheus` on the monitoring host and verify firewall configuration. |
| Prometheus targets UP but Grafana panels empty | Confirm that each panel in the dashboard JSON specifies `datasource.uid` explicitly; a null reference prevents query resolution in Grafana 10+. |
| Client launches but `tep_model_info` does not appear | Consult `~/inference_client.log` on the microcontroller. Typical causes are RKNN runtime initialisation failure or OPC UA connection timeout. |
