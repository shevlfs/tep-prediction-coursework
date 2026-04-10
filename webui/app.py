from flask import Flask, render_template, jsonify, request
import json
import os
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

app = Flask(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Hosts & URLs — overridable via env
RK3568_HOSTS = [h.strip() for h in os.environ.get("RK3568_HOSTS", "192.168.88.244").split(",") if h.strip()]
RK3568_HOST = RK3568_HOSTS[0]

X86_HOST = os.environ.get("X86_HOST", "192.168.88.241")
# OPC UA server, Prometheus and Grafana all run on the single x86 host (.241)
# via docker-compose.{yml,monitoring.yml}. Override with env if layout changes.
GRAFANA_URL = os.environ.get("GRAFANA_URL", f"http://{X86_HOST}:3000")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", f"http://{X86_HOST}:9090")
OPCUA_PORT = int(os.environ.get("OPCUA_PORT", 4840))
METRICS_PORT = int(os.environ.get("METRICS_PORT", 8000))

# Native inference client (runs on RK3568 as root via sudo+strace NPU workaround).
# No docker, no systemd — webUI supervises it directly.
OPCUA_URL = os.environ.get("OPCUA_URL", f"opc.tcp://{X86_HOST}:{OPCUA_PORT}")
BOARD_VENV_PYTHON = os.environ.get("BOARD_VENV_PYTHON", "/home/shevlfs/bench_env/bin/python3")
BOARD_CLIENT_SCRIPT = os.environ.get("BOARD_CLIENT_SCRIPT", "/home/shevlfs/inference_opcua_client.py")
BOARD_RKNN_DIR = os.environ.get("BOARD_RKNN_DIR", "/home/shevlfs/rknn")
BOARD_NORM_DIR = os.environ.get("BOARD_NORM_DIR", "/home/shevlfs/onnx")
BOARD_LOG_FILE = os.environ.get("BOARD_LOG_FILE", "/home/shevlfs/inference_client.log")
BOARD_SUDO_PASS = os.environ["BOARD_SUDO_PASS"]
CLIENT_PROC_PATTERN = "inference_opcua_client.py"

MODELS = {
    "tepnet": {"name": "TEPNet (1D CNN)", "params": "20K"},
    "tcn": {"name": "TCN (Temporal Conv)", "params": "101K"},
    "lstm": {"name": "BiLSTM", "params": "162K"},
    "transformer": {"name": "Transformer", "params": "105K"},
    "patchtst": {"name": "PatchTST", "params": "128K"},
}

ACCURACY = {
    "tepnet": {"fp32": 72.42, "fp16": 72.42, "int8": 3.03},
    "tcn": {"fp32": 78.18, "fp16": 78.18, "int8": 18.48},
    "lstm": {"fp32": 70.00, "fp16": 70.00, "int8": 4.85},
    "transformer": {"fp32": 77.58, "fp16": 77.58, "int8": 6.06},
    "patchtst": {"fp32": 82.99, "fp16": 82.99, "int8": 15.15},
}

FAULT_LABELS = {
    0: "Normal",
    1: "Fault 1: A/C feed ratio step",
    2: "Fault 2: B composition step",
    3: "Fault 3: D feed temp step",
    4: "Fault 4: Reactor cooling water inlet temp step",
    5: "Fault 5: Condenser cooling water inlet temp step",
    6: "Fault 6: A feed loss step",
    7: "Fault 7: C header pressure loss step",
    8: "Fault 8: A,B,C feed composition random",
    9: "Fault 9: D feed temp random",
    10: "Fault 10: C feed temp random",
    11: "Fault 11: Reactor cooling water inlet temp random",
    12: "Fault 12: Condenser cooling water inlet temp random",
    13: "Fault 13: Reaction kinetics slow drift",
    14: "Fault 14: Reactor cooling water valve sticking",
    15: "Fault 15: Condenser cooling water valve sticking",
    16: "Fault 16: Unknown",
    17: "Fault 17: Unknown",
    18: "Fault 18: Unknown",
    19: "Fault 19: Unknown",
    20: "Fault 20: Unknown",
}

deploy_lock = threading.Lock()
deploy_status = {"state": "idle", "message": "", "model": None, "quantization": None, "host": None}


def ssh_run(cmd, timeout=60, host=None):
    target = host or RK3568_HOST
    full = ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no", target, cmd]
    result = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,253}$")
MODEL_INFO_RE = re.compile(r'model_name="([^"]+)"[^}]*quantization="([^"]+)"')


def ssh_launch_native(host, model, quant):
    """Start the inference client natively in the background, detached from the SSH session."""
    inner = (
        f"echo {shlex.quote(BOARD_SUDO_PASS)} | sudo -S -E "
        f"strace -f -e trace=none -o /dev/null "
        f"{shlex.quote(BOARD_VENV_PYTHON)} {shlex.quote(BOARD_CLIENT_SCRIPT)} "
        f"{shlex.quote(OPCUA_URL)} "
        f"--model {shlex.quote(model)} --quantization {shlex.quote(quant)} "
        f"--rknn-dir {shlex.quote(BOARD_RKNN_DIR)} --norm-dir {shlex.quote(BOARD_NORM_DIR)}"
    )
    cmd = (
        f"nohup setsid bash -c {shlex.quote(inner)} "
        f"> {shlex.quote(BOARD_LOG_FILE)} 2>&1 < /dev/null & disown; echo launched"
    )
    return ssh_run(cmd, timeout=15, host=host)


def ssh_stop_native(host):
    """Kill any running inference client process. Safe if none is running."""
    cmd = (
        f"echo {shlex.quote(BOARD_SUDO_PASS)} | sudo -S pkill -f "
        f"{shlex.quote(CLIENT_PROC_PATTERN)} 2>/dev/null; sleep 1; true"
    )
    return ssh_run(cmd, timeout=15, host=host)


def poll_metrics_model(host):
    """Read tep_model_info from /metrics. Returns (model, quant) or (None, None) if no client up."""
    rc, out, _ = ssh_run(
        f"curl -s --max-time 3 http://localhost:{METRICS_PORT}/metrics",
        timeout=8, host=host,
    )
    if rc != 0 or not out:
        return None, None
    for line in out.split("\n"):
        if not line.startswith("tep_model_info"):
            continue
        m = MODEL_INFO_RE.search(line)
        if m:
            return m.group(1), m.group(2)
    return None, None


def pick_host():
    """Pick a host from the request (query/JSON) falling back to default.
    Accepts any valid hostname/IP — not restricted to RK3568_HOSTS — so users
    can target custom hosts added via the UI. Shell-injection safe by regex.
    """
    host = None
    if request.method == "GET":
        host = request.args.get("host")
    else:
        data = request.get_json(silent=True) or {}
        host = data.get("host") or request.args.get("host")
    if host and HOST_RE.match(host):
        return host
    return RK3568_HOST


def load_accuracy_override():
    """Prefer the rerun file if it exists (new INT8 calibration), else the original."""
    for name in ("accuracy_results_int8_rerun.json", "accuracy_results.json"):
        p = RESULTS_DIR / name
        if p.exists():
            try:
                return json.loads(p.read_text()), name
            except Exception:
                pass
    return None, None


def merged_accuracy():
    """Merge hardcoded ACCURACY with whatever's in the accuracy JSON file."""
    merged = {k: dict(v) for k, v in ACCURACY.items()}
    data, _ = load_accuracy_override()
    if not data:
        return merged
    for key, entry in data.items():
        # expect keys like "tepnet_int8" / "tepnet_fp16"
        if "_" not in key:
            continue
        model, quant = key.rsplit("_", 1)
        if model in merged and quant in ("fp16", "int8", "fp32"):
            acc = entry.get("accuracy") or entry.get("top1") or entry.get("acc")
            if isinstance(acc, (int, float)):
                merged[model][quant] = round(float(acc) * (100 if acc <= 1 else 1), 2)
    return merged


def load_benchmarks():
    path = RESULTS_DIR / "benchmark_results.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def load_e2e():
    path = RESULTS_DIR / "e2e_benchmark_results.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@app.route("/")
def index():
    benchmarks = load_benchmarks()
    e2e = load_e2e()
    _, acc_src = load_accuracy_override()
    return render_template(
        "index.html",
        models=MODELS,
        accuracy=merged_accuracy(),
        benchmarks=benchmarks,
        e2e=e2e,
        fault_labels=FAULT_LABELS,
        rk3568_host=RK3568_HOST,
        rk3568_hosts=RK3568_HOSTS,
        x86_host=X86_HOST,
        grafana_url=GRAFANA_URL,
        prometheus_url=PROMETHEUS_URL,
        opcua_port=OPCUA_PORT,
        metrics_port=METRICS_PORT,
        accuracy_source=acc_src or "hardcoded",
    )


@app.route("/api/infra")
def api_infra():
    return jsonify({
        "rk3568_hosts": RK3568_HOSTS,
        "default_rk3568": RK3568_HOST,
        "x86_host": X86_HOST,
        "grafana_url": GRAFANA_URL,
        "prometheus_url": PROMETHEUS_URL,
        "opcua_port": OPCUA_PORT,
        "metrics_port": METRICS_PORT,
    })


@app.route("/api/accuracy")
def api_accuracy():
    data, src = load_accuracy_override()
    return jsonify({"source": src or "hardcoded", "data": merged_accuracy()})


@app.route("/api/models")
def api_models():
    return jsonify(MODELS)


@app.route("/api/benchmarks")
def api_benchmarks():
    return jsonify(load_benchmarks())


@app.route("/api/select", methods=["POST"])
def api_select():
    data = request.json
    model = data.get("model", "patchtst")
    quant = data.get("quantization", "fp16")
    if model not in MODELS:
        return jsonify({"error": f"Unknown model: {model}"}), 400
    if quant not in ("fp16", "int8"):
        return jsonify({"error": f"Unknown quantization: {quant}"}), 400

    acc = ACCURACY.get(model, {}).get(quant, None)
    benchmarks = load_benchmarks()
    key = f"{model}_{quant}"
    bench = benchmarks.get(key, {})
    e2e = load_e2e()
    e2e_data = e2e.get(key, {})

    return jsonify({
        "model": model,
        "quantization": quant,
        "display_name": MODELS[model]["name"],
        "params": MODELS[model]["params"],
        "accuracy": acc,
        "npu_latency_ms": bench.get("npu", {}).get("mean_ms"),
        "npu_p95_ms": bench.get("npu", {}).get("p95_ms"),
        "e2e_mean_ms": e2e_data.get("mean_ms"),
    })


@app.route("/api/deploy", methods=["POST"])
def api_deploy():
    data = request.json
    model = data.get("model", "patchtst")
    quant = data.get("quantization", "fp16")
    if model not in MODELS:
        return jsonify({"error": f"Unknown model: {model}"}), 400
    if quant not in ("fp16", "int8"):
        return jsonify({"error": f"Unknown quantization: {quant}"}), 400

    if not deploy_lock.acquire(blocking=False):
        return jsonify({"error": "Deployment already in progress"}), 409

    host = pick_host()
    deploy_status.update(state="deploying", message=f"Deploying {model} ({quant})...", model=model, quantization=quant, host=host)

    def do_deploy():
        try:
            deploy_status["message"] = "Stopping existing inference client..."
            ssh_stop_native(host)

            # Wait for :8000 to actually free up before launching — otherwise
            # the new process will fail start_http_server with "address in use".
            for _ in range(20):
                rc, out, _ = ssh_run(
                    f"ss -ltnH 'sport = :{METRICS_PORT}' 2>/dev/null",
                    timeout=5, host=host,
                )
                if rc == 0 and not out.strip():
                    break
                time.sleep(0.5)
            else:
                deploy_status.update(state="error",
                                     message=f"Port :{METRICS_PORT} still busy after kill — someone else holds it")
                return

            deploy_status["message"] = f"Launching {model} ({quant}) natively on NPU..."
            rc, out, err = ssh_launch_native(host, model, quant)
            if rc != 0:
                deploy_status.update(state="error", message=f"Launch failed: {err or out}")
                return

            deploy_status["message"] = "Waiting for metrics endpoint to report new model..."
            for i in range(40):  # ~60s — startup loads RKNN + connects OPC UA + fills buffer
                time.sleep(1.5)
                m, q = poll_metrics_model(host)
                if m == model and q == quant:
                    deploy_status.update(state="deployed",
                                         message=f"{model} ({quant}) is running")
                    return
                deploy_status["message"] = (
                    f"Waiting for tep_model_info=({model},{quant}) ... "
                    f"{int((i + 1) * 1.5)}s"
                )

            deploy_status.update(state="error",
                                 message=f"Launched but no matching tep_model_info within 60s — check {BOARD_LOG_FILE}")
        except subprocess.TimeoutExpired:
            deploy_status.update(state="error", message="SSH command timed out")
        except Exception as e:
            deploy_status.update(state="error", message=str(e))
        finally:
            deploy_lock.release()

    threading.Thread(target=do_deploy, daemon=True).start()
    return jsonify({"status": "started", "message": f"Deploying {model} ({quant})..."})


@app.route("/api/deploy/status")
def api_deploy_status():
    return jsonify(deploy_status)


@app.route("/api/board/status")
def api_board_status():
    host = pick_host()
    try:
        fmt = "{{.Names}}\\t{{.Status}}\\t{{.Image}}\\t{{.Ports}}\\t{{.ID}}\\t{{.CreatedAt}}"
        # Use -a so the table shows Exited/Created containers too — otherwise
        # a dead stack looks identical to "nothing ever deployed".
        rc, out, err = ssh_run(f"docker ps -a --format '{fmt}'", timeout=10, host=host)
        if rc != 0:
            return jsonify({"online": False, "host": host, "error": err or "docker ps failed"})

        containers = []
        for line in out.split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            while len(parts) < 6:
                parts.append("")
            containers.append({
                "name": parts[0],
                "status": parts[1],
                "image": parts[2],
                "ports": parts[3],
                "id": parts[4][:12],
                "created": parts[5],
            })

        # Source of truth for "what's running" is the live /metrics endpoint
        # (tep_model_info gauge with labels). The native client isn't a docker
        # container, so docker inspect is irrelevant.
        current_model, current_quant = poll_metrics_model(host)

        # uname + uptime + load
        rc3, sysinfo, _ = ssh_run(
            "echo HOST=$(hostname); echo UPTIME=$(uptime -p 2>/dev/null || uptime); "
            "echo LOAD=$(cat /proc/loadavg | awk '{print $1,$2,$3}'); "
            "echo MEM=$(free -m | awk '/Mem:/{print $3\"/\"$2\" MB\"}')",
            timeout=10, host=host,
        )
        sysinfo_dict = {}
        if rc3 == 0:
            for line in sysinfo.split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    sysinfo_dict[k.strip().lower()] = v.strip()

        return jsonify({
            "online": True,
            "host": host,
            "containers": containers,
            "current_model": current_model,
            "current_quantization": current_quant,
            "system": sysinfo_dict,
        })
    except subprocess.TimeoutExpired:
        return jsonify({"online": False, "host": host, "error": "SSH timeout"})
    except Exception as e:
        return jsonify({"online": False, "host": host, "error": str(e)})


@app.route("/api/board/logs")
def api_board_logs():
    host = pick_host()
    container = request.args.get("container", "coursework-opcua-inference-client-1")
    try:
        lines = max(1, min(500, int(request.args.get("lines", 100))))
    except ValueError:
        lines = 100
    # crude shell-safety: only allow reasonable container name chars
    safe = "".join(ch for ch in container if ch.isalnum() or ch in "-_.")
    if not safe:
        return jsonify({"error": "invalid container"}), 400
    rc, out, err = ssh_run(f"docker logs --tail {lines} {safe} 2>&1", timeout=15, host=host)
    return jsonify({"container": safe, "host": host, "lines": out.split("\n"), "error": None if rc == 0 else err})


@app.route("/api/deploy/stop", methods=["POST"])
def api_deploy_stop():
    host = pick_host()
    if deploy_lock.locked():
        return jsonify({"error": "deployment in progress"}), 409
    rc, out, err = ssh_stop_native(host)
    if rc != 0:
        return jsonify({"error": err or "stop failed"}), 500
    deploy_status.update(state="idle", message="Stopped", model=None, quantization=None, host=host)
    return jsonify({"status": "stopped"})


@app.route("/api/board/metrics")
def api_board_metrics():
    host = pick_host()
    try:
        rc, out, _ = ssh_run(f"curl -s --max-time 3 http://localhost:{METRICS_PORT}/metrics", timeout=10, host=host)
        if rc != 0:
            return jsonify({"available": False})

        metrics = {}
        for line in out.split("\n"):
            if line.startswith("#"):
                continue
            if line.startswith("tep_current_prediction "):
                metrics["current_prediction"] = int(float(line.split()[-1]))
            elif line.startswith("tep_prediction_confidence "):
                metrics["confidence"] = float(line.split()[-1])
            elif line.startswith("tep_buffer_size "):
                metrics["buffer_size"] = int(float(line.split()[-1]))

        if "current_prediction" in metrics:
            metrics["fault_label"] = FAULT_LABELS.get(metrics["current_prediction"], "Unknown")

        return jsonify({"available": True, **metrics})
    except Exception:
        return jsonify({"available": False})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12345, debug=True)
