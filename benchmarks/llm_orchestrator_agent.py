#!/usr/bin/env python3
"""LLM-powered orchestrator agent for adaptive model management on RK3568 boards.

Runs on x86 server. Uses Ollama (local LLM) to:
1. Monitor inference metrics from Prometheus
2. Analyze prediction patterns and confidence trends
3. Decide whether to switch models on edge boards
4. Generate human-readable explanations of decisions

This demonstrates an AI-agent pipeline where an LLM on GPU orchestrates
NPU inference on edge devices via industrial protocols.

Usage (on x86):
    python3 llm_orchestrator_agent.py --prometheus-url http://localhost:9090 \
        --ollama-url http://localhost:11434 --board-url http://192.168.88.244:8001 \
        --output orchestrator_log.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path


FAULT_LABELS = {i: f"Fault {i}" if i > 0 else "Normal" for i in range(21)}

# Model characteristics for the agent's context
MODEL_PROFILES = {
    "tepnet": {"latency_ms": 1.9, "accuracy": 0.7035, "type": "1D CNN", "strength": "fastest"},
    "tcn": {"latency_ms": 2.7, "accuracy": 0.7540, "type": "Temporal CNN", "strength": "good balance"},
    "patchtst": {"latency_ms": 5.2, "accuracy": 0.7970, "type": "Patch Transformer", "strength": "most accurate"},
    "transformer": {"latency_ms": 3.4, "accuracy": 0.7800, "type": "Transformer", "strength": "accurate"},
    "lstm": {"latency_ms": 12.8, "accuracy": 0.7045, "type": "BiLSTM", "strength": "temporal patterns"},
}


def query_prometheus(prom_url: str, query: str) -> dict | None:
    """Execute a PromQL query."""
    try:
        url = f"{prom_url}/api/v1/query?query={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def get_system_state(prom_url: str) -> dict:
    """Collect current system metrics from Prometheus."""
    import urllib.parse

    state = {}

    # Current prediction confidence
    r = query_prometheus(prom_url, "tep_prediction_confidence")
    if r and r.get("data", {}).get("result"):
        state["current_confidence"] = float(r["data"]["result"][0]["value"][1])

    # Current prediction class
    r = query_prometheus(prom_url, "tep_current_prediction")
    if r and r.get("data", {}).get("result"):
        state["current_prediction"] = int(float(r["data"]["result"][0]["value"][1]))

    # Average confidence over last 5 minutes
    r = query_prometheus(prom_url, "avg_over_time(tep_prediction_confidence[5m])")
    if r and r.get("data", {}).get("result"):
        state["avg_confidence_5m"] = float(r["data"]["result"][0]["value"][1])

    # Inference rate
    r = query_prometheus(prom_url, "rate(tep_inference_total[5m])")
    if r and r.get("data", {}).get("result"):
        total_rate = sum(float(x["value"][1]) for x in r["data"]["result"])
        state["inference_rate_per_sec"] = round(total_rate, 2)

    # Prediction distribution over last 5 minutes
    r = query_prometheus(prom_url, "increase(tep_inference_total[5m])")
    if r and r.get("data", {}).get("result"):
        dist = {}
        for item in r["data"]["result"]:
            cls = item["metric"].get("predicted_class", "unknown")
            count = float(item["value"][1])
            if count > 0:
                dist[cls] = round(count)
        state["prediction_distribution_5m"] = dist

    # Inference latency p95
    r = query_prometheus(prom_url, "histogram_quantile(0.95, rate(tep_inference_latency_seconds_bucket[5m]))")
    if r and r.get("data", {}).get("result"):
        state["latency_p95_ms"] = round(float(r["data"]["result"][0]["value"][1]) * 1000, 2)

    # Model info
    r = query_prometheus(prom_url, "tep_model_info")
    if r and r.get("data", {}).get("result"):
        for item in r["data"]["result"]:
            state["current_model"] = item["metric"].get("model_name", "unknown")
            state["current_quantization"] = item["metric"].get("quantization", "unknown")

    # System metrics
    r = query_prometheus(prom_url, '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)')
    if r and r.get("data", {}).get("result"):
        state["cpu_usage_pct"] = round(float(r["data"]["result"][0]["value"][1]), 1)

    r = query_prometheus(prom_url, "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100")
    if r and r.get("data", {}).get("result"):
        state["memory_available_pct"] = round(float(r["data"]["result"][0]["value"][1]), 1)

    return state


def call_ollama(ollama_url: str, model: str, prompt: str, system: str = "") -> str:
    """Call Ollama API for LLM inference."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        }
    }).encode()

    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    latency = (time.perf_counter() - t0) * 1000

    return result.get("response", ""), latency


SYSTEM_PROMPT = """You are an AI orchestrator managing fault diagnosis models on an RK3568 edge NPU board.
You monitor metrics from Prometheus and decide if the current model should be switched.

Available models (on NPU, FP16 quantization):
- tepnet: 1.9ms latency, 70.35% accuracy — fastest, simple 1D CNN
- tcn: 2.7ms, 75.40% — good balance of speed and accuracy
- transformer: 3.4ms, 78.00% — accurate but slower
- patchtst: 5.2ms, 79.70% — most accurate but slowest
- lstm: 12.8ms, 70.45% — slow and low accuracy, avoid

Decision rules:
- If confidence is consistently high (>0.8) and predictions are stable → use fast model (tepnet/tcn)
- If confidence drops below 0.6 or predictions oscillate → switch to accurate model (patchtst)
- If latency is critical → prefer tepnet
- If accuracy is critical → prefer patchtst
- Never recommend lstm unless specifically needed for temporal pattern analysis

Respond in JSON format:
{
  "analysis": "brief analysis of current state",
  "recommendation": "keep_current" or "switch",
  "recommended_model": "model_name" (if switch),
  "reason": "one sentence explanation",
  "confidence_in_decision": 0.0-1.0,
  "alert_level": "normal" or "warning" or "critical"
}"""


def run_orchestrator_cycle(prom_url: str, ollama_url: str, ollama_model: str) -> dict:
    """Run one cycle of the orchestrator."""
    # Collect state
    state = get_system_state(prom_url)
    timestamp = datetime.now().isoformat()

    # Format state for LLM
    state_text = f"""Current system state at {timestamp}:

Current model: {state.get('current_model', 'unknown')} ({state.get('current_quantization', 'unknown')})
Current prediction: {FAULT_LABELS.get(state.get('current_prediction', -1), 'unknown')} (class {state.get('current_prediction', '?')})
Current confidence: {state.get('current_confidence', 'N/A')}
Average confidence (5min): {state.get('avg_confidence_5m', 'N/A')}
Inference rate: {state.get('inference_rate_per_sec', 'N/A')} inferences/sec
Latency P95: {state.get('latency_p95_ms', 'N/A')} ms
CPU usage: {state.get('cpu_usage_pct', 'N/A')}%
Memory available: {state.get('memory_available_pct', 'N/A')}%
Prediction distribution (last 5min): {json.dumps(state.get('prediction_distribution_5m', {}), indent=2)}

Based on this data, analyze the situation and recommend whether to keep the current model or switch."""

    # Call LLM
    llm_response, llm_latency_ms = call_ollama(ollama_url, ollama_model, state_text, SYSTEM_PROMPT)

    # Try to parse JSON from response
    decision = None
    try:
        # Find JSON in response
        start = llm_response.find("{")
        end = llm_response.rfind("}") + 1
        if start >= 0 and end > start:
            decision = json.loads(llm_response[start:end])
    except json.JSONDecodeError:
        decision = {"raw_response": llm_response, "parse_error": True}

    return {
        "timestamp": timestamp,
        "system_state": state,
        "llm_response": llm_response,
        "decision": decision,
        "llm_latency_ms": round(llm_latency_ms, 1),
        "llm_model": ollama_model,
    }


def run_benchmark_mode(ollama_url: str, ollama_model: str, num_scenarios: int = 5) -> list[dict]:
    """Run benchmark with synthetic scenarios (no Prometheus needed)."""
    scenarios = [
        {
            "name": "stable_normal_operation",
            "current_model": "tepnet", "current_quantization": "fp16",
            "current_prediction": 0, "current_confidence": 0.92,
            "avg_confidence_5m": 0.89, "inference_rate_per_sec": 5.2,
            "latency_p95_ms": 2.1, "cpu_usage_pct": 15.3, "memory_available_pct": 82.1,
            "prediction_distribution_5m": {"Normal": 45, "Fault 3": 2, "Fault 9": 1},
        },
        {
            "name": "confidence_dropping",
            "current_model": "tepnet", "current_quantization": "fp16",
            "current_prediction": 9, "current_confidence": 0.41,
            "avg_confidence_5m": 0.53, "inference_rate_per_sec": 5.1,
            "latency_p95_ms": 2.3, "cpu_usage_pct": 16.1, "memory_available_pct": 81.5,
            "prediction_distribution_5m": {"Fault 9": 18, "Normal": 12, "Fault 3": 10, "Fault 15": 8},
        },
        {
            "name": "oscillating_predictions",
            "current_model": "tcn", "current_quantization": "fp16",
            "current_prediction": 3, "current_confidence": 0.55,
            "avg_confidence_5m": 0.48, "inference_rate_per_sec": 5.0,
            "latency_p95_ms": 3.8, "cpu_usage_pct": 18.2, "memory_available_pct": 80.3,
            "prediction_distribution_5m": {"Fault 3": 12, "Fault 9": 11, "Fault 15": 10, "Normal": 8, "Fault 5": 7},
        },
        {
            "name": "accurate_model_stable",
            "current_model": "patchtst", "current_quantization": "fp16",
            "current_prediction": 5, "current_confidence": 0.94,
            "avg_confidence_5m": 0.91, "inference_rate_per_sec": 3.8,
            "latency_p95_ms": 5.8, "cpu_usage_pct": 22.1, "memory_available_pct": 79.8,
            "prediction_distribution_5m": {"Fault 5": 42, "Fault 4": 3, "Normal": 3},
        },
        {
            "name": "high_load_latency_pressure",
            "current_model": "patchtst", "current_quantization": "fp16",
            "current_prediction": 11, "current_confidence": 0.87,
            "avg_confidence_5m": 0.85, "inference_rate_per_sec": 3.5,
            "latency_p95_ms": 8.2, "cpu_usage_pct": 78.5, "memory_available_pct": 45.2,
            "prediction_distribution_5m": {"Fault 11": 38, "Fault 12": 5, "Normal": 2},
        },
    ]

    results = []
    for scenario in scenarios[:num_scenarios]:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*60}")

        state = {k: v for k, v in scenario.items() if k != "name"}
        timestamp = datetime.now().isoformat()

        state_text = f"""Current system state at {timestamp}:

Current model: {state['current_model']} ({state['current_quantization']})
Current prediction: {FAULT_LABELS.get(state['current_prediction'], 'unknown')} (class {state['current_prediction']})
Current confidence: {state['current_confidence']}
Average confidence (5min): {state['avg_confidence_5m']}
Inference rate: {state['inference_rate_per_sec']} inferences/sec
Latency P95: {state['latency_p95_ms']} ms
CPU usage: {state['cpu_usage_pct']}%
Memory available: {state['memory_available_pct']}%
Prediction distribution (last 5min): {json.dumps(state.get('prediction_distribution_5m', {}), indent=2)}

Based on this data, analyze the situation and recommend whether to keep the current model or switch."""

        llm_response, llm_latency_ms = call_ollama(ollama_url, ollama_model, state_text, SYSTEM_PROMPT)

        decision = None
        try:
            start = llm_response.find("{")
            end = llm_response.rfind("}") + 1
            if start >= 0 and end > start:
                decision = json.loads(llm_response[start:end])
        except json.JSONDecodeError:
            decision = {"raw_response": llm_response, "parse_error": True}

        result = {
            "scenario": scenario["name"],
            "timestamp": timestamp,
            "system_state": state,
            "llm_response": llm_response,
            "decision": decision,
            "llm_latency_ms": round(llm_latency_ms, 1),
            "llm_model": ollama_model,
        }
        results.append(result)

        if decision and not decision.get("parse_error"):
            print(f"  Analysis: {decision.get('analysis', 'N/A')}")
            print(f"  Recommendation: {decision.get('recommendation', 'N/A')}")
            if decision.get("recommendation") == "switch":
                print(f"  Switch to: {decision.get('recommended_model', 'N/A')}")
            print(f"  Reason: {decision.get('reason', 'N/A')}")
            print(f"  Alert: {decision.get('alert_level', 'N/A')}")
        else:
            print(f"  Raw response: {llm_response[:200]}...")

        print(f"  LLM latency: {llm_latency_ms:.0f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="LLM orchestrator agent")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--output", type=Path, default=Path("orchestrator_log.json"))
    parser.add_argument("--mode", choices=["live", "benchmark"], default="benchmark",
                        help="live: query real Prometheus; benchmark: use synthetic scenarios")
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between cycles in live mode")
    parser.add_argument("--cycles", type=int, default=5,
                        help="Number of cycles to run")
    args = parser.parse_args()

    print(f"LLM Orchestrator Agent")
    print(f"  Ollama: {args.ollama_url} ({args.ollama_model})")
    print(f"  Mode: {args.mode}")

    if args.mode == "benchmark":
        results = run_benchmark_mode(args.ollama_url, args.ollama_model, args.cycles)
    else:
        results = []
        for cycle in range(args.cycles):
            print(f"\n--- Cycle {cycle+1}/{args.cycles} ---")
            result = run_orchestrator_cycle(args.prometheus_url, args.ollama_url, args.ollama_model)
            results.append(result)

            if result["decision"] and not result["decision"].get("parse_error"):
                d = result["decision"]
                print(f"  Analysis: {d.get('analysis', 'N/A')}")
                print(f"  Recommendation: {d.get('recommendation', 'N/A')}")
                print(f"  Alert: {d.get('alert_level', 'N/A')}")
                print(f"  LLM latency: {result['llm_latency_ms']:.0f}ms")

            if cycle < args.cycles - 1:
                print(f"  Waiting {args.interval}s...")
                time.sleep(args.interval)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print("ORCHESTRATOR BENCHMARK SUMMARY")
    print(f"{'='*60}")
    latencies = [r["llm_latency_ms"] for r in results]
    print(f"LLM model: {args.ollama_model}")
    print(f"Scenarios tested: {len(results)}")
    print(f"LLM latency: mean={sum(latencies)/len(latencies):.0f}ms, "
          f"min={min(latencies):.0f}ms, max={max(latencies):.0f}ms")

    for r in results:
        d = r.get("decision", {})
        scenario = r.get("scenario", "live")
        rec = d.get("recommendation", "?")
        model = d.get("recommended_model", "-")
        alert = d.get("alert_level", "?")
        print(f"  {scenario:<35} → {rec:<15} model={model:<12} alert={alert}")


if __name__ == "__main__":
    main()
