#!/usr/bin/env python3
"""Evaluate LLM orchestrator decision accuracy on labeled scenarios.

Each scenario has a ground-truth correct action. We measure how often
the LLM makes the right call.

Usage (on x86):
    python3 orchestrator_accuracy_eval.py --ollama-model qwen2.5:7b
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

FAULT_LABELS = {i: f"Fault {i}" if i > 0 else "Normal" for i in range(21)}

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

Respond ONLY with JSON, no other text:
{
  "analysis": "brief analysis of current state",
  "recommendation": "keep_current" or "switch",
  "recommended_model": "model_name_if_switch_or_empty_string",
  "reason": "one sentence explanation",
  "confidence_in_decision": 0.0-1.0,
  "alert_level": "normal" or "warning" or "critical"
}"""

# Ground truth labeled scenarios
LABELED_SCENARIOS = [
    {
        "name": "1_fast_model_high_confidence",
        "state": {
            "current_model": "tepnet", "current_quantization": "fp16",
            "current_prediction": 0, "current_confidence": 0.95,
            "avg_confidence_5m": 0.92, "latency_p95_ms": 2.1,
            "prediction_distribution_5m": {"Normal": 48, "Fault 3": 1},
        },
        "correct_action": "keep_current",
        "correct_models": [],
        "explanation": "High confidence, stable predictions — fast model is fine",
    },
    {
        "name": "2_fast_model_low_confidence",
        "state": {
            "current_model": "tepnet", "current_quantization": "fp16",
            "current_prediction": 9, "current_confidence": 0.38,
            "avg_confidence_5m": 0.45, "latency_p95_ms": 2.3,
            "prediction_distribution_5m": {"Fault 9": 15, "Normal": 12, "Fault 3": 11, "Fault 15": 10},
        },
        "correct_action": "switch",
        "correct_models": ["patchtst", "transformer"],
        "explanation": "Low confidence + oscillating predictions → need more accurate model",
    },
    {
        "name": "3_oscillating_4_classes",
        "state": {
            "current_model": "tcn", "current_quantization": "fp16",
            "current_prediction": 3, "current_confidence": 0.52,
            "avg_confidence_5m": 0.49, "latency_p95_ms": 3.5,
            "prediction_distribution_5m": {"Fault 3": 13, "Fault 9": 12, "Fault 15": 11, "Normal": 9, "Fault 5": 5},
        },
        "correct_action": "switch",
        "correct_models": ["patchtst"],
        "explanation": "Predictions scattered across 5 classes with low confidence → need best model",
    },
    {
        "name": "4_patchtst_stable_can_downgrade",
        "state": {
            "current_model": "patchtst", "current_quantization": "fp16",
            "current_prediction": 5, "current_confidence": 0.97,
            "avg_confidence_5m": 0.95, "latency_p95_ms": 5.8,
            "prediction_distribution_5m": {"Fault 5": 47, "Normal": 1},
        },
        "correct_action": "keep_current",  # also acceptable: switch to tcn/tepnet
        "correct_models": ["tcn", "tepnet"],  # acceptable switch targets
        "explanation": "Very stable high confidence — could keep or downgrade to save latency, both OK",
    },
    {
        "name": "5_patchtst_confidence_dropping",
        "state": {
            "current_model": "patchtst", "current_quantization": "fp16",
            "current_prediction": 11, "current_confidence": 0.51,
            "avg_confidence_5m": 0.55, "latency_p95_ms": 6.2,
            "prediction_distribution_5m": {"Fault 11": 18, "Fault 12": 14, "Fault 13": 10, "Normal": 6},
        },
        "correct_action": "keep_current",
        "correct_models": [],
        "explanation": "Already on most accurate model — switching would make it worse. This is a hard case.",
    },
    {
        "name": "6_lstm_should_always_switch",
        "state": {
            "current_model": "lstm", "current_quantization": "fp16",
            "current_prediction": 7, "current_confidence": 0.72,
            "avg_confidence_5m": 0.68, "latency_p95_ms": 14.5,
            "prediction_distribution_5m": {"Fault 7": 30, "Fault 8": 10, "Normal": 8},
        },
        "correct_action": "switch",
        "correct_models": ["patchtst", "transformer", "tcn", "tepnet"],
        "explanation": "LSTM is slow and inaccurate — any other model is better",
    },
    {
        "name": "7_tepnet_moderate_confidence",
        "state": {
            "current_model": "tepnet", "current_quantization": "fp16",
            "current_prediction": 1, "current_confidence": 0.78,
            "avg_confidence_5m": 0.75, "latency_p95_ms": 2.0,
            "prediction_distribution_5m": {"Fault 1": 38, "Fault 2": 8, "Normal": 2},
        },
        "correct_action": "keep_current",
        "correct_models": ["tcn"],
        "explanation": "Moderate-high confidence, mostly stable — tepnet OK, tcn would be marginal upgrade",
    },
    {
        "name": "8_transformer_good_but_slow",
        "state": {
            "current_model": "transformer", "current_quantization": "fp16",
            "current_prediction": 4, "current_confidence": 0.91,
            "avg_confidence_5m": 0.88, "latency_p95_ms": 4.1,
            "prediction_distribution_5m": {"Fault 4": 44, "Fault 5": 3, "Normal": 1},
        },
        "correct_action": "keep_current",
        "correct_models": ["tcn", "tepnet"],
        "explanation": "Good accuracy and confidence — keep or optionally downgrade",
    },
    {
        "name": "9_sudden_fault_onset",
        "state": {
            "current_model": "tepnet", "current_quantization": "fp16",
            "current_prediction": 14, "current_confidence": 0.33,
            "avg_confidence_5m": 0.82, "latency_p95_ms": 2.1,
            "prediction_distribution_5m": {"Normal": 35, "Fault 14": 8, "Fault 13": 5},
        },
        "correct_action": "switch",
        "correct_models": ["patchtst", "transformer"],
        "explanation": "Was normal (high avg conf) but suddenly low conf on fault — need accurate model to confirm",
    },
    {
        "name": "10_all_normal_fast_model",
        "state": {
            "current_model": "tcn", "current_quantization": "fp16",
            "current_prediction": 0, "current_confidence": 0.99,
            "avg_confidence_5m": 0.97, "latency_p95_ms": 3.2,
            "prediction_distribution_5m": {"Normal": 50},
        },
        "correct_action": "keep_current",
        "correct_models": ["tepnet"],
        "explanation": "Perfect normal operation, very high confidence — fast model is ideal",
    },
    {
        "name": "11_rare_fault_low_support",
        "state": {
            "current_model": "tcn", "current_quantization": "fp16",
            "current_prediction": 20, "current_confidence": 0.44,
            "avg_confidence_5m": 0.51, "latency_p95_ms": 3.4,
            "prediction_distribution_5m": {"Fault 20": 12, "Fault 19": 10, "Fault 16": 9, "Normal": 8, "Fault 17": 6},
        },
        "correct_action": "switch",
        "correct_models": ["patchtst"],
        "explanation": "Rare faults (16-20) with low confidence and high spread — need best model",
    },
    {
        "name": "12_binary_oscillation",
        "state": {
            "current_model": "tepnet", "current_quantization": "fp16",
            "current_prediction": 3, "current_confidence": 0.56,
            "avg_confidence_5m": 0.58, "latency_p95_ms": 2.2,
            "prediction_distribution_5m": {"Fault 3": 24, "Normal": 22, "Fault 9": 2},
        },
        "correct_action": "switch",
        "correct_models": ["patchtst", "transformer", "tcn"],
        "explanation": "Oscillating between normal and fault 3 — unclear boundary, need better model",
    },
    {
        "name": "13_patchtst_perfect",
        "state": {
            "current_model": "patchtst", "current_quantization": "fp16",
            "current_prediction": 2, "current_confidence": 0.98,
            "avg_confidence_5m": 0.96, "latency_p95_ms": 5.5,
            "prediction_distribution_5m": {"Fault 2": 49, "Normal": 1},
        },
        "correct_action": "keep_current",
        "correct_models": ["tcn", "tepnet"],
        "explanation": "Near-perfect predictions — keep or downgrade to faster model both fine",
    },
    {
        "name": "14_tepnet_gradual_drift",
        "state": {
            "current_model": "tepnet", "current_quantization": "fp16",
            "current_prediction": 13, "current_confidence": 0.61,
            "avg_confidence_5m": 0.72, "latency_p95_ms": 2.1,
            "prediction_distribution_5m": {"Fault 13": 22, "Fault 3": 12, "Normal": 8, "Fault 12": 6},
        },
        "correct_action": "switch",
        "correct_models": ["patchtst", "transformer"],
        "explanation": "Confidence drifting down (0.72→0.61), spreading predictions — need escalation",
    },
    {
        "name": "15_tcn_acceptable_performance",
        "state": {
            "current_model": "tcn", "current_quantization": "fp16",
            "current_prediction": 6, "current_confidence": 0.84,
            "avg_confidence_5m": 0.81, "latency_p95_ms": 3.3,
            "prediction_distribution_5m": {"Fault 6": 40, "Fault 7": 5, "Normal": 3},
        },
        "correct_action": "keep_current",
        "correct_models": [],
        "explanation": "Good confidence on TCN, no reason to change",
    },
]


def call_ollama(ollama_url: str, model: str, prompt: str, system: str) -> tuple[str, float]:
    payload = json.dumps({
        "model": model, "prompt": prompt, "system": system,
        "stream": False, "options": {"temperature": 0.1, "num_predict": 300},
    }).encode()
    req = urllib.request.Request(
        f"{ollama_url}/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result.get("response", ""), (time.perf_counter() - t0) * 1000


def evaluate_decision(decision: dict, scenario: dict) -> dict:
    """Check if LLM decision matches ground truth."""
    correct_action = scenario["correct_action"]
    correct_models = scenario["correct_models"]
    llm_action = decision.get("recommendation", "")
    llm_model = decision.get("recommended_model", "")

    # Primary check: action matches
    action_correct = llm_action == correct_action

    # For "keep_current", switching to an acceptable model is also OK
    if correct_action == "keep_current" and llm_action == "switch":
        if llm_model in correct_models:
            action_correct = True  # acceptable alternative

    # For "switch", check if recommended model is in correct list
    model_correct = True
    if llm_action == "switch" and correct_action == "switch":
        model_correct = llm_model in correct_models

    # Overall score
    fully_correct = action_correct and model_correct

    return {
        "action_correct": action_correct,
        "model_correct": model_correct,
        "fully_correct": fully_correct,
        "llm_action": llm_action,
        "llm_model": llm_model,
        "expected_action": correct_action,
        "expected_models": correct_models,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--output", type=Path, default=Path("orchestrator_accuracy_results.json"))
    args = parser.parse_args()

    print(f"Evaluating LLM orchestrator accuracy: {args.ollama_model}")
    print(f"Scenarios: {len(LABELED_SCENARIOS)}\n")

    results = []
    correct_count = 0
    action_correct_count = 0

    for scenario in LABELED_SCENARIOS:
        state = scenario["state"]
        state_text = f"""Current system state:

Current model: {state['current_model']} ({state['current_quantization']})
Current prediction: {FAULT_LABELS.get(state.get('current_prediction', -1), 'unknown')} (class {state.get('current_prediction', '?')})
Current confidence: {state.get('current_confidence', 'N/A')}
Average confidence (5min): {state.get('avg_confidence_5m', 'N/A')}
Latency P95: {state.get('latency_p95_ms', 'N/A')} ms
Prediction distribution (last 5min): {json.dumps(state.get('prediction_distribution_5m', {}), indent=2)}

Based on this data, analyze the situation and recommend whether to keep the current model or switch."""

        response, latency_ms = call_ollama(args.ollama_url, args.ollama_model, state_text, SYSTEM_PROMPT)

        decision = None
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                decision = json.loads(response[start:end])
        except json.JSONDecodeError:
            decision = {"recommendation": "parse_error", "recommended_model": ""}

        if decision and "recommendation" in decision:
            eval_result = evaluate_decision(decision, scenario)
        else:
            eval_result = {"fully_correct": False, "action_correct": False, "model_correct": False,
                           "llm_action": "parse_error", "llm_model": "", "expected_action": scenario["correct_action"],
                           "expected_models": scenario["correct_models"]}

        if eval_result["fully_correct"]:
            correct_count += 1
        if eval_result["action_correct"]:
            action_correct_count += 1

        status = "OK" if eval_result["fully_correct"] else ("~OK action" if eval_result["action_correct"] else "WRONG")
        print(f"  [{status:>10}] {scenario['name']:<40} LLM: {eval_result['llm_action']:<15} "
              f"model={eval_result['llm_model']:<12} expected={scenario['correct_action']:<15} "
              f"({latency_ms:.0f}ms)")

        results.append({
            "scenario": scenario["name"],
            "ground_truth": scenario["explanation"],
            "llm_response": response,
            "decision": decision,
            "evaluation": eval_result,
            "latency_ms": round(latency_ms, 1),
        })

    # Save
    output_data = {
        "model": args.ollama_model,
        "total_scenarios": len(LABELED_SCENARIOS),
        "fully_correct": correct_count,
        "action_correct": action_correct_count,
        "accuracy_full": round(correct_count / len(LABELED_SCENARIOS), 4),
        "accuracy_action": round(action_correct_count / len(LABELED_SCENARIOS), 4),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"ACCURACY RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.ollama_model}")
    print(f"Full accuracy (action + model): {correct_count}/{len(LABELED_SCENARIOS)} = {correct_count/len(LABELED_SCENARIOS):.1%}")
    print(f"Action accuracy (keep/switch):  {action_correct_count}/{len(LABELED_SCENARIOS)} = {action_correct_count/len(LABELED_SCENARIOS):.1%}")
    avg_lat = sum(r["latency_ms"] for r in results) / len(results)
    print(f"Avg LLM latency: {avg_lat:.0f}ms")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
