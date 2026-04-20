#!/bin/bash
# Benchmark LLM inference on RK3568 ARM CPU
# Run ON the RK3568 board after setup_llama_cpp.sh
set -e

LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-cli"
export LD_LIBRARY_PATH="$HOME/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
MODEL_DIR="$HOME/models"
MODEL="${MODEL_DIR}/qwen2-0_5b-instruct-q4_0.gguf"
OUTPUT="$HOME/llm_benchmark_results.txt"

if [ ! -f "$LLAMA_BIN" ]; then
    echo "ERROR: llama.cpp not built. Run setup_llama_cpp.sh first."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    echo "Download it on x86 and copy via scp:"
    echo "  wget https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_0.gguf"
    echo "  scp qwen2-0_5b-instruct-q4_0.gguf 192.168.88.244:~/models/"
    exit 1
fi

echo "LLM Benchmark on RK3568" | tee "$OUTPUT"
echo "Model: $(basename $MODEL)" | tee -a "$OUTPUT"
echo "Date: $(date)" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

echo "System Info" | tee -a "$OUTPUT"
echo "CPU: $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2 | xargs)" | tee -a "$OUTPUT"
echo "Cores: $(nproc)" | tee -a "$OUTPUT"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

echo "Benchmark: 128 token generation" | tee -a "$OUTPUT"
$LLAMA_BIN \
    -m "$MODEL" \
    -p "Explain what a neural processing unit (NPU) is and why it is useful for edge AI inference:" \
    -n 128 \
    --threads $(nproc) \
    --temp 0.7 \
    2>&1 | tee -a "$OUTPUT"

echo "" | tee -a "$OUTPUT"
echo "Benchmark: batch processing (prompt eval)" | tee -a "$OUTPUT"

$LLAMA_BIN \
    -m "$MODEL" \
    -p "The Tennessee Eastman Process is a widely used benchmark for fault detection and diagnosis in chemical engineering. It simulates a realistic chemical plant with multiple operating modes and fault conditions. Machine learning models, particularly convolutional neural networks and recurrent neural networks, have been applied to this problem with varying degrees of success. Neural Processing Units offer hardware acceleration for these models at the edge." \
    -n 32 \
    --threads $(nproc) \
    --temp 0.0 \
    2>&1 | tee -a "$OUTPUT"

echo "" | tee -a "$OUTPUT"
echo "Benchmark complete" | tee -a "$OUTPUT"
echo "Results saved to $OUTPUT"
