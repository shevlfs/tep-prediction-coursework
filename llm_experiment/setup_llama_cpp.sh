#!/bin/bash
# Build llama.cpp for aarch64 on RK3568
# Run this ON the RK3568 board
set -e

BUILD_DIR="$HOME/llama.cpp"

echo "Installing build dependencies"
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget

if [ -d "$BUILD_DIR" ]; then
    echo "llama.cpp already cloned, pulling latest..."
    cd "$BUILD_DIR" && git pull
else
    echo "Cloning llama.cpp"
    git clone https://github.com/ggerganov/llama.cpp.git "$BUILD_DIR"
fi

cd "$BUILD_DIR"

echo "Building llama.cpp"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

echo ""
echo "Build complete"
echo "Binary: $BUILD_DIR/build/bin/llama-cli"
echo ""
echo "Next steps:"
echo "  1. Download a GGUF model on x86:"
echo "     wget https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_0.gguf"
echo "  2. Copy to RK3568:"
echo "     scp qwen2-0_5b-instruct-q4_0.gguf :~/models/"
echo "  3. Run benchmark:"
echo "     ./run_llm_benchmark.sh"
