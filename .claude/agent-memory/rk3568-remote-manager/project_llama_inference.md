---
name: llama.cpp inference setup on RK3568
description: llama.cpp build location, binary names, runtime flags, and measured inference performance on RK3568
type: project
---

llama.cpp is built at ~/llama.cpp/build/bin/ on the RK3568 (192.168.88.244). Shared libs are in the same directory — always set LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin before running.

**Why:** The .so files are not installed system-wide, so LD_LIBRARY_PATH is mandatory or binaries fail to start.

**How to apply:** Prefix every llama.cpp invocation with `LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin`.

## Available binaries (build b8604-6307ec07d)
- llama-cli — interactive chat (do NOT use for scripted benchmarks; runs in conversation loop)
- llama-simple — non-interactive completion with timing stats, best for benchmarks
- llama-simple-chat — chat variant
- llama-gemma3-cli — deprecated, use llama-mtmd-cli instead
- llama-llava-cli, llama-minicpmv-cli, llama-qwen2vl-cli — multimodal

## Non-interactive flag note
llama-cli does NOT accept --no-conversation; it prints an error and exits 255. Use llama-simple for scripted runs.

## Models on RK3568 ~/models/
- qwen2-0_5b-instruct-q4_0.gguf (337MB)
- gemma-3-1b-it-Q4_K_M.gguf (769MB) — unsloth quantization, Q4_K_M, 6.40 BPW

## Gemma 3 1B IT Q4_K_M benchmark results (4 CPU threads, no GPU offload)
- Model: 999.89M params, gemma3 arch, 26 layers, 1152 embd, 4 heads, 32768 ctx
- File size: 762.49 MiB on disk

### Function-calling prompt (49 prompt tokens, 128 gen tokens)
- Load time: 16788 ms
- Prompt eval: 3.48 t/s (287 ms/token)
- Generation: 0.62 t/s (1622 ms/token)
- Total: 176 tokens in 223 s

### NPU explanation prompt (16 prompt tokens, 64 gen tokens)
- Load time: 10325 ms
- Prompt eval: 2.02 t/s (494 ms/token)
- Generation: 0.23 t/s (4356 ms/token)
- Total: 79 tokens in 285 s

## RAM
- Total: 3.8 GiB physical + 3.8 GiB swap
- Available before inference: ~2.5 GiB
- During/after inference with Gemma 769MB model: used ~1.9 GiB physical, ~30 MiB swap

## HuggingFace access note
- google/gemma-3-* repos are gated (require HF token + accepted terms)
- bartowski/gemma-3-* also requires auth token
- unsloth/gemma-3-1b-it-GGUF is publicly accessible via hf-mirror.com without token
- Download via: wget -L "https://hf-mirror.com/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"
