---
name: RK3568 NPU Setup Requirements
description: Critical setup steps needed before running RKNN benchmarks on the RK3568 board (192.168.88.244)
type: reference
---

## RK3568 NPU Runtime Requirements

1. **DMA heap permissions**: After every reboot, `/dev/dma_heap/system` and `/dev/dma_heap/reserved` revert to root-only (`crw-------`). Must run:
   ```
   sudo chmod 666 /dev/dma_heap/system /dev/dma_heap/reserved
   ```

2. **core_mask not supported on RK3568**: The `init_runtime(core_mask=RKNNLite.NPU_CORE_0)` call fails with "core_mask is only supported by RK3588/RK3576". Use `init_runtime()` without arguments on RK3568.

3. **Python venv required**: System python is externally-managed (Debian policy). A venv exists at `~/bench_env/` with numpy, pandas, scikit-learn, rknn-toolkit-lite2 installed.

4. **Must run as root (sudo -S -E)**: Without sudo, the "failed to open rknpu module" error occurs because `/dev/dri/card2` (the RKNPU DRM device) needs root access or proper udev rules.

5. **RKNPU driver**: Kernel module `rknpu.ko` v0.9.8 at `/lib/modules/6.8.0-106-generic/extra/rknpu.ko`. Runtime `librknnrt.so` v2.3.2 at `/usr/lib/librknnrt.so`.

6. **llama.cpp**: Binary at `~/llama.cpp/build/bin/llama-cli`. Requires `LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin`. Model at `~/models/qwen2-0_5b-instruct-q4_0.gguf`. Board has only 3.8GB RAM -- running multiple llama-cli instances simultaneously causes OOM and reboot.

7. **llama-cli is interactive**: Use `printf "/exit\n" | timeout 120 llama-cli ...` to avoid hanging in interactive mode. `--no-conversation` flag is not supported; it suggests using `llama-completion` instead (not available in this build).
