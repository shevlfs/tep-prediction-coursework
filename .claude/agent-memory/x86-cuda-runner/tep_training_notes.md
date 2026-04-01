---
name: TEP model training notes and gotchas
description: Known issues and fixes for training/exporting TEP fault classification models on the x86 server
type: feedback
---

# TEP Training Workflow Notes

## Python environment for training
Use `~/miniconda3/envs/train_env/bin/python` — this has torch 2.11.0+cu128 which supports sm_120 (RTX 5070 Ti).
Do NOT use the `rknn` env for training — torch 2.4.0+cu121 in that env crashes with "no kernel image" on sm_120.

**Why:** RTX 5070 Ti is Blackwell (sm_120). PyTorch < 2.6 only supports up to sm_90.

## LSTM OOM during test evaluation
The LSTM model (162k params, bidirectional) OOM'd when evaluating all 97k test samples at once on GPU.

**Fix applied:** Patched `train_and_export.py` to use a batched DataLoader for test evaluation (using `batch_size=args.batch_size`). The patch is live on the server.

## ONNX export opset
PyTorch 2.11 exports at opset 18 by default (ignores opset_version=13 arg). This is fine for tepnet/tcn/transformer, but LSTM at opset 18 adds a `layout` attribute that rknn-toolkit2 2.3.2 does not recognize.

**Fix:** Re-export LSTM using the legacy exporter with `dynamo=False, opset_version=12`.

## RKNN conversion: calib-list-path collision
The default `--calib-list-path` in config.py is `small_tep/calibration` which collides with the pre-existing calibration directory.

**Fix:** Always pass explicit `--calib-list-path small_tep/calibration/dataset.txt` and `--calib-dir small_tep/calibration` to the conversion script.

## "Unknown op target: 0" in rknn build
Non-fatal warning — some ops fall back to CPU on rk3568. RKNN files are valid and usable.

## RK3568 NPU "failed to open rknpu module" race condition
On Ubuntu 24.04 (kernel 6.8.0-106-generic), rknnlite 2.3.2 fails to init_runtime without strace because librknnrt.so races its own DRM device scan thread against build_graph.

**Root cause:** librknnrt spawns an internal thread to scan DRM devices (card0/card1/card2). On a fast ARM core without ptrace overhead, build_graph is called before the scan completes, causing RKNN_ERR_FAIL. Under strace -f, ptrace overhead serializes the threads, allowing the scan to complete.

**Fix:** Run benchmarks under `sudo strace -f -e trace=none python ...` — the ptrace overhead (without actually tracing anything) is enough to win the race. No actual strace output is produced (stderr to /dev/null).

Command pattern:
```bash
echo ff7777 | sudo -S -E strace -f -e trace=none /path/to/python script.py 2>/dev/null
```

**Why:** strace with `-f` (follow forks/threads) uses ptrace which affects thread scheduling timing. The DRM scan thread completes before build_graph is called.

**Context:** rknpu driver 0.9.8 (2024-08-28) runs as a DRM device at /dev/dri/card2 + renderD129 (minor 2). No /dev/rknpu character device exists on this Ubuntu mainline kernel. NPU functionality is confirmed working — MobileNetV1 runs at 12.6ms inference.
