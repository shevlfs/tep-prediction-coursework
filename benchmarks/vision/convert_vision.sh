#!/bin/bash
# Convert vision ONNX models to RKNN (INT8 + FP16)
# Run on x86 machine
set -e

CONDA_ENV="${CONDA_ENV:-rknn}"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"

ONNX_BASE="${1:-$HOME/vision_onnx}"
RKNN_BASE="${2:-$HOME/vision_rknn}"

MODELS=("mobilenetv2" "resnet18")

CALIB_SCRIPT=$(cat <<'PYEOF'
import numpy as np
from pathlib import Path
import sys

calib_dir = Path(sys.argv[1])
calib_dir.mkdir(parents=True, exist_ok=True)

num_samples = 64
for i in range(num_samples):
    sample = np.random.randn(1, 3, 224, 224).astype(np.float32)
    np.save(calib_dir / f"calib_{i:04d}.npy", sample)

list_path = calib_dir / "dataset.txt"
with open(list_path, "w") as f:
    for i in range(num_samples):
        f.write(f"{(calib_dir / f'calib_{i:04d}.npy').resolve()}\n")

print(f"Generated {num_samples} calibration samples in {calib_dir}")
PYEOF
)

CALIB_DIR="$RKNN_BASE/calibration"
$PYTHON -c "$CALIB_SCRIPT" "$CALIB_DIR"

for model in "${MODELS[@]}"; do
    ONNX_PATH="$ONNX_BASE/$model/model.onnx"

    if [ ! -f "$ONNX_PATH" ]; then
        echo "[SKIP] $model: ONNX not found at $ONNX_PATH"
        continue
    fi

    echo "[CONVERT] $model fp16"
    $PYTHON -c "
from rknn.api import RKNN
rknn = RKNN()
rknn.config(target_platform='rk3568')
rknn.load_onnx(model='$ONNX_PATH')
rknn.build(do_quantization=False)
import os; os.makedirs('$RKNN_BASE/$model', exist_ok=True)
rknn.export_rknn('$RKNN_BASE/$model/model_fp16.rknn')
rknn.release()
print('[DONE] $model fp16')
"

    echo "[CONVERT] $model int8"
    $PYTHON -c "
from rknn.api import RKNN
rknn = RKNN()
rknn.config(target_platform='rk3568')
rknn.load_onnx(model='$ONNX_PATH')
rknn.build(do_quantization=True, dataset='$CALIB_DIR/dataset.txt')
import os; os.makedirs('$RKNN_BASE/$model', exist_ok=True)
rknn.export_rknn('$RKNN_BASE/$model/model_int8.rknn')
rknn.release()
print('[DONE] $model int8')
"
done

echo "All vision model conversions complete"
