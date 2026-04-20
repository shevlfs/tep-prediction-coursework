#!/bin/bash
set -e

CONDA_ENV="${CONDA_ENV:-rknn}"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$HOME/convert_onnx_to_rknn.py"

MODELS=("tepnet" "tcn" "lstm" "transformer")
ONNX_BASE="$HOME/onnx"
RKNN_BASE="$HOME/rknn"
DATASET="$HOME/small_tep/df.csv"

for model in "${MODELS[@]}"; do
    ONNX_PATH="$ONNX_BASE/$model/model.onnx"

    if [ ! -f "$ONNX_PATH" ]; then
        echo "SKIP $model: ONNX not found at $ONNX_PATH"
        continue
    fi

    echo "CONVERT $model — INT8 (quantized)"
    $PYTHON "$SCRIPT" \
        --onnx-path "$ONNX_PATH" \
        --rknn-path "$RKNN_BASE/$model/model_int8.rknn" \
        --dataset-path "$DATASET" \
        --target-platform rk3568 \
        --quantize \
        --window-size 32 \
        --num-calib-samples 256 \
        --log-level INFO

    echo "CONVERT $model — FP16 (no quantization)"
    $PYTHON "$SCRIPT" \
        --onnx-path "$ONNX_PATH" \
        --rknn-path "$RKNN_BASE/$model/model_fp16.rknn" \
        --dataset-path "$DATASET" \
        --target-platform rk3568 \
        --no-quantize \
        --log-level INFO

    echo "DONE $model"
    echo ""
done

echo "All conversions complete."
echo "Models saved to: $RKNN_BASE/{model_name}/model_{int8,fp16}.rknn"
