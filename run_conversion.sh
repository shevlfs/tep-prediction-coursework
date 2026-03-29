#!/bin/bash
set -e

cd ~

~/miniconda3/envs/rknn/bin/python convert_onnx_to_rknn.py \
    --onnx-path ~/model.onnx \
    --rknn-path ~/model.rknn \
    --dataset-path ~/small_tep/df.csv \
    --target-platform rk3568 \
    --quantize \
    --window-size 32 \
    --num-calib-samples 256 \
    --log-level INFO
