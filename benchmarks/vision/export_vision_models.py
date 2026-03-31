#!/usr/bin/env python3
"""Export standard vision models (MobileNetV2, ResNet18) to ONNX.

Run on x86 machine with PyTorch installed:
    python3 export_vision_models.py --output-dir ./vision_onnx
"""
from __future__ import annotations

import argparse
import torch
import torchvision.models as models
from pathlib import Path


VISION_MODELS = {
    "mobilenetv2": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
    "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
}

INPUT_SHAPE = (1, 3, 224, 224)


def export_model(name: str, model: torch.nn.Module, output_dir: Path):
    model.eval()
    dummy = torch.randn(*INPUT_SHAPE)
    out_path = output_dir / name / "model.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes=None,
    )
    print(f"Exported {name} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("vision_onnx"))
    parser.add_argument("--models", nargs="*", choices=list(VISION_MODELS.keys()),
                        default=list(VISION_MODELS.keys()))
    args = parser.parse_args()

    for name in args.models:
        print(f"Loading {name}...")
        model = VISION_MODELS[name]()
        export_model(name, model, args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
