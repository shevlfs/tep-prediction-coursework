from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _resolve_dataset_path(path_str: str, dataset_dir: Path) -> Path:
    candidate = _resolve_path(path_str)
    if candidate.exists():
        return candidate

    if path_str.startswith("/small_tep/"):
        alt = _resolve_path(path_str.lstrip("/"), base_dir=Path.cwd())
        if alt.exists():
            return alt

    fallback = (dataset_dir / "df.csv").resolve()
    if fallback.exists():
        return fallback

    return candidate


@dataclass(frozen=True)
class PathConfig:
    dataset_dir: Path
    dataset_path: Path
    onnx_dir: Path
    onnx_path: Path
    rknn_dir: Path
    rknn_path: Path
    calib_dir: Path
    calib_list_path: Path


@dataclass(frozen=True)
class ConversionConfig:
    paths: PathConfig
    target_platform: str
    quantize: bool
    feature_columns: Optional[List[str]]
    num_calib_samples: int
    window_size: Optional[int]
    window_stride: int
    max_rows: Optional[int]
    verbose_rknn: bool
    log_level: str

    @staticmethod
    def _parse_optional_int(value: Optional[str]) -> Optional[int]:
        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _parse_feature_columns(value: Optional[str]) -> Optional[List[str]]:
        if not value:
            return None
        columns = [col.strip() for col in value.split(",") if col.strip()]
        return columns or None

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        default_dataset_dir = _env("DATASET_DIR", "small_tep")
        default_onnx_dir = _env("ONNX_DIR", "onnx")
        default_rknn_dir = _env("RKNN_DIR", "rknn")

        default_dataset_path = _env("DATASET_PATH", "small_tep/df.csv")
        default_onnx_path = _env("ONNX_PATH", "onnx/model.onnx")
        default_rknn_path = _env("RKNN_PATH", "rknn/model.rknn")

        default_calib_dir = _env("CALIB_DIR", "small_tep/calibration")
        default_calib_list_path = _env("CALIB_LIST_PATH", "small_tep/calibration")

        default_target_platform = _env("TARGET_PLATFORM", "rk3568")
        default_feature_columns = _env("FEATURE_COLUMNS")
        default_num_calib_samples = _env_int("NUM_CALIB_SAMPLES", 256)
        default_window_size = cls._parse_optional_int(_env("WINDOW_SIZE"))
        default_window_stride = _env_int("WINDOW_STRIDE", 8)
        default_max_rows = cls._parse_optional_int(_env("MAX_ROWS"))
        default_quantize = _env_bool("DO_QUANTIZATION", True)
        default_log_level = _env("LOG_LEVEL", "INFO")

        parser = argparse.ArgumentParser(
            description="Convert ONNX to RKNN with optional INT8 quantization and CSV-based calibration"
        )
        parser.add_argument("--dataset-dir", default=default_dataset_dir, help="Directory with CSV dataset")
        parser.add_argument("--dataset-path", default=default_dataset_path, help="Path to CSV file (e.g. df.csv)")

        parser.add_argument("--onnx-dir", default=default_onnx_dir, help="Directory with ONNX model")
        parser.add_argument("--onnx-path", default=default_onnx_path, help="Path to ONNX model")

        parser.add_argument("--rknn-dir", default=default_rknn_dir, help="Output directory for RKNN model")
        parser.add_argument("--rknn-path", default=default_rknn_path, help="Output RKNN file path")

        parser.add_argument("--calib-dir", default=default_calib_dir, help="Directory for generated calibration .npy files")
        parser.add_argument("--calib-list-path", default=default_calib_list_path, help="Path to RKNN dataset list file")

        parser.add_argument("--target-platform", default=default_target_platform, help="RKNN target platform")

        quant_group = parser.add_mutually_exclusive_group()
        quant_group.add_argument("--quantize", dest="quantize", action="store_true", help="Enable INT8 quantization")
        quant_group.add_argument("--no-quantize", dest="quantize", action="store_false", help="Disable quantization")
        parser.set_defaults(quantize=default_quantize)

        parser.add_argument(
            "--feature-columns",
            default=default_feature_columns,
            help="Comma-separated feature columns from CSV (default: auto-select)",
        )
        parser.add_argument(
            "--num-calib-samples",
            type=int,
            default=default_num_calib_samples,
            help="Number of calibration windows to generate",
        )
        parser.add_argument(
            "--window-size",
            type=int,
            default=default_window_size,
            help="Time window length. If omitted, inferred from ONNX input shape",
        )
        parser.add_argument(
            "--window-stride",
            type=int,
            default=default_window_stride,
            help="Stride for sliding windows during calibration generation",
        )
        parser.add_argument(
            "--max-rows",
            type=int,
            default=default_max_rows,
            help="Optional cap on number of CSV rows to read",
        )

        parser.add_argument("--verbose-rknn", action="store_true", help="Enable verbose RKNN logs")
        parser.add_argument("--log-level", default=default_log_level, help="Python log level")

        return parser

    @classmethod
    def from_cli(cls, argv: Optional[Sequence[str]] = None) -> "ConversionConfig":
        parser = cls.build_parser()
        args = parser.parse_args(argv)

        dataset_dir = _resolve_path(args.dataset_dir)
        onnx_dir = _resolve_path(args.onnx_dir)
        rknn_dir = _resolve_path(args.rknn_dir)

        dataset_path_raw = args.dataset_path if args.dataset_path else str(dataset_dir / "df.csv")
        onnx_path_raw = args.onnx_path if args.onnx_path else str(onnx_dir / "model.onnx")
        rknn_path_raw = args.rknn_path if args.rknn_path else str(rknn_dir / "model.rknn")

        calib_dir_raw = args.calib_dir if args.calib_dir else str(dataset_dir / "calibration")
        calib_list_raw = (
            args.calib_list_path
            if args.calib_list_path
            else str(Path(calib_dir_raw) / "dataset.txt")
        )

        paths = PathConfig(
            dataset_dir=dataset_dir,
            dataset_path=_resolve_dataset_path(dataset_path_raw, dataset_dir),
            onnx_dir=onnx_dir,
            onnx_path=_resolve_path(onnx_path_raw),
            rknn_dir=rknn_dir,
            rknn_path=_resolve_path(rknn_path_raw),
            calib_dir=_resolve_path(calib_dir_raw),
            calib_list_path=_resolve_path(calib_list_raw),
        )

        return cls(
            paths=paths,
            target_platform=args.target_platform,
            quantize=bool(args.quantize),
            feature_columns=cls._parse_feature_columns(args.feature_columns),
            num_calib_samples=int(args.num_calib_samples),
            window_size=args.window_size,
            window_stride=int(args.window_stride),
            max_rows=args.max_rows,
            verbose_rknn=bool(args.verbose_rknn),
            log_level=args.log_level,
        )
