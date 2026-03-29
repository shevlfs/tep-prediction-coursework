from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import pandas as pd
import logging
from rknn.api import RKNN

from config import ConversionConfig
from logger import setup_logger


EXCLUDED_AUTO_COLUMNS = {"run_id", "sample", "target"}
LOGGER = logging.getLogger("onnx_to_rknn")


def get_onnx_input_shape(onnx_path: Path) -> Tuple[str, List[Optional[int]]]:
    model = onnx.load(str(onnx_path))
    if not model.graph.input:
        raise RuntimeError("ONNX model has no inputs")

    input_tensor = model.graph.input[0]
    dims: List[Optional[int]] = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            dims.append(None)

    return input_tensor.name, dims


def select_feature_columns(
    header: Sequence[str], requested_columns: Optional[Sequence[str]], expected_channels: Optional[int]
) -> List[str]:
    header_set = set(header)

    if requested_columns:
        missing = [col for col in requested_columns if col not in header_set]
        if missing:
            raise ValueError(f"Requested feature columns not found in CSV: {missing}")
        selected = list(requested_columns)
    else:
        candidates = [col for col in header if col.lower() not in EXCLUDED_AUTO_COLUMNS]
        if expected_channels is not None:
            if len(candidates) < expected_channels:
                raise ValueError(
                    f"CSV has only {len(candidates)} candidate feature columns, "
                    f"but ONNX expects {expected_channels} channels"
                )
            selected = candidates[:expected_channels]
        else:
            selected = candidates

    if expected_channels is not None and len(selected) != expected_channels:
        raise ValueError(
            f"Selected {len(selected)} feature columns, but ONNX expects {expected_channels} channels"
        )

    return selected


def read_csv_grouped_by_run(
    csv_path: Path,
    feature_columns: Sequence[str],
    max_rows: Optional[int],
) -> Tuple[Dict[str, np.ndarray], int]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    has_run_id = "run_id" in header

    usecols = list(feature_columns)
    if has_run_id:
        usecols.append("run_id")

    df = pd.read_csv(csv_path, usecols=usecols, nrows=max_rows)
    if df.empty:
        raise RuntimeError(f"No data rows were read from '{csv_path}'")

    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_drop = len(df)
    df = df.dropna(subset=list(feature_columns))
    if df.empty:
        raise RuntimeError("All rows contain NaN/invalid values in selected feature columns")

    if not has_run_id:
        df["run_id"] = "global"

    dropped = before_drop - len(df)
    if dropped > 0:
        LOGGER.warning("Dropped %d rows with non-numeric feature values", dropped)

    runs: Dict[str, np.ndarray] = {}
    for run_id, group in df.groupby("run_id", sort=True):
        runs[str(run_id)] = group[list(feature_columns)].to_numpy(dtype=np.float32, copy=False)

    return runs, len(df)


def build_calibration_dataset(
    runs: Dict[str, np.ndarray],
    calib_dir: Path,
    calib_list_path: Path,
    window_size: int,
    window_stride: int,
    num_calib_samples: int,
) -> int:
    calib_dir.mkdir(parents=True, exist_ok=True)
    calib_list_path.parent.mkdir(parents=True, exist_ok=True)

    for stale in calib_dir.glob("calib_*.npy"):
        stale.unlink()

    produced = 0
    with calib_list_path.open("w", encoding="utf-8") as out:
        for run_id in sorted(runs.keys()):
            data = runs[run_id]
            if data.ndim != 2 or data.shape[0] < window_size:
                continue

            for start in range(0, data.shape[0] - window_size + 1, window_stride):
                window = data[start : start + window_size].T
                sample = np.expand_dims(window, axis=0).astype(np.float32)

                sample_path = calib_dir / f"calib_{produced:06d}.npy"
                np.save(sample_path, sample)
                out.write(f"{sample_path.resolve()}\n")

                produced += 1
                if produced >= num_calib_samples:
                    return produced

    return produced


def run_conversion(
    onnx_path: Path,
    rknn_path: Path,
    target_platform: str,
    quantize: bool,
    calib_list_path: Optional[Path],
    verbose_rknn: bool,
) -> None:
    rknn_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=verbose_rknn)
    try:
        ret = rknn.config(target_platform=target_platform)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed with code {ret}")

        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

        if quantize:
            if calib_list_path is None:
                raise RuntimeError("Quantization is enabled, but calib_list_path is missing")
            ret = rknn.build(do_quantization=True, dataset=str(calib_list_path))
        else:
            ret = rknn.build(do_quantization=False)

        if ret != 0:
            raise RuntimeError(f"rknn.build failed with code {ret}")

        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed with code {ret}")
    finally:
        rknn.release()


def validate_config(config: ConversionConfig) -> None:
    if config.num_calib_samples <= 0:
        raise ValueError(f"num_calib_samples must be > 0, got {config.num_calib_samples}")
    if config.window_stride <= 0:
        raise ValueError(f"window_stride must be > 0, got {config.window_stride}")
    if config.window_size is not None and config.window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {config.window_size}")


def main() -> int:
    config = ConversionConfig.from_cli()
    logger = setup_logger("onnx_to_rknn", config.log_level)
    validate_config(config)

    paths = config.paths
    logger.info("DATASET_DIR  = %s", paths.dataset_dir)
    logger.info("DATASET_PATH = %s", paths.dataset_path)
    logger.info("ONNX_DIR     = %s", paths.onnx_dir)
    logger.info("ONNX_PATH    = %s", paths.onnx_path)
    logger.info("RKNN_DIR     = %s", paths.rknn_dir)
    logger.info("RKNN_PATH    = %s", paths.rknn_path)
    logger.info("CALIB_DIR    = %s", paths.calib_dir)
    logger.info("CALIB_LIST   = %s", paths.calib_list_path)
    logger.info("Quantization = %s", config.quantize)

    if not paths.onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {paths.onnx_path}")

    calib_list_path: Optional[Path] = None

    if config.quantize:
        if not paths.dataset_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {paths.dataset_path}")

        input_name, input_shape = get_onnx_input_shape(paths.onnx_path)
        if len(input_shape) != 3:
            raise RuntimeError(
                f"Expected 3D time-series input [B, C, T], got shape {input_shape} for input '{input_name}'"
            )

        expected_channels = input_shape[1]
        inferred_window_size = input_shape[2]

        if config.window_size is not None:
            window_size = config.window_size
        elif inferred_window_size is not None:
            window_size = inferred_window_size
        else:
            raise RuntimeError("Cannot infer window_size from ONNX input shape; provide --window-size")

        logger.info("ONNX input '%s' shape: %s", input_name, input_shape)

        header = pd.read_csv(paths.dataset_path, nrows=0).columns.tolist()
        feature_columns = select_feature_columns(header, config.feature_columns, expected_channels)
        logger.info("Selected %d feature columns for calibration", len(feature_columns))
        logger.info("Feature columns: %s", feature_columns)

        runs, rows_read = read_csv_grouped_by_run(
            csv_path=paths.dataset_path,
            feature_columns=feature_columns,
            max_rows=config.max_rows,
        )
        logger.info("Read %d rows from %d runs", rows_read, len(runs))

        produced = build_calibration_dataset(
            runs=runs,
            calib_dir=paths.calib_dir,
            calib_list_path=paths.calib_list_path,
            window_size=window_size,
            window_stride=config.window_stride,
            num_calib_samples=config.num_calib_samples,
        )
        if produced == 0:
            raise RuntimeError(
                "Failed to generate calibration samples. Check window size and dataset length per run."
            )

        logger.info("Generated %d calibration samples in %s", produced, paths.calib_dir)
        logger.info("Calibration list file: %s", paths.calib_list_path)
        calib_list_path = paths.calib_list_path

    run_conversion(
        onnx_path=paths.onnx_path,
        rknn_path=paths.rknn_path,
        target_platform=config.target_platform,
        quantize=config.quantize,
        calib_list_path=calib_list_path,
        verbose_rknn=config.verbose_rknn,
    )

    logger.info("RKNN export complete: %s", paths.rknn_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
