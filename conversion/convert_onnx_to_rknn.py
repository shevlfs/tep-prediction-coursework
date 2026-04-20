from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import pandas as pd
from rknn.api import RKNN

from config import ConversionConfig
from logger import setup_logger

EXCLUDED_AUTO_COLUMNS = {"run_id", "sample", "target"}
log = logging.getLogger(__name__)


def get_onnx_input_shape(onnx_path: Path) -> Tuple[str, List[Optional[int]]]:
    model = onnx.load(str(onnx_path))
    if not model.graph.input:
        raise RuntimeError("ONNX model has no inputs")

    inp = model.graph.input[0]
    dims: List[Optional[int]] = [
        int(d.dim_value) if d.dim_value > 0 else None
        for d in inp.type.tensor_type.shape.dim
    ]
    return inp.name, dims


def select_feature_columns(
    header: Sequence[str],
    requested: Optional[Sequence[str]],
    expected_channels: Optional[int],
) -> List[str]:
    if requested:
        missing = [c for c in requested if c not in set(header)]
        if missing:
            raise ValueError(f"Requested feature columns not found in CSV: {missing}")
        selected = list(requested)
    else:
        candidates = [c for c in header if c.lower() not in EXCLUDED_AUTO_COLUMNS]
        if expected_channels is not None and len(candidates) < expected_channels:
            raise ValueError(
                f"CSV has only {len(candidates)} candidate columns, "
                f"but ONNX expects {expected_channels}"
            )
        selected = candidates[:expected_channels] if expected_channels else candidates

    if expected_channels is not None and len(selected) != expected_channels:
        raise ValueError(
            f"Selected {len(selected)} feature columns, but ONNX expects {expected_channels}"
        )
    return selected


def read_csv_grouped_by_run(
    csv_path: Path,
    feature_columns: Sequence[str],
    max_rows: Optional[int],
) -> Tuple[Dict[str, np.ndarray], int]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    has_run_id = "run_id" in header

    usecols = list(feature_columns) + (["run_id"] if has_run_id else [])
    df = pd.read_csv(csv_path, usecols=usecols, nrows=max_rows)
    if df.empty:
        raise RuntimeError(f"No data rows were read from '{csv_path}'")

    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=list(feature_columns))
    if df.empty:
        raise RuntimeError("All rows contain NaN/invalid values in selected feature columns")
    dropped = before - len(df)
    if dropped > 0:
        log.warning("Dropped %d rows with non-numeric feature values", dropped)

    if not has_run_id:
        df["run_id"] = "global"

    runs = {
        str(run_id): group[list(feature_columns)].to_numpy(dtype=np.float32, copy=False)
        for run_id, group in df.groupby("run_id", sort=True)
    }
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

    # Build per-run window iterators and alternate them.
    # This ensures all runs (and thus all fault classes) contribute to
    # calibration instead of only the first few.
    def windows(data: np.ndarray):
        for start in range(0, data.shape[0] - window_size + 1, window_stride):
            yield np.expand_dims(data[start : start + window_size].T, axis=0).astype(np.float32)

    iterators = [
        windows(data)
        for data in (runs[r] for r in sorted(runs))
        if data.ndim == 2 and data.shape[0] >= window_size
    ]

    produced = 0
    with calib_list_path.open("w", encoding="utf-8") as out:
        while iterators and produced < num_calib_samples:
            next_iterators = []
            for it in iterators:
                sample = next(it, None)
                if sample is None:
                    continue
                path = calib_dir / f"calib_{produced:06d}.npy"
                np.save(path, sample)
                out.write(f"{path.resolve()}\n")
                produced += 1
                next_iterators.append(it)
                if produced >= num_calib_samples:
                    return produced
            iterators = next_iterators

    return produced


def _rknn_check(ret: int, step: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{step} failed with code {ret}")


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
        _rknn_check(rknn.config(target_platform=target_platform), "rknn.config")
        _rknn_check(rknn.load_onnx(model=str(onnx_path)), "rknn.load_onnx")
        if quantize:
            if calib_list_path is None:
                raise RuntimeError("Quantization enabled but calib_list_path is missing")
            _rknn_check(rknn.build(do_quantization=True, dataset=str(calib_list_path)), "rknn.build")
        else:
            _rknn_check(rknn.build(do_quantization=False), "rknn.build")
        _rknn_check(rknn.export_rknn(str(rknn_path)), "rknn.export_rknn")
    finally:
        rknn.release()


def prepare_calibration(config: ConversionConfig) -> Path:
    paths = config.paths
    if not paths.dataset_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {paths.dataset_path}")

    input_name, input_shape = get_onnx_input_shape(paths.onnx_path)
    if len(input_shape) != 3:
        raise RuntimeError(f"Expected 3D input [B, C, T], got shape {input_shape} for '{input_name}'")

    expected_channels, inferred_window = input_shape[1], input_shape[2]
    window_size = config.window_size or inferred_window
    if window_size is None:
        raise RuntimeError("Cannot infer window_size from ONNX shape; provide --window-size")

    log.info("ONNX input '%s' shape: %s", input_name, input_shape)

    header = pd.read_csv(paths.dataset_path, nrows=0).columns.tolist()
    feature_columns = select_feature_columns(header, config.feature_columns, expected_channels)
    log.info("Selected %d feature columns for calibration", len(feature_columns))

    runs, rows_read = read_csv_grouped_by_run(paths.dataset_path, feature_columns, config.max_rows)
    log.info("Read %d rows from %d runs", rows_read, len(runs))

    produced = build_calibration_dataset(
        runs=runs,
        calib_dir=paths.calib_dir,
        calib_list_path=paths.calib_list_path,
        window_size=window_size,
        window_stride=config.window_stride,
        num_calib_samples=config.num_calib_samples,
    )
    if produced == 0:
        raise RuntimeError("Failed to generate calibration samples. Check window size and dataset length.")

    log.info("Generated %d calibration samples → %s", produced, paths.calib_list_path)
    return paths.calib_list_path


def validate_config(config: ConversionConfig) -> None:
    if config.num_calib_samples <= 0:
        raise ValueError(f"num_calib_samples must be > 0, got {config.num_calib_samples}")
    if config.window_stride <= 0:
        raise ValueError(f"window_stride must be > 0, got {config.window_stride}")
    if config.window_size is not None and config.window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {config.window_size}")


def main() -> int:
    config = ConversionConfig.from_cli()
    setup_logger(__name__, config.log_level)
    validate_config(config)

    paths = config.paths
    log.info("ONNX: %s -> RKNN: %s", paths.onnx_path, paths.rknn_path)
    log.info("platform=%s, quantize=%s", config.target_platform, config.quantize)

    if not paths.onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {paths.onnx_path}")

    calib_list_path = prepare_calibration(config) if config.quantize else None

    run_conversion(
        onnx_path=paths.onnx_path,
        rknn_path=paths.rknn_path,
        target_platform=config.target_platform,
        quantize=config.quantize,
        calib_list_path=calib_list_path,
        verbose_rknn=config.verbose_rknn,
    )

    log.info("RKNN export complete: %s", paths.rknn_path)
    return 0


if __name__ == "__main__":
    main()
