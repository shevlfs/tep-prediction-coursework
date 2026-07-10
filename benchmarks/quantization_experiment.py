"""INT8 calibration experiment: bug or dataset feature?

Answers the question "why does INT8 collapse TEP accuracy to near-random —
is it a quantization bug or a genuine property of the dataset?" by measuring,
for each model, three conditions on the *same* normalized test set:

  1. fp16       -- no quantization (reference upper bound)
  2. int8_raw   -- INT8 calibrated on RAW sensor windows (reproduces the bug:
                   the quantizer sees the raw ~4-orders-of-magnitude range while
                   the model is actually fed standardized ~N(0,1) input)
  3. int8_norm  -- INT8 calibrated on windows standardized with the SAME saved
                   norm_mean/norm_std the runtime uses (the fix)

Everything runs on the x86 rknn-toolkit2 simulator (init_runtime(target=None)),
so no RK3568 board is required. Outputs a JSON, a Markdown report with a verdict,
and a bar chart to results/.

Decision rule:
  int8_norm ~= fp16 (gap within a few %)  -> it was a calibration BUG
  int8_norm still << fp16                 -> genuine 8-bit sensitivity of TEP

Run on the conversion box, e.g.:
  ~/miniconda3/envs/rknn/bin/python quantization_experiment.py \
      --onnx-dir ~/onnx --data-dir ~/small_tep --models tcn tepnet lstm transformer

The rknn import is lazy so the pure data/metric helpers can be imported and
unit-checked in a plain numpy/pandas environment.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

WINDOW_SIZE = 32
NUM_FEATURES = 52
NUM_CLASSES = 21
TARGET_PLATFORM = "rk3568"

CONDITIONS = ("fp16", "int8_raw", "int8_norm")

log = logging.getLogger("quant_experiment")


# --------------------------------------------------------------------------- #
# Data (pure numpy/pandas — importable and testable without rknn)             #
# --------------------------------------------------------------------------- #
def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(data_dir / "df.csv")
    target = pd.read_csv(data_dir / "target.csv")
    train_mask = pd.read_csv(data_dir / "train_mask.csv")

    feature_cols = [c for c in df.columns if c not in ("run_id", "sample")]
    if len(feature_cols) != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features, got {len(feature_cols)}")

    df = df.copy()
    df["target"] = target["target"].values
    df["train_mask"] = train_mask["train_mask"].values
    return df, feature_cols


def _make_windows(
    group: pd.DataFrame, feature_cols: Sequence[str], window: int, stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    x_raw = group[feature_cols].values.astype(np.float32)
    y_raw = group["target"].values.astype(np.int64)
    xs, ys = [], []
    for i in range(0, len(x_raw) - window + 1, stride):
        xs.append(x_raw[i : i + window].T)  # (C, T)
        ys.append(y_raw[i + window - 1])
    if not xs:
        return np.empty((0, len(feature_cols), window), np.float32), np.empty((0,), np.int64)
    return np.asarray(xs, np.float32), np.asarray(ys, np.int64)


def build_windows(
    df: pd.DataFrame, feature_cols: Sequence[str], mask_val: int, stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding windows for one split, grouped by run (matches training)."""
    subset = df[df["train_mask"] == mask_val]
    all_x, all_y = [], []
    for _, g in subset.groupby("run_id"):
        x, y = _make_windows(g, feature_cols, WINDOW_SIZE, stride)
        if len(x):
            all_x.append(x)
            all_y.append(y)
    if not all_x:
        raise RuntimeError(f"No windows produced for mask_val={mask_val}")
    return np.concatenate(all_x), np.concatenate(all_y)


def load_norm(norm_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.load(norm_dir / "norm_mean.npy").astype(np.float32).reshape(-1)
    std = np.load(norm_dir / "norm_std.npy").astype(np.float32).reshape(-1)
    return mean, std


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-channel standardization for windows shaped (N, C, T) or (C, T)."""
    if x.ndim == 3:
        return ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    if x.ndim == 2:
        return ((x - mean[:, None]) / std[:, None]).astype(np.float32)
    raise ValueError(f"Unexpected window ndim {x.ndim}")


def write_calibration(
    x_windows: np.ndarray,
    calib_dir: Path,
    num_samples: int,
    normalize: bool,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> Path:
    """Save up to num_samples calibration windows and an rknn dataset list.

    x_windows is the RAW (un-normalized) train windows (N, C, T). When
    normalize=True the same standardization used at runtime is applied first.
    """
    calib_dir.mkdir(parents=True, exist_ok=True)
    for stale in calib_dir.glob("calib_*.npy"):
        stale.unlink()

    n = min(num_samples, len(x_windows))
    # Evenly spaced picks across the split so all runs/faults contribute.
    idx = np.linspace(0, len(x_windows) - 1, n, dtype=int)

    list_path = calib_dir / "dataset.txt"
    with list_path.open("w", encoding="utf-8") as out:
        for j, i in enumerate(idx):
            sample = x_windows[i : i + 1]  # (1, C, T)
            if normalize:
                sample = standardize(sample, mean, std)
            p = calib_dir / f"calib_{j:06d}.npy"
            np.save(p, sample.astype(np.float32))
            out.write(f"{p.resolve()}\n")
    return list_path


# --------------------------------------------------------------------------- #
# Metrics (pure numpy)                                                        #
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    accuracy = float((y_true == y_pred).mean())

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    for c in range(num_classes):
        tp = int(cm[c, c])
        support = int(cm[c].sum())
        pred_c = int(cm[:, c].sum())
        precision = tp / pred_c if pred_c else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if support:
            per_class[c] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }

    # Collapse indicator: fraction of all predictions landing in the single
    # most-predicted class. ~1.0 means the model degenerated to one class.
    pred_counts = np.bincount(y_pred, minlength=num_classes)
    majority_class = int(pred_counts.argmax())
    majority_fraction = float(pred_counts.max() / len(y_pred))

    macro_f1 = float(np.mean([m["f1"] for m in per_class.values()])) if per_class else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "majority_pred_class": majority_class,
        "majority_pred_fraction": round(majority_fraction, 4),
        "num_test_samples": int(len(y_true)),
        "per_class": per_class,
    }


# --------------------------------------------------------------------------- #
# RKNN build + simulate (lazy import; needs rknn-toolkit2)                     #
# --------------------------------------------------------------------------- #
def simulate_predictions(
    onnx_path: Path,
    x_test_norm: np.ndarray,
    quantize: bool,
    calib_list_path: Optional[Path],
) -> np.ndarray:
    """Build the model in-memory and run it on the rknn PC simulator.

    Mirrors the production config in convert_onnx_to_rknn.py: rknn.config only
    sets target_platform (no mean/std), so the input fed here must already be
    standardized — exactly what the runtime does.
    """
    from rknn.api import RKNN  # lazy: only needed on the conversion box

    rknn = RKNN(verbose=False)
    try:
        assert rknn.config(target_platform=TARGET_PLATFORM) == 0, "rknn.config failed"
        assert rknn.load_onnx(model=str(onnx_path)) == 0, "rknn.load_onnx failed"
        if quantize:
            ret = rknn.build(do_quantization=True, dataset=str(calib_list_path))
        else:
            ret = rknn.build(do_quantization=False)
        assert ret == 0, "rknn.build failed"
        assert rknn.init_runtime(target=None) == 0, "rknn.init_runtime (simulator) failed"

        preds = np.empty((len(x_test_norm),), dtype=np.int64)
        for i, w in enumerate(x_test_norm):
            inp = np.expand_dims(w, axis=0).astype(np.float32)  # (1, C, T)
            outputs = rknn.inference(inputs=[inp])
            preds[i] = int(np.argmax(outputs[0][0]))
            if (i + 1) % 500 == 0:
                log.info("    simulated %d/%d", i + 1, len(x_test_norm))
        return preds
    finally:
        rknn.release()


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run_model(
    model: str,
    onnx_dir: Path,
    df: pd.DataFrame,
    feature_cols: List[str],
    args,
) -> dict:
    onnx_path = onnx_dir / model / "model.onnx"
    norm_dir = onnx_dir / model
    if not onnx_path.exists():
        return {"error": f"onnx not found: {onnx_path}"}

    mean, std = load_norm(norm_dir)

    # Runtime input: raw test windows standardized exactly like inference.
    x_test_raw, y_test = build_windows(df, feature_cols, mask_val=0, stride=args.test_stride)
    if args.max_test_samples and len(x_test_raw) > args.max_test_samples:
        idx = np.linspace(0, len(x_test_raw) - 1, args.max_test_samples, dtype=int)
        x_test_raw, y_test = x_test_raw[idx], y_test[idx]
    x_test_norm = standardize(x_test_raw, mean, std)
    log.info("[%s] test windows: %d (stride=%d)", model, len(x_test_norm), args.test_stride)

    # Calibration source: raw train windows (normalized on demand per condition).
    x_calib_raw, _ = build_windows(df, feature_cols, mask_val=1, stride=args.calib_stride)
    calib_dir = args.work_dir / model / "calib"

    conditions: Dict[str, dict] = {}
    for cond in CONDITIONS:
        log.info("[%s] condition=%s", model, cond)
        try:
            if cond == "fp16":
                preds = simulate_predictions(onnx_path, x_test_norm, quantize=False, calib_list_path=None)
            else:
                normalize = cond == "int8_norm"
                calib_list = write_calibration(
                    x_calib_raw, calib_dir, args.num_calib_samples, normalize, mean, std
                )
                preds = simulate_predictions(onnx_path, x_test_norm, quantize=True, calib_list_path=calib_list)
        except Exception as e:  # a single condition failing must not abort the run
            log.error("[%s] %s FAILED: %s", model, cond, e)
            conditions[cond] = {"error": str(e)}
            continue
        conditions[cond] = compute_metrics(y_test, preds, NUM_CLASSES)
        log.info(
            "[%s] %s: acc=%.4f macro_f1=%.4f majority_frac=%.2f",
            model, cond, conditions[cond]["accuracy"],
            conditions[cond]["macro_f1"], conditions[cond]["majority_pred_fraction"],
        )

    result = {"model": model, "conditions": conditions}
    if all("accuracy" in conditions.get(c, {}) for c in CONDITIONS):
        fp16_acc = conditions["fp16"]["accuracy"]
        result["fp16_minus_int8raw"] = round(fp16_acc - conditions["int8_raw"]["accuracy"], 4)
        result["fp16_minus_int8norm"] = round(fp16_acc - conditions["int8_norm"]["accuracy"], 4)
    return result


def render_report(results: List[dict], gap_threshold: float) -> str:
    lines = [
        "# INT8 quantization: bug or dataset feature?",
        "",
        "Accuracy on the standardized TEP test set, rknn simulator (rk3568).",
        "",
        "| model | fp16 | int8_raw (bug) | int8_norm (fix) | fp16−int8norm |",
        "|---|---|---|---|---|",
    ]
    ok = [r for r in results if "fp16_minus_int8norm" in r]
    for r in ok:
        c = r["conditions"]
        lines.append(
            f"| {r['model']} | {c['fp16']['accuracy']:.4f} | "
            f"{c['int8_raw']['accuracy']:.4f} | {c['int8_norm']['accuracy']:.4f} | "
            f"{r['fp16_minus_int8norm']:+.4f} |"
        )
    incomplete = [r for r in results if "fp16_minus_int8norm" not in r]
    if incomplete:
        lines += ["", "Skipped / partial (see JSON for details):"]
        for r in incomplete:
            reason = r.get("error") or ", ".join(
                f"{c}: {v['error']}" for c, v in r.get("conditions", {}).items() if "error" in v
            ) or "unknown"
            lines.append(f"- **{r.get('model','?')}** — {reason}")

    if ok:
        mean_gap = float(np.mean([r["fp16_minus_int8norm"] for r in ok]))
        mean_raw_gap = float(np.mean([r["fp16_minus_int8raw"] for r in ok]))
        verdict = (
            "**BUG confirmed (H1).** After matching calibration to the runtime "
            "normalization, INT8 recovers to ~FP16. The collapse was a "
            "calibration/normalization mismatch, not a property of the dataset."
            if mean_gap <= gap_threshold
            else "**Residual gap (H0 component).** Fixing calibration recovers most of "
            "the accuracy, but INT8 still trails FP16 — evidence of genuine 8-bit "
            "sensitivity in TEP fault signatures. This is the thread to develop."
        )
        lines += [
            "",
            f"- Mean gap fp16−int8_raw (bug): **{mean_raw_gap:+.4f}**",
            f"- Mean gap fp16−int8_norm (fix): **{mean_gap:+.4f}** "
            f"(threshold {gap_threshold})",
            "",
            verdict,
        ]
    return "\n".join(lines) + "\n"


def render_plot(results: List[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping plot")
        return

    ok = [r for r in results if "fp16_minus_int8norm" in r]
    if not ok:
        return
    models = [r["model"] for r in ok]
    x = np.arange(len(models))
    width = 0.26
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 4.5))
    for k, cond in enumerate(CONDITIONS):
        vals = [r["conditions"][cond]["accuracy"] for r in ok]
        ax.bar(x + (k - 1) * width, vals, width, label=cond)
    ax.axhline(1 / NUM_CLASSES, ls="--", lw=1, color="gray", label="random (1/21)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("FP16 vs INT8 (raw calib = bug) vs INT8 (normalized calib = fix)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("plot saved to %s", path)


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx-dir", type=Path, default=Path("onnx"),
                   help="Dir with {model}/model.onnx and {model}/norm_*.npy")
    p.add_argument("--data-dir", type=Path, default=Path("small_tep"),
                   help="Dir with df.csv, target.csv, train_mask.csv")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--work-dir", type=Path, default=Path("results") / "quant_experiment_work",
                   help="Scratch dir for generated calibration files")
    p.add_argument("--models", nargs="+", default=["tcn", "tepnet", "lstm", "transformer"])
    p.add_argument("--num-calib-samples", type=int, default=256)
    p.add_argument("--calib-stride", type=int, default=8)
    p.add_argument("--test-stride", type=int, default=32,
                   help="Stride for test windows (larger = faster simulation)")
    p.add_argument("--max-test-samples", type=int, default=0,
                   help="Optional cap on test windows (0 = no cap)")
    p.add_argument("--gap-threshold", type=float, default=0.03,
                   help="fp16−int8_norm gap at/below which the verdict is 'bug'")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    df, feature_cols = load_dataset(args.data_dir)
    log.info("loaded dataset: %d rows, %d features", len(df), len(feature_cols))

    results = []
    for m in args.models:
        try:
            results.append(run_model(m, args.onnx_dir, df, feature_cols, args))
        except Exception as e:  # keep going; still write the report for the rest
            log.error("[%s] model FAILED: %s", m, e)
            results.append({"model": m, "error": str(e)})

    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / "quantization_experiment.json"
    md_path = args.results_dir / "quantization_experiment.md"
    png_path = args.results_dir / "quantization_experiment.png"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(render_report(results, args.gap_threshold), encoding="utf-8")
    render_plot(results, png_path)

    log.info("wrote %s, %s, %s", json_path, md_path, png_path)
    print("\n" + render_report(results, args.gap_threshold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
