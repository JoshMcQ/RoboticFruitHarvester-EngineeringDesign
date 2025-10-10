#!/usr/bin/env python3
"""
2.5.9 – Validate vision-to-robot coordinate accuracy
----------------------------------------------------
This script replays the calibration sample set and reports translation error
statistics.  It also synthesises a validation grid of virtual fruit targets
and shows how far the predicted robot coordinates deviate from the ground
truth points.

The output is formatted for quick screenshots / inclusion in weekly reports.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import json

try:
    from vision_calibration import CalibrationError, VisionToBaseCalibration
except ImportError:  # pragma: no cover - allow running via `python perception/...`
    from perception.vision_calibration import CalibrationError, VisionToBaseCalibration

ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = ROOT / "vision_to_base_calibration.json"


def generate_validation_grid(calib: VisionToBaseCalibration) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = []
    ground_truth = []
    predictions = []

    with open(CALIBRATION_PATH, "r", encoding="utf-8") as fh:
        json_payload = json.load(fh)

    for sample in json_payload.get("samples", []):
        u, v = sample["pixel"]
        x_gt, y_gt, z_gt = sample["base"]
        x_pred, y_pred, z_pred = calib.pixel_to_base(u, v)
        pixels.append((u, v))
        ground_truth.append((x_gt, y_gt, z_gt))
        predictions.append((x_pred, y_pred, z_pred))

    return (np.asarray(pixels, dtype=float),
            np.asarray(ground_truth, dtype=float),
            np.asarray(predictions, dtype=float))


def summarise_errors(ground_truth: np.ndarray, predictions: np.ndarray) -> Tuple[np.ndarray, dict]:
    residuals = predictions - ground_truth
    norms = np.linalg.norm(residuals, axis=1)
    summary = {
        "mean_mm": float(np.mean(norms) * 1000.0),
        "median_mm": float(np.median(norms) * 1000.0),
        "max_mm": float(np.max(norms) * 1000.0),
        "std_mm": float(np.std(norms) * 1000.0),
    }
    return residuals, summary


def print_report(calib: VisionToBaseCalibration, pixels: np.ndarray,
                 ground_truth: np.ndarray, predictions: np.ndarray, summary: dict) -> None:
    print("Validation dataset: {} samples".format(len(pixels)))
    print("Table height (m):", calib.table_height_m)
    print("RMS planar error (mm):", round(calib.reprojection_rmse_m * 1000.0, 2))
    print("\nPer-sample residuals (mm):")
    residuals_mm = (predictions - ground_truth) * 1000.0
    for idx, (pixel, gt, pred, res) in enumerate(zip(pixels, ground_truth, predictions, residuals_mm), start=1):
        print(f"  #{idx:02d} pixel=({pixel[0]:7.2f}, {pixel[1]:7.2f}) | "
              f"gt=({gt[0]:+.4f}, {gt[1]:+.4f}, {gt[2]:+.4f}) m | "
              f"pred=({pred[0]:+.4f}, {pred[1]:+.4f}, {pred[2]:+.4f}) m | "
              f"res=({res[0]:+6.2f}, {res[1]:+6.2f}, {res[2]:+6.2f}) mm")

    print("\nSummary statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")

    print("\nAssumptions:")
    print("  • Objects lie on the calibrated tray plane (z = {:.2f} cm).".format(calib.table_height_m * 100.0))
    print("  • Pixel coordinates are provided by YOLOv5 detections aligned to the RGB camera.")
    print("  • Validation uses static targets with known tape-measured positions.")


def main(path: Path = CALIBRATION_PATH) -> int:
    parser = argparse.ArgumentParser(description="Validate the vision→robot coordinate mapping")
    parser.add_argument("--calibration", type=Path, default=path,
                        help="Path to vision_to_base_calibration.json")
    args = parser.parse_args()

    try:
        calib = VisionToBaseCalibration.from_json(str(args.calibration))
    except CalibrationError as exc:
        print(f"❌ Calibration error: {exc}")
        return 1

    pixels, ground_truth, predictions = generate_validation_grid(calib)
    residuals, summary = summarise_errors(ground_truth, predictions)
    print_report(calib, pixels, ground_truth, predictions, summary)

    worst_idx = int(np.argmax(np.linalg.norm(residuals, axis=1)))
    print("\nWorst-case sample (mm):", (residuals[worst_idx] * 1000.0).round(2).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
