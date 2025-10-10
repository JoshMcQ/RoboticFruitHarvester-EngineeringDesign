#!/usr/bin/env python3
"""
2.5.8 – Make transformation matrix for vision system
----------------------------------------------------
Utility script that loads the pixel→base calibration file and prints the
homogeneous 4×4 transform (T_base_cam) alongside a quick demo conversion.

The generated output is meant for documentation screenshots – no robot
connection is required.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from vision_calibration import CalibrationError, VisionToBaseCalibration
except ImportError:  # pragma: no cover - allow execution as standalone script
    from perception.vision_calibration import CalibrationError, VisionToBaseCalibration

ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = ROOT / "vision_to_base_calibration.json"
OUTPUT_JSON = ROOT / "transform_matrix_output.json"


def pretty_matrix(mat: np.ndarray) -> str:
    rows = []
    for row in mat:
        rows.append("[ " + ", ".join(f"{val: .6f}" for val in row) + " ]")
    return "\n".join(rows)


def compute_demo(calib: VisionToBaseCalibration) -> Tuple[Tuple[float, float], Tuple[float, float, float]]:
    """Project the image centre to the workspace plane for quick sanity check."""
    cx, cy = calib.camera_matrix[0, 2], calib.camera_matrix[1, 2]
    x_m, y_m, z_m = calib.pixel_to_base(cx, cy)
    return (float(cx), float(cy)), (x_m, y_m, z_m)


def save_json(calib: VisionToBaseCalibration, demo: Tuple[Tuple[float, float], Tuple[float, float, float]]) -> None:
    data = {
        "table_height_m": calib.table_height_m,
        "rotation_matrix": calib.rotation_base_from_cam.tolist(),
        "translation_m": calib.translation_base_from_cam.tolist(),
        "transform_matrix": calib.transform_matrix.tolist(),
        "homography": calib.homography_px_to_base.tolist(),
        "reprojection_rmse_m": calib.reprojection_rmse_m,
        "sample_residuals_m": calib.sample_errors_m.tolist(),
        "demo_input_pixel": demo[0],
        "demo_output_base_m": demo[1],
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def main(path: Path = CALIBRATION_PATH) -> int:
    parser = argparse.ArgumentParser(description="Print the camera→base transformation matrix")
    parser.add_argument("--calibration", type=Path, default=path,
                        help="Path to vision_to_base_calibration.json (default: perception folder)")
    args = parser.parse_args()

    print("📁 Loading calibration from", args.calibration)
    try:
        calib = VisionToBaseCalibration.from_json(str(args.calibration))
    except CalibrationError as exc:  # pragma: no cover - user reporting
        print(f"❌ Calibration error: {exc}")
        return 1

    print("\nRotation (base ← camera):")
    print(pretty_matrix(calib.rotation_base_from_cam))

    print("\nTranslation (m):")
    print(np.array2string(calib.translation_base_from_cam, precision=6))

    print("\nHomogeneous transform T_base_cam:")
    print(pretty_matrix(calib.transform_matrix))

    print("\nHomography (pixels→base-plane):")
    print(pretty_matrix(calib.homography_px_to_base))

    print(f"\nPlanar reprojection error: {calib.reprojection_rmse_m*1000:.2f} mm RMS")
    if calib.sample_errors_m.size:
        print("Sample residuals (mm):", np.array2string(calib.sample_errors_m * 1000, precision=2))

    demo_in, demo_out = compute_demo(calib)
    print("\nDemo: pixel → base plane")
    print(f"  Pixel centre: (u={demo_in[0]:.1f}, v={demo_in[1]:.1f})")
    print(f"  Base frame:  x={demo_out[0]:.3f} m, y={demo_out[1]:.3f} m, z={demo_out[2]:.3f} m")

    save_json(calib, (demo_in, demo_out))
    print(f"\n📝 Detailed output saved to {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
