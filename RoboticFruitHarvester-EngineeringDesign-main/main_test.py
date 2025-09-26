#!/usr/bin/env python3
"""
Unified test runner for Robotic Fruit Harvester vision system
Runs calibration check, coordinate math, YOLOv5 detection, and accuracy reporting
Best practices based on OpenCV and Ultralytics documentation (2024)
"""

import os
import sys
import numpy as np
import cv2
import json
from datetime import datetime

# Import perception functions
try:
    from perception.perception import (
        _get_model, _backproject, _cam_to_base, _get_cv_frame, 
        compute_object_pose_base, K, T_base_cam, calibration_status
    )
    print("✅ Successfully imported perception module")
except ImportError as e:
    print(f"❌ Failed to import perception module: {e}")
    sys.exit(1)

# Calibration check
status, fname = calibration_status()
if status:
    print(f"Calibration loaded from {fname}")
else:
    print("No calibration file found. Run calibrate_camera.py for best accuracy.")

# Camera matrix summary
print("\nCamera Intrinsics (K):")
print(K)

# Test 1: Coordinate math
print("\n🧮 TEST 1: Coordinate Math Accuracy")
test_pixels = [
    (320, 240, 0.1, "Image Center"),
    (480, 240, 0.2, "Right Side"),
    (320, 360, 0.15, "Bottom Center")
]
for u, v, depth, desc in test_pixels:
    Pc = _backproject(u, v, depth)
    print(f"{desc}: pixel=({u},{v}), depth={depth}m → Camera coords: {Pc[:3]}")

# Test 2: YOLOv5 detection
print("\n🔍 TEST 2: YOLOv5 Object Detection")
try:
    x, y, z = compute_object_pose_base()
    print(f"Detected object pose: x={x:.3f}, y={y:.3f}, z={z:.3f}")
    print("Detection snapshot saved as detection_snapshot.jpg")
except Exception as e:
    print(f"❌ Detection failed: {e}")

# Test 3: Repeatability
print("\n🔁 TEST 3: Detection Repeatability")
coords = []
for i in range(3):
    try:
        x, y, z = compute_object_pose_base()
        coords.append([x, y, z])
        print(f"Run {i+1}: ({x:.3f}, {y:.3f}, {z:.3f})")
    except Exception as e:
        print(f"Run {i+1}: Detection failed: {e}")
if len(coords) > 1:
    arr = np.array(coords)
    stds = np.std(arr, axis=0)
    print(f"Std deviation: x={stds[0]:.4f}, y={stds[1]:.4f}, z={stds[2]:.4f}")

# Test 4: Workspace validity
print("\n🤖 TEST 4: Workspace Validity")
workspace_tests = [
    ((160, 120), "Top-left quadrant"),
    ((480, 120), "Top-right quadrant"),
    ((160, 360), "Bottom-left quadrant"),
    ((480, 360), "Bottom-right quadrant"),
    ((320, 240), "Center")
]
for (u, v), desc in workspace_tests:
    Pc = _backproject(u, v, 0.15)
    Pb = _cam_to_base(Pc)
    reach = np.sqrt(Pb[0]**2 + Pb[1]**2)
    print(f"{desc}: ({Pb[0]:.3f}, {Pb[1]:.3f}, {Pb[2]:.3f}) → Reach={reach:.3f}m")

# Summary
print("\n==============================")
print("Unified test complete.")
print("Review detection_snapshot.jpg for YOLOv5 results.")
print("If calibration is missing, run calibrate_camera.py and update perception.py.")
print("For best accuracy, use real calibration and test with known object positions.")
