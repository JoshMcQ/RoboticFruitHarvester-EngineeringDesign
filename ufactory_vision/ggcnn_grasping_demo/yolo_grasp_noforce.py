#!/usr/bin/env python3
"""
yolo_grasp_noforce.py - YOLO fruit detection with xArm grasping (no force sensor)
Detects fruits using YOLOv5, converts pixel+depth to 3D coordinates, and picks them.
Gripper closes to a fixed position (no force feedback).
"""

import argparse
import cv2
import torch
import numpy as np
import time
import sys
import warnings
import math

sys.path.append('.')

from camera.depthai_camera import DepthAiCamera
from xarm.wrapper import XArmAPI

# Silence warnings
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

# Configuration
AVAILABLE_MODELS = ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]
KNOWN_FRUIT_LABELS = {"apple", "banana", "orange"}
DETECTION_INTERVAL = 5  # Run inference every N frames

# Safety limits (in mm, relative to robot base)
FLOOR_Z_MM = 77.0           # Table floor height
MIN_SAFE_Z_MM = 80.0        # Minimum Z to avoid floor collision
MAX_Z_MM = 500.0            # Maximum reach height
WORKSPACE_X_MIN = 150.0     # Workspace boundaries
WORKSPACE_X_MAX = 500.0
WORKSPACE_Y_MIN = -300.0
WORKSPACE_Y_MAX = 300.0

# Camera-to-gripper hand-eye calibration (from demo code)
# Camera is mounted ON the gripper, looking down
EULER_EEF_TO_COLOR_OPT = [0.0703, 0.0023, 0.0195, 0, 0, 1.579]  # [x, y, z, roll, pitch, yaw] meters/rad
EULER_COLOR_TO_DEPTH_OPT = [0.0375, 0, 0, 0, 0, 0]

# Robot positions (from demo, adjusted for 77mm floor)
DETECT_XYZ = [300, 0, 490]      # Initial scanning position [x, y, z] mm
RELEASE_XYZ = [400, 400, 360]   # Where to place fruits [x, y, z] mm
LIFT_OFFSET_Z = 100             # How much to lift after grasp, mm

# Gripper geometry and control (no force sensor)
GRIPPER_OPEN_POS = 850          # Fully open (xArm typical)
GRIPPER_CLOSE_POS = 300         # Fixed close position
GRIPPER_Z_MM = 70               # Distance from flange to gripper contact point, mm
GRASPING_MIN_Z = 76             # Floor height (77mm - safety margin), mm

# Workspace limits
GRASPING_RANGE = [180, 600, -200, 200]  # [x_min, x_max, y_min, y_max] mm

# Movement parameters
MOVE_SPEED = 50
MOVE_ACC = 250


def pixel_to_3d(pixel_x, pixel_y, depth_m, camera_intrinsics):
    """
    Convert pixel coordinates + depth to 3D camera coordinates.
    Returns (x, y, z) in camera frame (meters)
    """
    fx = camera_intrinsics[0][0]
    fy = camera_intrinsics[1][1]
    cx = camera_intrinsics[0][2]
    cy = camera_intrinsics[1][2]

    z_cam = depth_m
    x_cam = (pixel_x - cx) * z_cam / fx
    y_cam = (pixel_y - cy) * z_cam / fy
    return x_cam, y_cam, z_cam


def camera_to_robot(x_cam, y_cam, z_cam, current_robot_pose):
    """
    Transform camera coordinates to robot base coordinates (eye-in-hand, simplified).
    Returns (x, y, z) in robot base frame (mm)
    """
    # Convert camera coords to mm
    x_cam_mm = x_cam * 1000
    y_cam_mm = y_cam * 1000
    z_cam_mm = z_cam * 1000

    # Hand-eye transform: camera to end-effector frame using demo calibration
    eef_to_cam_x = EULER_EEF_TO_COLOR_OPT[0] * 1000
    eef_to_cam_y = EULER_EEF_TO_COLOR_OPT[1] * 1000
    eef_to_cam_z = EULER_EEF_TO_COLOR_OPT[2] * 1000

    # Simplified: camera looks down along Z of end-effector
    x_eef = x_cam_mm + eef_to_cam_x
    y_eef = y_cam_mm + eef_to_cam_y
    z_eef = z_cam_mm + eef_to_cam_z + GRIPPER_Z_MM

    # Transform from end-effector to robot base (assuming we know current pose)
    x_robot = current_robot_pose[0] + x_eef
    y_robot = current_robot_pose[1] + y_eef
    z_robot = z_eef
    return x_robot, y_robot, z_robot


def check_workspace_bounds(x, y, z):
    if z < MIN_SAFE_Z_MM:
        print(f"⚠ Z={z:.1f}mm is below safe minimum {MIN_SAFE_Z_MM}mm")
        return False
    if z > MAX_Z_MM:
        print(f"⚠ Z={z:.1f}mm exceeds maximum {MAX_Z_MM}mm")
        return False
    if not (WORKSPACE_X_MIN <= x <= WORKSPACE_X_MAX):
        print(f"⚠ X={x:.1f}mm outside workspace [{WORKSPACE_X_MIN}, {WORKSPACE_X_MAX}]")
        return False
    if not (WORKSPACE_Y_MIN <= y <= WORKSPACE_Y_MAX):
        print(f"⚠ Y={y:.1f}mm outside workspace [{WORKSPACE_Y_MIN}, {WORKSPACE_Y_MAX}]")
        return False
    return True


def grasp_fruit_no_force(arm, x, y, z):
    """
    Execute grasp sequence for detected fruit (no force sensor):
    - Open gripper
    - Move above target
    - Lower to grasp Z
    - Close to fixed position
    - Lift, move to release, open, return
    """
    print(f"\n{'='*60}")
    print(f"GRASPING (NO-FORCE) at X={x:.1f}, Y={y:.1f}, Z={z:.1f}mm")
    print(f"{'='*60}")

    if not (GRASPING_RANGE[0] <= x <= GRASPING_RANGE[1] and 
            GRASPING_RANGE[2] <= y <= GRASPING_RANGE[3]):
        print(f"✗ Target outside workspace range {GRASPING_RANGE}")
        return False
    if z < GRASPING_MIN_Z:
        print(f"✗ Target Z={z:.1f}mm below floor minimum {GRASPING_MIN_Z}mm")
        return False

    try:
        # 1) Open gripper
        print("1. Opening gripper...")
        arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)

        # 2) Move above target
        approach_z = DETECT_XYZ[2]
        print("2. Moving above target...")
        arm.set_position(x, y, approach_z, 180, 0, 0, speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)

        # 3) Lower to grasp height
        print(f"3. Lowering to Z={z:.1f}mm...")
        arm.set_position(x, y, z, 180, 0, 0, speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)

        # 4) Close to fixed position
        print(f"4. Closing gripper to {GRIPPER_CLOSE_POS}...")
        arm.set_gripper_position(GRIPPER_CLOSE_POS, wait=True, speed=6000)

        # 5) Set TCP load
        arm.set_tcp_load(0.3, [0, 0, 30])
        arm.set_state(0)

        # 6) Lift up
        lift_z = DETECT_XYZ[2] + LIFT_OFFSET_Z
        print(f"5. Lifting to Z={lift_z}mm...")
        arm.set_position(x, y, lift_z, 180, 0, 0, speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)

        # 7) Move to release
        print("6. Moving to release position...")
        arm.set_position(*RELEASE_XYZ, 180, 0, 0, speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)

        # 8) Open gripper to release
        print("7. Releasing object...")
        arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
        arm.set_tcp_load(0, [0, 0, 30])
        arm.set_state(0)

        # 9) Lift slightly before returning
        lift_release_z = RELEASE_XYZ[2] + LIFT_OFFSET_Z
        arm.set_position(RELEASE_XYZ[0], RELEASE_XYZ[1], lift_release_z, 180, 0, 0, speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)

        # 10) Return to scan
        print("8. Returning to scan position...")
        arm.set_position(*DETECT_XYZ, 180, 0, 0, speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)

        print("✓ Grasp (no-force) completed successfully")
        return True

    except Exception as e:
        print(f"✗ Error during grasp sequence: {e}")
        try:
            arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
            arm.set_position(*DETECT_XYZ, 180, 0, 0, wait=True)
        except:
            pass
        return False


def run_yolo_grasp_noforce(robot_ip, model_name="yolov5m"):
    """Main loop: detect fruits and grasp them (no force sensor)."""

    print("=" * 60)
    print("YOLOv5 Fruit Detection + Grasping (No Force Sensor)")
    print("=" * 60)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n1. Loading YOLOv5 model...")
    print(f"   Model: {model_name}")
    print(f"   Device: {device}")

    try:
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        model.to(device)
        model.eval()
        model.conf = 0.35  # Confidence for grasping
        model.iou = 0.45
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Initialize camera
    print("\n2. Initializing OAK-D camera...")
    try:
        camera = DepthAiCamera(width=640, height=400, disable_rgb=False)
        _, depth_intrinsics = camera.get_intrinsics()
        print("✓ Camera initialized")
        print(f"   Intrinsics: fx={depth_intrinsics[0][0]:.1f}, fy={depth_intrinsics[1][1]:.1f}")
    except Exception as e:
        print(f"✗ Failed to initialize camera: {e}")
        return

    # Initialize robot
    print(f"\n3. Connecting to xArm at {robot_ip}...")
    try:
        arm = XArmAPI(robot_ip)
        time.sleep(0.5)
        arm.set_mode(0)
        arm.set_state(0)
        arm.reset(wait=True)
        print("✓ Robot connected and ready")
    except Exception as e:
        print(f"✗ Failed to connect to robot: {e}")
        print("   Continuing in CAMERA-ONLY mode (no motion).")
        arm = None

    # Move to scan position (if robot connected)
    print(f"\n4. Moving to detection position at {DETECT_XYZ}...")
    if arm is not None:
        try:
            arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
            arm.set_position(*DETECT_XYZ, 180, 0, 0, speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)
        except Exception as e:
            print(f"✗ Failed to move to scan position: {e}")
    else:
        print("   Skipping robot move (not connected)")

    print("\n5. Starting detection loop...")
    print("   Controls: q=quit, g=grasp selected, s=next detection")
    print("-" * 40)

    # Ensure window is created and visible
    try:
        cv2.namedWindow('YOLO Grasp (No Force)', cv2.WINDOW_NORMAL)
    except Exception:
        pass

    frame_count = 0
    last_detections = None
    selected_idx = 0
    pending_pick = None  # {'xr','yr','zr','cx','cy','label','conf'} awaiting approval

    while True:
        color_image, depth_image = camera.get_images()
        if color_image is None:
            continue

        frame_count += 1

        # Run detection every N frames
        if frame_count % DETECTION_INTERVAL == 0:
            with torch.no_grad():
                results = model(color_image)

            detections = results.pandas().xyxy[0]
            if 'name' in detections.columns:
                mask = detections['name'].str.lower().isin(KNOWN_FRUIT_LABELS)
                fruit_detections = detections[mask]
            else:
                fruit_detections = detections.iloc[0:0]

            last_detections = fruit_detections.copy()

            if not fruit_detections.empty:
                print(f"\nFrame {frame_count}: Found {len(fruit_detections)} fruits")
                for i, (_, det) in enumerate(fruit_detections.iterrows()):
                    marker = "→" if i == selected_idx else " "
                    print(f"{marker} [{i}] {det['name']}: conf={det['confidence']:.2%}")

        # Draw detections
        display_image = color_image.copy()
        if last_detections is not None and not last_detections.empty:
            for i, (_, det) in enumerate(last_detections.iterrows()):
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                conf = det['confidence']
                label = det['name']

                color = (0, 255, 255) if i == selected_idx else (0, 200, 0)
                thickness = 4 if i == selected_idx else 2

                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.drawMarker(display_image, (center_x, center_y), color, cv2.MARKER_CROSS, 20, 2)
                label_text = f"[{i}] {label}: {conf:.2f}"
                cv2.putText(display_image, label_text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Controls and status
        status_lines = [
            "s=next, g=estimate, y=confirm, n=cancel, q=quit"
        ]
        if arm is None:
            status_lines.append("Robot: NOT connected (camera-only)")
        if pending_pick is not None:
            status_lines.append(
                f"Pending: X={pending_pick['xr']:.1f} Y={pending_pick['yr']:.1f} Z={pending_pick['zr']:.1f} mm"
            )
        for idx, text in enumerate(status_lines):
            cv2.putText(display_image, text, (10, 30 + idx*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Visualize pending target
        if pending_pick is not None:
            px, py = pending_pick['cx'], pending_pick['cy']
            cv2.circle(display_image, (px, py), 8, (0, 255, 255), 2)
            cv2.putText(display_image, "PENDING", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow('YOLO Grasp (No Force)', display_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and last_detections is not None and not last_detections.empty:
            selected_idx = (selected_idx + 1) % len(last_detections)
            print(f"Selected detection #{selected_idx}")
        elif key == ord('g') and last_detections is not None and not last_detections.empty:
            sel = last_detections.iloc[selected_idx]
            cx = int((sel['xmin'] + sel['xmax']) / 2)
            cy = int((sel['ymin'] + sel['ymax']) / 2)

            if depth_image is not None and 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                depth_val = depth_image[cy, cx]
                if depth_val is not None and not np.isnan(depth_val):
                    x_cam, y_cam, z_cam = pixel_to_3d(cx, cy, depth_val, depth_intrinsics)
                    current_pose = DETECT_XYZ + [180, 0, 0]
                    xr, yr, zr = camera_to_robot(x_cam, y_cam, z_cam, current_pose)

                    print(f"\nSelected: {sel['name']}")
                    print(f"  Pixel: ({cx}, {cy})  Depth: {depth_val:.3f}m")
                    print(f"  Camera: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f}) m")
                    print(f"  Robot:  ({xr:.1f}, {yr:.1f}, {zr:.1f}) mm")
                    # Store pending target for confirmation
                    pending_pick = {
                        'xr': xr, 'yr': yr, 'zr': zr,
                        'cx': cx, 'cy': cy,
                        'label': sel['name'], 'conf': float(sel['confidence']) if 'confidence' in sel else None
                    }
                    print("→ Pending pick stored. Press 'y' to confirm or 'n' to cancel.")
                else:
                    print("✗ No valid depth at selected point")
            else:
                print("✗ Depth image not available or index out of range")
        elif key == ord('y'):
            if pending_pick is None:
                continue
            if arm is None:
                print("✗ Robot not connected; cannot execute grasp")
                pending_pick = None
                continue
            xr, yr, zr = pending_pick['xr'], pending_pick['yr'], pending_pick['zr']
            print(f"✔ Approved. Executing grasp at X={xr:.1f}, Y={yr:.1f}, Z={zr:.1f} mm")
            grasp_fruit_no_force(arm, xr, yr, zr)
            pending_pick = None
        elif key == ord('n'):
            if pending_pick is not None:
                print("✗ Pending pick canceled")
            pending_pick = None

    # Cleanup
    cv2.destroyAllWindows()
    if 'arm' in locals() and arm is not None:
        try:
            arm.reset(wait=True)
        except:
            pass

    print("\n" + "=" * 60)
    print("Session completed")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO grasping without force sensor (fixed gripper close)")
    parser.add_argument("robot_ip", help="xArm IP address (e.g., 192.168.1.221)")
    parser.add_argument("--model", default="yolov5m", choices=AVAILABLE_MODELS, help="YOLOv5 model variant")
    parser.add_argument("--close-pos", type=int, default=GRIPPER_CLOSE_POS, help="Gripper close position (default 300)")
    parser.add_argument("--speed", type=int, default=MOVE_SPEED, help="Move speed mm/s (default 100)")
    parser.add_argument("--acc", type=int, default=MOVE_ACC, help="Move accel (default 500)")

    args = parser.parse_args()

    # Allow quick overrides via CLI
    if args.close_pos != GRIPPER_CLOSE_POS:
        GRIPPER_CLOSE_POS = args.close_pos  # type: ignore
    if args.speed != MOVE_SPEED:
        MOVE_SPEED = args.speed  # type: ignore
    if args.acc != MOVE_ACC:
        MOVE_ACC = args.acc  # type: ignore

    run_yolo_grasp_noforce(args.robot_ip, args.model)
