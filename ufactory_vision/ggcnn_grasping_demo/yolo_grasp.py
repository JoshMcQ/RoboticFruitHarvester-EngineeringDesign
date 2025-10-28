#!/usr/bin/env python3
"""
yolo_grasp.py - YOLO fruit detection with automated xArm grasping
Detects fruits using YOLOv5, converts pixel+depth to 3D coordinates, and picks them.
"""

import argparse
import cv2
import torch
import numpy as np
import time
import sys
import warnings
import serial
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

# Robot configuration
FORCE_PORT = "COM5"
FORCE_BAUD = 9600
FORCE_THRESHOLD = 100.0

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
DETECT_XYZ = [300, 0, 490]  # Initial scanning position [x, y, z] mm
RELEASE_XYZ = [400, 400, 360]  # Where to place fruits [x, y, z] mm
LIFT_OFFSET_Z = 100  # How much to lift after grasp, mm

# Gripper geometry
GRIPPER_Z_MM = 70  # Distance from flange to gripper contact point, mm
GRASPING_MIN_Z = 76  # Floor height (77mm - safety margin), mm

# Workspace limits
GRASPING_RANGE = [180, 600, -200, 200]  # [x_min, x_max, y_min, y_max] mm

# Movement parameters
MOVE_SPEED = 50
MOVE_ACC = 250


def pixel_to_3d(pixel_x, pixel_y, depth_m, camera_intrinsics):
    """
    Convert pixel coordinates + depth to 3D camera coordinates.
    
    Args:
        pixel_x, pixel_y: Pixel coordinates in image
        depth_m: Depth in meters at that pixel
        camera_intrinsics: 3x3 camera intrinsic matrix
    
    Returns:
        (x, y, z) in camera frame (meters)
    """
    fx = camera_intrinsics[0][0]
    fy = camera_intrinsics[1][1]
    cx = camera_intrinsics[0][2]
    cy = camera_intrinsics[1][2]
    
    # Convert pixel to camera coordinates
    z_cam = depth_m
    x_cam = (pixel_x - cx) * z_cam / fx
    y_cam = (pixel_y - cy) * z_cam / fy
    
    return x_cam, y_cam, z_cam


def camera_to_robot(x_cam, y_cam, z_cam, current_robot_pose):
    """
    Transform camera coordinates to robot base coordinates.
    Camera is mounted on the gripper (eye-in-hand), so we need to transform
    through the current end-effector pose.
    
    Args:
        x_cam, y_cam, z_cam: Coordinates in camera frame (meters)
        current_robot_pose: Current [x, y, z, roll, pitch, yaw] of end-effector
    
    Returns:
        (x, y, z) in robot base frame (mm)
    """
    # Convert camera coords to mm
    x_cam_mm = x_cam * 1000
    y_cam_mm = y_cam * 1000
    z_cam_mm = z_cam * 1000
    
    # Hand-eye transform: camera to end-effector frame
    # Using calibration from demo (EULER_EEF_TO_COLOR_OPT)
    eef_to_cam_x = EULER_EEF_TO_COLOR_OPT[0] * 1000  # Convert m to mm
    eef_to_cam_y = EULER_EEF_TO_COLOR_OPT[1] * 1000
    eef_to_cam_z = EULER_EEF_TO_COLOR_OPT[2] * 1000
    
    # Simplified transform (assumes camera looking straight down along Z)
    # In end-effector frame: camera sees positive X as robot positive X
    x_eef = x_cam_mm + eef_to_cam_x
    y_eef = y_cam_mm + eef_to_cam_y
    z_eef = z_cam_mm + eef_to_cam_z + GRIPPER_Z_MM  # Add gripper length
    
    # Transform from end-effector to robot base
    # Since we're at DETECT_XYZ looking down, this is simplified
    x_robot = current_robot_pose[0] + x_eef
    y_robot = current_robot_pose[1] + y_eef
    z_robot = z_eef  # Depth measurement gives absolute Z
    
    return x_robot, y_robot, z_robot


def check_workspace_bounds(x, y, z):
    """Check if target position is within safe workspace."""
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


def smooth_close_with_force(arm, ser, threshold=FORCE_THRESHOLD, start=850, end=0):
    """
    Adaptive gripper close with force feedback (from your test2.py)
    """
    import math, time

    EMA_ALPHA = 0.25
    BASE_STEP = 18
    MIN_STEP = 4
    V_MAX = 9000
    V_MIN = 1200
    SAMPLE_DT = 0.004
    WINDOW_S = 0.03
    HITS_NEEDED = 1
    HYST_RATIO = 0.92
    DWELL_S = 0.015

    def smoothstep01(x):
        x = 0.0 if x < 0 else (1.0 if x > 1 else x)
        return x*x*(3 - 2*x)

    def read_force(ser_obj, timeout=0.6):
        """Read force sensor value"""
        import re
        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                line = ser_obj.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                match = re.search(r"(\d+(?:\.\d+)?)", line)
                if match:
                    return float(match.group(1))
                time.sleep(0.001)
        except Exception as exc:
            print(f"[force] read error: {exc}")
        return None

    try:
        arm.set_gripper_enable(True)
        arm.set_gripper_mode(0)
    except Exception:
        pass

    f_ema, hits = 0.0, 0
    pos = start
    while pos >= end:
        f_ratio = f_ema / max(1e-6, float(threshold))
        s = smoothstep01(f_ratio)
        step = int(max(MIN_STEP, BASE_STEP*(1 - 0.8*s)))
        speed = int(V_MIN + (V_MAX - V_MIN)*(1 - s))

        arm.set_gripper_position(pos, wait=False, speed=speed)

        t_end, got = time.time() + WINDOW_S, 0
        while time.time() < t_end:
            v = read_force(ser, timeout=SAMPLE_DT)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                got += 1
                f_ema = EMA_ALPHA*v + (1-EMA_ALPHA)*f_ema
                if f_ema >= threshold:
                    hits += 1
                    if hits >= HITS_NEEDED:
                        print(f"[force] Contact detected at pos={pos}, stopping")
                        arm.set_gripper_position(pos, wait=True, speed=max(V_MIN, 800))
                        return True
                elif f_ema < threshold*HYST_RATIO:
                    hits = 0
            time.sleep(SAMPLE_DT)

        time.sleep(DWELL_S)
        pos -= step

    print("[force] Completed close without force threshold")
    arm.set_gripper_position(end, wait=True, speed=V_MIN)
    return False


def grasp_fruit(arm, ser, x, y, z):
    """
    Execute grasp sequence for detected fruit.
    Follows the demo's approach: move above target, lower, grasp, lift, place.
    
    Args:
        arm: XArmAPI instance
        ser: Serial connection for force sensor
        x, y, z: Target position in robot coordinates (mm)
    
    Returns:
        bool: True if grasp successful
    """
    print(f"\n{'='*60}")
    print(f"GRASPING FRUIT at X={x:.1f}, Y={y:.1f}, Z={z:.1f}mm")
    print(f"{'='*60}")
    
    # Safety check - use demo's GRASPING_RANGE
    if not (GRASPING_RANGE[0] <= x <= GRASPING_RANGE[1] and 
            GRASPING_RANGE[2] <= y <= GRASPING_RANGE[3]):
        print(f"✗ Target outside workspace range {GRASPING_RANGE}")
        return False
    
    if z < GRASPING_MIN_Z:
        print(f"✗ Target Z={z:.1f}mm below floor minimum {GRASPING_MIN_Z}mm")
        return False
    
    try:
        # 1. Open gripper
        print("1. Opening gripper...")
        arm.set_gripper_position(850, wait=True)
        
        # 2. Move above target (approach from detection height)
        approach_z = DETECT_XYZ[2]  # Use detection height
        print(f"2. Moving to approach position above target...")
        arm.set_position(x, y, approach_z, 180, 0, 0, 
                        speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)
        
        # 3. Lower to grasp height
        # Target Z already accounts for depth + gripper geometry
        print(f"3. Lowering to grasp height (Z={z:.1f}mm)...")
        arm.set_position(x, y, z, 180, 0, 0,
                        speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)
        
        # 4. Close gripper with force feedback
        print("4. Closing gripper with force feedback...")
        grasp_success = smooth_close_with_force(arm, ser)
        
        if not grasp_success:
            print("✗ No object detected in gripper")
            arm.set_gripper_position(850, wait=True)
            # Return to detection height
            arm.set_position(x, y, DETECT_XYZ[2], 180, 0, 0, wait=True)
            return False
        
        # 5. Set TCP load for grasped object
        arm.set_tcp_load(0.3, [0, 0, 30])
        arm.set_state(0)
        
        # 6. Lift up
        lift_z = DETECT_XYZ[2] + LIFT_OFFSET_Z
        print(f"5. Lifting to Z={lift_z}mm...")
        arm.set_position(x, y, lift_z, 180, 0, 0,
                        speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)
        
        # 7. Move to release position
        print("6. Moving to release position...")
        arm.set_position(*RELEASE_XYZ, 180, 0, 0,
                        speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)
        
        # 8. Release
        print("7. Releasing fruit...")
        arm.set_gripper_position(850, wait=True)
        arm.set_tcp_load(0, [0, 0, 30])
        arm.set_state(0)
        
        # 9. Lift before returning
        lift_release_z = RELEASE_XYZ[2] + LIFT_OFFSET_Z
        arm.set_position(RELEASE_XYZ[0], RELEASE_XYZ[1], lift_release_z, 180, 0, 0,
                        speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)
        
        # 10. Return to scan position
        print("8. Returning to scan position...")
        arm.set_position(*DETECT_XYZ, 180, 0, 0,
                        speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)
        
        print("✓ Grasp sequence completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error during grasp sequence: {e}")
        # Emergency: open gripper and return to safe height
        try:
            arm.set_gripper_position(850, wait=True)
            arm.set_position(*DETECT_XYZ, 180, 0, 0, wait=True)
        except:
            pass
        return False


def run_yolo_grasp(robot_ip, model_name="yolov5m"):
    """Main loop: detect fruits and grasp them."""
    
    print("=" * 60)
    print("YOLOv5 Fruit Detection + Automated Grasping")
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
        model.conf = 0.35  # Higher confidence for grasping
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
        return
    
    # Initialize force sensor
    print(f"\n4. Connecting to force sensor on {FORCE_PORT}...")
    try:
        ser = serial.Serial(FORCE_PORT, FORCE_BAUD, timeout=0.5)
        ser.reset_input_buffer()
        print("✓ Force sensor ready")
    except Exception as e:
        print(f"✗ Failed to connect to force sensor: {e}")
        print("   Continuing without force feedback...")
        ser = None
    
    # Move to scan position
    print(f"\n5. Moving to detection position at {DETECT_XYZ}...")
    try:
        arm.set_position(*DETECT_XYZ, 180, 0, 0, 
                        speed=MOVE_SPEED, mvacc=MOVE_ACC, wait=True)
    except Exception as e:
        print(f"✗ Failed to move to scan position: {e}")
        return
    
    print("\n6. Starting detection loop...")
    print("   Controls: q=quit, g=grasp selected fruit, s=skip current")
    print("-" * 40)
    
    frame_count = 0
    last_detections = None
    selected_fruit_idx = 0
    
    while True:
        color_image, depth_image = camera.get_images()
        
        if color_image is None:
            continue
        
        frame_count += 1
        
        # Run detection
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
                for idx, fruit in fruit_detections.iterrows():
                    marker = "→" if idx == selected_fruit_idx else " "
                    print(f"{marker} [{idx}] {fruit['name']}: conf={fruit['confidence']:.2%}")
        
        # Draw detections
        display_image = color_image.copy()
        
        if last_detections is not None and not last_detections.empty:
            for idx, (_, det) in enumerate(last_detections.iterrows()):
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                conf = det['confidence']
                label = det['name']
                
                # Highlight selected fruit
                if idx == selected_fruit_idx:
                    color = (0, 255, 255)  # Yellow for selected
                    thickness = 4
                else:
                    color = (0, 200, 0)
                    thickness = 2
                
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label
                label_text = f"[{idx}] {label}: {conf:.2f}"
                cv2.putText(display_image, label_text, (x1, max(y1 - 10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw center crosshair
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.drawMarker(display_image, (center_x, center_y), color,
                              cv2.MARKER_CROSS, 20, 2)
        
        # Status overlay
        cv2.putText(display_image, f"Frame: {frame_count} | Press 'g' to grasp, 'q' to quit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('YOLO Fruit Grasping', display_image)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g') and last_detections is not None and not last_detections.empty:
            # Grasp selected fruit
            selected = last_detections.iloc[selected_fruit_idx]
            center_x = int((selected['xmin'] + selected['xmax']) / 2)
            center_y = int((selected['ymin'] + selected['ymax']) / 2)
            
            # Get depth at fruit center
            if depth_image is not None and not np.isnan(depth_image[center_y, center_x]):
                depth_m = depth_image[center_y, center_x]
                
                # Convert pixel + depth to 3D camera coordinates
                x_cam, y_cam, z_cam = pixel_to_3d(center_x, center_y, depth_m, depth_intrinsics)
                
                # Get current robot pose (we're at DETECT_XYZ during scanning)
                current_pose = DETECT_XYZ + [180, 0, 0]  # [x, y, z, roll, pitch, yaw]
                
                # Transform to robot base coordinates
                x_robot, y_robot, z_robot = camera_to_robot(x_cam, y_cam, z_cam, current_pose)
                
                print(f"\nSelected: {selected['name']}")
                print(f"  Pixel: ({center_x}, {center_y})")
                print(f"  Depth: {depth_m:.3f}m")
                print(f"  Camera coords: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})m")
                print(f"  Robot coords: ({x_robot:.1f}, {y_robot:.1f}, {z_robot:.1f})mm")
                
                if ser is not None:
                    grasp_fruit(arm, ser, x_robot, y_robot, z_robot)
                else:
                    print("✗ Cannot grasp without force sensor")
            else:
                print("✗ No valid depth at fruit location")
        
        elif key == ord('s') and last_detections is not None:
            # Select next fruit
            selected_fruit_idx = (selected_fruit_idx + 1) % len(last_detections)
            print(f"Selected fruit #{selected_fruit_idx}")
    
    # Cleanup
    cv2.destroyAllWindows()
    if ser:
        ser.close()
    arm.reset(wait=True)
    
    print("\n" + "=" * 60)
    print("Session completed")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO fruit detection with automated grasping")
    parser.add_argument(
        "robot_ip",
        help="xArm IP address (e.g., 192.168.1.221)",
    )
    parser.add_argument(
        "--model",
        default="yolov5m",
        choices=AVAILABLE_MODELS,
        help="YOLOv5 model variant (default: yolov5m)",
    )
    
    args = parser.parse_args()
    run_yolo_grasp(args.robot_ip, args.model)
