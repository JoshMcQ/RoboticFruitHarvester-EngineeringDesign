#!/usr/bin/env python3
"""
Simple AprilTag position verification using known tag locations.
Uses the same transformation logic as the working YOLO code.
"""

import time
import argparse
import numpy as np
import cv2
from pupil_apriltags import Detector

from xarm.wrapper import XArmAPI
from camera.depthai_camera import DepthAiCamera

# ---- Robot setup ----
arm = XArmAPI('192.168.1.221')
time.sleep(0.5)
arm.set_mode(0)
arm.set_state(0)
arm.reset(wait=True)

# Observation pose (same as YOLO)
OBS_POSE = [357.4, 1.1, 231.7, 178.8, 0.3, 1.0]
arm.set_gripper_position(850, wait=True)
arm.set_position(*OBS_POSE, wait=True)

# ---- Current hand-eye constants from YOLO (camera to gripper) ----
EULER_EEF_TO_COLOR_OPT = [0.0703, 0.0023, 0.0195, 0, 0, 1.579]  # m/rad

# ---- Known AprilTag positions (from your apriltag file) ----
TAG_POSITIONS_MM = {
    0: [282.3,   44.6,  -89.3],  # BL
    3: [290.1, -118.3,  -89.5],  # BR
    1: [445.4, -128.4,  -86.0],  # ML
    5: [440.5,   45.3,  -86.3],  # MR
    2: [606.5,   54.0,  -83.2],  # TL
    4: [603.2, -131.6,  -81.0],  # TR
}

TAG_SIZE_M = 0.0058  # 5.8mm black square

# ---- Same transformation functions as YOLO ----

def _euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg):
    """Rotation matrix from roll/pitch/yaw in degrees (ZYX order)."""
    rd, pd, yd = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(rd), np.sin(rd)
    cp, sp = np.cos(pd), np.sin(pd)
    cy, sy = np.cos(yd), np.sin(yd)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def _euler_rad_to_rot(rx, ry, rz):
    """Rotation matrix from roll/pitch/yaw in radians (ZYX order)."""
    cr, sr = np.cos(rx), np.sin(rx)
    cp, sp = np.cos(ry), np.sin(ry)
    cy, sy = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def camera_to_robot(x_cam_m, y_cam_m, z_cam_m, eef_pose_xyzrpy):
    """
    Transform camera frame point to robot base frame.
    Same logic as YOLO code.
    """
    # Camera point in meters
    p_cam = np.array([x_cam_m, y_cam_m, z_cam_m], dtype=float)
    
    # EEF->Camera transform
    t_eef_cam = np.array(EULER_EEF_TO_COLOR_OPT[:3], dtype=float)  # meters
    rx_c, ry_c, rz_c = EULER_EEF_TO_COLOR_OPT[3:6]
    R_eef_cam = _euler_rad_to_rot(rx_c, ry_c, rz_c)
    
    # Transform to EEF frame
    p_eef = R_eef_cam @ p_cam + t_eef_cam
    
    # Base->EEF transform
    x_eef_mm, y_eef_mm, z_eef_mm, roll_deg, pitch_deg, yaw_deg = eef_pose_xyzrpy
    t_base_eef = np.array([x_eef_mm, y_eef_mm, z_eef_mm], dtype=float) / 1000.0  # to meters
    R_base_eef = _euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg)
    
    # Final point in base (meters)
    p_base = t_base_eef + R_base_eef @ p_eef
    
    # Return in mm
    return float(p_base[0] * 1000.0), float(p_base[1] * 1000.0), float(p_base[2] * 1000.0)

def main():
    parser = argparse.ArgumentParser(description='Verify AprilTag positions with known locations')
    parser.add_argument('--frames', type=int, default=10, help='Number of frames to average')
    args = parser.parse_args()
    
    # Initialize camera
    print("Starting OAK-D camera (640x400)...")
    cam = DepthAiCamera(width=640, height=400, disable_rgb=False)
    K_rgb, _ = cam.get_intrinsics()
    
    # Camera parameters for AprilTag detector
    cam_params = [
        float(K_rgb[0][0]),  # fx
        float(K_rgb[1][1]),  # fy
        float(K_rgb[0][2]),  # cx
        float(K_rgb[1][2]),  # cy
    ]
    
    # Initialize AprilTag detector
    print("Initializing AprilTag detector...")
    detector = Detector(
        families='tag36h11',
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )
    
    print(f"\nCollecting {args.frames} frames from observation pose...")
    print(f"Current pose: {OBS_POSE}\n")
    
    # Collect detections over multiple frames
    tag_detections = {}  # tag_id -> list of (x, y, z) in camera frame
    
    for frame_idx in range(args.frames):
        color, _ = cam.get_images()
        if color is None:
            continue
            
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=cam_params,
            tag_size=TAG_SIZE_M,
        )
        
        for det in detections:
            tag_id = det.tag_id
            if tag_id not in TAG_POSITIONS_MM:
                continue  # Skip unknown tags
                
            # Get 3D position in camera frame (meters)
            t_cam = det.pose_t.reshape(3)
            x_cam, y_cam, z_cam = float(t_cam[0]), float(t_cam[1]), float(t_cam[2])
            
            if tag_id not in tag_detections:
                tag_detections[tag_id] = []
            tag_detections[tag_id].append((x_cam, y_cam, z_cam))
        
        time.sleep(0.05)
    
    if not tag_detections:
        print("ERROR: No known AprilTags detected!")
        return
    
    print(f"Detected {len(tag_detections)} tags. Analyzing positions...\n")
    print("="*80)
    print("Tag ID | Detected Position (mm)      | Known Position (mm)         | Error (mm)")
    print("="*80)
    
    errors = []
    offset_sum = np.array([0.0, 0.0, 0.0])
    
    for tag_id in sorted(tag_detections.keys()):
        # Average all detections of this tag
        positions = tag_detections[tag_id]
        avg_cam = np.mean(positions, axis=0)
        x_cam, y_cam, z_cam = avg_cam
        
        # Transform to robot base frame using current calibration
        xr, yr, zr = camera_to_robot(x_cam, y_cam, z_cam, OBS_POSE)
        detected_pos = np.array([xr, yr, zr])
        
        # Get known position
        known_pos = np.array(TAG_POSITIONS_MM[tag_id])
        
        # Calculate error
        error_vec = detected_pos - known_pos
        error_mag = np.linalg.norm(error_vec)
        errors.append(error_mag)
        offset_sum += error_vec
        
        # Print results
        print(f"  {tag_id:2d}   | X:{xr:7.1f} Y:{yr:7.1f} Z:{zr:7.1f} | "
              f"X:{known_pos[0]:7.1f} Y:{known_pos[1]:7.1f} Z:{known_pos[2]:7.1f} | "
              f"{error_mag:6.1f}")
        print(f"       | Error vector: ΔX={error_vec[0]:+7.1f} ΔY={error_vec[1]:+7.1f} ΔZ={error_vec[2]:+7.1f}")
        print("-"*80)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Tags detected: {len(tag_detections)}")
    print(f"Mean error: {np.mean(errors):.1f} mm")
    print(f"Max error:  {np.max(errors):.1f} mm")
    print(f"Min error:  {np.min(errors):.1f} mm")
    print(f"Std error:  {np.std(errors):.1f} mm")
    
    # Calculate average offset (could be used as calibration correction)
    avg_offset = offset_sum / len(tag_detections)
    print(f"\nAverage position offset (detected - known):")
    print(f"  ΔX = {avg_offset[0]:+7.1f} mm")
    print(f"  ΔY = {avg_offset[1]:+7.1f} mm")
    print(f"  ΔZ = {avg_offset[2]:+7.1f} mm")
    
    print("\n" + "="*80)
    print("SUGGESTED CALIBRATION CORRECTION")
    print("="*80)
    print("To improve accuracy, you could:")
    print("1. Use these offsets directly in your code (like --dx, --dy, --dz args)")
    print("2. Update EULER_EEF_TO_COLOR_OPT translation component")
    print("3. Run this from multiple observation poses to verify consistency")
    print("\nSuggested bias correction (subtract this from detected positions):")
    print(f"  --dx {avg_offset[0]:.1f} --dy {avg_offset[1]:.1f} --dz {avg_offset[2]:.1f}")
    
    # Save annotated image
    color, _ = cam.get_images()
    if color is not None:
        display = color.copy()
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray, estimate_tag_pose=False)
        
        for det in detections:
            if det.tag_id not in TAG_POSITIONS_MM:
                continue
            corners = det.corners.astype(int)
            cv2.polylines(display, [corners], True, (0, 255, 0), 2)
            center = tuple(det.center.astype(int))
            cv2.circle(display, center, 5, (0, 0, 255), -1)
            cv2.putText(display, f"ID:{det.tag_id}", (center[0]-20, center[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        filename = f"apriltag_verify_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, display)
        print(f"\nSaved annotated image: {filename}")

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            arm.reset(wait=True)
        except Exception:
            pass