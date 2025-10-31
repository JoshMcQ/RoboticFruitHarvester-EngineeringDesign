#!/usr/bin/env python3
"""
Advanced: Compute/refine camera-to-gripper transformation using AprilTags at known positions.
This uses multiple poses and optimization to find the best transformation.
"""

import time
import argparse
import numpy as np
import cv2
from pupil_apriltags import Detector
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from xarm.wrapper import XArmAPI
from camera.depthai_camera import DepthAiCamera

# ---- Robot setup ----
arm = XArmAPI('192.168.1.221')
time.sleep(0.5)
arm.set_mode(0)
arm.set_state(0)
arm.reset(wait=True)

# Multiple observation poses - using all hover poses from original apriltag file
OBSERVATION_POSES = [
    [357.4, 1.1, 231.7, 178.8, 0.3, 1.0],      # Standard OBS_POSE
    [289.3, -39.7, 428.6, 178.9, -13.4, -6.9], # View-all pose
    [245.1,   23.6,  -45.0,  -179.3,   6.3,   3.2],  # Hover over tag 0 (BL)
    [253.7, -148.2,  -28.3,   174.5,   3.2,   1.4],  # Hover over tag 3 (BR)
    [384.3,   25.3,  -23.8,  -179.4,   6.0,   4.8],  # Hover over tag 5 (ML)
    [396.0, -149.6,  -25.9,   173.1,   5.6,  -0.4],  # Hover over tag 1 (MR)
    [557.3,   28.7,  -17.2,  -179.5,  -5.2,   2.6],  # Hover over tag 2 (TL)
    [552.8, -144.1,  -14.4,   175.4,  -5.0,   3.5],  # Hover over tag 4 (TR)
]

# ---- Current hand-eye constants (initial guess) ----
# Camera points down when Roll≈180°, Pitch≈0°, Yaw≈0°
# Since all observation poses have same orientation, fix rotation and optimize only translation
# Rotation: rx=0, ry=0, rz=π/2 means camera frame rotated 90° around Z from EEF frame
# NOTE: With Roll≈180° (inverted Z), increasing tz makes predictions more negative in Z
# To match tags at Z≈-85mm, tz should be small (~45mm from single-pose solve)
EULER_EEF_TO_COLOR_OPT_INIT = [0.031, -0.016, 0.045, 0, 0, 1.5708]  # m/rad (use last optimized tx/ty, tz≈45mm)

# ---- Known AprilTag positions (in robot base frame) ----
# Robot base is at (0, 0, 0)
# These positions are measured/verified in the robot's base coordinate system
TAG_POSITIONS_MM = {
    0: [282.3,   44.6,  -89.3],  # BL
    3: [290.1, -118.3,  -89.5],  # BR
    1: [445.4, -128.4,  -86.0],  # ML
    5: [440.5,   45.3,  -86.3],  # MR
    2: [606.5,   54.0,  -83.2],  # TL
    4: [603.2, -131.6,  -81.0],  # TR
}

TAG_SIZE_M = 0.0058

# ---- Transformation functions ----

def _euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg, order='ZYX'):
    """Rotation matrix from roll/pitch/yaw in degrees.
    order='ZYX': intrinsic rotations (Rz*Ry*Rx) - default for most robots
    order='XYZ': extrinsic rotations (Rx*Ry*Rz) - alternate convention
    """
    rd, pd, yd = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(rd), np.sin(rd)
    cp, sp = np.cos(pd), np.sin(pd)
    cy, sy = np.cos(yd), np.sin(yd)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    
    if order == 'ZYX':
        return Rz @ Ry @ Rx  # intrinsic Z-Y-X
    elif order == 'XYZ':
        return Rx @ Ry @ Rz  # extrinsic X-Y-Z
    else:
        raise ValueError(f"Unknown order: {order}")

def _euler_rad_to_rot(rx, ry, rz):
    """Rotation matrix from roll/pitch/yaw in radians (Rz*Ry*Rx order - matches working code)."""
    cr, sr = np.cos(rx), np.sin(rx)
    cp, sp = np.cos(ry), np.sin(ry)
    cy, sy = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # This matches the working code's rpy_to_rot

EULER_ORDER = 'ZYX'  # Global variable set by command line arg

def _euler_to_mat4x4(euler_xyzrpy):
    """Convert [x,y,z,rx,ry,rz] to 4x4 homogeneous transform matrix (matches working code)."""
    x, y, z, rx, ry, rz = euler_xyzrpy
    R = _euler_rad_to_rot(rx, ry, rz)
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = [x, y, z]
    return mat

def camera_to_robot(x_cam_m, y_cam_m, z_cam_m, eef_pose_xyzrpy, eef_to_cam_params, use_forward=False):
    """
    Transform camera frame point to robot base frame using 4x4 matrices (matches working code).
    eef_to_cam_params = [tx, ty, tz, rx, ry, rz] (meters and radians)
    
    use_forward=True: Use forward multiplication like the working code
    """
    # Build transform chain: Base->EEF * EEF->Camera
    x_eef_mm, y_eef_mm, z_eef_mm, roll_deg, pitch_deg, yaw_deg = eef_pose_xyzrpy
    
    # Base->EEF transform (convert mm to meters, degrees to radians)
    R_base_to_eef = _euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg, order=EULER_ORDER)
    euler_base_to_eef = [
        x_eef_mm / 1000.0,
        y_eef_mm / 1000.0,
        z_eef_mm / 1000.0,
        np.deg2rad(roll_deg),
        np.deg2rad(pitch_deg),
        np.deg2rad(yaw_deg)
    ]
    mat_base_to_eef = np.eye(4)
    mat_base_to_eef[:3, :3] = R_base_to_eef
    mat_base_to_eef[:3, 3] = [x_eef_mm / 1000.0, y_eef_mm / 1000.0, z_eef_mm / 1000.0]
    
    # EEF->Camera transform
    mat_eef_to_cam = _euler_to_mat4x4(eef_to_cam_params)
    
    # Combined transform: Base->Camera = Base->EEF @ EEF->Camera
    mat_base_to_cam = mat_base_to_eef @ mat_eef_to_cam
    
    # Transform point from camera frame to base frame
    p_cam_homogeneous = np.array([x_cam_m, y_cam_m, z_cam_m, 1.0])
    p_base_homogeneous = mat_base_to_cam @ p_cam_homogeneous
    p_base = p_base_homogeneous[:3]
    
    return p_base * 1000.0  # return in mm

def collect_measurements(cam, detector, cam_params, pose, num_frames=10):
    """
    Collect AprilTag measurements at a given robot pose.
    Returns dict: tag_id -> average (x, y, z) in camera frame (meters)
    """
    print(f"  Moving to pose: {pose}")
    arm.set_position(*pose, wait=True)
    time.sleep(0.3)
    
    tag_detections = {}
    
    for _ in range(num_frames):
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
                continue
                
            t_cam = det.pose_t.reshape(3)
            x_cam, y_cam, z_cam = float(t_cam[0]), float(t_cam[1]), float(t_cam[2])
            
            if tag_id not in tag_detections:
                tag_detections[tag_id] = []
                # DEBUG: Print first detection of each tag to see camera frame values
                print(f"    Tag {tag_id} in camera frame: x={x_cam:.3f}, y={y_cam:.3f}, z={z_cam:.3f}")
            tag_detections[tag_id].append((x_cam, y_cam, z_cam))
        
        time.sleep(0.02)
    
    # Average detections
    averaged = {}
    for tag_id, positions in tag_detections.items():
        avg_pos = np.mean(positions, axis=0)
        averaged[tag_id] = avg_pos
    
    return averaged

def optimization_objective(params, measurements, use_forward=False, fixed_rotation=None):
    """
    Objective function for optimization.
    params = [tx, ty, tz, rx, ry, rz] OR just [tx, ty, tz] if fixed_rotation is provided
    measurements = list of (eef_pose, tag_id, cam_position, known_position)
    fixed_rotation = [rx, ry, rz] if only optimizing translation
    """
    # Reconstruct full params
    if fixed_rotation is not None:
        full_params = np.concatenate([params, fixed_rotation])
    else:
        full_params = params
    
    total_error = 0.0
    
    for eef_pose, tag_id, cam_pos, known_pos in measurements:
        x_cam, y_cam, z_cam = cam_pos
        predicted_pos = camera_to_robot(x_cam, y_cam, z_cam, eef_pose, full_params, use_forward=use_forward)
        error = np.linalg.norm(predicted_pos - known_pos)
        total_error += error ** 2
    
    return total_error

def main():
    parser = argparse.ArgumentParser(description='Compute/refine camera calibration using AprilTags')
    parser.add_argument('--frames-per-pose', type=int, default=10, 
                       help='Frames to average at each pose')
    parser.add_argument('--optimize', action='store_true',
                       help='Run optimization to find best calibration parameters')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify current calibration, no optimization')
    parser.add_argument('--use-forward-transform', action='store_true',
                       help='Use forward transform (p_eef = R @ p_cam + t) instead of inverse')
    parser.add_argument('--euler-order', type=str, default='ZYX', choices=['ZYX', 'XYZ'],
                       help='Euler angle order: ZYX (intrinsic, default) or XYZ (extrinsic)')
    parser.add_argument('--optimize-translation-only', action='store_true',
                       help='Only optimize translation (tx,ty,tz), keep rotation fixed')
    args = parser.parse_args()
    
    global EULER_ORDER
    EULER_ORDER = args.euler_order
    print(f"Using Euler order: {EULER_ORDER}")
    
    # Initialize camera
    print("Starting OAK-D camera (640x400)...")
    cam = DepthAiCamera(width=640, height=400, disable_rgb=False)
    K_rgb, _ = cam.get_intrinsics()
    
    cam_params = [
        float(K_rgb[0][0]),
        float(K_rgb[1][1]),
        float(K_rgb[0][2]),
        float(K_rgb[1][2]),
    ]
    
    # Initialize detector
    print("Initializing AprilTag detector...")
    detector = Detector(
        families='tag36h11',
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )
    
    # Move to safe starting position
    arm.set_gripper_position(850, wait=True)
    
    # Collect measurements from all poses
    print(f"\nCollecting measurements from {len(OBSERVATION_POSES)} pose(s)...")
    all_measurements = []  # (eef_pose, tag_id, cam_position, known_position)
    
    for pose_idx, pose in enumerate(OBSERVATION_POSES):
        print(f"\nPose {pose_idx + 1}/{len(OBSERVATION_POSES)}:")
        detections = collect_measurements(cam, detector, cam_params, pose, args.frames_per_pose)
        
        if not detections:
            print("  WARNING: No tags detected at this pose")
            continue
            
        print(f"  Detected {len(detections)} tags: {sorted(detections.keys())}")
        
        for tag_id, cam_pos in detections.items():
            known_pos = np.array(TAG_POSITIONS_MM[tag_id])
            all_measurements.append((pose, tag_id, cam_pos, known_pos))
    
    if not all_measurements:
        print("\nERROR: No measurements collected!")
        return
    
    print(f"\nTotal measurements collected: {len(all_measurements)}")
    print(f"Unique tags seen: {len(set(m[1] for m in all_measurements))}")
    
    # Evaluate current calibration
    print("\n" + "="*80)
    print("CURRENT CALIBRATION PERFORMANCE")
    print("="*80)
    
    current_params = EULER_EEF_TO_COLOR_OPT_INIT
    errors = []
    
    for eef_pose, tag_id, cam_pos, known_pos in all_measurements:
        x_cam, y_cam, z_cam = cam_pos
        predicted_pos = camera_to_robot(x_cam, y_cam, z_cam, eef_pose, current_params, 
                                       use_forward=args.use_forward_transform)
        error = np.linalg.norm(predicted_pos - known_pos)
        errors.append(error)
    
    print(f"Mean error:   {np.mean(errors):.2f} mm")
    print(f"Median error: {np.median(errors):.2f} mm")
    print(f"Max error:    {np.max(errors):.2f} mm")
    print(f"Std error:    {np.std(errors):.2f} mm")
    print(f"\nCurrent params: {current_params}")
    
    if args.verify_only:
        print("\nVerification complete (--verify-only flag set)")
        print("\n" + "="*80)
        print("TESTING WITH KNOWN WORKING PARAMETERS FROM run_depthai_grasp.py")
        print("="*80)
        # Test with the working parameters: [0.0703, 0.0023, 0.0195, 0, 0, 1.579]
        working_params = [0.0703, 0.0023, 0.0195, 0, 0, 1.579]
        working_errors = []
        for eef_pose, tag_id, cam_pos, known_pos in all_measurements:
            x_cam, y_cam, z_cam = cam_pos
            predicted_pos = camera_to_robot(x_cam, y_cam, z_cam, eef_pose, working_params,
                                           use_forward=True)
            error = np.linalg.norm(predicted_pos - known_pos)
            working_errors.append(error)
        print(f"Mean error with working params:   {np.mean(working_errors):.2f} mm")
        print(f"Median error with working params: {np.median(working_errors):.2f} mm")
        return
    
    # Optimization
    if args.optimize:
        print("\n" + "="*80)
        print("RUNNING OPTIMIZATION")
        print("="*80)
        print("This may take a minute...")
        
        # Use current params as initial guess
        x0 = np.array(current_params)
        
        # Set bounds (wider ranges to avoid hitting limits)
        if args.optimize_translation_only:
            print("NOTE: Optimizing TRANSLATION ONLY (rotation fixed)")
            x0 = np.array(current_params[:3])  # Only tx, ty, tz
            bounds = [
                (-0.2, 0.2),   # tx (meters)
                (-0.2, 0.2),   # ty
                (-0.2, 0.5),   # tz - increased upper bound since it hit 0.2
            ]
            fixed_rotation = current_params[3:]  # rx, ry, rz
        else:
            bounds = [
                (-0.5, 0.5),   # tx (meters) - relaxed from 0.2
                (-0.5, 0.5),   # ty
                (-0.5, 0.5),   # tz - relaxed to allow more range
                (-np.pi, np.pi),  # rx (radians)
                (-np.pi, np.pi),  # ry
                (-np.pi, np.pi),  # rz
            ]
            fixed_rotation = None
        
        result = minimize(
            optimization_objective,
            x0,
            args=(all_measurements, args.use_forward_transform, fixed_rotation),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-8}
        )
        
        if result.success:
            print("✓ Optimization converged!")
            if fixed_rotation is not None:
                optimized_params = np.concatenate([result.x, fixed_rotation])
            else:
                optimized_params = result.x
            
            # Evaluate optimized calibration
            print("\n" + "="*80)
            print("OPTIMIZED CALIBRATION PERFORMANCE")
            print("="*80)
            
            opt_errors = []
            for eef_pose, tag_id, cam_pos, known_pos in all_measurements:
                x_cam, y_cam, z_cam = cam_pos
                predicted_pos = camera_to_robot(x_cam, y_cam, z_cam, eef_pose, optimized_params,
                                               use_forward=args.use_forward_transform)
                error = np.linalg.norm(predicted_pos - known_pos)
                opt_errors.append(error)
            
            print(f"Mean error:   {np.mean(opt_errors):.2f} mm")
            print(f"Median error: {np.median(opt_errors):.2f} mm")
            print(f"Max error:    {np.max(opt_errors):.2f} mm")
            print(f"Std error:    {np.std(opt_errors):.2f} mm")
            
            improvement = np.mean(errors) - np.mean(opt_errors)
            print(f"\nImprovement: {improvement:.2f} mm reduction in mean error")
            
            print("\n" + "="*80)
            print("NEW CALIBRATION PARAMETERS")
            print("="*80)
            print(f"EULER_EEF_TO_COLOR_OPT = {list(optimized_params)}")
            print("\nTranslation (meters):")
            print(f"  tx = {optimized_params[0]:.6f}")
            print(f"  ty = {optimized_params[1]:.6f}")
            print(f"  tz = {optimized_params[2]:.6f}")
            print("\nRotation (radians):")
            print(f"  rx = {optimized_params[3]:.6f}")
            print(f"  ry = {optimized_params[4]:.6f}")
            print(f"  rz = {optimized_params[5]:.6f}")
            print("\nRotation (degrees):")
            print(f"  rx = {np.rad2deg(optimized_params[3]):.3f}°")
            print(f"  ry = {np.rad2deg(optimized_params[4]):.3f}°")
            print(f"  rz = {np.rad2deg(optimized_params[5]):.3f}°")
            
            # Detailed comparison
            print("\n" + "="*80)
            print("DETAILED COMPARISON (Tag ID | Current Error | Optimized Error)")
            print("="*80)
            
            tag_errors_current = {}
            tag_errors_optimized = {}
            
            for eef_pose, tag_id, cam_pos, known_pos in all_measurements:
                x_cam, y_cam, z_cam = cam_pos
                
                pred_current = camera_to_robot(x_cam, y_cam, z_cam, eef_pose, current_params,
                                               use_forward=args.use_forward_transform)
                err_current = np.linalg.norm(pred_current - known_pos)
                
                pred_opt = camera_to_robot(x_cam, y_cam, z_cam, eef_pose, optimized_params,
                                          use_forward=args.use_forward_transform)
                err_opt = np.linalg.norm(pred_opt - known_pos)
                
                if tag_id not in tag_errors_current:
                    tag_errors_current[tag_id] = []
                    tag_errors_optimized[tag_id] = []
                
                tag_errors_current[tag_id].append(err_current)
                tag_errors_optimized[tag_id].append(err_opt)
            
            for tag_id in sorted(tag_errors_current.keys()):
                avg_current = np.mean(tag_errors_current[tag_id])
                avg_opt = np.mean(tag_errors_optimized[tag_id])
                print(f"Tag {tag_id}: {avg_current:6.2f} mm → {avg_opt:6.2f} mm "
                      f"({avg_current - avg_opt:+.2f} mm)")
            
            # Print detailed real vs predicted coordinates
            print("\n" + "="*80)
            print("DETAILED COORDINATES (Optimized)")
            print("="*80)
            print(f"{'Tag ID':<8} {'Real Position (mm)':<30} {'Predicted Pos (mm)':<30} {'Error':<10}")
            print("-" * 80)
            
            for eef_pose, tag_id, cam_pos, known_pos in all_measurements:
                x_cam, y_cam, z_cam = cam_pos
                pred_opt = camera_to_robot(x_cam, y_cam, z_cam, eef_pose, optimized_params,
                                          use_forward=args.use_forward_transform)
                error = np.linalg.norm(pred_opt - known_pos)
                
                real_str = f"[{known_pos[0]:7.1f}, {known_pos[1]:7.1f}, {known_pos[2]:7.1f}]"
                pred_str = f"[{pred_opt[0]:7.1f}, {pred_opt[1]:7.1f}, {pred_opt[2]:7.1f}]"
                
                print(f"Tag {tag_id:<4} {real_str:<30} {pred_str:<30} {error:6.1f} mm")
        
        else:
            print(f"✗ Optimization failed: {result.message}")
    
    else:
        print("\nTo run optimization and compute improved parameters, use --optimize flag")
        print("Example: python apriltag_calibrate_advanced.py --optimize")

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            arm.reset(wait=True)
        except Exception:
            pass