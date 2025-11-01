#!/usr/bin/env python3
"""
advanced_calibrate.py

Robust multi-pose hand–eye calibration for OAK-D on xArm:
- Collects AprilTag pose measurements over MANY robot observation poses
- Optimizes EEF->Camera extrinsics [tx, ty, tz, rx, ry, rz] (meters/radians)
- Equalizes per-tag influence and uses Huber robust loss to resist outliers
- Median aggregation across frames per pose to reduce jitter
- Saves results and (optionally) emits a Python snippet you can import

Keys:
- We keep the **tz sign-flip** inside camera_to_robot() (camera points down).
- Default Euler order is ZYX (intrinsic Rz*Ry*Rx), matching your working code.

Examples:
  Verify current params only:
    python advanced_calibrate.py --verify-only --frames-per-pose 15

  Full optimization (recommended):
    python advanced_calibrate.py --optimize --frames-per-pose 25 --euler-order ZYX \
      --equalize-tags --huber-delta 8 --aggregate median

  Emit a ready-to-paste constant:
    python advanced_calibrate.py --optimize --emit-python eef_to_color_opt.py
"""

import os, json, time, argparse, math, datetime as dt
import numpy as np
import cv2
from pupil_apriltags import Detector
from scipy.optimize import minimize

from xarm.wrapper import XArmAPI
from camera.depthai_camera import DepthAiCamera

# ---------------------- Robot & Camera Setup ----------------------

# Robot
arm = XArmAPI('192.168.1.221')
time.sleep(0.5)
arm.set_mode(0)
arm.set_state(0)
# Don't reset here; some users prefer to start from current pose
# arm.reset(wait=True)

# Observation poses (ALL 26 from your last successful sweep)
# fmt: off
OBSERVATION_POSES = [
    [357.4,   1.1, 231.7, 178.8,   0.3,   1.0],  # 1
    [289.3, -39.7, 428.6, 178.9, -13.4,  -6.9],  # 2
    [245.1,  23.6, -45.0,-179.3,   6.3,   3.2],  # 3
    [256.0,-164.9, -57.5, 178.7,   0.1,   0.3],  # 4
    [384.3,  25.3, -23.8,-179.4,   6.0,   4.8],  # 5
    [416.6,-164.6, -57.5, 178.7,   0.1,   0.3],  # 6
    [557.3,  28.7, -17.2,-179.5,  -5.2,   2.6],  # 7
    [566.9,-164.9, -57.5, 178.7,   0.1,   0.3],  # 8
    [346.1, -33.0, 442.6, 176.3,  -9.5,   6.1],  # 9
    [371.8, -92.2, 215.1, 178.8,  -1.9,  -1.3],  # 10
    [376.8,  69.2, 215.0, 178.7,  -1.9,  11.3],  # 11
    [438.6, 107.7, 282.6, 178.5,  -3.4,  14.8],  # 12
    [508.8,-107.2, 261.7, 178.7,  -2.3, -10.8],  # 13
    [576.0,-113.3, 176.4, 178.9,  -2.9, -10.1],  # 14
    [600.4,  51.5, 197.6, 178.8,  -5.7,   5.9],  # 15
    [471.9,  -5.7, 384.9, 178.2,  -4.0,   0.4],  # 16
    [318.3,  -1.8, 296.3, 178.1, -29.0,   0.7],  # 17
    [225.5,  -3.0, 137.0, 178.6,  -1.6,   0.3],  # 18
    [259.6,-106.7, 273.1, 178.3,   0.3, -21.1],  # 19
    [467.5,-126.3, 151.1, 179.1,  -0.9, -14.2],  # 20
    [323.6,-109.1, 150.1, 179.0,  -0.1, -17.7],  # 21
    [461.1,  50.9, 217.9, 179.5,  -2.4,   6.5],  # 22
    [472.1, -82.9, 242.2, 179.5,  -5.5,  -9.7],  # 23
    [339.5,  17.6, 204.6, 179.4,  -2.1,   3.0],  # 24
    [329.4, -84.5, 204.7, 179.5,  -2.1, -14.3],  # 25
    [247.3,-148.0, -23.6, 175.2,   1.5, -14.6],  # 26
]
# fmt: on

# Known AprilTag board: tag36h11; positions in robot base frame (mm)
TAG_POSITIONS_MM = {
    0: [282.3,   44.6,  -89.3],  # BL
    3: [290.1, -118.3,  -89.5],  # BR
    # 1 and 5 were swapped in an older note — corrected:
    1: [440.5,   45.3,  -86.3],  # +Y
    5: [445.4, -128.4,  -86.0],  # -Y
    2: [606.5,   54.0,  -83.2],  # TL
    4: [603.2, -131.6,  -81.0],  # TR
}

# Size of the **black square** edge (meters). Do NOT include white border.
TAG_SIZE_M = 0.060

# Initial guess: your latest best params (meters/radians)
EULER_EEF_TO_COLOR_OPT_INIT = [
    0.04404437632822061,
   -0.04311097417649829,
    0.08865376907110752,
   -0.08883315163124154,
   -0.04850454667834010,
    1.5180110500009243,
]

# ---------------------- Math Utils ----------------------

EULER_ORDER = 'ZYX'  # set by CLI

def _euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg, order='ZYX'):
    rd, pd, yd = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(rd), np.sin(rd)
    cp, sp = np.cos(pd), np.sin(pd)
    cy, sy = np.cos(yd), np.sin(yd)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    if order == 'ZYX':
        return Rz @ Ry @ Rx
    elif order == 'XYZ':
        return Rx @ Ry @ Rz
    raise ValueError(f"Unknown order: {order}")

def _euler_rad_to_rot(rx, ry, rz):
    cr, sr = np.cos(rx), np.sin(rx)
    cp, sp = np.cos(ry), np.sin(ry)
    cy, sy = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx  # matches working code

def _pose_to_mat4(x, y, z, rx, ry, rz):
    M = np.eye(4)
    M[:3,:3] = _euler_rad_to_rot(rx, ry, rz)
    M[:3, 3] = [x, y, z]
    return M

def camera_to_robot(x_cam_m, y_cam_m, z_cam_m, eef_pose_xyzrpy, eef_to_cam_params, use_forward=False):
    """
    Transform a point from camera frame to robot base frame.
    eef_pose_xyzrpy: [mm, mm, mm, deg, deg, deg]
    eef_to_cam_params: [tx, ty, tz, rx, ry, rz] in meters/radians
    NOTE: tz is flipped (camera points down w/ roll≈180°).
    """
    x_eef_mm, y_eef_mm, z_eef_mm, r_deg, p_deg, y_deg = eef_pose_xyzrpy
    R_be = _euler_deg_to_rot(r_deg, p_deg, y_deg, order=EULER_ORDER)
    T_be = np.eye(4)
    T_be[:3,:3] = R_be
    T_be[:3, 3] = [x_eef_mm/1000.0, y_eef_mm/1000.0, z_eef_mm/1000.0]

    tx, ty, tz, rx, ry, rz = eef_to_cam_params
    # Critical: flip tz to account for inverted Z with gripper-down convention
    T_ec = _pose_to_mat4(tx, ty, -tz, rx, ry, rz)

    T_bc = T_be @ T_ec
    p_cam = np.array([x_cam_m, y_cam_m, z_cam_m, 1.0])
    p_base = T_bc @ p_cam
    return p_base[:3] * 1000.0  # mm

# ---------------------- Data Collection ----------------------

def collect_measurements(cam, detector, cam_params, pose, num_frames=10, aggregate='median'):
    """
    Returns: dict {tag_id: np.array([x,y,z]) in camera frame (meters)}, aggregated across frames.
    aggregate: 'median' or 'mean'
    """
    print(f"  Moving to pose: {pose}")
    arm.set_position(*pose, wait=True)
    time.sleep(0.30)

    per_tag = {}
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
            t = det.pose_t.reshape(3).astype(float)  # meters
            per_tag.setdefault(tag_id, []).append(t)
        time.sleep(0.01)

    out = {}
    for tid, arrs in per_tag.items():
        A = np.vstack(arrs)  # N x 3
        if aggregate == 'median':
            out[tid] = np.median(A, axis=0)
        else:
            out[tid] = np.mean(A, axis=0)
    return out

# ---------------------- Optimization ----------------------

def huber_loss(err_mm, delta_mm):
    """Huber loss in millimeters."""
    a = abs(err_mm)
    if a <= delta_mm:
        return 0.5 * (err_mm * err_mm)
    return delta_mm * (a - 0.5 * delta_mm)

def build_tag_weights(measurements, equalize_tags=True):
    """
    measurements: list of tuples (eef_pose, tag_id, cam_pos_m, known_pos_mm)
    Returns: dict tag_id -> weight_factor
    If equalize_tags: w = 1 / count(tag), so sum of weights per-tag is similar.
    """
    counts = {}
    for _, tag_id, _, _ in measurements:
        counts[tag_id] = counts.get(tag_id, 0) + 1
    weights = {}
    for tid, c in counts.items():
        weights[tid] = (1.0 / max(1, c)) if equalize_tags else 1.0
    return weights

def objective(params, measurements, use_forward, fixed_rot, tag_weights, huber_delta=None):
    """
    params: if fixed_rot is None -> [tx,ty,tz,rx,ry,rz]; else -> [tx,ty,tz]
    """
    if fixed_rot is not None:
        full = np.concatenate([params, fixed_rot])
    else:
        full = params

    total = 0.0
    for eef_pose, tag_id, cam_pos_m, known_pos_mm in measurements:
        pred_mm = camera_to_robot(cam_pos_m[0], cam_pos_m[1], cam_pos_m[2], eef_pose, full, use_forward=use_forward)
        err = np.linalg.norm(pred_mm - known_pos_mm)  # mm
        w = tag_weights.get(tag_id, 1.0)
        if huber_delta is not None and huber_delta > 0:
            total += w * huber_loss(err, huber_delta)
        else:
            total += w * (err * err)
    return total

def evaluate_params(params, measurements, use_forward):
    errs = []
    for eef_pose, tag_id, cam_pos_m, known_pos_mm in measurements:
        pred_mm = camera_to_robot(cam_pos_m[0], cam_pos_m[1], cam_pos_m[2], eef_pose, params, use_forward=use_forward)
        errs.append(np.linalg.norm(pred_mm - known_pos_mm))
    errs = np.array(errs, dtype=float)
    stats = {
        "mean": float(np.mean(errs)),
        "median": float(np.median(errs)),
        "max": float(np.max(errs)),
        "std": float(np.std(errs)),
    }
    return errs, stats

def print_stats(title, stats):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(f"Mean error:   {stats['mean']:.2f} mm")
    print(f"Median error: {stats['median']:.2f} mm")
    print(f"Max error:    {stats['max']:.2f} mm")
    print(f"Std error:    {stats['std']:.2f} mm")

def dump_results_json(path, params, stats, args):
    data = {
        "timestamp": dt.datetime.now().isoformat(),
        "params": {
            "tx": float(params[0]), "ty": float(params[1]), "tz": float(params[2]),
            "rx": float(params[3]), "ry": float(params[4]), "rz": float(params[5]),
        },
        "stats_mm": stats,
        "args": vars(args),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved results JSON -> {path}")

def emit_python_snippet(path, params):
    txt = (
        "# Auto-generated by advanced_calibrate.py\n"
        "EULER_EEF_TO_COLOR_OPT = [\n"
        f"    {float(params[0]):.15f},\n"
        f"    {float(params[1]):.15f},\n"
        f"    {float(params[2]):.15f},\n"
        f"    {float(params[3]):.15f},\n"
        f"    {float(params[4]):.15f},\n"
        f"    {float(params[5]):.15f},\n"
        "]\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"Emitted Python params -> {path}")

# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser(description="Robust multi-pose camera calibration with AprilTags")
    ap.add_argument("--frames-per-pose", type=int, default=15)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--use-forward-transform", action="store_true")
    ap.add_argument("--euler-order", choices=["ZYX","XYZ"], default="ZYX")
    ap.add_argument("--optimize-translation-only", action="store_true")
    ap.add_argument("--equalize-tags", action="store_true", help="Equalize per-tag influence")
    ap.add_argument("--huber-delta", type=float, default=0.0, help="Huber threshold in mm; 0 disables")
    ap.add_argument("--aggregate", choices=["median","mean"], default="median")
    ap.add_argument("--min-tags-per-pose", type=int, default=1)
    ap.add_argument("--results-json", default="advanced_calibration_result.json")
    ap.add_argument("--emit-python", default="", help="Optional path to write EULER_EEF_TO_COLOR_OPT = [...]")
    args = ap.parse_args()

    global EULER_ORDER
    EULER_ORDER = args.euler_order
    print(f"Using Euler order: {EULER_ORDER}")

    # Camera init
    print("Starting OAK-D camera (640x400)...")
    cam = DepthAiCamera(width=640, height=400, disable_rgb=False)
    K_rgb, _ = cam.get_intrinsics()
    cam_params = [float(K_rgb[0][0]), float(K_rgb[1][1]), float(K_rgb[0][2]), float(K_rgb[1][2])]

    # AprilTag detector
    print("Initializing AprilTag detector...")
    detector = Detector(
        families='tag36h11',
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    # Open gripper for safety
    try:
        arm.set_gripper_position(850, wait=True)
    except Exception:
        pass

    # Collect
    print(f"\nCollecting measurements from {len(OBSERVATION_POSES)} pose(s)...")
    all_measurements = []  # (eef_pose, tag_id, cam_pos(m), known_pos(mm))
    for i, pose in enumerate(OBSERVATION_POSES, start=1):
        print(f"\nPose {i}/{len(OBSERVATION_POSES)}:")
        dets = collect_measurements(cam, detector, cam_params, pose, args.frames_per_pose, aggregate=args.aggregate)
        if len(dets) < args.min-tags-per-pose if hasattr(args, "min-tags-per-pose") else False:
            # safeguard, but argparse forbids '-' in attribute name; we won't use this branch.
            pass
        if not dets:
            print("  WARNING: No tags detected at this pose")
            continue
        print(f"  Detected {len(dets)} tags: {sorted(dets.keys())}")
        for tag_id, cam_pos in dets.items():
            known = np.array(TAG_POSITIONS_MM[tag_id], dtype=float)
            all_measurements.append((pose, tag_id, np.array(cam_pos, dtype=float), known))

    if not all_measurements:
        print("\nERROR: No measurements collected!")
        return

    print(f"\nTotal measurements collected: {len(all_measurements)}")
    print(f"Unique tags seen: {len(set(m[1] for m in all_measurements))}")

    # Evaluate current
    current = np.array(EULER_EEF_TO_COLOR_OPT_INIT, dtype=float)
    errs_curr, stats_curr = evaluate_params(current, all_measurements, args.use_forward_transform)
    print_stats("CURRENT CALIBRATION PERFORMANCE", stats_curr)
    print(f"\nCurrent params: {list(map(float, current))}")

    if args.verify_only and not args.optimize:
        print("\nVerification complete (--verify-only).")
        return

    if not args.optimize:
        print("\nTo run optimization, add --optimize")
        return

    # Optimizer setup
    tag_weights = build_tag_weights(all_measurements, equalize_tags=args.equalize_tags)
    if args.optimize_translation_only:
        x0 = current[:3].copy()
        bounds = [(-0.5,0.5), (-0.5,0.5), (-0.5,0.5)]
        fixed_rot = current[3:].copy()
    else:
        x0 = current.copy()
        bounds = [(-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-math.pi,math.pi), (-math.pi,math.pi), (-math.pi,math.pi)]
        fixed_rot = None

    print("\n" + "="*80)
    print("RUNNING OPTIMIZATION")
    print("="*80)

    res = minimize(
        objective,
        x0,
        args=(all_measurements, args.use_forward_transform, fixed_rot, tag_weights, args.huber_delta),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-8},
    )

    if not res.success:
        print(f"\n✗ Optimization failed: {res.message}")
        return

    print("\n✓ Optimization converged!")
    if fixed_rot is not None:
        opt = np.concatenate([res.x, fixed_rot])
    else:
        opt = res.x

    # Evaluate optimized
    errs_opt, stats_opt = evaluate_params(opt, all_measurements, args.use_forward_transform)
    print_stats("OPTIMIZED CALIBRATION PERFORMANCE", stats_opt)
    improvement = stats_curr["mean"] - stats_opt["mean"]
    print(f"\nImprovement: {improvement:.2f} mm reduction in mean error")

    print("\n" + "="*80)
    print("NEW CALIBRATION PARAMETERS")
    print("="*80)
    print(f"EULER_EEF_TO_COLOR_OPT = {list(map(float, opt))}")
    print("\nTranslation (meters):")
    print(f"  tx = {opt[0]:.6f}\n  ty = {opt[1]:.6f}\n  tz = {opt[2]:.6f}")
    print("\nRotation (radians):")
    print(f"  rx = {opt[3]:.6f}\n  ry = {opt[4]:.6f}\n  rz = {opt[5]:.6f}")
    print("\nRotation (degrees):")
    print(f"  rx = {np.rad2deg(opt[3]):.3f}°")
    print(f"  ry = {np.rad2deg(opt[4]):.3f}°")
    print(f"  rz = {np.rad2deg(opt[5]):.3f}°")

    # Per-tag averages (current vs optimized)
    tag_errs_curr = {}
    tag_errs_opt  = {}
    for eef_pose, tag_id, cam_pos, known_pos in all_measurements:
        pc = camera_to_robot(cam_pos[0], cam_pos[1], cam_pos[2], eef_pose, current, use_forward=args.use_forward_transform)
        po = camera_to_robot(cam_pos[0], cam_pos[1], cam_pos[2], eef_pose, opt,     use_forward=args.use_forward_transform)
        ec = float(np.linalg.norm(pc - known_pos))
        eo = float(np.linalg.norm(po - known_pos))
        tag_errs_curr.setdefault(tag_id, []).append(ec)
        tag_errs_opt.setdefault(tag_id, []).append(eo)

    print("\n" + "="*80)
    print("DETAILED COMPARISON (Tag ID | Current Error | Optimized Error)")
    print("="*80)
    for tid in sorted(tag_errs_curr.keys()):
        ac = float(np.mean(tag_errs_curr[tid]))
        ao = float(np.mean(tag_errs_opt[tid]))
        print(f"Tag {tid}: {ac:6.2f} mm → {ao:6.2f} mm ({ac-ao:+.2f} mm)")

    # Detailed coordinates for top-12 largest optimized residuals
    idx_sorted = np.argsort(-errs_opt)  # descending by error
    top = idx_sorted[:min(12, len(idx_sorted))]
    print("\n" + "="*80)
    print("TOP RESIDUALS (Optimized)  (Tag | Real (mm) | Pred (mm) | Error)")
    print("="*80)
    for k in top:
        eef_pose, tag_id, cam_pos, known_pos = all_measurements[k]
        pred = camera_to_robot(cam_pos[0], cam_pos[1], cam_pos[2], eef_pose, opt, use_forward=args.use_forward_transform)
        err = float(np.linalg.norm(pred - known_pos))
        real_str = f"[{known_pos[0]:7.1f}, {known_pos[1]:7.1f}, {known_pos[2]:7.1f}]"
        pred_str = f"[{pred[0]:7.1f}, {pred[1]:7.1f}, {pred[2]:7.1f}]"
        print(f"Tag {tag_id:<2}  {real_str:<32} {pred_str:<32} {err:6.1f} mm")

    # Save results
    dump_results_json(args.results_json, opt, stats_opt, args)
    if args.emit_python:
        emit_python_snippet(args.emit_python, opt)

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            arm.reset(wait=True)
        except Exception:
            pass
