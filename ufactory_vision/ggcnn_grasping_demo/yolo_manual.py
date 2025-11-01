#!/usr/bin/env python3
"""
Minimal: Move to a known pose, detect the top object once, and print estimated coordinates.
Asks for approval before picking: press 'p' to stage, then 'y' to confirm or 'n' to cancel.
"""

import time
import argparse
import warnings
import numpy as np
import torch
import cv2

from xarm.wrapper import XArmAPI
from camera.depthai_camera import DepthAiCamera

warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

# ---- Robot connect & move to pose ----
arm = XArmAPI('192.168.1.221')
time.sleep(0.5)
arm.set_mode(0)
arm.set_state(0)
arm.reset(wait=True)

# Move to observation pose (scan)
OBS_POSE = [357.4, 1.1, 231.7, 178.8, 0.3, 1.0]
arm.set_gripper_position(850, wait=True)
arm.set_position(*OBS_POSE, wait=True)

# ---- Hand-eye (from your optimizer; meters/radians). tz is flipped inside camera_to_robot() ----
#EULER_EEF_TO_COLOR_OPT = [
 #   0.04863833376845415,   # tx (m)
  # -0.04540044886615122,   # ty (m)
   # 0.08156322113763131,   # tz (m)  <-- flipped inside transform
   #-0.02791123564486829,   # rx (rad)
   #-0.0035033548509433654, # ry (rad)
   # 1.521339368142855      # rz (rad)
#]

# Use this in BOTH advanced_calibrate.py (INIT) and yolo_manual.py (FINAL)
EULER_EEF_TO_COLOR_OPT = [
    0.028331345999761292,     # tx (m)  <-- new
   -0.06602344853949635,      # ty (m)  <-- new
    0.07552896986298417,       # tz (m)  <-- new (remember: flip once in transform)
   -0.02791123564486829,       # rx (rad)  <-- from 9 mm run
   -0.0035033548509433654,     # ry (rad)  <-- from 9 mm run
    1.521339368142855          # rz (rad)  <-- from 9 mm run
]

GRIPPER_Z_MM = 70  # info

# --- Grasp & safety parameters ---
GRIPPER_OPEN_POS = 850
GRIPPER_CLOSE_POS = 300
HOVER_DELTA_MM   = 40.0
LIFT_DELTA_MM    = 100.0
MIN_Z_LIMIT_MM   = -250.0  # global floor
MAX_Z_LIMIT_MM   =  400.0  # global ceiling
Z_TABLE_LIMIT_MM = -85.0   # table plane (measured)
Z_CLEARANCE_MM   =   2.0   # keep at least 2 mm above table

def clamp_to_table(z_mm: float) -> float:
    """Keep Z at/above the table plane with clearance."""
    return max(z_mm, Z_TABLE_LIMIT_MM + Z_CLEARANCE_MM)

def clamp_motion(z_mm: float) -> float:
    z_mm = max(MIN_Z_LIMIT_MM, min(MAX_Z_LIMIT_MM, z_mm))
    return clamp_to_table(z_mm)

def pixel_to_3d(px, py, depth_m, K):
    fx = K[0][0]; fy = K[1][1]; cx = K[0][2]; cy = K[1][2]
    z = float(depth_m)
    x = (px - cx) * z / fx
    y = (py - cy) * z / fy
    return x, y, z  # meters

def _euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg):
    rd, pd, yd = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(rd), np.sin(rd)
    cp, sp = np.cos(pd), np.sin(pd)
    cy, sy = np.cos(yd), np.sin(yd)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy,  cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def _euler_rad_to_rot(rx, ry, rz):
    cr, sr = np.cos(rx), np.sin(rx)
    cp, sp = np.cos(ry), np.sin(ry)
    cy, sy = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy,  cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def camera_to_robot(x_cam_m, y_cam_m, z_cam_m, eef_pose_xyzrpy):
    """
    p_base = t_base_eef + R_base_eef @ (R_eef_cam @ p_cam + t_eef_cam)
    Returns base frame (x,y,z) in mm.
    IMPORTANT: flip tz to match calibrator’s convention (gripper points down).
    """
    p_cam = np.array([x_cam_m, y_cam_m, z_cam_m], dtype=float)

    tx, ty, tz, rx_c, ry_c, rz_c = EULER_EEF_TO_COLOR_OPT
    t_eef_cam = np.array([tx, ty, -tz], dtype=float)     # flip tz here
    R_eef_cam = _euler_rad_to_rot(rx_c, ry_c, rz_c)
    p_eef = R_eef_cam @ p_cam + t_eef_cam

    x_eef_mm, y_eef_mm, z_eef_mm, roll_deg, pitch_deg, yaw_deg = eef_pose_xyzrpy
    t_base_eef = np.array([x_eef_mm, y_eef_mm, z_eef_mm], dtype=float) / 1000.0
    R_base_eef = _euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg)

    p_base = t_base_eef + R_base_eef @ p_eef
    return float(p_base[0]*1000.0), float(p_base[1]*1000.0), float(p_base[2]*1000.0)

def median_valid_depth(depth_img, cx, cy, k=3):
    h, w = depth_img.shape[:2]
    x0 = max(0, cx - k); x1 = min(w, cx + k + 1)
    y0 = max(0, cy - k); y1 = min(h, cy + k + 1)
    patch = depth_img[y0:y1, x0:x1].astype(np.float32)
    vals = patch[np.isfinite(patch)]
    if vals.size == 0: return None
    return float(np.median(vals))

def _get_current_pose(arm: XArmAPI):
    try:
        res = arm.get_position()
        data = None
        if isinstance(res, dict) and 'data' in res: data = res.get('data')
        elif isinstance(res, (list, tuple)):
            if len(res) >= 2 and isinstance(res[1], (list, tuple)): data = res[1]
            elif len(res) >= 6 and all(isinstance(v, (int, float)) for v in res[:6]): data = res[:6]
        if data and len(data) >= 6:
            return [float(data[0]), float(data[1]), float(data[2]),
                    float(data[3]), float(data[4]), float(data[5])]
    except Exception:
        pass
    return OBS_POSE

def pick_last_estimate(arm: XArmAPI, est: dict):
    """Execute the pick sequence (uses safe/clamped Z)."""
    x = float(est['xr']); y = float(est['yr']); z_obj = float(est['zr'])
    roll, pitch, yaw = OBS_POSE[3], OBS_POSE[4], OBS_POSE[5]

    z_pick  = clamp_motion(z_obj)
    z_hover = clamp_motion(z_pick + HOVER_DELTA_MM)
    z_lift  = clamp_motion(z_hover + LIFT_DELTA_MM)

    print("\n--- PICK PLAN ---")
    print(f"Target XY=({x:.1f},{y:.1f})  Z_obj={z_obj:.1f}")
    print(f"HoverZ={z_hover:.1f}  GraspZ={z_pick:.1f}  LiftZ={z_lift:.1f}")
    if z_pick != z_obj:
        print(f"[SAFETY] Z pick clamped (table {Z_TABLE_LIMIT_MM} + {Z_CLEARANCE_MM}mm)")

    try:
        arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
        arm.set_position(x, y, OBS_POSE[2], roll, pitch, yaw, wait=True)
        arm.set_position(x, y, z_hover,    roll, pitch, yaw, wait=True)
        arm.set_position(x, y, z_pick,     roll, pitch, yaw, wait=True)
        arm.set_gripper_position(GRIPPER_CLOSE_POS, wait=True)
        time.sleep(0.25)
        arm.set_position(x, y, z_lift,     roll, pitch, yaw, wait=True)
    except Exception as e:
        print(f"Pick failed: {e}")
    finally:
        try: arm.set_position(*OBS_POSE, wait=True)
        except Exception: pass

def refine_at_hover(arm: XArmAPI, est: dict, cam: DepthAiCamera, model, Kd, calib_bias: np.ndarray):
    """Move laterally to target XY keeping current Z, re-detect, and recompute coordinates."""
    try:
        x = float(est['xr']); y = float(est['yr'])
    except Exception:
        print("Invalid last estimate; cannot refine.")
        return None

    cur_pose = _get_current_pose(arm)
    cur_z = float(cur_pose[2])
    roll, pitch, yaw = float(cur_pose[3]), float(cur_pose[4]), float(cur_pose[5])

    try:
        arm.set_position(x, y, cur_z, roll, pitch, yaw, wait=True)
        time.sleep(0.3)

        prev_cx = int(est.get('cx', 0)); prev_cy = int(est.get('cy', 0))
        best_est = None
        t_end = time.time() + 2.0

        while time.time() < t_end:
            color, depth = cam.get_images()
            if color is None: continue
            with torch.no_grad():
                results = model(color)
            df = results.pandas().xyxy[0]
            if df is None or df.empty: continue

            df = df.copy()
            df['cx'] = (df['xmin'] + df['xmax']) / 2.0
            df['cy'] = (df['ymin'] + df['ymax']) / 2.0
            df['dist2'] = (df['cx'] - prev_cx) ** 2 + (df['cy'] - prev_cy) ** 2
            df = df.sort_values(['dist2', 'confidence'], ascending=[True, False])
            sel = df.iloc[0]
            cx, cy = int(sel['cx']), int(sel['cy'])

            depth_m = None
            if depth is not None and 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                d = depth[cy, cx]
                depth_m = float(d) if np.isfinite(d) else median_valid_depth(depth, cx, cy, k=3)
            if depth_m is None: continue

            x_cam, y_cam, z_cam = pixel_to_3d(cx, cy, depth_m, Kd)
            eef_pose = [x, y, cur_z, roll, pitch, yaw]
            xr_raw, yr_raw, zr_raw = camera_to_robot(x_cam, y_cam, z_cam, eef_pose)
            xr, yr, zr = (np.array([xr_raw, yr_raw, zr_raw])).tolist()

            name = sel['name'] if 'name' in sel else 'object'
            conf = float(sel['confidence']) if 'confidence' in sel else 0.0
            best_est = {
                'cx': cx, 'cy': cy, 'depth_m': depth_m,
                'x_cam': x_cam, 'y_cam': y_cam, 'z_cam': z_cam,
                'xr': xr, 'yr': yr, 'zr': clamp_to_table(zr),  # keep safe Z
                'xr_raw': xr_raw, 'yr_raw': yr_raw, 'zr_raw': zr_raw,
                'name': name, 'conf': conf,
            }
            break

        if best_est is not None:
            print("\n=== Refined at hover ===")
            print(f"Detection: {best_est['name']} (conf {best_est['conf']:.2f})")
            print(f"Pixel:     ({best_est['cx']}, {best_est['cy']})  depth: {best_est['depth_m']:.3f} m")
            print(f"Camera3D:  x={best_est['x_cam']:.3f} m  y={best_est['y_cam']:.3f} m  z={best_est['z_cam']:.3f} m")
            print(f"RobotXYZ(raw): X={best_est['xr_raw']:.1f} Y={best_est['yr_raw']:.1f} Z={best_est['zr_raw']:.1f}")
            print(f"RobotXYZ(safe): X={best_est['xr']:.1f} Y={best_est['yr']:.1f} Z={best_est['zr']:.1f}")
        else:
            print("Refine at hover failed (no valid detection).")

        return best_est
    except Exception as e:
        print(f"Refine at hover error: {e}")
        return None

def main_once():
    parser = argparse.ArgumentParser(description='Manual YOLO with coordinate printout (confirm before pick)')
    parser.add_argument('--dy-pos', type=float, default=0.0, help='Y bias when yr_raw >= 0 (mm)')
    parser.add_argument('--dy-neg', type=float, default=0.0, help='Y bias when yr_raw < 0 (mm)')
    parser.add_argument('--verbose', action='store_true', help='Enable periodic console prints')
    parser.add_argument('--print-every', type=float, default=2.0, help='Seconds between prints with --verbose')
    args, _ = parser.parse_known_args()
    calib_bias = np.array([args.dx, args.dy, args.dz], dtype=float)

    if not args.verbose:
        print("Keys: 'p' stage pick, 'y' confirm, 'n' cancel, 'h' hover-refine, 'i' snapshot, 's' save, 'v' verbose, 'q' quit.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load YOLOv5
    print("Loading YOLOv5s model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to(device).eval()
    model.conf = 0.25
    model.iou = 0.45

    # Start OAK-D
    print("Starting OAK-D camera (640x400)...")
    cam = DepthAiCamera(width=640, height=400, disable_rgb=False)
    _, Kd = cam.get_intrinsics()  # use depth intrinsics

    # UI state
    try: cv2.namedWindow('Manual YOLO', cv2.WINDOW_NORMAL)
    except Exception: pass
    DETECTION_INTERVAL = 5
    frame_count = 0
    last_df = None
    last_est = None
    verbose = bool(args.verbose)
    last_print_time = 0.0

    # Pick confirmation state
    awaiting_confirm = False
    pending_est = None
    pending_plan_lines = []

    while True:
        color, depth = cam.get_images()
        if color is None:
            continue

        frame_count += 1
        if frame_count % DETECTION_INTERVAL == 0:
            with torch.no_grad():
                results = model(color)
            df = results.pandas().xyxy[0]
            last_df = df

            if df is not None and not df.empty:
                best = df.iloc[df['confidence'].astype(float).idxmax()]
                x1, y1, x2, y2 = map(int, [best['xmin'], best['ymin'], best['xmax'], best['ymax']])
                cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)

                # Depth for center
                depth_m = None
                if depth is not None and 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                    d = depth[cy, cx]
                    depth_m = float(d) if np.isfinite(d) else median_valid_depth(depth, cx, cy, k=3)

                if depth_m is not None:
                    x_cam, y_cam, z_cam = pixel_to_3d(cx, cy, depth_m, Kd)
                    xr_raw, yr_raw, zr_raw = camera_to_robot(x_cam, y_cam, z_cam, OBS_POSE)
                    xr, yr, zr = (np.array([xr_raw, yr_raw, zr_raw]) + calib_bias).tolist()
                    name = best['name'] if 'name' in best else 'object'
                    conf = float(best['confidence']) if 'confidence' in best else 0.0
                    last_est = {
                        'cx': cx, 'cy': cy, 'depth_m': depth_m,
                        'x_cam': x_cam, 'y_cam': y_cam, 'z_cam': z_cam,
                        'xr': xr, 'yr': yr, 'zr': clamp_to_table(zr),    # safe Z stored
                        'xr_raw': xr_raw, 'yr_raw': yr_raw, 'zr_raw': zr_raw,
                        'name': name, 'conf': conf,
                    }

                    now = time.time()
                    if verbose and (now - last_print_time) >= max(0.2, float(args.print_every)):
                        last_print_time = now
                        print("\n=== Estimated target ===")
                        print(f"Detection: {name} (conf {conf:.2f})")
                        print(f"Pixel:     ({cx}, {cy})  depth: {depth_m:.3f} m")
                        print(f"Camera3D:  x={x_cam:.3f} m  y={y_cam:.3f} m  z={z_cam:.3f} m")
                        print(f"RobotXYZ(raw):  X={xr_raw:.1f} Y={yr_raw:.1f} Z={zr_raw:.1f}")
                        print(f"RobotXYZ(safe): X={xr:.1f}   Y={yr:.1f}   Z={last_est['zr']:.1f}")

        # Draw overlay
        display = color.copy()
        if last_df is not None and not last_df.empty:
            for _, det in last_df.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 2)
                label = det.get('name', 'obj'); conf = float(det.get('confidence', 0.0))
                cv2.putText(display, f"{label}:{conf:.2f}", (x1, max(20, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)

        if last_est is not None:
            cx, cy = last_est['cx'], last_est['cy']
            cv2.drawMarker(display, (cx, cy), (0,255,255), cv2.MARKER_CROSS, 18, 2)
            hud = [
                f"{last_est['name']} ({last_est['conf']:.2f})",
                f"px=({cx},{cy}) d={last_est['depth_m']:.3f}m",
                f"X={last_est['xr']:.1f} Y={last_est['yr']:.1f} Z={last_est['zr']:.1f} mm",
            ]
            hud.append(f"raw Z={last_est['zr_raw']:.1f}")
            for i, txt in enumerate(hud):
                cv2.putText(display, txt, (10, 24 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # Show pending pick confirmation
        if awaiting_confirm and pending_plan_lines:
            y0 = 24 + 5*20
            for i, line in enumerate(["CONFIRM PICK: press 'y' to execute, 'n' to cancel"] + pending_plan_lines):
                cv2.putText(display, line, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow('Manual YOLO', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            fname = f"manual_yolo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, display)
            print(f"Saved {fname}")

        elif key == ord('p'):
            if last_est is None:
                print("No estimate yet; can't stage pick.")
            else:
                # Build staged plan (no motion yet)
                x = float(last_est['xr']); y = float(last_est['yr']); z_obj = float(last_est['zr'])
                z_pick  = clamp_motion(z_obj)
                z_hover = clamp_motion(z_pick + HOVER_DELTA_MM)
                z_lift  = clamp_motion(z_hover + LIFT_DELTA_MM)
                pending_plan_lines = [
                    f"Target XY=({x:.1f},{y:.1f})  Z_obj={z_obj:.1f}",
                    f"HoverZ={z_hover:.1f}  GraspZ={z_pick:.1f}  LiftZ={z_lift:.1f}",
                ]
                if z_pick != z_obj:
                    pending_plan_lines.append(f"[SAFETY] Z pick clamped (table {Z_TABLE_LIMIT_MM}+{Z_CLEARANCE_MM}mm)")
                pending_est = dict(last_est)  # keep same schema
                awaiting_confirm = True
                print("\n[PENDING PICK] " + "  ".join(pending_plan_lines))

        elif key == ord('y'):
            if awaiting_confirm and pending_est is not None:
                print("Pick confirmed.")
                awaiting_confirm = False
                try:
                    pick_last_estimate(arm, pending_est)
                finally:
                    pending_est = None
                    pending_plan_lines = []
            # else ignore 'y'

        elif key == ord('n'):
            if awaiting_confirm:
                print("Pick canceled.")
                awaiting_confirm = False
                pending_est = None
                pending_plan_lines = []

        elif key == ord('h'):
            if last_est is None:
                print("No estimate yet; can't hover-refine.")
            else:
                new_est = refine_at_hover(arm, last_est, cam, model, Kd, np.zeros(3))
                if new_est is not None:
                    last_est = new_est

        elif key == ord('v'):
            verbose = not verbose
            print(f"Verbose printing: {'ON' if verbose else 'OFF'}")

        elif key == ord('i'):
            if last_est is None:
                print("No estimate yet.")
            else:
                e = last_est
                print("\n=== Current estimate (snapshot) ===")
                print(f"Detection: {e['name']} (conf {e['conf']:.2f})")
                print(f"Pixel:     ({e['cx']}, {e['cy']})  depth: {e['depth_m']:.3f} m")
                print(f"Camera3D:  x={e['x_cam']:.3f} m  y={e['y_cam']:.3f} m  z={e['z_cam']:.3f} m")
                print(f"RobotXYZ(raw):  X={e['xr_raw']:.1f} Y={e['yr_raw']:.1f} Z={e['zr_raw']:.1f}")
                print(f"RobotXYZ(safe): X={e['xr']:.1f}   Y={e['yr']:.1f}   Z={e['zr']:.1f}")

if __name__ == "__main__":
    try:
        main_once()
    finally:
        try:
            arm.reset(wait=True)
        except Exception:
            pass
