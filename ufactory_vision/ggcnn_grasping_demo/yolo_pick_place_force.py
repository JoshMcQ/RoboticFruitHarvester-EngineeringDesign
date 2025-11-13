#!/usr/bin/env python3
"""
yolo_pick_place_force.py
Detect fruit with YOLO11 (Ultralytics) -> map to robot -> force-limited grasp -> place in bin.

Keys:
  p : stage pick plan from current best detection
  y : execute pick & place (force sensor required)
  n : cancel staged plan
  s : save overlay screenshot
  q : quit
"""

import time, argparse, warnings, re
import numpy as np
import torch, cv2, serial
from ultralytics import YOLO
import pandas as pd

from xarm.wrapper import XArmAPI
from camera.depthai_camera import DepthAiCamera

warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

# ---------------- Robot setup ----------------
ROBOT_IP = "192.168.1.221"
arm = XArmAPI(ROBOT_IP)
time.sleep(0.5)
arm.set_gripper_enable(True)
arm.set_mode(0)
arm.set_state(0)
arm.reset(wait=True)
## Re-enable the gripper
OBS_POSE = [357.4, 1.1, 231.7, 178.8, 0.3, 1.0]
GRIPPER_OPEN_POS = 850

arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
arm.set_position(*OBS_POSE, wait=True)

# ---------------- Hand-eye params ----------------
# Try import; otherwise use latest optimized values you printed.
try:
    from eef_to_color_opt import EULER_EEF_TO_COLOR_OPT  # [tx,ty,tz, rx,ry,rz]
except Exception:
    EULER_EEF_TO_COLOR_OPT = [
        0.044707024930037,   # tx (m)
        -0.040881623923506,  # ty (m)
        0.088584379244258,   # tz (m)
        -0.053468293190528,  # rx (rad)
        -0.049247341299073,  # ry (rad)
        1.534069196206855,   # rz (rad)
    ]

# --- Grasp & safety parameters ---
HOVER_DELTA_MM   = 40.0
LIFT_DELTA_MM    = 120.0
MIN_Z_LIMIT_MM   = -250.0
MAX_Z_LIMIT_MM   =  400.0
Z_TABLE_LIMIT_MM =  -85.0
Z_CLEARANCE_MM   =    2.0

def clamp_to_table(z_mm: float) -> float:
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
    IMPORTANT: flip tz to match calibrator’s convention (gripper points down).
    Returns base-frame (x,y,z) in mm.
    """
    p_cam = np.array([x_cam_m, y_cam_m, z_cam_m], dtype=float)

    tx, ty, tz, rx_c, ry_c, rz_c = EULER_EEF_TO_COLOR_OPT
    t_eef_cam = np.array([tx, ty, -tz], dtype=float)  # flip tz once
    R_eef_cam = _euler_rad_to_rot(rx_c, ry_c, rz_c)
    p_eef = R_eef_cam @ p_cam + t_eef_cam

    x_eef_mm, y_eef_mm, z_eef_mm, roll_deg, pitch_deg, yaw_deg = eef_pose_xyzrpy
    t_base_eef = np.array([x_eef_mm, y_eef_mm, z_eef_mm], dtype=float) / 1000.0
    R_base_eef = _euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg)

    p_base = t_base_eef + R_base_eef @ p_eef
    return float(p_base[0] * 1000.0), float(p_base[1] * 1000.0), float(p_base[2] * 1000.0)

def median_valid_depth(depth_img, cx, cy, k=3):
    h, w = depth_img.shape[:2]
    x0 = max(0, cx - k); x1 = min(w, cx + k + 1)
    y0 = max(0, cy - k); y1 = min(h, cy + k + 1)
    patch = depth_img[y0:y1, x0:x1].astype(np.float32)
    vals = patch[np.isfinite(patch)]
    if vals.size == 0: return None
    return float(np.median(vals))

# ---------- Bias helpers ----------
def apply_bias(xr_raw, yr_raw, zr_raw, args, cx=None, img_w=None):
    """
    Piecewise Y bias:
      - if dy_pos/dy_neg provided, use by sign of yr_raw
      - else use uniform dy
    Always apply dx, dz. Optionally linear Y vs. pixel X: y_bias_per_px*(cx - img_w/2).
    """
    dx = float(getattr(args, 'dx', 0.0))
    dz = float(getattr(args, 'dz', 0.0))

    dy_pos = getattr(args, 'dy_pos', None)
    dy_neg = getattr(args, 'dy_neg', None)
    if dy_pos is not None and dy_neg is not None:
        dy = float(dy_pos) if yr_raw >= 0 else float(dy_neg)
    else:
        dy = float(getattr(args, 'dy', 0.0))

    y_bias_per_px = float(getattr(args, 'y_bias_per_px', 0.0))
    if y_bias_per_px != 0.0 and cx is not None and img_w is not None:
        dy += y_bias_per_px * (float(cx) - float(img_w) / 2.0)

    xr = xr_raw + dx
    yr = yr_raw + dy
    zr = clamp_to_table(zr_raw + dz)
    return xr, yr, zr

# ========= Force Sensing (COM port) =========
def read_force(ser: serial.Serial, timeout: float = 0.006) -> float | None:
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line: continue
            m = re.search(r"(\d+(?:\.\d+)?)", line)
            if m: return float(m.group(1))
    except Exception:
        return None
    return None

def smooth_close_with_force(arm: XArmAPI, ser: serial.Serial,
                            threshold: float,
                            start=GRIPPER_OPEN_POS, end=300):
    """Adaptive, debounced close that stops at threshold (force-only)."""
    EMA_ALPHA   = 0.25
    BASE_STEP   = 18
    MIN_STEP    = 4
    # Use a conservative max speed; some xArm grippers ignore overly high speeds on subsequent cycles
    # V_MAX = 9000
    V_MAX       = 5000
    V_MIN       = 1200
    SAMPLE_DT   = 0.004
    WINDOW_S    = 0.03
    HITS_NEEDED = 1
    HYST_RATIO  = 0.92
    DWELL_S     = 0.015

    def smoothstep01(x):
        x = 0 if x < 0 else (1 if x > 1 else x)
        return x*x*(3 - 2*x)

    try:
        # Ensure fresh serial data each cycle
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        # Reinitialize gripper each close to avoid post-first-run no-op behavior on some firmware
        arm.set_gripper_enable(True)
        arm.set_gripper_mode(0)
        # Ensure we're starting from an open position
        arm.set_gripper_position(start, wait=True, speed=V_MAX)
        time.sleep(0.05)
    except Exception:
        pass

    f_ema, hits = 0.0, 0
    pos = start
    while pos >= end:
        ratio = f_ema / max(1e-6, threshold)
        s = smoothstep01(ratio)
        step  = int(max(MIN_STEP, BASE_STEP*(1 - 0.8*s)))
        speed = int(V_MIN + (V_MAX - V_MIN)*(1 - s))
        arm.set_gripper_position(pos, wait=False, speed=speed)

        t_end = time.time() + WINDOW_S
        while time.time() < t_end:
            v = read_force(ser, timeout=SAMPLE_DT)
            if v is not None:
                f_ema = EMA_ALPHA*v + (1-EMA_ALPHA)*f_ema
                if f_ema >= threshold:
                    hits += 1
                    if hits >= HITS_NEEDED:
                        arm.set_gripper_position(pos, wait=True, speed=max(V_MIN, 800))
                        return True
                elif f_ema < threshold*HYST_RATIO:
                    hits = 0
            time.sleep(SAMPLE_DT)

        time.sleep(DWELL_S)
        pos -= step

    arm.set_gripper_position(end, wait=True, speed=V_MIN)
    return False

# ========= Main loop (manual trigger like yolo_manual2) =========
def main_once():
    ap = argparse.ArgumentParser(description='Manual YOLO pick&place with force-only close')
    # Detection
    ap.add_argument('--classes', nargs='+',
                    default=['sports ball','orange','apple'],
                    help='YOLO class names to treat as fruit')
    ap.add_argument('--conf', type=float, default=0.35, help='min confidence')
    ap.add_argument('--detect-every', type=int, default=5, help='run YOLO every N frames')

    # Bias controls
    ap.add_argument('--dx', type=float, default=0.0)
    ap.add_argument('--dy', type=float, default=0.0)
    ap.add_argument('--dz', type=float, default=0.0)
    ap.add_argument('--dy-pos', type=float, default=None)
    ap.add_argument('--dy-neg', type=float, default=None)
    ap.add_argument('--y-bias-per-px', type=float, default=0.0)

    # Force sensor (REQUIRED)
    ap.add_argument('--force-port', default='COM5')
    ap.add_argument('--force-baud', type=int, default=9600)
    ap.add_argument('--force-threshold', type=float, default=100.0)

    # Bin / travel (defaults set to your exact numbers)
    ap.add_argument('--bin-x', type=float, default=313.4)
    ap.add_argument('--bin-y', type=float, default=-353.5)
    ap.add_argument('--bin-z', type=float, default=-54.6)          # final drop height
    ap.add_argument('--bin-approach-z', type=float, default=156.1)  # approach height over bin
    ap.add_argument('--travel-z', type=float, default=380.0)       # clear everything

    # Camera
    ap.add_argument('--align-to-rgb', action='store_true', default=True,
                    help='Request Stereo depth aligned to RGB (requires updated DepthAiCamera)')
    # Verbose/printing & UI
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--print-every', type=float, default=2.0)
    args, _ = ap.parse_known_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Start OAK-D (aligned if supported)
    print("Starting OAK-D camera (640x400)...")
    try:
        cam = DepthAiCamera(width=640, height=400, disable_rgb=False, align_to_rgb=bool(args.align_to_rgb))
    except TypeError:
        cam = DepthAiCamera(width=640, height=400, disable_rgb=False)
    try:
        aligned_flag = bool(getattr(cam, "depth_aligned_to_rgb", False))
    except Exception:
        aligned_flag = False

    K_rgb, K_depth = cam.get_intrinsics()
    K_use = K_rgb if aligned_flag else K_depth
    print(f"Depth aligned to RGB: {aligned_flag}")
    if not aligned_flag:
        print("[WARN] Depth is NOT aligned to RGB; expect lateral bias. Update camera/depthai_camera.py to enable alignment.")

    # YOLO11 (Ultralytics)
    print("Loading YOLO11 model (yolo11n.pt)...")
    model = YOLO("yolo11n.pt")
    model.to(device)

    # Force sensor (REQUIRED: exit if not available)
    try:
        ser = serial.Serial(args.force_port, args.force_baud, timeout=0.5)
        ser.reset_input_buffer()
        print(f"[force] Connected {args.force_port} @ {args.force_baud}")
    except Exception as e:
        print(f"[force] ERROR: {e}")
        print("[force] Force sensor is required (force-only closing). Exiting.")
        try: arm.reset(wait=True)
        except Exception: pass
        raise SystemExit(2)

    # UI
    try: cv2.namedWindow('Manual YOLO', cv2.WINDOW_NORMAL)
    except Exception: pass
    DETECTION_INTERVAL = max(1, int(args.detect_every))
    frame_count = 0
    last_df = None
    last_est = None
    verbose = bool(args.verbose)
    last_print_time = 0.0

    awaiting_confirm = False
    pending_est = None
    pending_plan_lines = []

    print("Keys: 'p' stage, 'y' confirm, 'n' cancel, 's' save, 'q' quit.")

    while True:
        color, depth = cam.get_images()
        if color is None:
            continue

        frame_count += 1
        if frame_count % DETECTION_INTERVAL == 0:
            # Run YOLO11 inference
            with torch.no_grad():
                results = model(color, conf=float(args.conf), iou=0.45, verbose=False)

            r = results[0]
            det_rows = []
            if r.boxes is not None and len(r.boxes):
                boxes = r.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss  = boxes.cls.cpu().numpy().astype(int)
                names = getattr(r, 'names', None) or getattr(model, 'names', None)
                for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
                    label = str(cls_id)
                    if isinstance(names, dict) and cls_id in names:
                        label = names[cls_id]
                    elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                        label = names[cls_id]
                    det_rows.append({
                        'xmin': float(x1), 'ymin': float(y1), 'xmax': float(x2), 'ymax': float(y2),
                        'confidence': float(c), 'name': label
                    })

            df_all = pd.DataFrame(det_rows)
            last_df = df_all

            # Filter desired classes
            if not df_all.empty:
                df = df_all[df_all['name'].isin(args.classes)].copy()
            else:
                df = pd.DataFrame(columns=['xmin','ymin','xmax','ymax','confidence','name'])

            if not df.empty:
                idx = df['confidence'].astype(float).idxmax()
                best = df.loc[idx]
                x1, y1, x2, y2 = map(int, [best['xmin'], best['ymin'], best['xmax'], best['ymax']])
                cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)

                depth_m = None
                if depth is not None and 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                    d = depth[cy, cx]
                    depth_m = float(d) if np.isfinite(d) else median_valid_depth(depth, cx, cy, k=3)

                if depth_m is not None and np.isfinite(depth_m):
                    x_cam, y_cam, z_cam = pixel_to_3d(cx, cy, depth_m, K_use)
                    xr_raw, yr_raw, zr_raw = camera_to_robot(x_cam, y_cam, z_cam, OBS_POSE)
                    xr, yr, zr = apply_bias(xr_raw, yr_raw, zr_raw, args, cx=cx, img_w=color.shape[1])
                    name = best['name']
                    conf = float(best['confidence'])
                    last_est = {
                        'cx': cx, 'cy': cy, 'depth_m': depth_m,
                        'x_cam': x_cam, 'y_cam': y_cam, 'z_cam': z_cam,
                        'xr': xr, 'yr': yr, 'zr': clamp_to_table(zr),
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
                        print(f"RobotXYZ(biased): X={xr:.1f}   Y={yr:.1f}   Z={last_est['zr']:.1f}")

        # Draw overlay
        display = color.copy()
        if last_df is not None and not last_df.empty:
            for _, det in last_df.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 2)
                label = det.get('name', 'obj'); conf = float(det.get('confidence', 0.0))
                cv2.putText(display, f"{label}:{conf:.2f}", (x1, max(20, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)

        status = f"AlignedDepth={'YES' if aligned_flag else 'NO'} | yBias/px={getattr(args,'y_bias_per_px',0.0):.3f} mm"
        cv2.putText(display, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)

        if last_est is not None:
            cx, cy = last_est['cx'], last_est['cy']
            cv2.drawMarker(display, (cx, cy), (0,255,255), cv2.MARKER_CROSS, 18, 2)
            hud = [
                f"{last_est['name']} ({last_est['conf']:.2f})",
                f"px=({cx},{cy}) d={last_est['depth_m']:.3f}m",
                f"X={last_est['xr']:.1f} Y={last_est['yr']:.1f} Z={last_est['zr']:.1f} mm",
                f"raw Z={last_est['zr_raw']:.1f}"
            ]
            for i, txt in enumerate(hud):
                cv2.putText(display, txt, (10, 44 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # Show pending pick confirmation
        if 'awaiting_confirm' in locals() and awaiting_confirm and pending_plan_lines:
            y0 = 44 + 4*20
            for i, line in enumerate(["CONFIRM PICK: 'y' execute, 'n' cancel"] + pending_plan_lines):
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
                x = float(last_est['xr'])
                y = float(last_est['yr'])
                # Previous behavior (using detected Z with -10 mm offset):
                # z_obj = float(last_est['zr']) - 10.0
                # New behavior: always use a fixed pick height at Z = -68.7 mm
                z_obj = -68.7
                z_pick  = clamp_motion(z_obj)
                z_hover = clamp_motion(z_pick + HOVER_DELTA_MM)
                z_lift  = clamp_motion(z_hover + LIFT_DELTA_MM)

                pending_plan_lines[:] = [
                    f"Target XY=({x:.1f},{y:.1f})  Z_obj={z_obj:.1f}",
                    f"HoverZ={z_hover:.1f}  GraspZ={z_pick:.1f}  LiftZ={z_lift:.1f}",
                    f"Bin=(X={args.bin_x:.1f}, Y={args.bin_y:.1f})  travelZ={args.travel_z:.1f}  approachZ={args.bin_approach_z:.1f}  placeZ={args.bin_z:.1f}"
                ]
                if z_pick != z_obj:
                    pending_plan_lines.append(f"[SAFETY] Z clamped (table {Z_TABLE_LIMIT_MM}+{Z_CLEARANCE_MM}mm)")
                # Keep a copy of the latest estimate; force Z to the fixed pick height for execution
                pending_est = dict(last_est)
                # Previous: pending_est['zr'] = float(last_est['zr']) - 10.0
                pending_est['zr'] = -68.7
                awaiting_confirm = True
                print("\n[PENDING PICK] " + "  ".join(pending_plan_lines))

        elif key == ord('y'):
            if awaiting_confirm and pending_est is not None:
                print("Pick confirmed.")
                awaiting_confirm = False
                try:
                    # --- Execute pick ---
                    x = float(pending_est['xr'])
                    y = float(pending_est['yr'])
                    # Use staged/detected Z (with -10 mm applied when staged)
                    z_pick  = clamp_motion(float(pending_est['zr']))
                    # Previous fallback (fixed Z):
                    # z_pick  = clamp_motion(-68.7)
                    z_hover = clamp_motion(z_pick + HOVER_DELTA_MM)
                    z_lift  = clamp_motion(z_hover + LIFT_DELTA_MM)

                    # Move & grasp
                    arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
                    arm.set_position(x, y, OBS_POSE[2], OBS_POSE[3], OBS_POSE[4], OBS_POSE[5], wait=True)
                    arm.set_position(x, y, z_hover,     OBS_POSE[3], OBS_POSE[4], OBS_POSE[5], wait=True)
                    arm.set_position(x, y, z_pick,      OBS_POSE[3], OBS_POSE[4], OBS_POSE[5], wait=True)

                    # Force-only close
                    smooth_close_with_force(arm, ser, threshold=float(args.force_threshold))
                    time.sleep(0.2)

                    # Lift
                    arm.set_position(x, y, z_lift, OBS_POSE[3], OBS_POSE[4], OBS_POSE[5], wait=True)

                    # --- Place sequence 
                    # Clear over everything
                    arm.set_position(args.bin_x, args.bin_y, clamp_motion(args.travel_z), OBS_POSE[3], OBS_POSE[4], OBS_POSE[5], wait=True)
                    # Approach height over bin
                    arm.set_position(args.bin_x, args.bin_y, clamp_motion(args.bin_approach_z), OBS_POSE[3], OBS_POSE[4], OBS_POSE[5], wait=True)
                    # Place height 
                    arm.set_position(args.bin_x, args.bin_y, clamp_motion(args.bin_z), OBS_POSE[3], OBS_POSE[4], OBS_POSE[5], wait=True)

                    # Open gripper to drop
                    arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
                    time.sleep(0.2)

                    # Back to clear height, then OBS
                    arm.set_position(args.bin_x, args.bin_y, clamp_motion(args.travel_z), OBS_POSE[3], OBS_POSE[4], OBS_POSE[5], wait=True)
                    arm.set_position(*OBS_POSE, wait=True)

                except Exception as e:
                    print(f"[pick/place] error: {e}")
                    try: arm.set_position(*OBS_POSE, wait=True)
                    except Exception: pass
                finally:
                    pending_est = None
                    pending_plan_lines = []

        elif key == ord('n'):
            if awaiting_confirm:
                print("Pick canceled.")
                awaiting_confirm = False
                pending_est = None
                pending_plan_lines = []

    # cleanup
    try:
        ser.close()
    except Exception:
        pass
    try:
        arm.reset(wait=True)
    except Exception:
        pass

if __name__ == "__main__":
    try:
        main_once()
    except KeyboardInterrupt:
        try: arm.reset(wait=True)
        except Exception: pass
