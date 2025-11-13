#!/usr/bin/env python3
"""
Simple manual YOLO11 picking with OAK-D + xArm.

- Moves to a fixed observation pose
- Detects the top object with YOLO11
- Uses depth + hand–eye calibration to convert to robot base coordinates
- Lets you confirm the pick with keyboard

Keys:
  q : quit
  s : save screenshot
  p : stage pick (plan only)
  y : execute staged pick
  n : cancel staged pick
  i : print current estimate
"""

import time
import warnings
import numpy as np
import torch
import cv2

from ultralytics import YOLO
from xarm.wrapper import XArmAPI
from camera.depthai_camera import DepthAiCamera

warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

# --------------------------------------------------------------------------
# Robot connect & move to observation pose
# --------------------------------------------------------------------------

arm = XArmAPI("192.168.1.221")
time.sleep(0.5)
arm.set_mode(0)
arm.set_state(0)
arm.reset(wait=True)

OBS_POSE = [357.4, 1.1, 231.7, 178.8, 0.3, 1.0]
arm.set_gripper_position(850, wait=True)
arm.set_position(*OBS_POSE, wait=True)

# --------------------------------------------------------------------------
# Hand–eye calibration (EEF -> camera), meters/radians
# (This is the set that worked well for you in yolo_manual2.py)
# tz is flipped INSIDE camera_to_robot()
# --------------------------------------------------------------------------

EULER_EEF_TO_COLOR_OPT = [
    0.047979961751551,
    -0.041452806329264,
    0.090504323450518,
    -0.030203336543533,
    -0.020719773728618,
    1.505089859105352,
]


# --------------------------------------------------------------------------
# Grasp & safety parameters
# --------------------------------------------------------------------------

GRIPPER_OPEN_POS = 850
GRIPPER_CLOSE_POS = 300

HOVER_DELTA_MM = 40.0
LIFT_DELTA_MM  = 100.0

MIN_Z_LIMIT_MM = -250.0
MAX_Z_LIMIT_MM =  400.0
Z_TABLE_LIMIT_MM = -85.0
Z_CLEARANCE_MM   =   2.0

def clamp_to_table(z_mm: float) -> float:
    return max(z_mm, Z_TABLE_LIMIT_MM + Z_CLEARANCE_MM)

def clamp_motion(z_mm: float) -> float:
    z_mm = max(MIN_Z_LIMIT_MM, min(MAX_Z_LIMIT_MM, z_mm))
    return clamp_to_table(z_mm)

# --------------------------------------------------------------------------
# Camera / math helpers
# --------------------------------------------------------------------------

def pixel_to_3d(px, py, depth_m, K):
    fx = K[0][0]; fy = K[1][1]
    cx = K[0][2]; cy = K[1][2]
    z = float(depth_m)
    x = (px - cx) * z / fx
    y = (py - cy) * z / fy
    return x, y, z  # meters

def _euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg):
    rd, pd, yd = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(rd), np.sin(rd)
    cp, sp = np.cos(pd), np.sin(pd)
    cy, sy = np.cos(yd), np.sin(yd)
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    return Rz @ Ry @ Rx

def _euler_rad_to_rot(rx, ry, rz):
    cr, sr = np.cos(rx), np.sin(rx)
    cp, sp = np.cos(ry), np.sin(ry)
    cy, sy = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    return Rz @ Ry @ Rx

def camera_to_robot(x_cam_m, y_cam_m, z_cam_m, eef_pose_xyzrpy):
    """
    Convert a point in CAMERA coordinates (meters) to ROBOT BASE coordinates (mm).

    p_base = t_base_eef + R_base_eef @ (R_eef_cam @ p_cam + t_eef_cam)

    - EULER_EEF_TO_COLOR_OPT gives us t_eef_cam, R_eef_cam
    - We flip tz ONCE here because the camera looks down.
    """
    p_cam = np.array([x_cam_m, y_cam_m, z_cam_m], dtype=float)

    tx, ty, tz, rx_c, ry_c, rz_c = EULER_EEF_TO_COLOR_OPT
    t_eef_cam = np.array([tx, ty, -tz], dtype=float)   # flip tz here
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
    if vals.size == 0:
        return None
    return float(np.median(vals))

# --------------------------------------------------------------------------
# Robot helpers
# --------------------------------------------------------------------------

def _get_current_pose(arm: XArmAPI):
    try:
        res = arm.get_position()
        data = None
        if isinstance(res, dict) and "data" in res:
            data = res.get("data")
        elif isinstance(res, (list, tuple)):
            if len(res) >= 2 and isinstance(res[1], (list, tuple)):
                data = res[1]
            elif len(res) >= 6 and all(isinstance(v, (int, float)) for v in res[:6]):
                data = res[:6]
        if data and len(data) >= 6:
            return [float(data[0]), float(data[1]), float(data[2]),
                    float(data[3]), float(data[4]), float(data[5])]
    except Exception:
        pass
    return OBS_POSE

def pick_last_estimate(arm: XArmAPI, est: dict):
    x = float(est["xr"])
    y = float(est["yr"])
    # Apply a deeper safety offset: pick 20 mm lower than the detected surface
    z_obj = float(est["zr"]) - 20.0
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
        try:
            arm.set_position(*OBS_POSE, wait=True)
        except Exception:
            pass

# --------------------------------------------------------------------------
# YOLO11 helper
# --------------------------------------------------------------------------

def run_yolo11(model, img_bgr, conf=0.25, iou=0.45):
    """
    Run YOLO11 and return a list of detections as dicts:
      {"xmin","ymin","xmax","ymax","confidence","name"}
    """
    with torch.no_grad():
        results = model(img_bgr, conf=conf, iou=iou, verbose=False)

    r = results[0]
    dets = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss  = boxes.cls.cpu().numpy().astype(int)

    names = getattr(r, "names", None)
    if names is None:
        names = getattr(model, "names", None)

    for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
        label = str(cls_id)
        if isinstance(names, dict) and cls_id in names:
            label = names[cls_id]
        elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            label = names[cls_id]

        dets.append({
            "xmin": float(x1),
            "ymin": float(y1),
            "xmax": float(x2),
            "ymax": float(y2),
            "confidence": float(c),
            "name": label,
        })
    return dets

# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main_once():
    print("Keys: 'p' stage pick, 'y' confirm, 'n' cancel, 'i' info, 's' save, 'q' quit.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load YOLO11
    print("Loading YOLO11 model (yolo11n.pt)...")
    model = YOLO("yolo11n.pt")
    model.to(device)

    # Start OAK-D
    print("Starting OAK-D camera (640x400)...")
    cam = DepthAiCamera(width=640, height=400, disable_rgb=False)
    _, Kd = cam.get_intrinsics()  # depth intrinsics

    try:
        cv2.namedWindow("Manual YOLO11", cv2.WINDOW_NORMAL)
    except Exception:
        pass

    DETECTION_INTERVAL = 5
    frame_count = 0
    last_dets = []
    last_est = None

    awaiting_confirm = False
    pending_est = None
    pending_plan_lines = []

    while True:
        color, depth = cam.get_images()
        if color is None:
            continue

        frame_count += 1
        if frame_count % DETECTION_INTERVAL == 0:
            dets = run_yolo11(model, color, conf=0.25, iou=0.45)
            last_dets = dets

            if dets:
                best = max(dets, key=lambda d: d["confidence"])
                x1, y1, x2, y2 = map(int, [best["xmin"], best["ymin"], best["xmax"], best["ymax"]])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                depth_m = None
                if depth is not None and 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                    dval = depth[cy, cx]
                    depth_m = float(dval) if np.isfinite(dval) else median_valid_depth(depth, cx, cy, k=3)

                if depth_m is not None:
                    x_cam, y_cam, z_cam = pixel_to_3d(cx, cy, depth_m, Kd)
                    xr_raw, yr_raw, zr_raw = camera_to_robot(x_cam, y_cam, z_cam, OBS_POSE)

                    name = best["name"]
                    conf = float(best["confidence"])

                    last_est = {
                        "cx": cx, "cy": cy, "depth_m": depth_m,
                        "x_cam": x_cam, "y_cam": y_cam, "z_cam": z_cam,
                        "xr": xr_raw, "yr": yr_raw, "zr": clamp_to_table(zr_raw),
                        "xr_raw": xr_raw, "yr_raw": yr_raw, "zr_raw": zr_raw,
                        "name": name, "conf": conf,
                    }

                    print("\n=== Estimated target ===")
                    print(f"Detection: {name} (conf {conf:.2f})")
                    print(f"Pixel:     ({cx}, {cy})  depth: {depth_m:.3f} m")
                    print(f"Camera3D:  x={x_cam:.3f} m  y={y_cam:.3f} m  z={z_cam:.3f} m")
                    print(f"RobotXYZ:  X={xr_raw:.1f} Y={yr_raw:.1f} Z={zr_raw:.1f} (mm)")

        # Draw overlay
        display = color.copy()

        if last_dets:
            for det in last_dets:
                x1 = int(det["xmin"]); y1 = int(det["ymin"])
                x2 = int(det["xmax"]); y2 = int(det["ymax"])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 2)
                label = det.get("name", "obj")
                conf = float(det.get("confidence", 0.0))
                cv2.putText(display, f"{label}:{conf:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

        if last_est is not None:
            cx, cy = last_est["cx"], last_est["cy"]
            cv2.drawMarker(display, (cx, cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 18, 2)
            hud = [
                f"{last_est['name']} ({last_est['conf']:.2f})",
                f"px=({cx},{cy}) d={last_est['depth_m']:.3f}m",
                f"X={last_est['xr']:.1f} Y={last_est['yr']:.1f} Z={last_est['zr']:.1f} mm",
            ]
            for i, txt in enumerate(hud):
                cv2.putText(display, txt, (10, 24 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if awaiting_confirm and pending_plan_lines:
            y0 = 24 + 5 * 20
            lines = ["CONFIRM PICK: press 'y' to execute, 'n' to cancel"] + pending_plan_lines
            for i, line in enumerate(lines):
                cv2.putText(display, line, (10, y0 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Manual YOLO11", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            fname = f"manual_yolo11_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, display)
            print(f"Saved {fname}")

        elif key == ord("p"):
            if last_est is None:
                print("No estimate yet; can't stage pick.")
            else:
                x = float(last_est["xr"])
                y = float(last_est["yr"])
                z_obj = float(last_est["zr"])

                z_pick  = clamp_motion(z_obj)
                z_hover = clamp_motion(z_pick + HOVER_DELTA_MM)
                z_lift  = clamp_motion(z_hover + LIFT_DELTA_MM)

                pending_plan_lines = [
                    f"Target XY=({x:.1f},{y:.1f})  Z_obj={z_obj:.1f}",
                    f"HoverZ={z_hover:.1f}  GraspZ={z_pick:.1f}  LiftZ={z_lift:.1f}",
                ]
                if z_pick != z_obj:
                    pending_plan_lines.append(
                        f"[SAFETY] Z pick clamped (table {Z_TABLE_LIMIT_MM}+{Z_CLEARANCE_MM}mm)"
                    )
                pending_est = dict(last_est)
                awaiting_confirm = True
                print("\n[PENDING PICK] " + "  ".join(pending_plan_lines))

        elif key == ord("y"):
            if awaiting_confirm and pending_est is not None:
                print("Pick confirmed.")
                awaiting_confirm = False
                try:
                    pick_last_estimate(arm, pending_est)
                finally:
                    pending_est = None
                    pending_plan_lines = []

        elif key == ord("n"):
            if awaiting_confirm:
                print("Pick canceled.")
                awaiting_confirm = False
                pending_est = None
                pending_plan_lines = []

        elif key == ord("i"):
            if last_est is None:
                print("No estimate yet.")
            else:
                e = last_est
                print("\n=== Current estimate (snapshot) ===")
                print(f"Detection: {e['name']} (conf {e['conf']:.2f})")
                print(f"Pixel:     ({e['cx']}, {e['cy']})  depth: {e['depth_m']:.3f} m")
                print(f"Camera3D:  x={e['x_cam']:.3f} m  y={e['y_cam']:.3f} m  z={e['z_cam']:.3f} m")
                print(f"RobotXYZ:  X={e['xr_raw']:.1f} Y={e['yr_raw']:.1f} Z={e['zr_raw']:.1f} mm")

if __name__ == "__main__":
    try:
        main_once()
    finally:
        try:
            arm.reset(wait=True)
        except Exception:
            pass
