#!/usr/bin/env python3
"""
Minimal: Move to a known pose, detect the top object once, and print estimated coordinates.
"""

import time
import sys
import argparse
import warnings
import numpy as np
import torch
import cv2

from xarm.wrapper import XArmAPI
from camera.depthai_camera import DepthAiCamera

warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

# ---- Robot connect & move to pose ----
# arm = XArmAPI('192.168.1.202')
arm = XArmAPI('192.168.1.221')

time.sleep(0.5)
arm.set_mode(0)
arm.set_state(0)
arm.reset(wait=True)

# Move to observation pose (scan)
OBS_POSE = [357.4, 1.1, 231.7, 178.8, 0.3, 1.0]
arm.set_gripper_position(850, wait=True)
arm.set_position(*OBS_POSE, wait=True)

# ---- Minimal detection and coordinate estimation ----

# Hand-eye constants (camera mounted on gripper, looking down)
EULER_EEF_TO_COLOR_OPT = [0.0703, 0.0023, 0.0195, 0, 0, 1.579]  # m/rad
GRIPPER_Z_MM = 70  # mm

# --- Simple grasp parameters ---
GRIPPER_OPEN_POS = 850
GRIPPER_CLOSE_POS = 300
HOVER_DELTA_MM = 40.0   # hover above detected Z by this many mm
LIFT_DELTA_MM  = 100.0  # lift up after grasp
MIN_Z_LIMIT_MM = -250.0 # safety floor (do not go below)
MAX_Z_LIMIT_MM = 400.0  # safety ceiling

def pixel_to_3d(px, py, depth_m, K):
	fx = K[0][0]; fy = K[1][1]; cx = K[0][2]; cy = K[1][2]
	z = float(depth_m)
	x = (px - cx) * z / fx
	y = (py - cy) * z / fy
	return x, y, z  # meters

def _euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg):
	"""Rotation matrix from roll/pitch/yaw in degrees.
	Uses yaw-pitch-roll (ZYX) order: R = Rz(yaw) * Ry(pitch) * Rx(roll).
	This matches common RPY conventions in many robot controllers.
	"""
	rd, pd, yd = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
	cr, sr = np.cos(rd), np.sin(rd)
	cp, sp = np.cos(pd), np.sin(pd)
	cy, sy = np.cos(yd), np.sin(yd)
	Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
	Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
	Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
	return Rz @ Ry @ Rx

def _euler_rad_to_rot(rx, ry, rz):
	"""Rotation matrix from roll/pitch/yaw in radians.
	Uses yaw-pitch-roll (ZYX) order to align with _euler_deg_to_rot.
	"""
	cr, sr = np.cos(rx), np.sin(rx)
	cp, sp = np.cos(ry), np.sin(ry)
	cy, sy = np.cos(rz), np.sin(rz)
	Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
	Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
	Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
	return Rz @ Ry @ Rx

def camera_to_robot(x_cam_m, y_cam_m, z_cam_m, eef_pose_xyzrpy):
	"""Full eye-in-hand transform.
	p_base = t_base_eef + R_base_eef @ (R_eef_cam @ p_cam + t_eef_cam)
	Returns base frame (x,y,z) in mm.
	"""
	# Camera point in meters
	p_cam = np.array([x_cam_m, y_cam_m, z_cam_m], dtype=float)

	# EEF->Camera transform (meters and radians from EULER_EEF_TO_COLOR_OPT)
	t_eef_cam = np.array(EULER_EEF_TO_COLOR_OPT[:3], dtype=float)  # meters
	rx_c, ry_c, rz_c = EULER_EEF_TO_COLOR_OPT[3:6]
	R_eef_cam = _euler_rad_to_rot(rx_c, ry_c, rz_c)

	# Transform point into EEF frame
	p_eef = R_eef_cam @ p_cam + t_eef_cam

	# Base->EEF transform (EEF pose is in mm and degrees per OBS_POSE / xArm defaults)
	x_eef_mm, y_eef_mm, z_eef_mm, roll_deg, pitch_deg, yaw_deg = eef_pose_xyzrpy
	t_base_eef = np.array([x_eef_mm, y_eef_mm, z_eef_mm], dtype=float) / 1000.0  # to meters
	R_base_eef = _euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg)

	# Final point in base (meters)
	p_base = t_base_eef + R_base_eef @ p_eef

	# Return in mm
	return float(p_base[0] * 1000.0), float(p_base[1] * 1000.0), float(p_base[2] * 1000.0)

def median_valid_depth(depth_img, cx, cy, k=3):
	"""Return median depth (m) in a kxk window around (cx,cy) ignoring NaNs."""
	h, w = depth_img.shape[:2]
	x0 = max(0, cx - k)
	x1 = min(w, cx + k + 1)
	y0 = max(0, cy - k)
	y1 = min(h, cy + k + 1)
	patch = depth_img[y0:y1, x0:x1].astype(np.float32)
	vals = patch[np.isfinite(patch)]
	if vals.size == 0:
		return None
	return float(np.median(vals))

def _clamp(v, lo, hi):
	return max(lo, min(hi, v))

def pick_last_estimate(arm: XArmAPI, est: dict):
	"""Pick sequence using last_est (expects keys 'xr','yr','zr')."""
	x = float(est['xr']); y = float(est['yr']); z_obj = float(est['zr'])
	roll, pitch, yaw = OBS_POSE[3], OBS_POSE[4], OBS_POSE[5]

	hover_z = _clamp(z_obj + HOVER_DELTA_MM, MIN_Z_LIMIT_MM, MAX_Z_LIMIT_MM)
	grasp_z = _clamp(z_obj + 0.0, MIN_Z_LIMIT_MM, MAX_Z_LIMIT_MM)
	lift_z  = _clamp(hover_z + LIFT_DELTA_MM, MIN_Z_LIMIT_MM, MAX_Z_LIMIT_MM)

	print("\n--- PICK PLAN ---")
	print(f"Target XY=({x:.1f},{y:.1f})  Z_obj={z_obj:.1f}")
	print(f"HoverZ={hover_z:.1f}  GraspZ={grasp_z:.1f}  LiftZ={lift_z:.1f}")

	try:
		# Ensure open
		arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)

		# Move above target at current observation height
		arm.set_position(x, y, OBS_POSE[2], roll, pitch, yaw, wait=True)
		# Descend to hover
		arm.set_position(x, y, hover_z, roll, pitch, yaw, wait=True)
		# Descend to grasp
		arm.set_position(x, y, grasp_z, roll, pitch, yaw, wait=True)
		# Close gripper
		arm.set_gripper_position(GRIPPER_CLOSE_POS, wait=True)
		time.sleep(0.25)
		# Lift
		arm.set_position(x, y, lift_z, roll, pitch, yaw, wait=True)
	except Exception as e:
		print(f"Pick failed: {e}")
	finally:
		# Return to observation pose for safety
		try:
			arm.set_position(*OBS_POSE, wait=True)
		except Exception:
			pass

def main_once():
	# --- Args: optional calibration biases (mm) ---
	parser = argparse.ArgumentParser(description='Manual YOLO with coordinate printout')
	parser.add_argument('--dx', type=float, default=-7.7, help='Calibration bias in X (mm)')
	parser.add_argument('--dy', type=float, default=98.5, help='Calibration bias in Y (mm)')
	parser.add_argument('--dz', type=float, default=139.9, help='Calibration bias in Z (mm)')
	parser.add_argument('--verbose', action='store_true', help='Enable periodic console prints of estimates')
	parser.add_argument('--print-every', type=float, default=2.0, help='Seconds between prints when --verbose is set')
	args, _ = parser.parse_known_args()
	calib_bias = np.array([args.dx, args.dy, args.dz], dtype=float)
	if np.any(calib_bias != 0.0):
		print(f"Using calibration bias (mm): dx={args.dx:.1f}, dy={args.dy:.1f}, dz={args.dz:.1f}")
	if not args.verbose:
		print("Quiet mode: no per-frame console spam. Press 'v' to toggle verbose, 'i' to print current estimate, 'p' to pick, 'q' to quit.")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	# Load YOLOv5 (lightweight)
	print("Loading YOLOv5s model...")
	model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
	model.to(device).eval()
	model.conf = 0.25
	model.iou = 0.45

	# Start OAK-D
	print("Starting OAK-D camera (640x400)...")
	cam = DepthAiCamera(width=640, height=400, disable_rgb=False)
	_, Kd = cam.get_intrinsics()  # use depth intrinsics

	# Live preview like yolo.py (press 'q' to quit, 's' to save)
	try:
		cv2.namedWindow('Manual YOLO', cv2.WINDOW_NORMAL)
	except Exception:
		pass

	DETECTION_INTERVAL = 5
	frame_count = 0
	last_df = None
	last_best = None
	last_est = None  # dict with keys: cx, cy, depth_m, x_cam, y_cam, z_cam, xr, yr, zr, name, conf
	verbose = bool(args.verbose)
	last_print_time = 0.0

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
				last_best = best
				x1, y1, x2, y2 = map(int, [best['xmin'], best['ymin'], best['xmax'], best['ymax']])
				cx = int((x1 + x2) / 2)
				cy = int((y1 + y2) / 2)

				depth_m = None
				if depth is not None and 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
					d = depth[cy, cx]
					if np.isfinite(d):
						depth_m = float(d)
					else:
						depth_m = median_valid_depth(depth, cx, cy, k=3)

				if depth_m is not None:
					x_cam, y_cam, z_cam = pixel_to_3d(cx, cy, depth_m, Kd)
					xr_raw, yr_raw, zr_raw = camera_to_robot(x_cam, y_cam, z_cam, OBS_POSE)
					xr, yr, zr = (np.array([xr_raw, yr_raw, zr_raw]) + calib_bias).tolist()
					name = best['name'] if 'name' in best else 'object'
					conf = float(best['confidence']) if 'confidence' in best else 0.0
					last_est = {
						'cx': cx, 'cy': cy, 'depth_m': depth_m,
						'x_cam': x_cam, 'y_cam': y_cam, 'z_cam': z_cam,
						'xr': xr, 'yr': yr, 'zr': zr,
						'xr_raw': xr_raw, 'yr_raw': yr_raw, 'zr_raw': zr_raw,
						'name': name, 'conf': conf,
					}
					# Optional periodic console print in verbose mode only
					now = time.time()
					if verbose and (now - last_print_time) >= max(0.2, float(args.print_every)):
						last_print_time = now
						print("\n=== Estimated target ===")
						print(f"Detection: {name} (conf {conf:.2f})")
						print(f"Pixel:     ({cx}, {cy})  depth: {depth_m:.3f} m")
						print(f"Camera3D:  x={x_cam:.3f} m  y={y_cam:.3f} m  z={z_cam:.3f} m")
						print(f"RobotXYZ(raw): X={xr_raw:.1f} mm  Y={yr_raw:.1f} mm  Z={zr_raw:.1f} mm")
						if np.any(calib_bias != 0.0):
							print(f"RobotXYZ(cal): X={xr:.1f} mm  Y={yr:.1f} mm  Z={zr:.1f} mm  (bias applied)")

		# Draw overlay
		display = color.copy()
		if last_df is not None and not last_df.empty:
			for _, det in last_df.iterrows():
				x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
				cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 2)
				label = det['name'] if 'name' in det else 'obj'
				conf = float(det['confidence']) if 'confidence' in det else 0.0
				cv2.putText(display, f"{label}:{conf:.2f}", (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)

		if last_est is not None:
			cx, cy = last_est['cx'], last_est['cy']
			cv2.drawMarker(display, (cx, cy), (0,255,255), cv2.MARKER_CROSS, 18, 2)
			lines = [
				f"{last_est['name']} ({last_est['conf']:.2f})",
				f"px=({cx},{cy}) d={last_est['depth_m']:.3f}m",
				f"X={last_est['xr']:.1f} Y={last_est['yr']:.1f} Z={last_est['zr']:.1f} mm",
			]
			if 'xr_raw' in last_est:
				lines.append(f"raw: X={last_est['xr_raw']:.1f} Y={last_est['yr_raw']:.1f} Z={last_est['zr_raw']:.1f}")
			for i, txt in enumerate(lines):
				cv2.putText(display, txt, (10, 24 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

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
				print("No estimate yet; can't pick.")
			else:
				print("Executing pick on last estimate (press q to abort loop)…")
				pick_last_estimate(arm, last_est)
		elif key == ord('v'):
			verbose = not verbose
			state = 'ON' if verbose else 'OFF'
			print(f"Verbose printing: {state} (print-every={args.print_every}s)")
		elif key == ord('i'):
			if last_est is None:
				print("No estimate yet.")
			else:
				e = last_est
				print("\n=== Current estimate (snapshot) ===")
				print(f"Detection: {e['name']} (conf {e['conf']:.2f})")
				print(f"Pixel:     ({e['cx']}, {e['cy']})  depth: {e['depth_m']:.3f} m")
				print(f"Camera3D:  x={e['x_cam']:.3f} m  y={e['y_cam']:.3f} m  z={e['z_cam']:.3f} m")
				print(f"RobotXYZ(raw): X={e['xr_raw']:.1f} mm  Y={e['yr_raw']:.1f} mm  Z={e['zr_raw']:.1f} mm")
				print(f"RobotXYZ(cal): X={e['xr']:.1f} mm  Y={e['yr']:.1f} mm  Z={e['zr']:.1f} mm")

if __name__ == "__main__":
	try:
		main_once()
	finally:
		try:
			arm.reset(wait=True)
		except Exception:
			pass
