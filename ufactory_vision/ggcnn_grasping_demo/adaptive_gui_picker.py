#!/usr/bin/env python3
"""
adaptive_gui_picker.py
Tkinter GUI for YOLO11 pick & place with adaptive force thresholds per object type.

Usage:
    python adaptive_gui_picker.py
"""

import time, warnings, re, threading
import numpy as np
import torch, cv2, serial
from ultralytics import YOLO
import pandas as pd
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from xarm.wrapper import XArmAPI
from camera.depthai_camera import DepthAiCamera

warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

# ---------------- Constants ----------------
ROBOT_IP = "192.168.1.221"
OBS_POSE = [357.4, 1.1, 231.7, 178.8, 0.3, 1.0]
GRIPPER_OPEN_POS = 850
HOVER_DELTA_MM = 40.0
LIFT_DELTA_MM = 120.0
MIN_Z_LIMIT_MM = -250.0
MAX_Z_LIMIT_MM = 400.0
Z_TABLE_LIMIT_MM = -85.0
Z_CLEARANCE_MM = 2.0

# Bin parameters
BIN_X = 313.4
BIN_Y = -353.5
BIN_Z = -54.6
BIN_APPROACH_Z = 156.1
TRAVEL_Z = 380.0

# Force sensor
FORCE_PORT = "COM5"

# Adaptive force thresholds per object type
FORCE_THRESHOLDS = {
    'apple': 120.0,      # Firm, higher threshold
    'orange': 100.0,     # Medium firmness
    'sports ball': 80.0, # Softer, lower threshold
}
DEFAULT_FORCE_THRESHOLD = 100.0

# Hand-eye calibration
try:
    from eef_to_color_opt import EULER_EEF_TO_COLOR_OPT
except Exception:
    EULER_EEF_TO_COLOR_OPT = [
        0.044707024930037, -0.040881623923506, 0.088584379244258,
        -0.053468293190528, -0.049247341299073, 1.534069196206855,
    ]

# ---------------- Helper Functions ----------------
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
    return x, y, z

def _euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg):
    rd, pd, yd = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(rd), np.sin(rd)
    cp, sp = np.cos(pd), np.sin(pd)
    cy, sy = np.cos(yd), np.sin(yd)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def _euler_rad_to_rot(rx, ry, rz):
    cr, sr = np.cos(rx), np.sin(rx)
    cp, sp = np.cos(ry), np.sin(ry)
    cy, sy = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def camera_to_robot(x_cam_m, y_cam_m, z_cam_m, eef_pose_xyzrpy):
    p_cam = np.array([x_cam_m, y_cam_m, z_cam_m], dtype=float)
    tx, ty, tz, rx_c, ry_c, rz_c = EULER_EEF_TO_COLOR_OPT
    t_eef_cam = np.array([tx, ty, -tz], dtype=float)
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

def get_force_threshold(object_name):
    """Get adaptive force threshold based on object type."""
    return FORCE_THRESHOLDS.get(object_name, DEFAULT_FORCE_THRESHOLD)

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

def smooth_close_with_force(arm: XArmAPI, ser: serial.Serial, threshold: float,
                            start=GRIPPER_OPEN_POS, end=300):
    EMA_ALPHA = 0.25
    BASE_STEP = 18
    MIN_STEP = 4
    V_MAX = 5000
    V_MIN = 1200
    SAMPLE_DT = 0.004
    WINDOW_S = 0.03
    HITS_NEEDED = 1
    HYST_RATIO = 0.92
    DWELL_S = 0.015

    def smoothstep01(x):
        x = 0 if x < 0 else (1 if x > 1 else x)
        return x*x*(3 - 2*x)

    try:
        ser.reset_input_buffer()
        arm.set_gripper_enable(True)
        arm.set_gripper_mode(0)
        arm.set_gripper_position(start, wait=True, speed=V_MAX)
        time.sleep(0.05)
    except Exception:
        pass

    f_ema, hits = 0.0, 0
    pos = start
    while pos >= end:
        ratio = f_ema / max(1e-6, threshold)
        s = smoothstep01(ratio)
        step = int(max(MIN_STEP, BASE_STEP*(1 - 0.8*s)))
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

# ---------------- GUI Application ----------------
class AdaptivePickerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO11 Pick & Place - Adaptive Force")
        self.root.geometry("1100x750")
        
        # State variables
        self.running = False
        self.staged_pick = None
        self.arm = None
        self.cam = None
        self.model = None
        self.ser = None
        self.K_use = None
        
        # Create GUI elements
        self.create_widgets()
        
        # Start camera/detection thread
        self.detection_thread = None
        
    def create_widgets(self):
        # Top control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.start_btn = ttk.Button(control_frame, text="▶️ Start System", command=self.start_system)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop", command=self.stop_system, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.stage_btn = ttk.Button(control_frame, text="📌 Stage Pick", command=self.stage_pick, state=tk.DISABLED)
        self.stage_btn.pack(side=tk.LEFT, padx=5)
        
        self.execute_btn = ttk.Button(control_frame, text="✅ Execute", command=self.execute_pick, state=tk.DISABLED)
        self.execute_btn.pack(side=tk.LEFT, padx=5)
        
        self.cancel_btn = ttk.Button(control_frame, text="❌ Cancel", command=self.cancel_pick, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Adaptive force indicator
        ttk.Label(control_frame, text="🎯 Adaptive Force:", font=("", 9, "bold")).pack(side=tk.LEFT, padx=(5, 2))
        self.adaptive_force_label = ttk.Label(control_frame, text="--", foreground="blue", font=("", 9))
        self.adaptive_force_label.pack(side=tk.LEFT)
        
        # Main content area
        content_frame = ttk.Frame(self.root)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Video feed
        video_frame = ttk.LabelFrame(content_frame, text="Live Camera Feed", padding="5")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()
        
        # Right: Info panel
        info_frame = ttk.Frame(content_frame, width=350)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Detection info
        det_frame = ttk.LabelFrame(info_frame, text="Detection Info", padding="10")
        det_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        self.det_text = tk.Text(det_frame, height=10, width=40, font=("Consolas", 9))
        self.det_text.pack()
        
        # Force threshold table
        force_frame = ttk.LabelFrame(info_frame, text="Force Thresholds by Object", padding="10")
        force_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        force_table_text = "Object          Threshold\n"
        force_table_text += "─" * 30 + "\n"
        for obj, thresh in FORCE_THRESHOLDS.items():
            force_table_text += f"{obj:15} {thresh:6.0f}\n"
        force_table_text += f"\nDefault:        {DEFAULT_FORCE_THRESHOLD:6.0f}"
        
        force_table = tk.Text(force_frame, height=6, width=40, font=("Consolas", 8))
        force_table.insert(1.0, force_table_text)
        force_table.config(state=tk.DISABLED)
        force_table.pack()
        
        # Staged pick info
        staged_frame = ttk.LabelFrame(info_frame, text="Staged Pick", padding="10")
        staged_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        self.staged_text = tk.Text(staged_frame, height=7, width=40, font=("Consolas", 9))
        self.staged_text.pack()
        
        # Status log
        log_frame = ttk.LabelFrame(info_frame, text="Status Log", padding="10")
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, width=40, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="⚪ Idle")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        
    def start_system(self):
        self.log("🚀 Starting system...")
        self.start_btn.config(state=tk.DISABLED)
        
        try:
            # Initialize robot
            self.log("Connecting to robot...")
            self.arm = XArmAPI(ROBOT_IP)
            time.sleep(0.5)
            self.arm.set_gripper_enable(True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            self.arm.reset(wait=True)
            self.arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
            self.arm.set_position(*OBS_POSE, wait=True)
            self.log("✅ Robot initialized")
            
            # Initialize camera
            self.log("Starting camera...")
            self.cam = DepthAiCamera(width=640, height=400, disable_rgb=False, align_to_rgb=True)
            K_rgb, K_depth = self.cam.get_intrinsics()
            self.K_use = K_rgb
            self.log("✅ Camera started")
            
            # Initialize YOLO
            self.log("Loading YOLO11 model...")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = YOLO("yolo11n.pt")
            self.model.to(device)
            self.log(f"✅ YOLO loaded on {device}")
            
            # Initialize force sensor
            self.log("Connecting force sensor...")
            self.ser = serial.Serial(FORCE_PORT, 9600, timeout=0.5)
            self.ser.reset_input_buffer()
            self.log(f"✅ Force sensor on {FORCE_PORT}")
            self.log("🎯 Adaptive force thresholds enabled")
            
            # Start detection loop
            self.running = True
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            
            self.status_var.set("🟢 Running - Adaptive Force Mode")
            self.stop_btn.config(state=tk.NORMAL)
            self.stage_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.log(f"❌ Error: {e}")
            self.status_var.set("⚪ Idle")
            self.start_btn.config(state=tk.NORMAL)
            
    def stop_system(self):
        self.running = False
        self.log("⏹️ Stopping...")
        
        try:
            if self.arm:
                self.arm.reset(wait=True)
            if self.ser:
                self.ser.close()
        except:
            pass
        
        self.status_var.set("⚪ Idle")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.stage_btn.config(state=tk.DISABLED)
        self.execute_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.DISABLED)
        self.staged_pick = None
        self.adaptive_force_label.config(text="--")
        
    def detection_loop(self):
        fruit_classes = ['sports ball', 'orange', 'apple']
        
        while self.running:
            try:
                color, depth = self.cam.get_images()
                if color is None:
                    continue
                
                # Run YOLO detection
                with torch.no_grad():
                    results = self.model(color, conf=0.35, iou=0.45, verbose=False)
                
                r = results[0]
                display = color.copy()
                best_detection = None
                
                if r.boxes is not None and len(r.boxes):
                    boxes = r.boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                    names = getattr(r, 'names', None) or getattr(self.model, 'names', None)
                    
                    det_rows = []
                    for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
                        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                        det_rows.append({
                            'xmin': float(x1), 'ymin': float(y1),
                            'xmax': float(x2), 'ymax': float(y2),
                            'confidence': float(c), 'name': label
                        })
                        
                        # Draw all detections
                        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(display, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                        cv2.putText(display, f"{label}:{c:.2f}", (x1i, max(20, y1i-8)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Find best fruit
                    df = pd.DataFrame(det_rows)
                    df_fruit = df[df['name'].isin(fruit_classes)]
                    
                    if not df_fruit.empty:
                        idx = df_fruit['confidence'].astype(float).idxmax()
                        best = df_fruit.loc[idx]
                        x1, y1, x2, y2 = map(int, [best['xmin'], best['ymin'], best['xmax'], best['ymax']])
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        
                        # Get depth
                        depth_m = None
                        if depth is not None and 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                            d = depth[cy, cx]
                            depth_m = float(d) if np.isfinite(d) else median_valid_depth(depth, cx, cy, k=3)
                        
                        if depth_m is not None and np.isfinite(depth_m):
                            x_cam, y_cam, z_cam = pixel_to_3d(cx, cy, depth_m, self.K_use)
                            xr, yr, zr = camera_to_robot(x_cam, y_cam, z_cam, OBS_POSE)
                            
                            # Get adaptive force threshold for this object
                            obj_name = best['name']
                            adaptive_threshold = get_force_threshold(obj_name)
                            
                            best_detection = {
                                'cx': cx, 'cy': cy, 'depth_m': depth_m,
                                'xr': xr, 'yr': yr, 'zr': clamp_to_table(zr - 10.0),
                                'name': obj_name, 'conf': float(best['confidence']),
                                'force_threshold': adaptive_threshold
                            }
                            
                            # Draw crosshair
                            cv2.drawMarker(display, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
                            
                            # Update adaptive force label
                            self.adaptive_force_label.config(text=f"{obj_name} → {adaptive_threshold:.0f}")
                            
                            # Update detection text
                            det_info = f"Detection: {obj_name}\n"
                            det_info += f"Confidence: {best_detection['conf']:.2f}\n"
                            det_info += f"Pixel: ({cx}, {cy})\n"
                            det_info += f"Depth: {depth_m:.3f} m\n\n"
                            det_info += f"Robot Coords:\n"
                            det_info += f"  X: {best_detection['xr']:.1f} mm\n"
                            det_info += f"  Y: {best_detection['yr']:.1f} mm\n"
                            det_info += f"  Z: {best_detection['zr']:.1f} mm\n\n"
                            det_info += f"🎯 Force Threshold:\n"
                            det_info += f"  {adaptive_threshold:.0f} (adaptive)"
                            
                            self.det_text.delete(1.0, tk.END)
                            self.det_text.insert(1.0, det_info)
                
                # Update video feed
                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(display_rgb)
                img = img.resize((640, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=photo)
                self.video_label.image = photo
                
                # Store latest detection for staging
                self.latest_detection = best_detection
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                self.log(f"⚠️ Detection error: {e}")
                time.sleep(0.1)
                
    def stage_pick(self):
        if hasattr(self, 'latest_detection') and self.latest_detection:
            self.staged_pick = self.latest_detection
            sp = self.staged_pick
            
            staged_info = f"Ready to pick:\n\n"
            staged_info += f"{sp['name']} ({sp['conf']:.2f})\n\n"
            staged_info += f"Target:\n"
            staged_info += f"  X: {sp['xr']:.1f} mm\n"
            staged_info += f"  Y: {sp['yr']:.1f} mm\n"
            staged_info += f"  Z: {sp['zr']:.1f} mm\n\n"
            staged_info += f"🎯 Force: {sp['force_threshold']:.0f}"
            
            self.staged_text.delete(1.0, tk.END)
            self.staged_text.insert(1.0, staged_info)
            
            self.execute_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.NORMAL)
            
            self.log(f"📌 Staged: {sp['name']} at ({sp['xr']:.1f}, {sp['yr']:.1f}, {sp['zr']:.1f}) | Force={sp['force_threshold']:.0f}")
        else:
            self.log("⚠️ No detection to stage")
            
    def cancel_pick(self):
        self.staged_pick = None
        self.staged_text.delete(1.0, tk.END)
        self.staged_text.insert(1.0, "No pick staged")
        self.execute_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.DISABLED)
        self.log("❌ Pick canceled")
        
    def execute_pick(self):
        if not self.staged_pick:
            return
        
        self.log(f"🤖 Executing pick & place with force={self.staged_pick['force_threshold']:.0f}...")
        self.execute_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.DISABLED)
        
        # Run in thread to not block GUI
        threading.Thread(target=self._execute_pick_thread, daemon=True).start()
        
    def _execute_pick_thread(self):
        try:
            sp = self.staged_pick
            x, y = sp['xr'], sp['yr']
            z_pick = clamp_motion(sp['zr'])
            z_hover = clamp_motion(z_pick + HOVER_DELTA_MM)
            z_lift = clamp_motion(z_hover + LIFT_DELTA_MM)
            adaptive_force = sp['force_threshold']
            
            # Pick sequence
            self.arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
            self.arm.set_position(x, y, OBS_POSE[2], *OBS_POSE[3:], wait=True)
            self.arm.set_position(x, y, z_hover, *OBS_POSE[3:], wait=True)
            self.arm.set_position(x, y, z_pick, *OBS_POSE[3:], wait=True)
            
            # Use adaptive force threshold
            smooth_close_with_force(self.arm, self.ser, adaptive_force)
            time.sleep(0.2)
            
            self.arm.set_position(x, y, z_lift, *OBS_POSE[3:], wait=True)
            
            # Place sequence
            self.arm.set_position(BIN_X, BIN_Y, clamp_motion(TRAVEL_Z), *OBS_POSE[3:], wait=True)
            self.arm.set_position(BIN_X, BIN_Y, clamp_motion(BIN_APPROACH_Z), *OBS_POSE[3:], wait=True)
            self.arm.set_position(BIN_X, BIN_Y, clamp_motion(BIN_Z), *OBS_POSE[3:], wait=True)
            
            self.arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
            time.sleep(0.2)
            
            self.arm.set_position(BIN_X, BIN_Y, clamp_motion(TRAVEL_Z), *OBS_POSE[3:], wait=True)
            self.arm.set_position(*OBS_POSE, wait=True)
            
            self.log(f"✅ Pick & place complete! (used force={adaptive_force:.0f})")
            self.staged_pick = None
            self.staged_text.delete(1.0, tk.END)
            self.staged_text.insert(1.0, "No pick staged")
            
        except Exception as e:
            self.log(f"❌ Error during pick: {e}")
            try:
                self.arm.set_position(*OBS_POSE, wait=True)
            except:
                pass

# ---------------- Main ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AdaptivePickerGUI(root)
    root.mainloop()
