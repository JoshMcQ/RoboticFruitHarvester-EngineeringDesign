#!/usr/bin/env python3
"""
streamlit_pick_place_live.py
Streamlit UI with live camera feed for YOLO11 pick & place with force sensing.

Usage:
    streamlit run streamlit_pick_place_live.py
"""

import time, warnings, re, threading, queue
import numpy as np
import torch, cv2, serial
from ultralytics import YOLO
import pandas as pd
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xarm.wrapper import XArmAPI
from camera.depthai_camera import DepthAiCamera

warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

# Page config
st.set_page_config(
    page_title="YOLO Pick & Place Live",
    page_icon="🤖",
    layout="wide"
)

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

# Hand-eye calibration params
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

# ---------------- Streamlit UI ----------------
st.title("🤖 YOLO11 Pick & Place with Live Camera Feed")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    force_threshold = st.number_input(
        "Force Threshold",
        min_value=10.0,
        max_value=500.0,
        value=100.0,
        step=10.0
    )
    
    force_port = st.text_input("Force Sensor Port", value="COM5")
    
    with st.expander("📍 Fixed Parameters"):
        st.code("""
Bin: X=313.4, Y=-353.5, Z=-54.6
Approach Z: 156.1
Travel Z: 380.0
Camera: Align to RGB
        """)
    
    st.markdown("---")
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶️ Start", type="primary")
    with col2:
        stop_btn = st.button("⏹️ Stop")
    
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'staged_pick' not in st.session_state:
        st.session_state.staged_pick = None
    
    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
        st.session_state.staged_pick = None
    
    st.markdown("---")
    
    # Pick controls (only when running)
    if st.session_state.running:
        st.subheader("🎯 Pick Controls")
        stage_pick_btn = st.button("📌 Stage Pick")
        execute_pick_btn = st.button("✅ Execute Pick", 
                                     disabled=st.session_state.staged_pick is None,
                                     type="primary")
        cancel_pick_btn = st.button("❌ Cancel", 
                                    disabled=st.session_state.staged_pick is None)
        
        if stage_pick_btn:
            st.session_state.stage_pick_trigger = True
        if execute_pick_btn:
            st.session_state.execute_pick_trigger = True
        if cancel_pick_btn:
            st.session_state.staged_pick = None
            st.success("Pick canceled")

# Main content area
col_video, col_info = st.columns([2, 1])

with col_video:
    st.subheader("📹 Live Camera Feed")
    video_placeholder = st.empty()

with col_info:
    st.subheader("📊 Detection Info")
    info_placeholder = st.empty()
    
    st.subheader("🎯 Staged Pick")
    staged_placeholder = st.empty()
    
    st.subheader("📝 Status Log")
    log_placeholder = st.empty()

# Status indicator
status_placeholder = st.empty()

# Initialize hardware when starting
if st.session_state.running:
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        with st.spinner("Initializing hardware..."):
            try:
                # Robot
                st.session_state.arm = XArmAPI(ROBOT_IP)
                time.sleep(0.5)
                st.session_state.arm.set_gripper_enable(True)
                st.session_state.arm.set_mode(0)
                st.session_state.arm.set_state(0)
                st.session_state.arm.reset(wait=True)
                st.session_state.arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
                st.session_state.arm.set_position(*OBS_POSE, wait=True)
                
                # Camera
                st.session_state.cam = DepthAiCamera(width=640, height=400, disable_rgb=False, align_to_rgb=True)
                K_rgb, K_depth = st.session_state.cam.get_intrinsics()
                st.session_state.K_use = K_rgb
                
                # YOLO
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                st.session_state.model = YOLO("yolo11n.pt")
                st.session_state.model.to(device)
                
                # Force sensor
                st.session_state.ser = serial.Serial(force_port, 9600, timeout=0.5)
                st.session_state.ser.reset_input_buffer()
                
                st.session_state.initialized = True
                st.session_state.log = ["✅ Hardware initialized"]
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {e}")
                st.session_state.running = False
                st.session_state.initialized = False
    
    # Main detection loop
    if st.session_state.get('initialized', False):
        try:
            arm = st.session_state.arm
            cam = st.session_state.cam
            model = st.session_state.model
            ser = st.session_state.ser
            K_use = st.session_state.K_use
            
            # Get camera frame
            color, depth = cam.get_images()
            
            if color is not None:
                # Run YOLO detection
                with torch.no_grad():
                    results = model(color, conf=0.35, iou=0.45, verbose=False)
                
                r = results[0]
                det_rows = []
                display = color.copy()
                
                if r.boxes is not None and len(r.boxes):
                    boxes = r.boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                    names = getattr(r, 'names', None) or getattr(model, 'names', None)
                    
                    for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
                        label = str(cls_id)
                        if isinstance(names, dict) and cls_id in names:
                            label = names[cls_id]
                        
                        det_rows.append({
                            'xmin': float(x1), 'ymin': float(y1),
                            'xmax': float(x2), 'ymax': float(y2),
                            'confidence': float(c), 'name': label
                        })
                        
                        # Draw detection
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display, f"{label}:{c:.2f}", (x1, max(20, y1-8)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                df_all = pd.DataFrame(det_rows)
                
                # Filter for fruit classes
                fruit_classes = ['sports ball', 'orange', 'apple']
                if not df_all.empty:
                    df = df_all[df_all['name'].isin(fruit_classes)].copy()
                else:
                    df = pd.DataFrame()
                
                best_detection = None
                if not df.empty:
                    idx = df['confidence'].astype(float).idxmax()
                    best = df.loc[idx]
                    x1, y1, x2, y2 = map(int, [best['xmin'], best['ymin'], best['xmax'], best['ymax']])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # Get depth
                    depth_m = None
                    if depth is not None and 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                        d = depth[cy, cx]
                        depth_m = float(d) if np.isfinite(d) else median_valid_depth(depth, cx, cy, k=3)
                    
                    if depth_m is not None and np.isfinite(depth_m):
                        x_cam, y_cam, z_cam = pixel_to_3d(cx, cy, depth_m, K_use)
                        xr, yr, zr = camera_to_robot(x_cam, y_cam, z_cam, OBS_POSE)
                        
                        best_detection = {
                            'cx': cx, 'cy': cy, 'depth_m': depth_m,
                            'xr': xr, 'yr': yr, 'zr': clamp_to_table(zr - 10.0),
                            'name': best['name'], 'conf': float(best['confidence'])
                        }
                        
                        # Draw crosshair on best
                        cv2.drawMarker(display, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
                
                # Display frame
                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_rgb, channels="RGB", use_container_width=True)
                
                # Display detection info
                if best_detection:
                    info_placeholder.markdown(f"""
**Detection:** {best_detection['name']}  
**Confidence:** {best_detection['conf']:.2f}  
**Pixel:** ({best_detection['cx']}, {best_detection['cy']})  
**Depth:** {best_detection['depth_m']:.3f} m  
**Robot XYZ:**  
- X: {best_detection['xr']:.1f} mm  
- Y: {best_detection['yr']:.1f} mm  
- Z: {best_detection['zr']:.1f} mm  
                    """)
                else:
                    info_placeholder.info("No fruit detected")
                
                # Handle stage pick trigger
                if st.session_state.get('stage_pick_trigger', False):
                    st.session_state.stage_pick_trigger = False
                    if best_detection:
                        st.session_state.staged_pick = best_detection
                        st.session_state.log.append(f"📌 Staged: {best_detection['name']} at ({best_detection['xr']:.1f}, {best_detection['yr']:.1f}, {best_detection['zr']:.1f})")
                
                # Display staged pick
                if st.session_state.staged_pick:
                    sp = st.session_state.staged_pick
                    staged_placeholder.markdown(f"""
**Ready to pick:**  
{sp['name']} (conf {sp['conf']:.2f})  
**Target:** X={sp['xr']:.1f}, Y={sp['yr']:.1f}, Z={sp['zr']:.1f} mm  
                    """)
                else:
                    staged_placeholder.info("No pick staged")
                
                # Handle execute pick trigger
                if st.session_state.get('execute_pick_trigger', False):
                    st.session_state.execute_pick_trigger = False
                    if st.session_state.staged_pick:
                        sp = st.session_state.staged_pick
                        st.session_state.log.append(f"🤖 Executing pick...")
                        
                        try:
                            x, y = sp['xr'], sp['yr']
                            z_pick = clamp_motion(sp['zr'])
                            z_hover = clamp_motion(z_pick + HOVER_DELTA_MM)
                            z_lift = clamp_motion(z_hover + LIFT_DELTA_MM)
                            
                            # Pick sequence
                            arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
                            arm.set_position(x, y, OBS_POSE[2], *OBS_POSE[3:], wait=True)
                            arm.set_position(x, y, z_hover, *OBS_POSE[3:], wait=True)
                            arm.set_position(x, y, z_pick, *OBS_POSE[3:], wait=True)
                            smooth_close_with_force(arm, ser, force_threshold)
                            arm.set_position(x, y, z_lift, *OBS_POSE[3:], wait=True)
                            
                            # Place sequence
                            arm.set_position(313.4, -353.5, clamp_motion(380.0), *OBS_POSE[3:], wait=True)
                            arm.set_position(313.4, -353.5, clamp_motion(156.1), *OBS_POSE[3:], wait=True)
                            arm.set_position(313.4, -353.5, clamp_motion(-54.6), *OBS_POSE[3:], wait=True)
                            arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
                            time.sleep(0.2)
                            arm.set_position(313.4, -353.5, clamp_motion(380.0), *OBS_POSE[3:], wait=True)
                            arm.set_position(*OBS_POSE, wait=True)
                            
                            st.session_state.log.append(f"✅ Pick & place complete!")
                            st.session_state.staged_pick = None
                            
                        except Exception as e:
                            st.session_state.log.append(f"❌ Error: {e}")
                            try:
                                arm.set_position(*OBS_POSE, wait=True)
                            except:
                                pass
                
                # Display log
                log_placeholder.text_area("Status Log", value="\n".join(st.session_state.log[-10:]), height=200, label_visibility="collapsed")
                
                status_placeholder.success("🟢 Running")
                
        except Exception as e:
            status_placeholder.error(f"❌ Error: {e}")
            st.session_state.running = False

else:
    video_placeholder.info("Click ▶️ Start to begin")
    status_placeholder.info("⚪ Idle")
    if st.session_state.get('initialized', False):
        # Cleanup
        try:
            st.session_state.arm.reset(wait=True)
            st.session_state.ser.close()
        except:
            pass
        st.session_state.initialized = False

# Auto-refresh when running (with delay to reduce flicker)
if st.session_state.running:
    time.sleep(0.1)  # Slightly longer delay
    st.rerun()
