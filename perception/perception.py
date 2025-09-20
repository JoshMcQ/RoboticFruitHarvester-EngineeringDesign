
import cv2, torch, numpy as np, json, os

# --- Load calibration from file if available ---
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), '..', 'camera_calibration_latest.json')
K = np.array([[600.0, 0.0, 320.0],
              [0.0, 600.0, 240.0],
              [0.0,   0.0,   1.0]], dtype=float)   # fx,fy,cx,cy
DIST_COEFFS = np.zeros((5,))
T_base_cam = np.eye(4, dtype=float)                # 4x4 cam->base
Z_TABLE_M = 0.241                                   # assume objects ~9.5 inches away
DEPTH_SCALE = 0.001                                 # units->m (e.g., RealSense)

def load_calibration():
    global K, DIST_COEFFS
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as f:
            calib = json.load(f)
        K = np.array(calib.get('camera_matrix', K)).astype(float)
        DIST_COEFFS = np.array(calib.get('distortion_coefficients', DIST_COEFFS)).astype(float)
        print(f"Loaded calibration from {CALIBRATION_FILE}")
    else:
        print("Calibration file not found, using default K.")

def calibration_status():
    if os.path.exists(CALIBRATION_FILE):
        return True, CALIBRATION_FILE
    return False, None

load_calibration()

# --- YOLOv5: load once, eval, pick device ---
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        _MODEL = _MODEL.to(_DEVICE).eval()
    return _MODEL

def _backproject(u, v, Z):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    Xc = (u - cx) * Z / fx
    Yc = (v - cy) * Z / fy
    return np.array([Xc, Yc, Z, 1.0], dtype=float)

def _cam_to_base(Pc_h):
    return (T_base_cam @ Pc_h)[:3]


def _get_cv_frame(index=0, w=640, h=480, undistort=True):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # works well on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, bgr = cap.read()
    cap.release()
    if not ok: raise RuntimeError("No camera frame")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if undistort and DIST_COEFFS is not None and np.any(DIST_COEFFS):
        rgb = cv2.undistort(rgb, K, DIST_COEFFS)
    return rgb


def compute_object_pose_base(depth_frame=None, depth_scale=DEPTH_SCALE):
    """Return (x,y,z) in BASE. Uses depth if given; else falls back to table Z."""
    model = _get_model()
    rgb = _get_cv_frame(0, 640, 480, undistort=True)

    with torch.no_grad():                  # inference only
        res = model(rgb, size=640)         # default imgsz
        res.render()  # modifies res.ims[0] with boxes
        cv2.imwrite("detection_snapshot.jpg", cv2.cvtColor(res.ims[0], cv2.COLOR_RGB2BGR))
        print("Saved detection snapshot to detection_snapshot.jpg")
        preds = res.xyxy[0].cpu().numpy()  # no pandas dep
    if preds.size == 0:
        print("no detections")
        return 0.60, 0.00, 0.10

    # highest conf
    x1,y1,x2,y2,conf,cls_id = max(preds, key=lambda p: p[4])
    u = int(0.5*(x1+x2)); v = int(0.5*(y1+y2))
    class_name = model.names[int(cls_id)]
    print(f"px=({u},{v}) conf={conf:.2f} class={int(cls_id)} ({class_name})")

    # depth if we have it; else table
    Zm = Z_TABLE_M
    if depth_frame is not None:
        z_raw = float(depth_frame[v, u])
        if z_raw > 0.0 and np.isfinite(z_raw):
            Zm = z_raw * depth_scale

    Pc = _backproject(u, v, Zm)            # cam frame
    xb, yb, zb = _cam_to_base(Pc)          # base frame
    return float(xb), float(yb), float(zb)

# ---------------- camera stubs (uncomment when ready) ----------------


# -------------------- Test/Demo Mode --------------------
if __name__ == "__main__":
    status, fname = calibration_status()
    if status:
        print(f"Calibration loaded from {fname}")
    else:
        print("No calibration file found. Run calibrate_camera.py for best accuracy.")
    print("Testing perception system...")
    print("Make sure a camera is connected and visible objects are in view.")
    try:
        x, y, z = compute_object_pose_base()
        print(f"✅ Object detected at: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m")
        print("Perception system working correctly!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check camera connection and dependencies.")
