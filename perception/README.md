# Perception Module

This module handles computer vision and object detection for the robotic fruit harvester. It uses YOLOv5 for object detection and provides coordinate transformation from camera pixels to robot base frame coordinates.

## Overview

The perception system works in several stages:
1. **Capture** image from camera
2. **Detect** objects using YOLOv5
3. **Transform** pixel coordinates to 3D robot coordinates
4. **Return** object pose for robot control

## Files

- `perception.py` - Main perception module with object detection and coordinate transformation
- `README.md` - This documentation file

## Core Functions

### `compute_object_pose_base(depth_frame=None, depth_scale=DEPTH_SCALE)`

**Main entry point** - Returns the 3D position of the detected object in robot base coordinates.

**Parameters:**
- `depth_frame` (optional): Depth image array for accurate Z-coordinate
- `depth_scale` (optional): Scale factor to convert depth units to meters

**Returns:**
- `(x, y, z)` tuple in meters, robot base frame coordinates

**Example:**
```python
from perception import compute_object_pose_base

# Basic usage (uses table height assumption)
x, y, z = compute_object_pose_base()

# With depth camera data
x, y, z = compute_object_pose_base(depth_frame=depth_image, depth_scale=0.001)
```

## Key Design Patterns

### Singleton Pattern for Model Loading

The `_get_model()` function implements a **singleton pattern** to efficiently manage the YOLOv5 model:

```python
_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        _MODEL = _MODEL.to(_DEVICE).eval()
    return _MODEL
```

**Why use singleton pattern here?**

1. **Performance**: YOLOv5 models are large (14-140MB) and loading takes 1-3 seconds
2. **Memory**: Avoid loading multiple copies of the same model
3. **GPU Efficiency**: Moving model to GPU once vs. repeated transfers
4. **Thread Safety**: Single model instance prevents conflicts

**Benefits:**
- ✅ First call: loads model (slow)
- ✅ Subsequent calls: instant return (fast)
- ✅ Automatic GPU/CPU selection
- ✅ Model stays in eval mode for inference

## Coordinate System Transformations

### Camera Calibration Matrix (K)

```python
K = np.array([[600.0, 0.0, 320.0],
              [0.0, 600.0, 240.0],
              [0.0,   0.0,   1.0]], dtype=float)
```

This is the **camera intrinsics matrix** containing:
- `fx, fy` (600.0): Focal lengths in pixels
- `cx, cy` (320.0, 240.0): Principal point (image center)

**Note:** These are placeholder values. Real calibration required for accuracy.

### Backprojection: Pixels → 3D Camera Frame

The `_backproject(u, v, Z)` function converts 2D pixel coordinates to 3D camera coordinates:

```python
def _backproject(u, v, Z):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    Xc = (u - cx) * Z / fx
    Yc = (v - cy) * Z / fy
    return np.array([Xc, Yc, Z, 1.0], dtype=float)
```

**Math explanation:**
- `Xc = (u - cx) * Z / fx` - Convert pixel u to camera X using focal length
- `Yc = (v - cy) * Z / fy` - Convert pixel v to camera Y using focal length
- `Z` - Depth (from depth camera or table assumption)
- Returns homogeneous coordinates `[Xc, Yc, Z, 1]`

### Coordinate Frame Transformation

The `_cam_to_base(Pc_h)` function transforms from camera frame to robot base frame:

```python
T_base_cam = np.eye(4, dtype=float)  # 4x4 transformation matrix

def _cam_to_base(Pc_h):
    return (T_base_cam @ Pc_h)[:3]
```

**`T_base_cam` is a 4×4 transformation matrix:**
```
[R11 R12 R13 tx]   [Rotation    Translation]
[R21 R22 R23 ty] = [  3x3         3x1      ]
[R31 R32 R33 tz]   [            
[ 0   0   0   1]   [ 0  0  0        1      ]
```

**Currently identity matrix** - means camera and robot frames are aligned. **Needs calibration for real setup.**

## Detection Pipeline

### 1. Image Capture
```python
def _get_cv_frame(index=0, w=640, h=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, bgr = cap.read()
    cap.release()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
```

- Opens camera with specified resolution
- `cv2.CAP_DSHOW` optimized for Windows
- Converts BGR→RGB for YOLOv5
- **One-shot capture** - opens, captures, closes camera

### 2. YOLOv5 Inference
```python
with torch.no_grad():
    res = model(rgb, size=640)
    preds = res.xyxy[0].cpu().numpy()
```

- `torch.no_grad()` - disables gradients for faster inference
- `size=640` - input image size for YOLOv5
- `res.xyxy[0]` - bounding box format: [x1, y1, x2, y2, confidence, class_id]
- `.cpu().numpy()` - moves results to CPU as numpy array

### 3. Best Detection Selection
```python
x1,y1,x2,y2,conf,cls_id = max(preds, key=lambda p: p[4])
u = int(0.5*(x1+x2)); v = int(0.5*(y1+y2))
```

- Finds detection with highest confidence (`p[4]`)
- Calculates bounding box center as target point
- `(u, v)` = pixel coordinates of object center

## Depth Handling

### Fallback Strategy
```python
Zm = Z_TABLE_M  # Default table height
if depth_frame is not None:
    z_raw = float(depth_frame[v, u])
    if z_raw > 0.0 and np.isfinite(z_raw):
        Zm = z_raw * depth_scale
```

**Two modes:**
1. **Table assumption**: Uses fixed height `Z_TABLE_M = 0.10m`
2. **Depth camera**: Uses actual depth measurement

**Depth validation:**
- Checks `z_raw > 0.0` (valid depth)
- Checks `np.isfinite()` (not NaN/infinite)
- Applies `depth_scale` conversion factor

## Camera Integration Stubs

The module includes ready-to-use templates for popular depth cameras:

### Intel RealSense
```python
# def get_realsense_frames():
#     # - Pipeline configuration for color + depth
#     # - Automatic depth scale detection
#     # - Depth-to-color alignment
#     # - Returns: rgb, depth_array, depth_scale
```

### OAK-D (DepthAI)
```python
# def get_oakd_frames():
#     # - StereoDepth pipeline
#     # - RGB-depth alignment
#     # - XLinkOut queues for data
```

### ZED Camera
```python
# def get_zed_frames():
#     # - grab() and retrieve_measure() pattern
#     # - Built-in depth alignment
#     # - Direct numpy conversion
```

## Setup and Dependencies

### Required Packages
```bash
pip install torch torchvision opencv-python numpy
```

### Optional (for depth cameras)
```bash
# Intel RealSense
pip install pyrealsense2

# OAK-D
pip install depthai

# ZED Camera  
pip install pyzed
```

## Calibration Requirements

### Camera Intrinsics Calibration

**Current placeholder values need replacement:**
```python
K = np.array([[600.0, 0.0, 320.0],    # fx, 0, cx
              [0.0, 600.0, 240.0],    # 0, fy, cy  
              [0.0,   0.0,   1.0]])   # 0, 0, 1
```

**How to calibrate:**
1. Use OpenCV's `cv2.calibrateCamera()`
2. Print checkerboard pattern
3. Capture 20+ images from different angles
4. Extract fx, fy, cx, cy values

### Hand-Eye Calibration

**Current identity matrix needs replacement:**
```python
T_base_cam = np.eye(4, dtype=float)  # Camera = Robot base (wrong!)
```

**How to calibrate:**
1. Mount camera on robot or fixed position
2. Move robot to known positions
3. Observe same object from different robot poses
4. Solve for transformation matrix using least squares

## Usage Examples

### Basic Object Detection
```python
from perception import compute_object_pose_base

# Detect any object, use table height
x, y, z = compute_object_pose_base()
print(f"Object at: {x:.3f}, {y:.3f}, {z:.3f}")
```

### With Depth Camera
```python
import pyrealsense2 as rs
from perception import compute_object_pose_base

# Get frames from RealSense
pipeline = rs.pipeline()
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
depth_array = np.asanyarray(depth_frame.get_data())
depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

# Detect with real depth
x, y, z = compute_object_pose_base(depth_array, depth_scale)
```

### Integration with Main Robot Code
```python
# In main.py, uncomment these lines:
try:
    from perception import compute_object_pose_base  
    HAS_PERCEPTION = True
except Exception:
    HAS_PERCEPTION = False

# Later in main():
if HAS_PERCEPTION:
    try:
        OBJECT_X, OBJECT_Y, OBJECT_Z = compute_object_pose_base()
        print(f"Camera detected object at: x={OBJECT_X:.3f}, y={OBJECT_Y:.3f}, z={OBJECT_Z:.3f}")
    except Exception as e:
        print("Perception failed; using predefined pose:", e)
```

## Performance Considerations

### Model Loading
- **First call**: ~2-3 seconds (downloads + loads model)
- **Subsequent calls**: ~50-200ms (inference only)
- **GPU acceleration**: Automatic if CUDA available

### Camera Capture
- **USB cameras**: ~30-100ms per frame
- **Depth cameras**: ~50-150ms per frame
- **Resolution impact**: 640×480 optimal for speed/accuracy balance

### Memory Usage
- **YOLOv5s**: ~14MB model + ~100MB GPU memory
- **Image processing**: ~5MB per 640×480 RGB frame
- **Total**: ~120MB typical usage

## Troubleshooting

### Common Issues

**"No camera frame" error:**
- Check camera is connected and not used by other apps
- Try different camera index: `_get_cv_frame(index=1)`
- Verify camera permissions

**"No detections" frequently:**
- Check lighting conditions
- Verify object is in camera view
- Lower confidence threshold in YOLOv5
- Try different YOLOv5 model size (yolov5m, yolov5l)

**Coordinates seem wrong:**
- Calibrate camera intrinsics matrix K
- Perform hand-eye calibration for T_base_cam
- Check depth camera alignment

**Slow performance:**
- Install CUDA for GPU acceleration
- Reduce camera resolution
- Use smaller YOLOv5 model (yolov5n)

### Debug Mode

Add debug prints to understand detection:
```python
print(f"Detections found: {len(preds)}")
print(f"Best detection: conf={conf:.2f}, class={int(cls_id)}")
print(f"Pixel coords: u={u}, v={v}")
print(f"Camera coords: {Pc}")
print(f"Robot coords: {xb:.3f}, {yb:.3f}, {zb:.3f}")
```

## Future Enhancements

- [ ] **Multiple object tracking** - Track objects across frames
- [ ] **Object classification** - Filter by specific fruit types  
- [ ] **Pose estimation** - 6DOF object orientation
- [ ] **Occlusion handling** - Detect partially hidden objects
- [ ] **Active vision** - Move camera to get better view
- [ ] **Uncertainty estimation** - Confidence bounds on coordinates
