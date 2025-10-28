# YOLO Fruit Grasping with xArm

Automated fruit detection and grasping using YOLOv5 + OAK-D camera mounted on the xArm gripper.

## Setup

**Hardware:**
- xArm robot arm
- OAK-D camera mounted on gripper (eye-in-hand)
- Force sensor on gripper (COM5)

**Software:**
```powershell
cd ufactory_vision\ggcnn_grasping_demo
pip install pyserial  # If not already installed
```

## Usage

```powershell
python yolo_grasp.py <ROBOT_IP>
```

Example:
```powershell
python yolo_grasp.py 192.168.1.221
```

With different model:
```powershell
python yolo_grasp.py 192.168.1.221 --model yolov5l
```

## Controls

- **`g`** - Grasp the currently selected (yellow) fruit
- **`s`** - Select next fruit
- **`q`** - Quit

## How It Works

1. **Detection Phase**
   - Robot moves to scan position (300, 0, 490mm)
   - Camera (mounted on gripper) looks down at workspace
   - YOLOv5 detects fruits in real-time
   - Fruits are numbered and color-coded

2. **Selection**
   - Press `s` to cycle through detected fruits
   - Selected fruit highlighted in yellow
   - Others shown in green

3. **Grasping Sequence** (when you press `g`)
   - Converts pixel + depth → 3D camera coords
   - Transforms camera coords → robot base coords (using hand-eye calibration)
   - Checks workspace bounds (must be within safe range)
   - Executes grasp: approach → lower → close with force feedback → lift → place → return

## Configuration

Key parameters in `yolo_grasp.py`:

### Safety Limits
```python
GRASPING_MIN_Z = 76  # Floor height (77mm with 1mm safety margin)
GRASPING_RANGE = [180, 600, -200, 200]  # [x_min, x_max, y_min, y_max]
```

### Robot Positions
```python
DETECT_XYZ = [300, 0, 490]  # Scanning position
RELEASE_XYZ = [400, 400, 360]  # Where to place fruits
LIFT_OFFSET_Z = 100  # Lift height after grasp
```

### Hand-Eye Calibration
```python
EULER_EEF_TO_COLOR_OPT = [0.0703, 0.0023, 0.0195, 0, 0, 1.579]  # [x, y, z, r, p, y] m/rad
GRIPPER_Z_MM = 70  # Distance from flange to gripper contact
```

### Force Sensor
```python
FORCE_PORT = "COM5"
FORCE_THRESHOLD = 100.0  # Adjust based on your sensor
```

## Calibration Notes

The camera-to-gripper transform (`EULER_EEF_TO_COLOR_OPT`) is pre-calibrated from the demo code. If your camera mount is different, you'll need to recalibrate:

1. Place a known object at a measured position
2. Run detection and compare detected vs actual position
3. Adjust `EULER_EEF_TO_COLOR_OPT` values
4. Repeat until positions match

## Safety

- ✅ **Floor protection**: Won't move below 76mm (77mm floor minus 1mm margin)
- ✅ **Workspace limits**: Checks X/Y bounds before every grasp
- ✅ **Force feedback**: Stops gripper close when contact detected
- ✅ **Emergency handling**: Opens gripper and returns to safe position on error

## Troubleshooting

**"No valid depth at fruit location"**
- Fruit may be too far/close for stereo depth
- Try different lighting or fruit position

**"Target outside workspace range"**
- Fruit is beyond robot reach
- Adjust `GRASPING_RANGE` or reposition fruit

**"Cannot grasp without force sensor"**
- Check `FORCE_PORT` setting
- Verify sensor is connected and powered

**Coordinates seem off**
- Hand-eye calibration may need adjustment
- Check `EULER_EEF_TO_COLOR_OPT` values
- Verify `GRIPPER_Z_MM` matches your gripper

## Comparison with yolo.py

| Feature | yolo.py | yolo_grasp.py |
|---------|---------|---------------|
| Purpose | Detection only | Detection + grasping |
| Camera | Any mount | Must be on gripper |
| Robot | Not required | xArm required |
| Output | Visual + logs | Physical pick & place |
| Controls | View only | Interactive grasp |

Use `yolo.py` for testing detection, `yolo_grasp.py` for full automation.
