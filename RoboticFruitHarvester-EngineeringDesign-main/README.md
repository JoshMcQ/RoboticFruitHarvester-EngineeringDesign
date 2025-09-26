
# Robotic Fruit Harvester – Engineering Design

This project contains scripts and utilities for controlling a Kinova robotic arm for automated fruit harvesting. The system includes gripper commands, Cartesian waypoint trajectory execution, and a complete pick-and-place workflow.

## Project Structure

- `main.py` – **Main pick-and-place workflow** that orchestrates the complete fruit harvesting sequence
- `Kinova/gripper_command.py` – Example for sending gripper commands to the Kinova arm
- `Kinova/send_cartesian_waypoint_trajectory.py` – Example for sending Cartesian waypoint trajectories
- `Kinova/utilities.py` – Utilities for device connection and argument parsing

## How the System Works

### Main Workflow (`main.py`)

The main script implements a complete automated pick sequence:

1. **Connection & Initialization**
   - Connects to the Kinova arm using TCP connection
   - Parses command-line arguments for robot IP, username, and password

2. **Home Position**
   - Moves the robot arm to a safe "Home" position (if configured on the robot)
   - Continues gracefully if Home position is not available

3. **Approach & Descent**
   - Moves above the target object at a safe approach height
   - Descends to the grasp position using predefined coordinates

4. **Grasp Operation**
   - Executes gripper closing sequence to grasp the object
   - Uses position and speed control for reliable grasping

5. **Lift & Complete**
   - Lifts the grasped object to a safe height
   - Reports successful completion of the pick sequence

### Key Components

#### Configuration Parameters
```python
OBJECT_X, OBJECT_Y, OBJECT_Z = 0.60, 0.00, 0.10  # Object position (meters)
TOOL_TX, TOOL_TY, TOOL_TZ = 90.0, 0.0, 90.0      # Tool orientation (degrees)
APPROACH_H, LIFT_H = 0.12, 0.15                   # Safety heights (meters)
```

#### Waypoint Execution
- Reuses existing Kinova example functions for robust trajectory execution
- Validates waypoints before execution to prevent collisions
- Implements timeout handling and error recovery

#### Future Expansion
- Commented sections ready for camera-based perception system
- Placeholder for force feedback integration
- Modular design for easy feature additions


## Setup

### Prerequisites
- **Python 3.7+** with pip installed
- **Kinova Gen3 robot** with network connectivity
- **Windows, Linux, or macOS** development environment

### 1. Download the Kortex Python API Wheel (.whl)

You must first download the Kortex Python API `.whl` file from the Kinova Artifactory:

- [Kinova Artifactory – kortex_api](https://artifactory.kinovaapps.com/ui/native/generic-public/kortex/API/2.6.0/)

Save the `.whl` file to your project directory.

### 2. Clone and Setup Environment

Clone this repository and navigate to the project directory:

```sh
git clone https://github.com/JoshMcQ/RoboticFruitHarvester-EngineeringDesign.git
cd RoboticFruitHarvester-EngineeringDesign
```

### 3. (Recommended) Set up and activate a virtual environment

On **Windows**:

```sh
python -m venv venv
.\venv\Scripts\activate
```

On **Linux/Mac**:

```sh
python3 -m venv venv
source venv/bin/activate
```

### 4. Install the Kortex Python API and dependencies

On **Windows**:

```sh
python -m pip install <whl relative fullpath name>.whl
```

On **Linux**:

```sh
python3 -m pip install <whl relative fullpath name>.whl
# Note: root privilege is usually required to install a new module under Linux.
```

Replace `<whl relative fullpath name>.whl` with the actual filename you downloaded.

## Usage

### Running the Main Pick Sequence

To run the complete automated fruit harvesting workflow:

```powershell
python main.py --ip <ROBOT_IP> -u <USERNAME> -p <PASSWORD>
```

**Default values:**
- IP: `192.168.1.10`
- Username: `admin`  
- Password: `admin`

**Example:**
```powershell
python main.py --ip 192.168.1.10 -u admin -p admin
```

### Running Individual Components

Each script in the `Kinova/` directory can be run independently for testing:

**Gripper control:**
```powershell
python Kinova/gripper_command.py --ip <ROBOT_IP> -u <USERNAME> -p <PASSWORD>
```

**Waypoint trajectory:**
```powershell
python Kinova/send_cartesian_waypoint_trajectory.py --ip <ROBOT_IP> -u <USERNAME> -p <PASSWORD>
```

### Customizing the Pick Sequence

Edit the configuration parameters in `main.py`:

```python
# Object position in base frame (meters)
OBJECT_X, OBJECT_Y, OBJECT_Z = 0.60, 0.00, 0.10

# Tool orientation (degrees)  
TOOL_TX, TOOL_TY, TOOL_TZ = 90.0, 0.0, 90.0

# Safety parameters (meters)
APPROACH_H = 0.12  # Height above object for approach
LIFT_H = 0.15      # Height to lift after grasping
```

## Safety Considerations

⚠️ **IMPORTANT SAFETY GUIDELINES:**

1. **Clear Workspace**: Ensure the robot workspace is clear of people and obstacles
2. **Emergency Stop**: Keep the emergency stop button accessible at all times
3. **Test Positions**: Verify all target positions are reachable and collision-free
4. **Start Slow**: Begin with conservative speeds and small movements
5. **Supervision**: Always supervise the robot during operation

## Troubleshooting

### Common Issues

**Connection Failed:**
- Verify robot IP address and network connectivity
- Check username/password credentials
- Ensure robot is powered on and in proper mode

**Import Errors:**
- Confirm Kortex API is installed: `python -c "import kortex_api"`
- Check Python path and virtual environment activation
- Verify all files are in correct directory structure

**Movement Errors:**
- Check if "Home" position is configured on the robot
- Verify target coordinates are within robot workspace
- Ensure robot is not in protective stop mode

**Waypoint Validation Failed:**
- Review target positions for reachability
- Check for potential collisions with robot base or environment
- Adjust approach height or tool orientation

### Getting Help

- Check robot status lights and display messages
- Review terminal output for specific error messages
- Consult Kinova documentation for robot-specific issues

## Development

### Code Architecture

The project follows a modular architecture:

- **`main.py`**: High-level orchestration and workflow logic
- **`Kinova/utilities.py`**: Low-level connection and session management  
- **`Kinova/send_cartesian_waypoint_trajectory.py`**: Cartesian movement primitives
- **`Kinova/gripper_command.py`**: Gripper control primitives

### Adding New Features

**Perception System:**
Uncomment and implement the perception hooks in `main.py`:
```python
# Future: perception hook for camera-based object detection
try:
    from perception import compute_object_pose_base  
    HAS_PERCEPTION = True
except Exception:
    HAS_PERCEPTION = False
```

**Force Feedback:**
The gripper system can be enhanced with force sensors:
```python
# Future: add force feedback threshold in step 4
```

**Advanced Trajectories:**
Add more complex movement patterns by extending `_execute_waypoints_via_examples()`.

### Testing

**Simulation Mode:**
For development without hardware, consider implementing a simulation backend that mimics the Kortex API responses.

## Future Enhancements

### Planned Features
- [ ] Camera-based object detection and pose estimation
- [ ] Force feedback for adaptive grasping
- [ ] Multi-object harvesting sequences  
- [ ] Collision avoidance and path planning
- [ ] Real-time monitoring and logging
- [ ] Web-based control interface

### Research Areas
- Machine learning for fruit recognition
- Adaptive grasping strategies
- Optimal harvest path planning
- Integration with mobile platforms

## License

This project includes code and examples from Kinova Inc. and is subject to the BSD 3-Clause license. See individual files for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## Contact

For questions about this project, please open an issue on GitHub or contact the development team.