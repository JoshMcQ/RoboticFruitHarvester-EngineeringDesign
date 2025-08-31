
# Robotic Fruit Harvester – Engineering Design

This project contains scripts and utilities for controlling a Kinova robotic arm, including gripper commands and Cartesian waypoint trajectory execution. It is intended for use in the development and testing of a robotic fruit harvesting system.

## Project Structure

- `Kinova/gripper_command.py` – Example for sending gripper commands to the Kinova arm.
- `Kinova/send_cartesian_waypoint_trajectory.py` – Example for sending Cartesian waypoint trajectories.
- `Kinova/utilities.py` – Utilities for device connection and argument parsing.


## Setup

### 1. Download the Kortex Python API Wheel (.whl)

You must first download the Kortex Python API `.whl` file from the Kinova Artifactory:

- [Kinova Artifactory – kortex_api](https://artifactory.kinovaapps.com/ui/native/generic-public/kortex/API/2.6.0/)

Save the `.whl` file to your project directory.

### 2. Clone this repository and navigate to the project directory

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

Each script in the `Kinova/` directory can be run directly. For example:

```sh
python Kinova/gripper_command.py --ip <ROBOT_IP> -u <USERNAME> -p <PASSWORD>
```

Replace `<ROBOT_IP>`, `<USERNAME>`, and `<PASSWORD>` with your robot's connection details. Default values are set in the scripts.

## License

This project includes code and examples from Kinova Inc. and is subject to the BSD 3-Clause license. See individual files for details.