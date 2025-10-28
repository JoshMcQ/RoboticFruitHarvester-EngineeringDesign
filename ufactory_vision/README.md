# ufactory_vision

[中文版说明 (Chinese Version)](./README_ZH.md)

## Project Overview

`ufactory_vision` is a vision-based grasping demo project based on UFACTORY robot arms. Users can quickly implement vision-based object detection and grasping with this project.

[![Watch the video](https://img.youtube.com/vi/ijnuqsNcfUY/0.jpg)](https://www.youtube.com/watch?v=ijnuqsNcfUY)

## Hardware Requirements

### Hardware Configuration for Example Scripts

| Robot Arm Model             | Camera Model                                      | End Effector          |
| --------------------------- | ------------------------------------------------- | --------------------- |
| xArm 5/6/7 or 850           | Intel Realsense D435 / Luxonis OAK-D-Pro-PoE      | UFACTORY Gripper      |
| Lite 6                      | Intel Realsense D435 / Luxonis OAK-D-Pro-PoE      | Lite 6 Vacuum Gripper |

### Configuration with Intel Realsense D435 Camera

-   **Robot Arm**: [UFACTORY xArm 6](https://www.ufactory.cc/xarm-collaborative-robot/)
-   **Gripper**: [UFACTORY xArm Gripper](https://www.ufactory.cc/product-page/ufactory-xarm-gripper/)
-   **Camera**: [Intel RealSense D435](https://www.intelrealsense.com/depth-camera-d435/)
-   **Camera Mount**: Provided by UFACTORY (available for purchase or 3D printing)
    -   Purchase Link: [UFACTORY Camera Stand](https://www.ufactory.cc/product-page/ufactory-xarm-camera-stand/)
    -   3D File Download: [Realsense_Camera_Stand.STEP](https://www.ufactory.cc/wp-content/uploads/2024/05/CameraStand_1300.zip)

### Configuration with Luxonis OAK-D-Pro-PoE Camera

-   **Robot Arm**: [UFACTORY 850](https://www.ufactory.cc/ufactory-850/)
-   **Gripper**: [UFACTORY xArm Gripper](https://www.ufactory.cc/product-page/ufactory-xarm-gripper/)
-   **Camera**: [OAK-D Pro PoE](https://shop.luxonis.com/products/oak-d-pro-poe?variant=42469208883423)
-   **PoE Injector**: Tenda POE30G-AT, 1000M, 30W (Other brands will also work)
-   **Camera Mount**: Provided by UFACTORY (3D printing or CNC with 3D file)
    -   3D File Download: [OAK_Camera_Stand.STEP](https://www.ufactory.cc/wp-content/uploads/2025/05/oak_camera_stand.zip)

## Software

### Supported Python Versions

Supported Python versions: 3.8-3.11 (Recommended: 3.11).

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/xArm-Developer/ufactory_vision.git
cd ufactory_vision
```

### 2. Create a Python Virtual Environment

It is recommended to use a virtual environment for running this project.

#### **Windows (Using Anaconda)**

```bash
conda create --name ufactory_vision python=3.11
conda activate ufactory_vision
```

#### **Linux (Using venv)**

```bash
python3.11 -m venv ufactory_vision
source ufactory_vision/bin/activate
```

### 3. Install Dependencies and Run Examples

Please follow the corresponding installation and execution steps based on the camera model you are using.
First, ensure you are in the `ggcnn_grasping_demo` directory:

```bash
cd ggcnn_grasping_demo
```

#### **Using Intel Realsense D435 Camera**

1.  **Install Dependencies**

    ```bash
    pip install -r requirements_rs.txt
    ```

2.  **Run Example**

    Replace `192.168.1.xxx` with the actual IP address of your robot arm controller.

    *   **UFACTORY 850 or xArm 5/6/7 Series Robot Arm**
        ```bash
        python run_rs_grasp.py 192.168.1.xxx
        ```
    *   **UFACTORY Lite 6 Robot Arm**
        ```bash
        python run_rs_grasp_lite6.py 192.168.1.xxx
        ```

#### **Using Luxonis OAK-D-Pro-PoE Camera**

1.  **Install Dependencies**

    ```bash
    pip install -r requirements_depthai.txt
    ```

2.  **Run Example**

    Replace `192.168.1.xxx` with the actual IP address of your robot arm controller.

    *   **UFACTORY 850 or xArm 5/6/7 Series Robot Arm**
        ```bash
        python run_depthai_grasp.py 192.168.1.xxx
        ```
    *   **UFACTORY Lite 6 Robot Arm**
        ```bash
        python run_depthai_grasp_lite6.py 192.168.1.xxx
        ```

## Important Notes

*   **TCP/Coordinate Offset**: Do not set TCP offset or coordinate offset, otherwise you may need to fine-tune the code.
*   **TCP Payload**: Set TCP payload to avoid false collision detection.
*   **Collision Detection**: Before running the example, ensure that collision detection is enabled. It is recommended to set the collision sensitivity to 3 or higher.

## License

This project is licensed under the **BSD 3-Clause License**. For details, please check the [LICENSE](LICENSE) file.

## Acknowledgements

Our demo project is built based on the following open-source projects:

-   [GGCNN](https://github.com/dougsm/ggcnn)
-   [ggcnn_kinova_grasping](https://github.com/dougsm/ggcnn_kinova_grasping)
