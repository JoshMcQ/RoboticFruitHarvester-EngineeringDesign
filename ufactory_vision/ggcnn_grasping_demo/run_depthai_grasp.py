import sys
import cv2
import time
import numpy as np
from camera.depthai_camera import DepthAiCamera
from camera.utils import get_combined_img
from grasp.ggcnn_torch import TorchGGCNN
from grasp.robot_grasp_depthai import RobotGrasp
from queue import Queue

WIN_NAME = 'DEPTHAI'
CAM_WIDTH = 640
CAM_HEIGHT = 400
DISABLE_RGB = True

MODEL_FILE = 'models/ggcnn_epoch_23_cornell'    # GGCNN
# MODEL_FILE = 'models/epoch_50_cornell'          # GGCNN2
# use open-loop solution when robot height is over OPEN_LOOP_HEIGHT
OPEN_LOOP_HEIGHT = 500 # mm
GGCNN_IN_THREAD = False

# EULER_EEF_TO_COLOR_OPT = [0.075, 0, 0.021611456, 0, 0, 1.5708] # xyzrpy meters_rad
EULER_EEF_TO_COLOR_OPT = [0.0703, 0.0023, 0.0195, 0, 0, 1.579] # xyzrpy meters_rad
EULER_COLOR_TO_DEPTH_OPT = [0.0375, 0, 0, 0, 0, 0]

# The range of motion of the robot grasping
# If it exceeds the range, it will return to the initial detection position.
GRASPING_RANGE = [180, 600, -200, 200] # [x_min, x_max, y_min, y_max]

# initial detection position
DETECT_XYZ = [300, 0, 400] # [x, y, z]

# release grasping pos
RELEASE_XYZ = [400, 400, 270]

# lift offset based on DETECT_XYZ[2] after grasping or release
LIFT_OFFSET_Z = 100 # lift_height = DETECT_XYZ[2] + LIFT_OFFSET_Z

# The distance between the gripping point of the robot grasping and the end of the robot arm flange
# The value needs to be fine-tuned according to the actual situation.
GRIPPER_Z_MM = 150 # mm

# minimum z for grasping
GRASPING_MIN_Z = 175 # mm

MOVE_SPEED = 200
MOVE_ACC = 1000

DETECT_STEPS = 2


def main():
    if len(sys.argv) < 2:
        print('Usage: {} {{robot_ip}}'.format(sys.argv[0]))
        exit(1)
    
    robot_ip = sys.argv[1]

    depth_img_que = Queue(1)
    ggcnn_cmd_que = Queue(1)
    
    camera = DepthAiCamera(width=CAM_WIDTH, height=CAM_HEIGHT, disable_rgb=DISABLE_RGB)
    _, depth_intrin = camera.get_intrinsics()
    ggcnn_config = {
        'MODEL_FILE': MODEL_FILE,
        'OPEN_LOOP_HEIGHT': OPEN_LOOP_HEIGHT,
        'GGCNN_IN_THREAD': GGCNN_IN_THREAD,
        'DEPTH_CAM_K': depth_intrin,
    }
    ggcnn = TorchGGCNN(ggcnn_config, depth_img_que, ggcnn_cmd_que)
    time.sleep(2)
    euler_opt = {
        'EULER_EEF_TO_COLOR_OPT': EULER_EEF_TO_COLOR_OPT,
        'EULER_COLOR_TO_DEPTH_OPT': EULER_COLOR_TO_DEPTH_OPT,
    }
    grasp_config = {
        'GRASPING_RANGE': GRASPING_RANGE,
        'DETECT_XYZ': DETECT_XYZ,
        'RELEASE_XYZ': RELEASE_XYZ,
        'LIFT_OFFSET_Z': LIFT_OFFSET_Z,
        'GRIPPER_Z_MM': GRIPPER_Z_MM,
        'GRASPING_MIN_Z': GRASPING_MIN_Z,
        'DETECT_STEPS': DETECT_STEPS,
        'MOVE_SPEED': MOVE_SPEED,
        'MOVE_ACC': MOVE_ACC,
    }
    grasp = RobotGrasp(robot_ip, ggcnn_cmd_que, euler_opt, grasp_config)

    while grasp.is_alive():
        color_image, depth_image = camera.get_images()
        robot_pos = grasp.get_eef_pose_m()
        if GGCNN_IN_THREAD:
            if not depth_img_que.empty():
                depth_img_que.get()
            depth_img_que.put([robot_pos, depth_image])
            if ggcnn.grasp_img is not None:
                if DISABLE_RGB:
                    combined_img = get_combined_img(depth_image, ggcnn.grasp_img)
                else:
                    combined_img = get_combined_img(color_image, ggcnn.grasp_img)
                cv2.imshow(WIN_NAME, combined_img)
            else:
                if DISABLE_RGB:
                    cv2.imshow(WIN_NAME, depth_image) # 显示深度图像
                else:
                    cv2.imshow(WIN_NAME, color_image) # 显示彩色图像
        else:
            grasp_img, result = ggcnn.get_grasp_img(depth_image, depth_intrin, robot_pos[2])
            if result:
                if not ggcnn_cmd_que.empty():
                    ggcnn_cmd_que.get()
                ggcnn_cmd_que.put([robot_pos, result])

            if DISABLE_RGB:
                combined_img = get_combined_img(depth_image, grasp_img)
            else:
                combined_img = get_combined_img(color_image, grasp_img)
            cv2.imshow(WIN_NAME, combined_img)
        
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            camera.stop()
            break

if __name__ == '__main__':
    main()