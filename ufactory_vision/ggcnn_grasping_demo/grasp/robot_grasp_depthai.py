import time
import math
import threading
import numpy as np
from .helpers.matrix_funcs import euler2mat, convert_pose
from xarm.wrapper import XArmAPI


# 3次滑动平均，每组输入为4个变量。[x,y,z,yaw]
class Averager():
    def __init__(self, inputs, time_steps):
        self.buffer = np.zeros((time_steps, inputs))
        self.steps = time_steps
        self.curr = 0
        self.been_reset = True

    def update(self, v):
        if self.steps == 1:
            self.buffer = v
            return v
        self.buffer[self.curr, :] = v
        self.curr += 1
        if self.been_reset:
            self.been_reset = False
            while self.curr != 0:
                self.update(v)
        if self.curr >= self.steps:
            self.curr = 0
        pos = self.buffer.mean(axis=0)
        return pos

    def evaluate(self):
        if self.steps == 1:
            return self.buffer
        return self.buffer.mean(axis=0)

    def reset(self):
        self.buffer *= 0
        self.curr = 0
        self.been_reset = True


class MinPos():
    def __init__(self, inputs, time_steps):
        self.buffer = np.zeros((time_steps, inputs))
        self.steps = time_steps
        self.curr = 0
        self.been_reset = True
        self.inputs = inputs
        self.prev_pos = [0, 0, 0, 0]

    def update(self, v):
        if self.steps == 1:
            self.buffer = v
            return v
        self.buffer[self.curr, :] = v
        self.curr += 1
        if self.been_reset:
            self.been_reset = False
            while self.curr != 0:
                self.update(v)
        if self.curr >= self.steps:
            self.curr = 0
        min_inx = 0
        min_dis = 9999
        for i in range(self.steps):
            dis = (
                (self.buffer[i][0] - self.prev_pos[0]) ** 2
                + (self.buffer[i][1] - self.prev_pos[1]) ** 2
                + (self.buffer[i][2] - self.prev_pos[2]) ** 2
            )
            if dis < min_dis:
                min_dis = dis
                min_inx = i
        self.prev_pos[0] = self.buffer[min_inx][0]
        self.prev_pos[1] = self.buffer[min_inx][1]
        self.prev_pos[2] = self.buffer[min_inx][2]
        self.prev_pos[3] = self.buffer[min_inx][3]
        return self.prev_pos

    def evaluate(self):
        if self.steps == 1:
            return self.buffer
        return self.prev_pos

    def reset(self):
        self.buffer *= 0
        self.curr = 0
        self.prev = 0
        self.been_reset = True


class GraspPos:
    def __init__(self, threshold_x=20, threshold_y=20):
        self.threshold_x = threshold_x
        self.threshold_y = threshold_y
        self.grasp_pos_a = [0] * 6
        self.grasp_pos_b = [0] * 6
        self.grasp_pos_c = [0] * 6
        self.step = 0

    def set_step(self, step):
        self.step = step
        print('set_step: {}'.format(step))

    def set_pos_a(self, pos):
        for i in range(6):
            self.grasp_pos_a[i] = pos[i]

    def set_pos_b(self, pos):
        for i in range(6):
            self.grasp_pos_b[i] = pos[i]

    def set_pos_c(self, pos):
        for i in range(6):
            self.grasp_pos_c[i] = pos[i]

    def update_pos_a_from_b(self):
        for i in range(6):
            self.grasp_pos_a[i] = self.grasp_pos_b[i]

    def update_pos_a_from_bc(self):
        for i in range(6):
            self.grasp_pos_a[i] = self.grasp_pos_c[i]
        self.grasp_pos_a[2] = self.grasp_pos_b[2]

    def check_ab(self):
        if (abs(self.grasp_pos_a[0] - self.grasp_pos_b[0]) > self.threshold_x
                or abs(self.grasp_pos_a[1] - self.grasp_pos_b[1]) > self.threshold_y):
            return True
        return False

    def check_bc(self):
        if (abs(self.grasp_pos_b[0] - self.grasp_pos_c[0]) > self.threshold_x
                or abs(self.grasp_pos_b[1] - self.grasp_pos_c[1]) > self.threshold_y):
            return True
        return False


class RobotGrasp(object):
    CURR_POS = [300, 0, 350, 180, 0, 0]
    GOAL_POS = [0, 0, 0, 0, 0, 0]

    def __init__(self, robot_ip, ggcnn_cmd_que, euler_opt, grasp_config):
        self.arm = XArmAPI(robot_ip, report_type='real')
        self.ggcnn_cmd_que = ggcnn_cmd_que
        self.euler_eef_to_color_opt = euler_opt['EULER_EEF_TO_COLOR_OPT']
        self.euler_color_to_depth_opt = euler_opt['EULER_COLOR_TO_DEPTH_OPT']
        self.grasping_range = grasp_config['GRASPING_RANGE']
        self.detect_xyz = grasp_config['DETECT_XYZ']
        self.release_xyz = grasp_config['RELEASE_XYZ']
        self.lift_height = self.detect_xyz[2] + grasp_config['LIFT_OFFSET_Z']
        self.gripper_z_mm = grasp_config['GRIPPER_Z_MM']
        self.grasping_min_z = grasp_config['GRASPING_MIN_Z']  # set in launcher (e.g., 80 mm)
        self.detect_steps = grasp_config.get('DETECT_STEPS', 3)
        self.move_speed = grasp_config.get('MOVE_SPEED', 200)
        self.move_acc = grasp_config.get('MOVE_ACC', 1000)
        if self.detect_steps not in [1, 2, 3]:
            self.detect_steps = 3
        self.use_vacuum_gripper = grasp_config.get('USE_VACUUM_GRIPPER', False)
        # self.pose_averager = Averager(4, 3)
        self.pose_averager = MinPos(4, 3)
        self.ready_check = False
        self.ready_grasp = False
        self.alive = True
        self.last_grasp_time = 0
        self.is_over_range = False
        self.grasp_pos = GraspPos()
        self.pos_t = threading.Thread(target=self.update_pos_loop, daemon=True)
        self.pos_t.start()
        self.ggcnn_t = threading.Thread(target=self.handle_ggcnn_loop, daemon=True)
        self.ggcnn_t.start()
        self.check_t = threading.Thread(target=self.check_loop, daemon=True)
        self.check_t.start()

    def is_alive(self):
        return self.alive

    def handle_ggcnn_loop(self):
        while self.arm.connected and self.alive:
            cmd = self.ggcnn_cmd_que.get()
            self.grasp(cmd)

    def update_pos_loop(self):
        self.arm.motion_enable(True)
        self.arm.clean_error()
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.5)
        self.arm.set_position(z=self.detect_xyz[2], wait=True)
        self.arm.set_position(x=self.detect_xyz[0], y=self.detect_xyz[1], z=self.detect_xyz[2],
                              roll=180, pitch=0, yaw=0, wait=True)
        time.sleep(0.5)

        self.place()
        time.sleep(0.5)

        self.ready_grasp = True
        self.arm.set_state(0)
        time.sleep(0.5)

        self.ready_check = True
        self.grasp_pos.set_step(1)

        while self.arm.connected and self.arm.error_code == 0:
            _, pos = self.arm.get_position()
            self.arm.get_err_warn_code()
            self.CURR_POS = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]
            time.sleep(0.01)
        self.alive = False
        if self.arm.error_code != 0:
            print('ERROR_CODE: {}'.format(self.arm.error_code))
        print('*************** PROGRAM OVER *******************')

        self.arm.disconnect()

    def check_loop(self):
        while self.arm.connected and self.arm.error_code == 0:
            self._check()
            time.sleep(0.01)

    def _check(self):
        if not self.ready_check or not self.alive:
            return
        x = self.CURR_POS[0]
        y = self.CURR_POS[1]
        z = self.CURR_POS[2]
        roll = self.CURR_POS[3]
        pitch = self.CURR_POS[4]
        yaw = self.CURR_POS[5]

        # reset to start position if moved outside of boundary
        if ((self.last_grasp_time != 0 and (time.monotonic() - self.last_grasp_time) > 5)
            or x < self.grasping_range[0] or x > self.grasping_range[1]
            or y < self.grasping_range[2] or y > self.grasping_range[3]):
            if ((time.monotonic() - self.last_grasp_time) > 3
                and abs(x - self.detect_xyz[0]) < 2 and abs(y - self.detect_xyz[1]) < 2 and abs(z - self.detect_xyz[2]) < 2
                and abs(abs(roll) - 180) < 2 and abs(pitch) < 2 and abs(yaw) < 2):
                self.last_grasp_time = time.monotonic()
                return
            print('[STOP] MOVE TO INITIAL DETECT POSINT, CURR_POS={}, last_grasp_time={}'.format(self.CURR_POS, self.last_grasp_time))
            self.ready_grasp = False
            self.arm.set_state(4)
            time.sleep(1)
            self.arm.set_state(0)
            time.sleep(1)
            self.arm.set_position(z=self.detect_xyz[2], speed=self.move_speed, wait=True)
            self.arm.set_position(x=self.detect_xyz[0], y=self.detect_xyz[1], z=self.detect_xyz[2],
                                  roll=180, pitch=0, yaw=0, speed=self.move_speed, wait=True)
            time.sleep(0.25)
            self.pose_averager.reset()
            self.arm.set_state(0)
            time.sleep(1)

            self.ready_grasp = True
            self.last_grasp_time = time.monotonic()
            self.grasp_pos.set_step(1)
            return

        if 2 < self.grasp_pos.step < 3:
            pos = self.grasp_pos.grasp_pos_a
            if abs(x - pos[0] + 50) < 2 and abs(y - pos[1]) < 2 and self.arm.state != 1:
                time.sleep(1)
                self.grasp_pos.set_step(3)
        if 5 < self.grasp_pos.step < 6:
            pos = self.grasp_pos.grasp_pos_b
            pos_z = max(pos[2], 320)
            if abs(x - pos[0] + 50) < 2 and abs(y - pos[1]) < 2 and abs(z - pos_z) < 2 and self.arm.state != 1:
                time.sleep(1)
                self.grasp_pos.set_step(6)

        if self.grasp_pos.step == 4:
            print('===========step4=============')
            print(self.grasp_pos.grasp_pos_a)
            print(self.grasp_pos.grasp_pos_b)

            if self.grasp_pos.check_ab():
                self.grasp_pos.update_pos_a_from_b()
                self.grasp_pos.set_step(2)
            else:
                self.grasp_pos.set_step(5)
        elif self.grasp_pos.step == 7:
            print('===========step7==============')
            print(self.grasp_pos.grasp_pos_b)
            print(self.grasp_pos.grasp_pos_c)
            if self.grasp_pos.check_bc():
                self.grasp_pos.update_pos_a_from_bc()
                self.grasp_pos.set_step(2)
            else:
                self.grasp_pos.set_step(8)

        # Stop Conditions: stop when we reached target Z or dipped below the floor.
        if self.grasp_pos.step == 9 or z < self.grasping_min_z or z - 1 < self.GOAL_POS[2]:
            if not self.ready_grasp:
                return
            self.ready_grasp = False
            print('[GRASP] CURR_POS={}'.format(self.CURR_POS))
            self.arm.set_state(4)
            time.sleep(1)
            self.arm.set_state(0)
            time.sleep(0.1)

            self.pick()
            time.sleep(0.5)

            self.arm.set_position(z=self.lift_height, speed=self.move_speed, wait=True)
            self.arm.set_position(x=self.release_xyz[0], y=self.release_xyz[1],
                                  roll=180, pitch=0, yaw=0, speed=self.move_speed, wait=True)
            self.arm.set_position(z=self.release_xyz[2], speed=self.move_speed, wait=True)

            self.place()

            self.arm.set_position(z=self.lift_height, speed=self.move_speed, wait=True)
            self.arm.set_position(x=self.detect_xyz[0], y=self.detect_xyz[1], z=self.detect_xyz[2],
                                  roll=180, pitch=0, yaw=0, speed=self.move_speed, wait=True)

            self.pose_averager.reset()
            self.arm.set_state(0)
            time.sleep(2)

            self.ready_grasp = True
            self.last_grasp_time = time.monotonic()
            self.grasp_pos.set_step(1)

    def pick(self):
        if self.use_vacuum_gripper:
            self.arm.set_vacuum_gripper(on=True)
        else:
            self.arm.set_gripper_position(0, wait=True)

    def place(self):
        if self.use_vacuum_gripper:
            self.arm.set_vacuum_gripper(on=False)
        else:
            self.arm.set_gripper_position(800, wait=True)

    def get_eef_pose_m(self):
        _, eef_pos_mm = self.arm.get_position(is_radian=True)
        eef_pos_m = [
            eef_pos_mm[0] * 0.001,
            eef_pos_mm[1] * 0.001,
            eef_pos_mm[2] * 0.001,
            eef_pos_mm[3], eef_pos_mm[4], eef_pos_mm[5]
        ]
        return eef_pos_m

    def grasp(self, data):
        if not self.alive or not self.ready_check or not self.ready_grasp:
            return
        if self.grasp_pos.step <= 0:
            return

        euler_base_to_eef = data[0]
        d = list(data[1])

        # PBVS Method.
        # euler_base_to_eef = self.get_eef_pose_m()

        if d[2] > 0.2:  # Min effective depth range
            gp = [d[0], d[1], d[2], 0, 0, -1 * d[3]]  # xyzrpy in meters

            # Transform grasp pose into robot base frame
            mat_depthOpt_in_base = (
                euler2mat(euler_base_to_eef)
                * euler2mat(self.euler_eef_to_color_opt)
                * euler2mat(self.euler_color_to_depth_opt)
            )
            gp_base = convert_pose(gp, mat_depthOpt_in_base)

            if gp_base[5] < -np.pi:
                gp_base[5] += np.pi
            elif gp_base[5] > 0:
                gp_base[5] -= np.pi

            av = self.pose_averager.update(
                np.array([gp_base[0], gp_base[1], gp_base[2], gp_base[5]])
            )
        else:
            gp_base = [0] * 6
            av = self.pose_averager.evaluate()

        # Drop invalid averages
        if not np.isfinite(av).all():
            return

        # Average pose in base frame.
        ang = av[3] - np.pi / 2  # We don't want to align, we want to grip.

        # FIXED: use x,y,z correctly (no x=y=z)
        gp_base = [av[0], av[1], av[2], np.pi, 0, ang]

        # Build goal pose (mm/deg for xArm)
        self.GOAL_POS = [
            av[0] * 1000,
            av[1] * 1000,
            av[2] * 1000 + self.gripper_z_mm,
            180, 0,
            math.degrees(ang + np.pi)
        ]

        # Early reject if goal below floor
        if self.GOAL_POS[2] < self.grasping_min_z:
            return

        # Final hard clamp to [floor, ceiling]
        Z_MIN_SAFE = self.grasping_min_z
        Z_MAX_SAFE = 700  # adjust to your cell
        if self.GOAL_POS[2] < Z_MIN_SAFE:
            print('[BLOCK] Goal Z below floor:', self.GOAL_POS[2], '→', Z_MIN_SAFE)
            self.GOAL_POS[2] = Z_MIN_SAFE
        elif self.GOAL_POS[2] > Z_MAX_SAFE:
            self.GOAL_POS[2] = Z_MAX_SAFE

        # XY envelope check
        if (self.GOAL_POS[0] < self.grasping_range[0] or self.GOAL_POS[0] > self.grasping_range[1]
                or self.GOAL_POS[1] < self.grasping_range[2] or self.GOAL_POS[1] > self.grasping_range[3]):
            if not self.is_over_range:
                print('[WARN] TARGET IS OVER RANGE:', self.GOAL_POS)
            self.is_over_range = True
            return

        self.is_over_range = False
        self.last_grasp_time = time.monotonic()

        if self.detect_steps == 1:
            # 直接去到目标点进行抓取操作
            self.grasp_pos.set_pos_c(self.GOAL_POS)
            self.grasp_pos.set_step(8)

        if self.grasp_pos.step == 1:
            self.grasp_pos.set_pos_a(self.GOAL_POS)
            print('[1]', time.time(), self.GOAL_POS)
            self.grasp_pos.set_step(2)

        if self.grasp_pos.step == 2:
            pos = self.grasp_pos.grasp_pos_a
            self.last_grasp_time = 0
            self.arm.set_position(x=pos[0] - 50, y=pos[1], z=self.CURR_POS[2],
                                  roll=self.CURR_POS[3], pitch=self.CURR_POS[4], yaw=self.CURR_POS[5],
                                  speed=self.move_speed, acc=self.move_acc, wait=False)
            self.grasp_pos.set_step(2.1)
            self.last_grasp_time = time.monotonic()

        if self.grasp_pos.step == 3:
            self.grasp_pos.set_pos_b(self.GOAL_POS)
            print('[3]', time.time(), self.GOAL_POS)
            self.grasp_pos.set_step(4)

        if self.grasp_pos.step == 5:
            if self.detect_steps == 2:
                # 直接去到目标点进行抓取操作
                self.grasp_pos.set_pos_c(self.GOAL_POS)
                self.grasp_pos.set_step(8)
            else:
                pos = self.grasp_pos.grasp_pos_b
                self.last_grasp_time = 0
                self.arm.set_position(x=pos[0] - 50, y=pos[1], z=max(pos[2], 320),
                                      roll=self.CURR_POS[3], pitch=self.CURR_POS[4], yaw=self.CURR_POS[5],
                                      speed=self.move_speed, acc=self.move_acc, wait=False)
                self.grasp_pos.set_step(5.1)
            self.last_grasp_time = time.monotonic()

        if self.grasp_pos.step == 6:
            print('[6]', time.time(), self.GOAL_POS)
            self.grasp_pos.set_pos_c(self.GOAL_POS)
            self.grasp_pos.set_step(7)

        if self.grasp_pos.step == 8:
            pos = self.grasp_pos.grasp_pos_c
            self.last_grasp_time = 0

            # Hover first, then slow descend (safer)
            Z_HOVER_OFFSET = 80  # mm
            hover_z = max(self.CURR_POS[2], pos[2] + Z_HOVER_OFFSET)

            self.arm.set_position(x=pos[0], y=pos[1], z=hover_z,
                                  roll=pos[3], pitch=pos[4], yaw=pos[5],
                                  speed=int(self.move_speed * 0.5),
                                  acc=int(self.move_acc * 0.5), wait=True)

            self.arm.set_position(z=pos[2],
                                  speed=int(self.move_speed * 0.25),
                                  acc=int(self.move_acc * 0.25), wait=True)

            self.grasp_pos.set_step(9)
            self.last_grasp_time = time.monotonic()
