#!/usr/bin/env python3
"""
Hand-eye calibration using multiple AprilTags placed in the workspace.

Assumptions
- Tags are tag36h11 family
- Each tag is 5.8 mm x 5.8 mm (size = 0.0058 m)
- You provide the base-frame XYZ of each tag center in TAG_POSITIONS_MM

Flow
1) Update TAG_POSITIONS_MM with your actual tag locations
2) Update default_calibration_poses() with robot waypoints above your tags
3) Run script: visits robot poses, detects tags, runs calibrateHandEye
4) Validates by comparing predicted vs known tag world positions
5) Saves EULER_EEF_TO_COLOR_OPT compatible array: [tx, ty, tz, rx, ry, rz]

Hot tips for small tags (5.8 mm)
- Keep camera close (short working distance)
- Good lighting and focus matter a lot
- On OAK-D, avoid image decimation in the tag detector
"""

import time
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import cv2
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

from xarm.wrapper import XArmAPI
from camera.depthai_camera import DepthAiCamera


# ---- Configuration ----
ROBOT_IP = '192.168.1.221'

# AprilTag parameters for your setup
TAG_FAMILY = 'tag36h11'
TAG_SIZE_M = 0.0058  # 5.8 mm = 0.0058 m
NUM_TAGS = 6        # expected unique tag IDs (for planning only)

# Known tag world positions (mm) in robot base frame: center of each tag
# Replace with your measured values for your actual tag placement.
TAG_POSITIONS_MM: Dict[int, List[float]] = {
    # tag_id: [X_mm, Y_mm, Z_mm]
    0: [300, -50, -73],
    1: [350, -50, -73],
    2: [400, -50, -73],
    3: [300,  50, -73],
    4: [350,  50, -73],
    5: [400,  50, -73],
}


def euler_xyz_deg_to_rot(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Rotation matrix from roll/pitch/yaw in degrees using XYZ order.
    Use the same convention consistently for both data collection and testing.
    """
    if R is None:
        # Minimal fallback using numpy and composition (XYZ intrinsic: Rx @ Ry @ Rz)
        rd, pd, yd = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
        cr, sr = np.cos(rd), np.sin(rd)
        cp, sp = np.cos(pd), np.sin(pd)
        cy, sy = np.cos(yd), np.sin(yd)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rx @ Ry @ Rz
    return R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()


class MultiTagCalibrator:
    def __init__(self,
                 robot_ip: str = ROBOT_IP,
                 tag_size_m: float = TAG_SIZE_M,
                 tag_positions_mm: Dict[int, List[float]] = TAG_POSITIONS_MM):
        if Detector is None:
            raise RuntimeError("pupil_apriltags not installed. Run: pip install pupil-apriltags")
        if R is None:
            raise RuntimeError("scipy not installed. Run: pip install scipy")

        # Robot
        self.arm = XArmAPI(robot_ip)
        time.sleep(0.3)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.reset(wait=True)

        # AprilTag detector tuned for small tags
        self.detector = Detector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=1.0,   # no decimation for tiny tags
            quad_sigma=0.0,      # no blur
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        # Camera
        print("Starting OAK-D camera (640x400)…")
        self.cam = DepthAiCamera(width=640, height=400, disable_rgb=False)
        self.K_rgb, self.K_depth = self.cam.get_intrinsics()

        self.tag_size_m = float(tag_size_m)
        self.tag_positions_mm = dict(tag_positions_mm)
        self.measurements: List[dict] = []

    # ---------- Helpers ----------
    def _get_pose_rot_trans(self, pose: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        x, y, z, roll, pitch, yaw = pose
        R_g2b = euler_xyz_deg_to_rot(roll, pitch, yaw)
        t_g2b = np.array([[x], [y], [z]], dtype=float) / 1000.0
        return R_g2b, t_g2b

    def _camera_params(self) -> List[float]:
        return [
            float(self.K_rgb[0][0]),  # fx
            float(self.K_rgb[1][1]),  # fy
            float(self.K_rgb[0][2]),  # cx
            float(self.K_rgb[1][2]),  # cy
        ]

    # ---------- Detection ----------
    def detect_all_tags(self, show_image: bool = False) -> Dict[int, dict]:
        """Detect AprilTags, average over a few frames, return dict per tag_id:
        {'R': R_tag2cam, 't': t_tag2cam, 'num_detections': n, 'world_pos': [x,y,z] m}
        """
        by_id: Dict[int, List] = {}
        for _ in range(6):
            color, depth = self.cam.get_images()
            if color is None:
                continue
            # Ensure grayscale
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            dets = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self._camera_params(),
                tag_size=self.tag_size_m,
            )
            for d in dets:
                by_id.setdefault(d.tag_id, []).append(d)

            if show_image:
                disp = color.copy()
                for d in dets:
                    c = d.corners.astype(int)
                    cv2.polylines(disp, [c], True, (0, 255, 0), 2)
                    ctr = tuple(d.center.astype(int))
                    dist_mm = float(np.linalg.norm(d.pose_t)) * 1000.0
                    cv2.putText(disp, f"ID:{d.tag_id}", (ctr[0]-20, ctr[1]-18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                    cv2.putText(disp, f"{dist_mm:.0f}mm", (ctr[0]-25, ctr[1]+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
                cv2.imshow('AprilTags', disp)
                cv2.waitKey(1)
            time.sleep(0.04)

        averaged: Dict[int, dict] = {}
        for tid, lst in by_id.items():
            if tid not in self.tag_positions_mm:
                print(f"  Warn: detected tag {tid} not in TAG_POSITIONS_MM; skipping")
                continue
            # Average rotation via quaternions
            R_list = [d.pose_R for d in lst]
            quats = [R.from_matrix(Ri).as_quat() for Ri in R_list]
            q = np.mean(quats, axis=0)
            q = q / np.linalg.norm(q)
            R_tag2cam = R.from_quat(q).as_matrix()
            # Average translation
            t_list = [d.pose_t.reshape(3,1) for d in lst]
            t_tag2cam = np.mean(t_list, axis=0)

            averaged[tid] = {
                'R': R_tag2cam,
                't': t_tag2cam,
                'num_detections': len(lst),
                'world_pos': np.array(self.tag_positions_mm[tid], dtype=float) / 1000.0,
            }
        return averaged

    # ---------- Data collection ----------
    def collect_at_pose(self, pose: List[float], pose_num: int, show_image: bool = True):
        """Collect measurements at a single pose"""
        print(f"\n-- Pose {pose_num}: {pose}")
        self.arm.set_position(*pose, wait=True)
        time.sleep(0.4)
        det = self.detect_all_tags(show_image=show_image)
        if len(det) < 2:
            print(f"  Only {len(det)} tags seen; skipping")
            return
        print(f"  Detected tags: {list(det.keys())}")
        R_g2b, t_g2b = self._get_pose_rot_trans(pose)
        for tid, d in det.items():
            self.measurements.append({
                'R_gripper2base': R_g2b.astype(np.float64),
                't_gripper2base': t_g2b.astype(np.float64),
                'R_tag2cam': d['R'].astype(np.float64),
                't_tag2cam': d['t'].astype(np.float64),
                'tag_id': tid,
                'tag_world_pos': d['world_pos'].astype(np.float64),
                'pose_index': pose_num,
            })
        print(f"  Total measurements: {len(self.measurements)}")
    
    def collect(self, show_image: bool = True) -> int:
        """Visit all calibration poses and collect measurements"""
        print("Collecting calibration data...")
        
        self.arm.set_position(*[350, 0, 250, 180, 0, 0], wait=True)
        self.collect_at_pose([350, 0, 250, 180, 0, 0], 1, show_image)
        
        self.arm.set_position(*[350, 0, 200, 180, 0, 0], wait=True)
        self.collect_at_pose([350, 0, 200, 180, 0, 0], 2, show_image)
        
        self.arm.set_position(*[300, 0, 230, 180, 0, 0], wait=True)
        self.collect_at_pose([300, 0, 230, 180, 0, 0], 3, show_image)
        
        self.arm.set_position(*[400, 0, 230, 180, 0, 0], wait=True)
        self.collect_at_pose([400, 0, 230, 180, 0, 0], 4, show_image)
        
        self.arm.set_position(*[350, -50, 230, 180, 0, 0], wait=True)
        self.collect_at_pose([350, -50, 230, 180, 0, 0], 5, show_image)
        
        self.arm.set_position(*[350, 50, 230, 180, 0, 0], wait=True)
        self.collect_at_pose([350, 50, 230, 180, 0, 0], 6, show_image)
        
        self.arm.set_position(*[320, -30, 220, 175, 5, 5], wait=True)
        self.collect_at_pose([320, -30, 220, 175, 5, 5], 7, show_image)
        
        self.arm.set_position(*[380, 30, 240, 175, -5, -5], wait=True)
        self.collect_at_pose([380, 30, 240, 175, -5, -5], 8, show_image)
        
        self.arm.set_position(*[350, 0, 230, 180, 0, 10], wait=True)
        self.collect_at_pose([350, 0, 230, 180, 0, 10], 9, show_image)
        
        self.arm.set_position(*[350, 0, 230, 180, 0, -10], wait=True)
        self.collect_at_pose([350, 0, 230, 180, 0, -10], 10, show_image)
        
        self.arm.set_position(*[330, -40, 210, 170, 10, 0], wait=True)
        self.collect_at_pose([330, -40, 210, 170, 10, 0], 11, show_image)
        
        self.arm.set_position(*[370, 40, 210, 185, -10, 0], wait=True)
        self.collect_at_pose([370, 40, 210, 185, -10, 0], 12, show_image)
        
        cv2.destroyAllWindows()
        print(f"Done. Measurements collected: {len(self.measurements)}")
        return len(self.measurements)

    # ---------- Calibration ----------
    def calibrate(self) -> dict:
        if len(self.measurements) < 20:
            raise RuntimeError("Not enough measurements for calibration (need >= 20)")

        print("\n=== Hand-Eye Calibration (multi-method) ===")
        R_g2b = [m['R_gripper2base'] for m in self.measurements]
        t_g2b = [m['t_gripper2base'] for m in self.measurements]
        R_t2c = [m['R_tag2cam'] for m in self.measurements]
        t_t2c = [m['t_tag2cam'] for m in self.measurements]

        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
            (cv2.CALIB_HAND_EYE_PARK, "Park"),
            (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
        ]
        results = []
        for mcode, mname in methods:
            try:
                R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=mcode)
                err_mm = self._validate_world_error(R_c2g, t_c2g)
                print(f"  {mname}: mean world error = {err_mm:.2f} mm")
                results.append({'name': mname, 'R': R_c2g, 't': t_c2g, 'err': err_mm})
            except Exception as e:
                print(f"  {mname} failed: {e}")

        if not results:
            raise RuntimeError("All hand-eye methods failed")
        best = min(results, key=lambda r: r['err'])
        print(f"Best: {best['name']} (error {best['err']:.2f} mm)")

        # Convert to Euler XYZ radians
        eul = R.from_matrix(best['R']).as_euler('xyz', degrees=False)
        return {
            'rotation_matrix': best['R'].tolist(),
            'translation': best['t'].reshape(3).tolist(),
            'euler_angles': eul.tolist(),
            'method': best['name'],
            'error_mm': float(best['err']),
            'num_measurements': len(self.measurements),
            'num_tags_used': len(set(m['tag_id'] for m in self.measurements)),
        }

    def _validate_world_error(self, R_c2g: np.ndarray, t_c2g: np.ndarray) -> float:
        errs = []
        for m in self.measurements:
            p_cam = m['t_tag2cam']           # (3,1)
            p_grip = R_c2g @ p_cam + t_c2g    # (3,1)
            p_world = m['R_gripper2base'] @ p_grip + m['t_gripper2base']
            p_world_mm = p_world.reshape(3) * 1000.0
            actual_mm = m['tag_world_pos'].reshape(3) * 1000.0
            errs.append(float(np.linalg.norm(p_world_mm - actual_mm)))
        return float(np.mean(errs)) if errs else 1e9

    # ---------- Testing ----------
    def test_once(self, calib: dict, pose_mmdeg: List[float]):
        print("\n=== Test Calibration at pose ===")
        print(pose_mmdeg)
        self.arm.set_position(*pose_mmdeg, wait=True)
        time.sleep(0.4)
        det = self.detect_all_tags(show_image=True)
        if not det:
            print("No tags detected at test pose")
            return
        R_c2g = np.array(calib['rotation_matrix'])
        t_c2g = np.array(calib['translation']).reshape(3,1)
        R_g2b, t_g2b = self._get_pose_rot_trans(pose_mmdeg)
        print("ID |    Predicted (mm)            |     Actual (mm)           |  Error (mm)")
        print("-"*72)
        for tid, d in det.items():
            p_cam = d['t']
            p_grip = R_c2g @ p_cam + t_c2g
            p_world = R_g2b @ p_grip + t_g2b
            pred = p_world.reshape(3) * 1000.0
            actual = np.array(self.tag_positions_mm[tid], dtype=float)
            err = float(np.linalg.norm(pred - actual))
            print(f"{tid:2d} | X={pred[0]:7.1f} Y={pred[1]:7.1f} Z={pred[2]:7.1f} | X={actual[0]:7.1f} Y={actual[1]:7.1f} Z={actual[2]:7.1f} | {err:7.2f}")
        cv2.destroyAllWindows()


def default_calibration_poses() -> List[List[float]]:
    """Return robot poses [X, Y, Z, Roll, Pitch, Yaw] in mm and degrees.
    Update these with waypoints that hover above your actual tag positions.
    Each pose should allow the camera to see at least 2 tags clearly.
    """
    return [
        [350, 0, 250, 180, 0, 0],
        [350, 0, 200, 180, 0, 0],
        [300, 0, 230, 180, 0, 0],
        [400, 0, 230, 180, 0, 0],
        [350, -50, 230, 180, 0, 0],
        [350, 50, 230, 180, 0, 0],
        [320, -30, 220, 175, 5, 5],
        [380, 30, 240, 175, -5, -5],
        [350, 0, 230, 180, 0, 10],
        [350, 0, 230, 180, 0, -10],
        [330, -40, 210, 170, 10, 0],
        [370, 40, 210, 185, -10, 0],
    ]


def main():
    print("="*60)
    print("Multi-Tag AprilTag Hand-Eye Calibration (5.8 mm tags, tag36h11)")
    print("="*60)

    parser = argparse.ArgumentParser(description='AprilTag multi-tag hand-eye calibration')
    parser.add_argument('--tag-size-mm', type=float, default=5.8, help='AprilTag edge length in millimeters (default 5.8)')
    args, _ = parser.parse_known_args()

    cal = MultiTagCalibrator(tag_size_m=float(args.tag_size_mm) / 1000.0)

    if cal.collect(show_image=True) < 20:
        print("Not enough measurements; try more poses or ensure at least 2 tags are visible per pose.")
        return

    result = cal.calibrate()
    print("\n=== Calibration Result ===")
    print(f"method: {result['method']}")
    print(f"error_mm: {result['error_mm']:.2f}")
    print(f"translation (m): {result['translation']}")
    print(f"euler_xyz (rad): {result['euler_angles']}")
    print(f"euler_xyz (deg): {np.rad2deg(result['euler_angles']).tolist()}")

    if input("Test calibration now? (y/n): ").strip().lower() == 'y':
        cal.test_once(result, [350, 0, 230, 178, 0, 1])

    if input("Save calibration to JSON? (y/n): ").strip().lower() == 'y':
        simple = {
            'translation': result['translation'],
            'rotation': result['euler_angles'],  # radians
            'method': result['method'],
            'error_mm': result['error_mm'],
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
        with open('hand_eye_calibration.json', 'w') as f:
            json.dump(simple, f, indent=2)
        print("Saved to hand_eye_calibration.json")
        t = result['translation']
        r = result['euler_angles']
        print(f"\nEULER_EEF_TO_COLOR_OPT = {t + r}")


if __name__ == '__main__':
        main()
        arm = XArmAPI(ROBOT_IP)
        arm.reset(wait=True)
