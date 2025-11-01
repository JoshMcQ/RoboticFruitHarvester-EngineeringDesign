#!/usr/bin/env python3
import os, time, argparse, json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import cv2
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI

from camera.depthai_camera import DepthAiCamera

ROBOT_IP = '192.168.1.221'
TAG_FAMILY = 'tag36h11'
TAG_SIZE_M = 0.0058  # black square only

VIEW_ALL_POSE_MMDEG = [289.3, -39.7, 428.6, 178.9, -13.4, -6.9]

TAG_POSITIONS_MM = {
    0: [282.3,   44.6,  -89.3],  # BL  (id 0)
    3: [290.1, -118.3,  -89.5],  # BR  (id 3)
    1: [445.4, -128.4,  -86.0],  # ML  (id 1)
    5: [440.5,   45.3,  -86.3],  # MR  (id 5)
    2: [606.5,   54.0,  -83.2],  # TL  (id 2)
    4: [603.2, -131.6,  -81.0],  # TR  (id 4)
}

# Hover offsets already applied (Y −20mm, Z +20mm from your earlier set)
HOVER_OVER_TAG_POSES_MMDEG = [
    [245.1,   23.6,  -45.0,  -179.3,   6.3,   3.2],  # 0 BL
    [253.7, -148.2,  -28.3,   174.5,   3.2,   1.4],  # 1 BR
    [384.3,   25.3,  -23.8,  -179.4,   6.0,   4.8],  # 2 ML
    [396.0, -149.6,  -25.9,   173.1,   5.6,  -0.4],  # 3 MR
    [557.3,   28.7,  -17.2,  -179.5,  -5.2,   2.6],  # 4 TL
    [552.8, -144.1,  -14.4,   175.4,  -5.0,   3.5],  # 5 TR
]

def euler_deg_to_rot(roll_deg: float, pitch_deg: float, yaw_deg: float, order: str = 'xyz') -> np.ndarray:
    return R.from_euler(order, [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()

class MultiTagCalibrator:
    def __init__(
        self,
        robot_ip: str,
        tag_size_m: float,
        tag_positions_mm: Dict[int, List[float]],
        session_dir: str,
        save_images: bool = True,
        avg_frames: int = 6,
    ):
        self.session_dir = session_dir
        self.save_images = save_images
        self.avg_frames = int(avg_frames)
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "poses"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "test"), exist_ok=True)

        # Robot
        self.arm = XArmAPI(robot_ip)
        time.sleep(0.2)
        self._ensure_ready()

        # Detector
        self.detector = Detector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        # Camera
        print("Starting OAK-D camera (640x400)…")
        self.cam = DepthAiCamera(width=640, height=400, disable_rgb=False)
        self.K_rgb, self.K_depth = self.cam.get_intrinsics()
        with open(os.path.join(self.session_dir, "intrinsics.json"), "w") as f:
            json.dump(
                {
                    "K_rgb": np.asarray(self.K_rgb).tolist(),
                    "K_depth": np.asarray(self.K_depth).tolist(),
                    "fx": float(self.K_rgb[0][0]),
                    "fy": float(self.K_rgb[1][1]),
                    "cx": float(self.K_rgb[0][2]),
                    "cy": float(self.K_rgb[1][2]),
                },
                f,
                indent=2,
            )

        self.tag_size_m = float(tag_size_m)
        self.tag_positions_mm = dict(tag_positions_mm)
        self.measurements: List[dict] = []

    # ---------- Robot helpers ----------
    def _ensure_ready(self):
        # Be resilient to ControllerError 1
        try:
            self.arm.motion_enable(True)
            self.arm.clean_error()
            self.arm.set_mode(0)
            self.arm.set_state(0)
        except Exception:
            pass

    def _safe_move(self, pose):
        try:
            code = self.arm.set_position(*pose, wait=True)
            if code != 0:
                print(f"[set_position] code={code}, attempting recovery…")
                self._ensure_ready()
                code2 = self.arm.set_position(*pose, wait=True)
                if code2 != 0:
                    print(f"[set_position] retry failed, code={code2}")
        except Exception as e:
            print(f"[set_position] exception: {e}")
            self._ensure_ready()

    # ---------- Math helpers ----------
    def _to_proper_rotation(self, Rm: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(Rm)
        R_fix = U @ Vt
        if np.linalg.det(R_fix) < 0:
            U[:, -1] *= -1.0
            R_fix = U @ Vt
        return R_fix

    def _get_pose_rot_trans(self, pose: List[float], order: str = 'xyz') -> Tuple[np.ndarray, np.ndarray]:
        x, y, z, roll, pitch, yaw = pose
        R_g2b = euler_deg_to_rot(roll, pitch, yaw, order=order)
        t_g2b = np.array([[x], [y], [z]], dtype=float) / 1000.0
        return R_g2b, t_g2b

    def _camera_params(self) -> List[float]:
        return [
            float(self.K_rgb[0][0]),
            float(self.K_rgb[1][1]),
            float(self.K_rgb[0][2]),
            float(self.K_rgb[1][2]),
        ]

    # ---------- Detection & saving ----------
    def _annotate(self, color, dets):
        disp = color.copy()
        for d in dets:
            c = d.corners.astype(int)
            cv2.polylines(disp, [c], True, (0, 255, 0), 2)
            ctr = tuple(d.center.astype(int))
            dist_mm = float(np.linalg.norm(d.pose_t)) * 1000.0
            cv2.putText(disp, f"ID:{d.tag_id}", (ctr[0]-20, ctr[1]-18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            cv2.putText(disp, f"{dist_mm:.0f}mm", (ctr[0]-25, ctr[1]+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
        return disp

    def detect_all_tags(self, show_image: bool, pose_idx: int) -> Dict[int, dict]:
        pose_dir = os.path.join(self.session_dir, "poses", f"pose_{pose_idx:02d}")
        os.makedirs(pose_dir, exist_ok=True)
        per_frame_records = []

        by_id: Dict[int, List] = {}
        for fidx in range(self.avg_frames):
            color, _ = self.cam.get_images()
            if color is None:
                continue
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            dets = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self._camera_params(),
                tag_size=self.tag_size_m,
            )

            if self.save_images:
                raw_path = os.path.join(pose_dir, f"frame_{fidx:02d}_raw.png")
                cv2.imwrite(raw_path, color)
                ann = self._annotate(color, dets)
                ann_path = os.path.join(pose_dir, f"frame_{fidx:02d}_ann.png")
                cv2.imwrite(ann_path, ann)

            # record detections
            frame_rec = []
            for d in dets:
                by_id.setdefault(d.tag_id, []).append(d)
                frame_rec.append({
                    "id": int(d.tag_id),
                    "center": [float(d.center[0]), float(d.center[1])],
                    "t": np.asarray(d.pose_t).reshape(-1).tolist(),
                    "R": np.asarray(d.pose_R).tolist(),
                    "corners": np.asarray(d.corners).reshape(-1,2).tolist()
                })
            per_frame_records.append({"frame": fidx, "detections": frame_rec})

            if show_image:
                ann_show = self._annotate(color, dets)
                cv2.imshow('AprilTags', ann_show)
                cv2.waitKey(1)
            time.sleep(0.04)

        if self.save_images:
            with open(os.path.join(pose_dir, "detections.json"), "w") as f:
                json.dump(per_frame_records, f, indent=2)

        averaged: Dict[int, dict] = {}
        for tid, lst in by_id.items():
            if tid not in self.tag_positions_mm:
                print(f"  Warn: detected tag {tid} not in TAG_POSITIONS_MM; skipping")
                continue
            R_list = [d.pose_R for d in lst]
            quats = [R.from_matrix(Ri).as_quat() for Ri in R_list]
            q = np.mean(quats, axis=0); q = q / np.linalg.norm(q)
            R_tag2cam = R.from_quat(q).as_matrix()
            t_list = [d.pose_t.reshape(3,1) for d in lst]
            t_tag2cam = np.mean(t_list, axis=0)

            averaged[tid] = {
                'R': R_tag2cam,
                't': t_tag2cam,
                'num_detections': len(lst),
                'world_pos': np.array(self.tag_positions_mm[tid], dtype=float) / 1000.0,
            }

        return averaged

    # ---------- Mapping helper ----------
    def suggest_mapping_by_image(self, frames: int = 12) -> None:
        accum: Dict[int, List[np.ndarray]] = {}
        for _ in range(frames):
            color, _ = self.cam.get_images()
            if color is None:
                continue
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            dets = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self._camera_params(),
                tag_size=self.tag_size_m,
            )
            for d in dets:
                accum.setdefault(d.tag_id, []).append(np.array(d.center, dtype=float))
            ann = self._annotate(color, dets)
            cv2.imshow('AprilTags (mapping)', ann); cv2.waitKey(1); time.sleep(0.04)
        cv2.destroyAllWindows()

        centers = []
        for tid, lst in accum.items():
            ctr = np.mean(np.stack(lst, axis=0), axis=0)
            centers.append((tid, float(ctr[0]), float(ctr[1])))
        centers.sort(key=lambda t: t[2])
        rows = [centers[0:2], centers[2:4], centers[4:6]]
        rows = [sorted(r, key=lambda t: t[1]) for r in rows]
        def id_or_none(r,i): return r[i][0] if i < len(r) else None
        tl, tr = id_or_none(rows[0],0), id_or_none(rows[0],1)
        ml, mr = id_or_none(rows[1],0), id_or_none(rows[1],1)
        bl, br = id_or_none(rows[2],0), id_or_none(rows[2],1)
        print("\nSuggested image-order mapping (TL/TR/ML/MR/BL/BR):")
        print(f"  Top    : L={tl}  R={tr}")
        print(f"  Middle : L={ml}  R={mr}")
        print(f"  Bottom : L={bl}  R={br}")

    # ---------- Data collection ----------
    def collect_at_pose(self, pose: List[float], pose_num: int, show_image: bool = True):
        print(f"\n-- Pose {pose_num}: {pose}")
        self._safe_move(pose)
        time.sleep(0.3)
        det = self.detect_all_tags(show_image=show_image, pose_idx=pose_num)
        if len(det) < 1:
            print("  No tags seen; skipping")
            return
        print(f"  Detected tags: {list(det.keys())}")
        for tid, d in det.items():
            self.measurements.append({
                'pose_mmdeg': np.array(pose, dtype=np.float64),
                'R_tag2cam': d['R'].astype(np.float64),
                't_tag2cam': d['t'].astype(np.float64),
                'tag_id': tid,
                'tag_world_pos': d['world_pos'].astype(np.float64),
                'pose_index': pose_num,
            })
        print(f"  Total measurements: {len(self.measurements)}")

    def collect(self, show_image: bool = True) -> int:
        print("Collecting calibration data...")
        for i, pose in enumerate(HOVER_OVER_TAG_POSES_MMDEG):
            self.collect_at_pose(pose, i, show_image)
        # Extra diverse views (your set)
        extras = [
            [346.1, -33,   442.6, 176.3, -9.5,   6.1],
            [371.8, -92.2, 215.1, 178.8, -1.9,  -1.3],
            [376.8,  69.2, 215.0, 178.7, -1.9,  11.3],
            [438.6, 107.7, 282.6, 178.5, -3.4,  14.8],
            [508.8,-107.2, 261.7, 178.7, -2.3, -10.8],
            [576.0,-113.3, 176.4, 178.9, -2.9, -10.1],
            [600.4,  51.5, 197.6, 178.8, -5.7,   5.9],
            [471.9,  -5.7, 384.9, 178.2, -4.0,   0.4],
            [318.3,  -1.8, 296.3, 178.1,-29.0,   0.7],
            [225.5,  -3.0, 137.0, 178.6, -1.6,   0.3],
            [259.6,-106.7, 273.1, 178.3,  0.3, -21.1],
            [467.5,-126.3, 151.1, 179.1, -0.9, -14.2],
            [323.6,-109.1, 150.1, 179.0, -0.1, -17.7],
            [461.1,  50.9, 217.9, 179.5, -2.4,   6.5],
            [472.1, -82.9, 242.2, 179.5, -5.5,  -9.7],
            [339.5,  17.6, 204.6, 179.4, -2.1,   3.0],
            [329.4, -84.5, 204.7, 179.5, -2.1, -14.3],
            [247.3,-148.0, -23.6, 175.2,  1.5, -14.6],
        ]
        base = len(HOVER_OVER_TAG_POSES_MMDEG)
        for i, pose in enumerate(extras, start=base):
            self.collect_at_pose(pose, i, show_image)
        cv2.destroyAllWindows()
        print(f"Done. Measurements collected: {len(self.measurements)}")
        return len(self.measurements)

    # ---------- Calibration ----------
    def _validate_world_error(self, R_c2g: np.ndarray, t_c2g: np.ndarray, euler_order: str = 'xyz') -> float:
        errs = []
        for m in self.measurements:
            p_cam = m['t_tag2cam']
            p_grip = R_c2g @ p_cam + t_c2g
            R_g2b, t_g2b = self._get_pose_rot_trans(m['pose_mmdeg'].tolist(), order=euler_order)
            p_world = R_g2b @ p_grip + t_g2b
            pred_mm = p_world.reshape(3) * 1000.0
            actual_mm = np.array(self.tag_positions_mm[m['tag_id']], dtype=float)
            errs.append(float(np.linalg.norm(pred_mm - actual_mm)))
        return float(np.mean(errs)) if errs else 1e9

    def calibrate(self) -> dict:
        print("\n=== Hand-Eye Calibration ===")
        tag_counts: Dict[int, int] = {}
        for m in self.measurements:
            tag_counts[m['tag_id']] = tag_counts.get(m['tag_id'], 0) + 1
        if not tag_counts:
            raise RuntimeError("No tag measurements available")
        dominant_tag = max(tag_counts.items(), key=lambda kv: kv[1])[0]
        num_dom = tag_counts[dominant_tag]
        print("Per-tag counts:", {k: tag_counts[k] for k in sorted(tag_counts.keys())})
        print(f"Using single tag ID {dominant_tag} for hand-eye (samples={num_dom})")
        if num_dom < 4:
            raise RuntimeError("Too few samples of one tag (need ≥4).")

        dom_meas = [m for m in self.measurements if m['tag_id'] == dominant_tag]
        order = 'xyz'
        R_g2b_all, t_g2b_all = [], []
        for m in dom_meas:
            Rg, tg = self._get_pose_rot_trans(m['pose_mmdeg'].tolist(), order=order)
            R_g2b_all.append(Rg); t_g2b_all.append(tg)

        R_t2c_all = [m['R_tag2cam'] for m in dom_meas]
        t_t2c_all = [m['t_tag2cam'] for m in dom_meas]
        R_c2t_all = [Rt.T for Rt in R_t2c_all]
        t_c2t_all = [-(Rt.T @ tt) for Rt, tt in zip(R_t2c_all, t_t2c_all)]

        candidates = []

        def try_one(R_tc, t_tc, tag_conv: str):
            R_c2g_raw, t_c2g_raw = cv2.calibrateHandEye(
                R_g2b_all, t_g2b_all, R_tc, t_tc, method=cv2.CALIB_HAND_EYE_PARK
            )
            errA = self._validate_world_error(R_c2g_raw, t_c2g_raw, euler_order=order)
            candidates.append({"variant":"as-doc","tag_conv":tag_conv,"error_mm":float(errA),
                               "R_c2g":np.asarray(R_c2g_raw).tolist(),"t_c2g":np.asarray(t_c2g_raw).reshape(3).tolist()})
            # Inverted possibility
            R_g2c = R_c2g_raw.T
            t_g2c = -R_g2c @ t_c2g_raw
            R_c2g_fix = R_g2c.T
            t_c2g_fix = -R_c2g_fix @ t_g2c
            errB = self._validate_world_error(R_c2g_fix, t_c2g_fix, euler_order=order)
            candidates.append({"variant":"inverted","tag_conv":tag_conv,"error_mm":float(errB),
                               "R_c2g":np.asarray(R_c2g_fix).tolist(),"t_c2g":np.asarray(t_c2g_fix).reshape(3).tolist()})

        try_one(R_t2c_all, t_t2c_all, "tag->cam")
        try_one(R_c2t_all, t_c2t_all, "cam->tag")

        # Save all candidates for offline diffing
        with open(os.path.join(self.session_dir, "calibration_candidates.json"), "w") as f:
            json.dump(candidates, f, indent=2)

        best = min(candidates, key=lambda c: c["error_mm"])
        print(f"  Best candidate: {best['variant']} using {best['tag_conv']}; mean world error = {best['error_mm']:.2f} mm")

        R_best = self._to_proper_rotation(np.asarray(best["R_c2g"]))
        t_best = np.asarray(best["t_c2g"]).reshape(3,1)
        eul_xyz = R.from_matrix(R_best).as_euler('xyz', degrees=False)

        return {
            'rotation_matrix': R_best.tolist(),
            'translation': t_best.reshape(3).tolist(),
            'euler_angles': eul_xyz.tolist(),
            'method': f"Park/{best['variant']}/{best['tag_conv']}",
            'error_mm': float(best["error_mm"]),
            'num_measurements': len(self.measurements),
            'num_tags_used': len(set(m['tag_id'] for m in self.measurements)),
            'config': {'euler_order': order, 'dominant_tag': dominant_tag},
        }

    # ---------- Test ----------
    def test_once(self, calib: dict, pose_mmdeg: List[float]):
        print("\n=== Test Calibration at pose ===")
        print(pose_mmdeg)
        self._safe_move(pose_mmdeg)
        time.sleep(0.3)
        color, _ = self.cam.get_images()
        det = self.detect_all_tags(show_image=True, pose_idx=999)  # uses saving
        if not det:
            print("No tags detected at test pose")
            return
        # re-draw on the original color so we have a single annotated test image
        ann = color.copy()
        R_c2g = np.array(calib['rotation_matrix'])
        t_c2g = np.array(calib['translation']).reshape(3,1)
        order = calib.get('config', {}).get('euler_order', 'xyz')
        R_g2b, t_g2b = self._get_pose_rot_trans(pose_mmdeg, order=order)

        print("ID |    Predicted (mm)            |     Actual (mm)           |  Error (mm)")
        print("-"*72)
        rows = []
        for tid, d in det.items():
            p_cam = d['t']
            p_grip = R_c2g @ p_cam + t_c2g
            p_world = R_g2b @ p_grip + t_g2b
            pred = p_world.reshape(3) * 1000.0
            actual = np.array(self.tag_positions_mm[tid], dtype=float)
            err = float(np.linalg.norm(pred - actual))
            rows.append([int(tid), float(pred[0]), float(pred[1]), float(pred[2]),
                         float(actual[0]), float(actual[1]), float(actual[2]), err])
            print(f"{tid:2d} | X={pred[0]:7.1f} Y={pred[1]:7.1f} Z={pred[2]:7.1f} | "
                  f"X={actual[0]:7.1f} Y={actual[1]:7.1f} Z={actual[2]:7.1f} | {err:7.2f}")

        test_dir = os.path.join(self.session_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        # Save raw test frame
        cv2.imwrite(os.path.join(test_dir, "test_raw.png"), color)
        # Save CSV report
        with open(os.path.join(test_dir, "report.csv"), "w") as f:
            f.write("id,pred_x,pred_y,pred_z,act_x,act_y,act_z,error_mm\n")
            for r in rows:
                f.write(",".join(map(str, r)) + "\n")

# -------------------- Main --------------------

def main():
    print("="*60)
    print("Multi-Tag AprilTag Hand-Eye Calibration (5.8 mm tags, tag36h11)")
    print("="*60)

    ap = argparse.ArgumentParser()
    ap.add_argument('--tag-size-mm', type=float, default=5.8,
                    help='AprilTag black-square edge length in millimeters (default 5.8)')
    ap.add_argument('--map-tags', action='store_true',
                    help='Suggest TL/TR/ML/MR/BL/BR mapping by observing image positions and exit')
    ap.add_argument('--auto-test', action='store_true',
                    help='After calibration, run a visibility test at a sample pose')
    ap.add_argument('--save-json', action='store_true',
                    help='Save hand-eye result to hand_eye_calibration.json (also saved in debug dir)')
    ap.add_argument('--out-dir', type=str, default='debug_runs',
                    help='Root directory for debug artifacts (images, JSON)')
    ap.add_argument('--no-save-images', action='store_true',
                    help='Disable saving images (raw/annotated)')
    ap.add_argument('--avg-frames', type=int, default=6,
                    help='Frames to average at each pose')
    args, _ = ap.parse_known_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.out_dir, ts)

    cal = MultiTagCalibrator(
        robot_ip=ROBOT_IP,
        tag_size_m=float(args.tag_size_mm) / 1000.0,
        tag_positions_mm=TAG_POSITIONS_MM,
        session_dir=session_dir,
        save_images=not args.no_save_images,
        avg_frames=args.avg_frames,
    )

    if args.map_tags:
        print("Moving to the 'see-everything' pose for mapping…")
        cal._safe_move(VIEW_ALL_POSE_MMDEG)
        time.sleep(0.4)
        print("Sampling a short sequence to infer TL/TR/ML/MR/BL/BR mapping…")
        cal.suggest_mapping_by_image(frames=15)
        print(f"Artifacts at: {session_dir}")
        return

    n = cal.collect(show_image=True)
    if n < 6:
        print(f"Warning: only {n} measurements collected. Aim for many views of the SAME tag (≥4).")

    result = cal.calibrate()
    print("\n=== Calibration Result ===")
    print(f"method: {result['method']}")
    print(f"error_mm: {result['error_mm']:.2f}")
    print(f"translation (m): {result['translation']}")
    print(f"euler_xyz (rad): {result['euler_angles']}")
    print(f"euler_xyz (deg): {np.rad2deg(result['euler_angles']).tolist()}")

    # Save outputs (both in CWD and session_dir for convenience)
    simple = {
        'translation': result['translation'],
        'rotation': result['euler_angles'],  # radians (xyz)
        'method': result['method'],
        'error_mm': result['error_mm'],
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    if args.save_json:
        with open('hand_eye_calibration.json', 'w') as f:
            json.dump(simple, f, indent=2)
        print("Saved to hand_eye_calibration.json")
    with open(os.path.join(session_dir, 'hand_eye_calibration.json'), 'w') as f:
        json.dump(simple, f, indent=2)

    if args.auto_test:
        cal.test_once(result, [350, 0, 230, 178, 0, 1])

    print(f"\nAll artifacts saved under: {session_dir}")

if __name__ == '__main__':
    main()
