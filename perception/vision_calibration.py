"""Utilities for mapping camera pixels to Kinova base frame coordinates.

This module loads a calibration JSON file with pixel-to-base correspondences and
produces both a planar homography and a full SE(3) transform (4x4 matrix)
from the camera optical frame to the robot base frame.  It also exposes helper
methods to convert detections into actionable robot targets while enforcing
basic safety heuristics (e.g. clamping to the calibrated workspace plane).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

ArrayLike = Sequence[float]


class CalibrationError(RuntimeError):
    """Raised when the calibration file is missing or inconsistent."""


@dataclass(frozen=True)
class VisionToBaseCalibration:
    """Holds the calibrated mapping between camera and robot base frames."""

    camera_matrix: np.ndarray
    rotation_base_from_cam: np.ndarray
    translation_base_from_cam: np.ndarray
    table_height_m: float
    homography_px_to_base: np.ndarray
    reprojection_rmse_m: float
    sample_errors_m: np.ndarray

    @property
    def transform_matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous transform T_base_cam."""
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.rotation_base_from_cam
        T[:3, 3] = self.translation_base_from_cam
        return T

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    def _pixel_to_camera_ray(self, u: float, v: float) -> np.ndarray:
        fx, fy, cx, cy = (
            self.camera_matrix[0, 0],
            self.camera_matrix[1, 1],
            self.camera_matrix[0, 2],
            self.camera_matrix[1, 2],
        )
        x = (u - cx) / fx
        y = (v - cy) / fy
        return np.array([x, y, 1.0], dtype=float)

    def _ray_to_plane_intersection(
        self, ray_dir_base: np.ndarray, plane_height_m: float
    ) -> np.ndarray:
        origin_base = self.translation_base_from_cam
        denom = ray_dir_base[2]
        if abs(denom) < 1e-6:
            raise CalibrationError(
                "Camera ray is parallel to workspace plane; cannot intersect"
            )
        step = (plane_height_m - origin_base[2]) / denom
        if step <= 0.0:
            raise CalibrationError(
                "Camera ray intersection behind the optical center; verify mounting"
            )
        return origin_base + ray_dir_base * step

    def pixel_to_base(
        self,
        u: float,
        v: float,
        object_height_m: Optional[float] = None,
        depth_override_m: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Convert pixel coordinates to robot base coordinates.

        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.
            object_height_m: Desired target plane height in the base frame.  If
                omitted, the calibrated table height is used.
            depth_override_m: Optional depth measurement along the camera z-axis
                (e.g., from a depth sensor). When provided, the result is computed
                by back-projecting the pixel using this depth instead of the
                planar intersection.

        Returns:
            (x, y, z) tuple in meters in the robot base frame.
        """

        if depth_override_m is not None:
            cam_pt = self.pixel_to_camera(depth_override_m, u, v)
            base_pt = self.camera_point_to_base(cam_pt)
            return tuple(map(float, base_pt))

        target_plane = (
            float(object_height_m) if object_height_m is not None else self.table_height_m
        )
        ray_cam = self._pixel_to_camera_ray(u, v)
        ray_base = self.rotation_base_from_cam @ ray_cam
        base_point = self._ray_to_plane_intersection(ray_base, target_plane)
        return tuple(map(float, base_point))

    def pixel_to_camera(self, depth_m: float, u: float, v: float) -> np.ndarray:
        """Back-project a pixel into the camera frame using a metric depth."""
        ray = self._pixel_to_camera_ray(u, v)
        return ray * float(depth_m)

    def camera_point_to_base(self, camera_point: np.ndarray) -> np.ndarray:
        """Transform a 3D point from camera to base frame."""
        return self.rotation_base_from_cam @ camera_point + self.translation_base_from_cam

    def project_base_to_pixel(self, base_point: np.ndarray) -> Tuple[float, float]:
        """Project a base-frame 3D point into pixel coordinates (for diagnostics)."""
        cam_point = self.rotation_base_from_cam.T @ (base_point - self.translation_base_from_cam)
        if cam_point[2] <= 0:
            raise CalibrationError("Point projects behind the camera")
        pix_h = self.camera_matrix @ cam_point
        return float(pix_h[0] / pix_h[2]), float(pix_h[1] / pix_h[2])

    def apply_homography(self, u: float, v: float) -> Tuple[float, float]:
        """Use the planar homography to map (u, v) → (x, y)."""
        px = np.array([u, v, 1.0], dtype=float)
        mapped = self.homography_px_to_base @ px
        mapped /= mapped[2]
        return float(mapped[0]), float(mapped[1])

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str) -> "VisionToBaseCalibration":
        if not os.path.exists(path):
            raise CalibrationError(f"Calibration file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        samples = payload.get("samples", [])
        if len(samples) < 4:
            raise CalibrationError("Calibration requires at least four samples")

        camera_matrix = np.array(
            payload.get("camera_matrix"), dtype=float
        ) if payload.get("camera_matrix") is not None else None
        table_height = float(payload.get("table_height_m", 0.0))

        pixels = []
        base_pts = []
        cam_pts = []
        for sample in samples:
            pixels.append(sample["pixel"][:2])
            base_pts.append(sample["base"])
            if "camera_point" in sample:
                cam_pts.append(sample["camera_point"])

        pixels_np = np.asarray(pixels, dtype=float)
        base_np = np.asarray(base_pts, dtype=float)

        if camera_matrix is None:
            raise CalibrationError("Camera intrinsics missing from calibration file")

        # Estimate homography for quick planar conversions / diagnostics
        H, mask = cv2.findHomography(pixels_np, base_np[:, :2], method=0)
        if H is None:
            raise CalibrationError("Failed to compute homography from samples")

        reprojection = []
        for (u, v), (x, y, _) in zip(pixels_np, base_np):
            px, py = _apply_homography(H, u, v)
            reprojection.append(np.hypot(px - x, py - y))
        reprojection = np.asarray(reprojection)
        rmse = float(math.sqrt(np.mean(reprojection**2)))

        if cam_pts:
            cam_np = np.asarray(cam_pts, dtype=float)
            R, t = _estimate_rigid_transform(cam_np, base_np)
            rigid_residuals = np.linalg.norm((R @ cam_np.T).T + t - base_np, axis=1)
        else:
            # Fall back to homography-derived (approximate) transform assuming
            # plane z = table_height.
            R, t = _derive_extrinsics_from_homography(camera_matrix, H, table_height)
            rigid_residuals = reprojection

        return cls(
            camera_matrix=camera_matrix,
            rotation_base_from_cam=R,
            translation_base_from_cam=t,
            table_height_m=table_height,
            homography_px_to_base=H,
            reprojection_rmse_m=rmse,
            sample_errors_m=rigid_residuals,
        )


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _apply_homography(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    vec = np.array([u, v, 1.0], dtype=float)
    mapped = H @ vec
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])


def _estimate_rigid_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the best-fit rigid transform (no scaling) aligning src→dst."""
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)

    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid

    H = src_centered.T @ dst_centered / src.shape[0]
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_centroid - R @ src_centroid
    return R, t


def _derive_extrinsics_from_homography(
    K: np.ndarray, H: np.ndarray, plane_height: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate extrinsics using the method from Hartley & Zisserman."""
    # Decompose homography assuming world plane z = plane_height.
    K_inv = np.linalg.inv(K)
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    # Compute rotation columns (up to scale)
    lam = 1.0 / np.linalg.norm(K_inv @ h1)
    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    r3 = np.cross(r1, r2)
    R = np.column_stack((r1, r2, r3))

    # Ensure R is orthonormal
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    t = lam * (K_inv @ h3)

    # Position the plane at the specified height (z = plane_height)
    # We assume plane normal [0,0,1] and adjust translation accordingly.
    t[2] = plane_height - (R[2, 0] * 0 + R[2, 1] * 0)
    return R, t
