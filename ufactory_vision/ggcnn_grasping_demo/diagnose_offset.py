#!/usr/bin/env python3
"""
Diagnose camera offset by computing it from actual measurements.
"""
import numpy as np

# From your verification run - Pose 3
eef_world_pos = np.array([245.1, 23.6, -45.0]) / 1000.0  # Convert to meters
eef_world_rpy_deg = np.array([-179.3, 6.3, 3.2])

# Tag detection
tag0_world_pos = np.array([282.3, 44.6, -89.3]) / 1000.0  # meters
camera_sees_tag_at = np.array([0.004, 0.000, 0.013])  # meters in camera frame

# Camera's world position (assuming camera Z-axis points toward tag)
# If camera pointing down, camera Z+ points down, so:
# camera_world_Z = tag_world_Z + camera_distance
camera_world_z = tag0_world_pos[2] + camera_sees_tag_at[2]

# Camera XY in world (approximately tag XY + small offset from camera detection)
# This is approximate because we don't know exact camera orientation yet
camera_world_x = tag0_world_pos[0] + camera_sees_tag_at[0]
camera_world_y = tag0_world_pos[1] + camera_sees_tag_at[1]

camera_world_pos = np.array([camera_world_x, camera_world_y, camera_world_z])

print("="*80)
print("CAMERA OFFSET DIAGNOSIS")
print("="*80)
print(f"\nEEF world position: {eef_world_pos * 1000} mm")
print(f"EEF orientation (deg): Roll={eef_world_rpy_deg[0]:.1f}, Pitch={eef_world_rpy_deg[1]:.1f}, Yaw={eef_world_rpy_deg[2]:.1f}")
print(f"\nTag 0 world position: {tag0_world_pos * 1000} mm")
print(f"Camera sees tag at (camera frame): {camera_sees_tag_at * 1000} mm")
print(f"\nEstimated camera world position: {camera_world_pos * 1000} mm")

offset_world = camera_world_pos - eef_world_pos
print(f"\nOffset in WORLD frame: {offset_world * 1000} mm")

# Now compute offset in EEF LOCAL frame
# Build rotation matrix for EEF orientation
roll_rad, pitch_rad, yaw_rad = np.deg2rad(eef_world_rpy_deg)
cr, sr = np.cos(roll_rad), np.sin(roll_rad)
cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)

Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
R_world_to_eef = Rz @ Ry @ Rx

# Transform offset from world frame to EEF local frame
# offset_eef_local = R^T @ offset_world
offset_eef_local = R_world_to_eef.T @ offset_world

print(f"\nOffset in EEF LOCAL frame: {offset_eef_local * 1000} mm")
print(f"\nThis should be close to your physical measurement:")
print(f"  X (to the side): {offset_eef_local[0] * 1000:.1f} mm (you said ~5mm)")
print(f"  Y (forward?):    {offset_eef_local[1] * 1000:.1f} mm")
print(f"  Z (up):          {offset_eef_local[2] * 1000:.1f} mm (you said ~10mm)")

print("\n" + "="*80)
print("RECOMMENDED INITIAL GUESS for EULER_EEF_TO_COLOR_OPT:")
print("="*80)
print(f"Translation: [{offset_eef_local[0]:.6f}, {offset_eef_local[1]:.6f}, {offset_eef_local[2]:.6f}]  # meters")
print(f"Rotation: Camera pointing down means approx [0, 0, pi/2] if camera frame aligned with EEF")
print("\nUpdate your EULER_EEF_TO_COLOR_OPT_INIT in advanced_calibrate.py to:")
print(f"EULER_EEF_TO_COLOR_OPT_INIT = [{offset_eef_local[0]:.6f}, {offset_eef_local[1]:.6f}, {offset_eef_local[2]:.6f}, 0, 0, 1.5708]")
