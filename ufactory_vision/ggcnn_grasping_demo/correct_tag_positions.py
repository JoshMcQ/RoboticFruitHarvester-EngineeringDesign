#!/usr/bin/env python3
"""
Calculate corrected TAG_POSITIONS_MM accounting for 150mm gripper tip offset.
The original measurements were EEF flange positions when gripper TIP touched each tag.
We need to add the gripper offset transformed through each measurement pose's rotation.
"""

import numpy as np

# Gripper tip offset in EEF local frame (from GRIPPER_Z_MM in run_depthai_grasp.py)
GRIPPER_OFFSET_EEF = np.array([0.0, 0.0, 150.0])  # mm along EEF Z-axis

def euler_deg_to_rot(roll_deg, pitch_deg, yaw_deg):
    """Rotation matrix from roll/pitch/yaw in degrees (ZYX order)."""
    rd, pd, yd = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(rd), np.sin(rd)
    cp, sp = np.cos(pd), np.sin(pd)
    cy, sy = np.cos(yd), np.sin(yd)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # ZYX intrinsic rotation

# Measurement data: (tag_id, eef_x, eef_y, eef_z, roll, pitch, yaw)
# These are the EEF FLANGE positions when gripper TIP touched each tag
measurements = [
    (0, 282.3,  44.6,  -89.3, -180.0,  0.5,  8.4),  # Tag 0 (BL)
    (3, 290.1, -118.3, -89.5,  178.3,  1.5, -0.6),  # Tag 3 (BR)
    (1, 445.4, -128.4, -86.0,  179.8,  4.0,  1.8),  # Tag 1 (ML)
    (5, 440.5,  45.3,  -86.3,  178.7,  3.6,  3.5),  # Tag 5 (MR)
    (2, 606.5,  54.0,  -83.2,  178.2, -0.4,  3.1),  # Tag 2 (TL)
    (4, 603.2, -131.6, -81.0,  177.9,  0.3,  1.3),  # Tag 4 (TR)
]

print("="*80)
print("CORRECTING TAG_POSITIONS_MM FOR GRIPPER TIP OFFSET")
print("="*80)
print(f"\nGripper tip offset in EEF frame: {GRIPPER_OFFSET_EEF} mm")
print("\nOriginal measurements were EEF FLANGE positions when TIP touched tags.")
print("True tag position = EEF flange position + R_base_to_eef @ gripper_offset\n")

corrected_positions = {}

for tag_id, x_eef, y_eef, z_eef, roll, pitch, yaw in measurements:
    # Build rotation matrix from base to EEF
    R_base_to_eef = euler_deg_to_rot(roll, pitch, yaw)
    
    # Transform gripper offset from EEF local frame to world frame
    gripper_offset_world = R_base_to_eef @ GRIPPER_OFFSET_EEF
    
    # True tag position = flange position + gripper offset in world frame
    eef_flange = np.array([x_eef, y_eef, z_eef])
    tag_position_world = eef_flange + gripper_offset_world
    
    corrected_positions[tag_id] = tag_position_world
    
    print(f"Tag {tag_id}:")
    print(f"  EEF flange (measured):     [{x_eef:7.1f}, {y_eef:7.1f}, {z_eef:7.1f}] mm")
    print(f"  Orientation:               Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°")
    print(f"  Gripper offset (world):    [{gripper_offset_world[0]:7.1f}, {gripper_offset_world[1]:7.1f}, {gripper_offset_world[2]:7.1f}] mm")
    print(f"  TRUE tag position:         [{tag_position_world[0]:7.1f}, {tag_position_world[1]:7.1f}, {tag_position_world[2]:7.1f}] mm")
    print()

print("="*80)
print("CORRECTED TAG_POSITIONS_MM FOR advanced_calibrate.py")
print("="*80)
print("\nTAG_POSITIONS_MM = {")
for tag_id in sorted(corrected_positions.keys()):
    pos = corrected_positions[tag_id]
    print(f"    {tag_id}: [{pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:7.1f}],")
print("}")

print("\n" + "="*80)
print("COMPARISON: Original vs Corrected")
print("="*80)

original_positions = {
    0: [282.3,   44.6,  -89.3],  # BL
    3: [290.1, -118.3,  -89.5],  # BR
    1: [445.4, -128.4,  -86.0],  # ML
    5: [440.5,   45.3,  -86.3],  # MR
    2: [606.5,   54.0,  -83.2],  # TL
    4: [603.2, -131.6,  -81.0],  # TR
}

print(f"{'Tag':<5} {'Original (mm)':<30} {'Corrected (mm)':<30} {'Difference (mm)'}")
print("-" * 90)
for tag_id in sorted(corrected_positions.keys()):
    orig = np.array(original_positions[tag_id])
    corr = corrected_positions[tag_id]
    diff = corr - orig
    
    orig_str = f"[{orig[0]:7.1f}, {orig[1]:7.1f}, {orig[2]:7.1f}]"
    corr_str = f"[{corr[0]:7.1f}, {corr[1]:7.1f}, {corr[2]:7.1f}]"
    diff_str = f"[{diff[0]:+7.1f}, {diff[1]:+7.1f}, {diff[2]:+7.1f}]"
    
    print(f"{tag_id:<5} {orig_str:<30} {corr_str:<30} {diff_str}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Copy the corrected TAG_POSITIONS_MM dict above")
print("2. Replace the TAG_POSITIONS_MM in advanced_calibrate.py (around line 53)")
print("3. Re-run calibration:")
print("   python advanced_calibrate.py --optimize --frames-per-pose 10 --use-forward-transform --optimize-translation-only")
print("4. Expected result: Mean error should drop significantly (<50mm target)")
print("="*80)
