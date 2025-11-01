#!/usr/bin/env python3
"""
Test script to verify robot orientation convention.
Move robot to simple known poses and check if math matches reality.
"""
import time
from xarm.wrapper import XArmAPI
import numpy as np

arm = XArmAPI('192.168.1.221')
time.sleep(0.5)
arm.set_mode(0)
arm.set_state(0)
arm.reset(wait=True)

print("="*80)
print("ROBOT ORIENTATION TEST")
print("="*80)
print("\nThis will help determine xArm's Euler angle convention.")
print("\nTest 1: Move to pose with Roll=0, Pitch=0, Yaw=0")
print("Expected: EEF Z-axis points DOWN (toward table)")

# Move to upright pose
pose1 = [300, 0, 200, 0, 0, 0]  # Roll=Pitch=Yaw=0
print(f"\nMoving to: {pose1}")
arm.set_position(*pose1, wait=True)
input("Press Enter when you've observed the EEF orientation...")

print("\nTest 2: Move to pose with Roll=180°, Pitch=0, Yaw=0")
print("Expected: EEF flipped upside down")

pose2 = [300, 0, 200, 180, 0, 0]  # Roll=180
print(f"\nMoving to: {pose2}")
arm.set_position(*pose2, wait=True)
input("Press Enter when you've observed the EEF orientation...")

print("\nTest 3: Move to pose with Roll=90°, Pitch=0, Yaw=0")
print("Expected: EEF rotated 90° around its X-axis")

pose3 = [300, 0, 200, 90, 0, 0]  # Roll=90
print(f"\nMoving to: {pose3}")
arm.set_position(*pose3, wait=True)
input("Press Enter when you've observed the EEF orientation...")

print("\n" + "="*80)
print("VERIFICATION")
print("="*80)
print("\nNow verify: does your camera point straight DOWN when:")
print("A) Roll=0, Pitch=0, Yaw=0?")
print("B) Roll=180, Pitch=0, Yaw=0?")
print("C) Roll=0, Pitch=90, Yaw=0?")
print("D) Some other angle?")
print("\nOnce you know which angles make the camera point down,")
print("we can correctly set up the calibration.")

arm.reset(wait=True)
print("\nDone!")
