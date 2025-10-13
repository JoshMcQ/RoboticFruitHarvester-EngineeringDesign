#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Example: Gripper Control
Please make sure that the gripper is attached to the end.
"""

import re
import serial

import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI
from configparser import ConfigParser
parser = ConfigParser()
parser.read('../robot.conf')
try:
    ip = parser.get('xArm', 'ip')
except:
    ip = input('Please input the xArm ip address[192.168.1.194]:')
    if not ip:
        ip = '192.168.1.221'


arm = XArmAPI(ip)
arm.motion_enable(True)
arm.clean_error()
arm.set_mode(0)
arm.set_state(0)
time.sleep(1)

code = arm.set_gripper_mode(0)
print('set gripper mode: location mode, code={}'.format(code))

code = arm.set_gripper_enable(True)
print('set gripper enable, code={}'.format(code))

code = arm.set_gripper_speed(5000)
print('set gripper speed, code={}'.format(code))

code = arm.set_gripper_position(600, wait=False)
print('[wait]set gripper pos, code={}'.format(code))

#code = arm.set_gripper_position(200, wait=False, speed=8000)
#print('[no wait]set gripper pos, code={}'.format(code))

FORCE_PORT = "COM5"
FORCE_BAUD = 9600
FORCE_THRESHOLD = 480.0  # adjust based on sensor units



def read_force(ser: serial.Serial) -> float | None:
	"""Read a single force measurement from serial, return None if unavailable."""
	try:
		line = ser.readline().decode("utf-8", errors="ignore").strip()
		if not line:
			return None
		match = re.search(r"(\d+(?:\.\d+)?)", line)
		if match:
			return float(match.group(1))
	except Exception as exc:
		print(f"[force] read error: {exc}")
	return None


def close_with_force_feedback(arm: XArmAPI) -> None:
	try:
		with serial.Serial(FORCE_PORT, FORCE_BAUD, timeout=0.1) as ser:
			ser.reset_input_buffer()
			print(f"[force] Monitoring {FORCE_PORT} with threshold {FORCE_THRESHOLD}")
			for position in range(850, -1, -10):
				arm.set_gripper_position(position, wait=True)
				force_val = read_force(ser)
				if force_val is not None:
					print(f"[force] pos={position} reading={force_val:.2f}")
					if force_val >= FORCE_THRESHOLD:
						print("[force] Threshold reached, stopping gripper close.")
						break
	except serial.SerialException as exc:
		print(f"[force] Serial unavailable ({exc}), running open-loop sweep.")
		for position in range(500, -1, -50):
			arm.set_gripper_position(position, wait=True)


#close_with_force_feedback(arm)