#!/usr/bin/env python3

import re
import time

import serial

from xarm.wrapper import XArmAPI

FORCE_PORT = "COM5"
FORCE_BAUD = 9600
FORCE_THRESHOLD = 100.0  # adjust based on sensor units

def read_force(ser: serial.Serial, timeout: float = 0.6) -> float | None:
	"""Read until a numeric force value is observed or the timeout elapses."""
	deadline = time.time() + timeout
	first_text_line = None
	try:
		while time.time() < deadline:
			line = ser.readline().decode("utf-8", errors="ignore").strip()
			if not line:
				continue
			match = re.search(r"(\d+(?:\.\d+)?)", line)
			if match:
				return float(match.group(1))
			if first_text_line is None:
				first_text_line = line
			# allow a short breather before trying again to avoid busy looping
			time.sleep(0.02)
	except Exception as exc:
		print(f"[force] read error: {exc}")
	else:
		if first_text_line:
			print(f"[force] skipped text line: {first_text_line}")
	return None


def close_with_force_feedback(arm: XArmAPI) -> None:
	try:
		with serial.Serial(FORCE_PORT, FORCE_BAUD, timeout=0.5) as ser:
			ser.reset_input_buffer()
			print(f"[force] Monitoring {FORCE_PORT} with threshold {FORCE_THRESHOLD}")
			reached_threshold = False
			for position in range(850, -1, -25):
				arm.set_gripper_position(position, wait=True)
				position_deadline = time.time() + 0.35
				readings_this_step = 0
				while time.time() < position_deadline:
					force_val = read_force(ser)
					if force_val is None:
						continue
					readings_this_step += 1
					print(f"[force] pos={position} reading={force_val:.2f}")
					if force_val >= FORCE_THRESHOLD:
						print("[force] Threshold reached, stopping gripper close.")
						reached_threshold = True
						break
					# small pause so we don't overwhelm the serial buffer
					time.sleep(0.04)
				if readings_this_step == 0:
					print(f"[force] pos={position} no numeric reading within window")
				if reached_threshold:
					break
	except serial.SerialException as exc:
		print(f"[force] Serial unavailable ({exc}), running open-loop sweep.")
		for position in range(500, -1, -50):
			arm.set_gripper_position(position, wait=True)

# arm = XArmAPI('192.168.1.202')
arm = XArmAPI('192.168.1.221')

time.sleep(0.5)
arm.set_mode(0)
arm.set_state(0)
arm.reset(wait=True)

#move to pick the object
arm.set_gripper_position(850, wait=True)
arm.set_position(*[347.2, -172.3, 137.3, 179.9, 0, 0.6], wait=True)
arm.set_position(*[372.9, -185.7, 81.5, 180, 0, 0.5], wait=True)
close_with_force_feedback(arm)
#arm.set_gripper_position(300, wait=True)

#set tcp payload
arm.set_tcp_load(0.3, [0, 0, 30])
arm.set_state(0)

#move to place the object
arm.set_position(*[300, 0, 400, 180, 0, 0], wait=True)
# arm.set_position(*[300, -150, 400, 180, 0, 0], wait=True)
# arm.set_position(*[300, -150, 300, 180, 0, 0], wait=True)

arm.set_position(*[372.9, 120, 81.5, 180, 0, 0.5], wait=True)
arm.set_gripper_position(850, wait=True)
arm.set_tcp_load(0, [0, 0, 30])
arm.set_state(0)
arm.set_position(*[300, -150, 400, 180, 0, 0], wait=True)
arm.set_gripper_position(850, wait=True)
arm.reset(wait=True)
