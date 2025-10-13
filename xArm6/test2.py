#!/usr/bin/env python3

import re
import time

import serial

from xarm.wrapper import XArmAPI

FORCE_PORT = "COM5"
FORCE_BAUD = 9600 
FORCE_THRESHOLD = 100.0 # adjust based on sensor units

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
			time.sleep(0.001)
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
			for position in range(850, -1, -15):
				arm.set_gripper_position(position, wait=False, speed=8000) #true
				position_deadline = time.time() + 0.004
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

# Smooth, adaptive, debounced close (works with your XArmAPI)
def smooth_close_with_force(arm: XArmAPI, ser: serial.Serial,
                            threshold=FORCE_THRESHOLD,
                            start=850, end=0):
    import math, time

    # --- Tunables ---
    EMA_ALPHA     = 0.25    # force smoothing (0..1)
    BASE_STEP     = 18      # coarse step far from contact
    MIN_STEP      = 4       # tiny step near contact
    V_MAX         = 9000    # r/min, far from contact
    V_MIN         = 1200    # r/min, near contact
    SAMPLE_DT     = 0.004   # s, force poll cadence
    WINDOW_S      = 0.03    # s, per-step window
    HITS_NEEDED   = 1       # debounce (filtered samples ≥ threshold)
    HYST_RATIO    = 0.92    # drop below this to clear hits
    DWELL_S       = 0.015   # brief settle to avoid stick-slip

    def smoothstep01(x):
        x = 0.0 if x < 0 else (1.0 if x > 1 else x)
        return x*x*(3 - 2*x)

    # Ensure gripper is on and in position (location) mode
    try:
        arm.set_gripper_enable(True)
        arm.set_gripper_mode(0)  # location mode
    except Exception:
        pass  # safe to proceed if firmware already set

    f_ema, hits = 0.0, 0
    pos = start
    while pos >= end:
        # Contact likelihood from filtered force → adaptive step & speed
        f_ratio = f_ema / max(1e-6, float(threshold))
        s = smoothstep01(f_ratio)                   # 0 (far) → 1 (at contact)
        step  = int(max(MIN_STEP, BASE_STEP*(1 - 0.8*s)))
        speed = int(V_MIN + (V_MAX - V_MIN)*(1 - s))

        # Command motion toward target pos (non-blocking)
        arm.set_gripper_position(pos, wait=False, speed=speed)

        t_end, got, last_val = time.time() + WINDOW_S, 0, None
        while time.time() < t_end:
            v = read_force(ser, timeout=SAMPLE_DT)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                last_val = v
                got += 1
                f_ema = EMA_ALPHA*v + (1-EMA_ALPHA)*f_ema
                if f_ema >= threshold:
                    hits += 1
                    if hits >= HITS_NEEDED:
                        print(f"[force] pos={pos} ema={f_ema:.2f} -> smooth stop")
                        # No explicit stop API; hold softly at current pos:
                        arm.set_gripper_position(pos, wait=True, speed=max(V_MIN, 800))
                        return
                elif f_ema < threshold*HYST_RATIO:
                    hits = 0
            time.sleep(SAMPLE_DT)

        if got == 0:
            print(f"[force] pos={pos} no reading")
        else:
            print(f"[force] pos={pos} ema={f_ema:.2f} speed={speed} step={step}")

        time.sleep(DWELL_S)
        pos -= step

    print("[force] Completed sweep without reaching threshold.")
    arm.set_gripper_position(end, wait=True, speed=V_MIN)


# arm = XArmAPI('192.168.1.202')
arm = XArmAPI('192.168.1.221')

time.sleep(0.5)
arm.set_mode(0)
arm.set_state(0)
arm.reset(wait=True)

#move to pick the object
arm.set_gripper_position(850, wait=True)
arm.set_position(*[347.2, -172.3, 237.3, 179.9, 0, 0.6], wait=True)##
arm.set_position(*[372.9, -185.7, 81.5, 180, 0, 0.5], wait=True)
smooth_close_with_force(arm, ser:=serial.Serial(FORCE_PORT, FORCE_BAUD, timeout=0.5))
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
