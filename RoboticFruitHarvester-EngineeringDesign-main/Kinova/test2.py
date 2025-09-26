import os
import sys
import time
import threading
import re

# --- Arduino force sensor integration (START) ---
import serial

# Accepts either "Reading: 123" or "F=123"
READING_RE = re.compile(r"^\s*(?:Reading:|F=)\s*(\d+)\s*\)?\s*$")

def _parse_force_line(line: str):
    m = READING_RE.match(line)
    return int(m.group(1)) if m else None

# NOTE: Base_pb2 is imported below; these functions won't run until after import.
def _gripper_speed(base, speed_01):
    from kortex_api.autogen.messages import Base_pb2
    cmd = Base_pb2.GripperCommand()
    cmd.mode = Base_pb2.GRIPPER_SPEED
    f = cmd.gripper.finger.add()
    f.finger_identifier = 1
    f.value = float(speed_01)  # +close / -open
    base.SendGripperCommand(cmd)

def _gripper_pos_measured(base):
    from kortex_api.autogen.messages import Base_pb2
    req = Base_pb2.GripperRequest()
    req.mode = Base_pb2.GRIPPER_POSITION
    meas = base.GetMeasuredGripperMovement(req)
    return (meas.finger[0].value if len(meas.finger) else None)

def close_touch_then_wait_force(base, ser,
                                close_speed=0.20,          # gentle close (0..1)
                                touch_eps=0.0025,          # plateau tolerance on measured pos
                                touch_hits=6,              # consecutive plateaus = touch
                                touch_max_time=6.0,        # max seconds to search for touch
                                force_threshold=500,       # Arduino threshold
                                force_window=8.0,          # seconds to wait after touch
                                backoff_open_time=0.06):   # small open after threshold
    """
    Phase 1: Close slowly until the gripper's *measured* position plateaus (touch/contact).
    Phase 2: Stop and wait up to `force_window` for Arduino force >= threshold.
    Returns: 'touched_and_forced' | 'touched_no_force' | 'no_touch'
    """
    print("[Touch] closing slowly to detect contact...")
    _gripper_speed(base, +close_speed)

    hits = 0
    t0 = time.time()
    prev = None
    while (time.time() - t0) < touch_max_time:
        time.sleep(0.06)  # ~16 Hz poll; don't hammer the API
        cur = _gripper_pos_measured(base)
        if cur is None:
            continue
        if prev is not None:
            if abs(cur - prev) < touch_eps:
                hits += 1
            else:
                hits = 0
        prev = cur
        if hits >= touch_hits:
            print(f"[Touch] contact detected at measured={cur:.3f}")
            break
    else:
        # never broke -> no touch
        base.Stop()
        print("[Touch] no contact before timeout")
        return "no_touch"

    # Stop motion at touch point
    base.Stop()

    # ---- Phase 2: wait for external force threshold (Arduino) ----
    print(f"[Force] waiting up to {force_window:.1f}s for Arduino >= {force_threshold}...")
    t1 = time.time()
    consec = 0
    need = 3  # require a few consecutive frames at/above threshold to avoid noise
    while (time.time() - t1) < force_window:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        val = _parse_force_line(line)
        if val is None:
            # Optional: uncomment to see all serial lines
            # print(f"[Force][raw] {line}")
            continue
        # print(f"[Force] {val}")  # verbose if needed
        if val >= force_threshold:
            consec += 1
            if consec >= need:
                print(f"[Force] threshold reached: {val}")
                if backoff_open_time > 0:
                    _gripper_speed(base, -0.25)  # tiny open nudge
                    time.sleep(backoff_open_time)
                    base.Stop()
                return "touched_and_forced"
        else:
            consec = 0

    print("[Force] window elapsed without threshold")
    return "touched_no_force"
# --- Arduino force sensor integration (END) ---

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2

TIMEOUT_DURATION = 30.0

def check_for_end_or_abort(e):
    def _cb(notif, e=e):
        print("EVENT:", Base_pb2.ActionEvent.Name(notif.action_event))
        if notif.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
            e.set()
    return _cb

def populate_cartesian(waypoint):
    x, y, z, blending, tx, ty, tz = waypoint
    w = Base_pb2.CartesianWaypoint()
    w.pose.x = x
    w.pose.y = y
    w.pose.z = z
    w.pose.theta_x = tx
    w.pose.theta_y = ty
    w.pose.theta_z = tz
    w.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    w.blending_radius = blending
    return w

def execute_waypoints(base: BaseClient, waypoints_def):
    # Set servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # Build list
    wl = Base_pb2.WaypointList()
    wl.duration = 0.0
    wl.use_optimal_blending = False

    for idx, wp_def in enumerate(waypoints_def):
        wp = wl.waypoints.add()
        wp.name = f"waypoint_{idx}"
        wp.cartesian_waypoint.CopyFrom(populate_cartesian(wp_def))

    # Validate (optional but useful)
    result = base.ValidateWaypointList(wl)
    if len(result.trajectory_error_report.trajectory_error_elements):
        print("Trajectory validation failed:")
        result.trajectory_error_report.PrintDebugString()
        return False

    # Execute and wait
    done_evt = threading.Event()
    handle = base.OnNotificationActionTopic(check_for_end_or_abort(done_evt), Base_pb2.NotificationOptions())
    print("Moving cartesian trajectory...")
    base.ExecuteWaypointTrajectory(wl)

    finished = done_evt.wait(TIMEOUT_DURATION)
    base.Unsubscribe(handle)
    if not finished:
        print("Timeout waiting for trajectory to finish")
    return finished

def gripper_set_position(base: BaseClient, position: float, force_threshold=500, ser=None):
    """
    If called with position >= 0.9 and a serial port, run the two-phase routine:
    1) close slowly until *touch* (firmware-measured plateau), then
    2) wait for Arduino force threshold.
    Otherwise, just command the target position.
    Returns True if force threshold confirmed, False otherwise.
    """
    if position >= 0.9 and ser is not None:
        result = close_touch_then_wait_force(
            base, ser,
            close_speed=0.20,         # tune as needed
            touch_eps=0.0025,
            touch_hits=6,
            touch_max_time=6.0,
            force_threshold=force_threshold,
            force_window=8.0,
            backoff_open_time=0.06
        )
        if result == "touched_and_forced":
            return True
        elif result == "touched_no_force":
            print("Warning: Touched object but no external force within window.")
            return False
        else:
            print("Warning: No touch detected before timeout.")
            return False

    # Normal open/close (no force logic)
    cmd = Base_pb2.GripperCommand()
    cmd.mode = Base_pb2.GRIPPER_POSITION
    finger = cmd.gripper.finger.add()
    finger.finger_identifier = 1
    finger.value = position
    base.SendGripperCommand(cmd)
    time.sleep(0.5)
    return True

def main():
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        # Create Base client from router (required to send commands)
        base = BaseClient(router)

        # Your exact waypoints
        first_route = (
            (0.497, -0.024, 0.263, 0.0, 98.035,  4.639, 123.538),  # 1: approach
            (0.495, -0.023, 0.189, 0.0, 102.728, -3.498, 123.519), # 2: lower (grip here)
            (0.460, -0.170, 0.325, 0.0, 94.091, 11.431, 105.063),  # 3: lift
            (0.462, -0.171, 0.169, 0.0, 103.837, -5.858, 105.634), # 4: place (optional)
            (0.465, -0.174, 0.277, 0.0, 97.079, 6.004, 105.651),   # 5: lift after place (optional)
            (0.13,  -0.068, 0.118, 0.0, 10.777, 177.812, 82.762),  # 6: Retract
        )

        gripper_closed = False
        try:
            # Open serial connection once for all grips
            # Make sure this matches your Arduino Serial.begin(....)
            ser = serial.Serial('COM3', 9600, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset

            # Optional: open gripper before approach
            print("Opening gripper...")
            gripper_set_position(base, 0.0)

            # Move to first two waypoints (approach → lower)
            if not execute_waypoints(base, first_route[:2]):
                ser.close()
                return 1

            # Close gripper with touch-then-force behavior
            print("Starting touch-then-force gripping using Arduino force sensor...")
            FORCE_THRESHOLD = 500
            if gripper_set_position(base, 1.0, force_threshold=FORCE_THRESHOLD, ser=ser):
                gripper_closed = True
            else:
                print("Warning: Gripper touch/force phase did not confirm threshold")

            # Execute the next waypoints (lift → place)
            if not execute_waypoints(base, first_route[2:4]):
                ser.close()
                return 1

            # Brief settle before release to ensure stability at place pose
            time.sleep(0.5)

            print("Opening gripper to release...")
            gripper_set_position(base, 0.0)
            gripper_closed = False

            # Lift-away after release (waypoint 5) to clear above the placed object
            if not execute_waypoints(base, [first_route[4]]):
                ser.close()
                return 1

            # Return to the place pose (waypoint 4) to re-pick the object
            if not execute_waypoints(base, [first_route[3]]):
                ser.close()
                return 1

            print("Closing gripper to re-pick at place (touch-then-force)...")
            if gripper_set_position(base, 1.0, force_threshold=FORCE_THRESHOLD, ser=ser):
                gripper_closed = True
            else:
                print("Warning: Re-pick touch/force phase did not confirm threshold")

            # Transit back to the original area: go to original lift (waypoint 3)
            if not execute_waypoints(base, [first_route[2]]):
                ser.close()
                return 1

            # Lower to the original lower pose (waypoint 2) to re-place at origin
            if not execute_waypoints(base, [first_route[1]]):
                ser.close()
                return 1

            # Brief settle before re-release
            time.sleep(0.5)

            print("Opening gripper to release at original location...")
            gripper_set_position(base, 0.0)
            gripper_closed = False

            # Lift back up to original lift (waypoint 3) to clear
            if not execute_waypoints(base, [first_route[2]]):
                ser.close()
                return 1

            # Finally, retract to safe pose (waypoint 6)
            if not execute_waypoints(base, [first_route[5]]):
                ser.close()
                return 1

            print("Done.")
            ser.close()
            return 0
        finally:
            # Ensure the gripper is opened at the end even if an error occurred after closing
            if gripper_closed:
                try:
                    print("Finalizing: opening gripper at end...")
                    gripper_set_position(base, 0.0)
                except Exception as e:
                    print(f"Finalization gripper open failed: {e}")

if __name__ == "__main__":
    sys.exit(main())
