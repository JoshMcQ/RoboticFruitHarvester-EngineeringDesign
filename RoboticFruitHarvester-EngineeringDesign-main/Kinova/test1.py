import os
import sys
import time
import threading

# --- Arduino force sensor integration (START) ---
import serial

def grip_with_force_feedback(base, ser, threshold=300, timeout=15):
    """
    Close gripper with force feedback.
    Returns True if threshold reached, False if no object (plateau or timeout).
    """
    try:
        print("[DEBUG] Entered grip_with_force_feedback")
        ser.reset_input_buffer()
        t0 = time.time()

        position = 0.0
        step = 0.05
        max_pos = 1.0

        threshold_reached = False
        plateau_detected = False

        print("Closing gripper with force feedback (smooth)...")
        prev_measured = None
        stable_count = 0
        STABLE_EPS = 0.003
        STABLE_NEEDED = 5       # consecutive near-equal measured positions = fingertips touched/nothing inside
        OPEN_NUDGE_SPEED = -0.25
        OPEN_NUDGE_TIME  = 0.08

        while position <= max_pos and (time.time() - t0) < timeout:
            print(f"[DEBUG] Setting gripper position: {position:.2f}")
            cmd = Base_pb2.GripperCommand()
            cmd.mode = Base_pb2.GRIPPER_POSITION
            finger = cmd.gripper.finger.add()
            finger.finger_identifier = 1
            finger.value = position
            base.SendGripperCommand(cmd)

            # let mechanics settle before reading
            time.sleep(0.2)

            # ----- Check measured gripper movement (plateau near closed) -----
            req = Base_pb2.GripperRequest()
            req.mode = Base_pb2.GRIPPER_POSITION
            meas = base.GetMeasuredGripperMovement(req)
            if len(meas.finger):
                measured = meas.finger[0].value
                print(f"[DEBUG] Measured gripper position: {measured:.3f}")

                # Only check plateau if nearly closed
                if position > 0.90:
                    if prev_measured is not None and abs(measured - prev_measured) < STABLE_EPS:
                        stable_count += 1
                    else:
                        stable_count = 0
                    prev_measured = measured

                    if stable_count >= STABLE_NEEDED:
                        print("[DEBUG] Gripper plateau detected (fingers touched, no object)")
                        plateau_detected = True
                        break

            # ----- Read a few serial lines for force -----
            for _ in range(3):
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"[DEBUG] Serial line: {line}")
                if "Reading:" in line:                 # keep this to match your Arduino format
                    try:
                        value = int(line.split("Reading:")[1].replace(")", "").strip())
                        print(f"[Arduino] Force reading: {value}")
                        if value >= threshold:
                            print(f"[Arduino] Threshold {threshold} reached!")
                            threshold_reached = True
                            break
                    except ValueError:
                        print(f"[DEBUG] Could not parse force value from: {line}")
                        continue

            if threshold_reached:
                break

            position = min(max_pos, position + step)

        # ================== FINALIZATION (the important fix) ==================
        if threshold_reached:
            # Hard stop then back off slightly so you’re not squeezing
            base.Stop()
            cmd = Base_pb2.GripperCommand()
            cmd.mode = Base_pb2.GRIPPER_SPEED
            f = cmd.gripper.finger.add(); f.finger_identifier = 1; f.value = OPEN_NUDGE_SPEED
            base.SendGripperCommand(cmd); time.sleep(OPEN_NUDGE_TIME); base.Stop()
            print(f"[DEBUG] Exiting grip_with_force_feedback, threshold_reached=True")
            return True

        if plateau_detected:
            # Do NOT go to max_pos. Hold here or back off a hair.
            base.Stop()
            # optional: open ~0.02 for clearance
            hold_pos = max(0.0, position - 0.02)
            cmd = Base_pb2.GripperCommand()
            cmd.mode = Base_pb2.GRIPPER_POSITION
            finger = cmd.gripper.finger.add()
            finger.finger_identifier = 1
            finger.value = hold_pos
            base.SendGripperCommand(cmd)
            time.sleep(0.2)
            print("[DEBUG] Exiting grip_with_force_feedback, no object (plateau).")
            return False

        # Timeout or reached max_pos without threshold: treat as no-object
        base.Stop()
        # optional: don’t clamp shut; slightly open for safety
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = max(0.0, min(position, 0.95))  # keep near current, not hard-closed
        base.SendGripperCommand(cmd)
        time.sleep(0.2)
        print("[DEBUG] Exiting grip_with_force_feedback, no threshold (timeout or max).")
        return False

    except Exception as e:
        print(f"[Arduino] Serial error: {e}")
        return False

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
    # position in [0.0 (open), 1.0 (closed)]
    if position >= 0.9 and ser is not None:
        # Use force feedback for closing
        return grip_with_force_feedback(base, ser, threshold=force_threshold)
    else:
        # Use normal open/close
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
            (0.465, -0.174, 0.277, 0.0, 97.079, 6.004, 105.651),# 5: lift after place (optional)
            (0.13, -0.068, 0.118, 0.0, 10.777, 177.812, 82.762),# 6: Retract 
        )
        
        


        gripper_closed = False
        try:
            # Open serial connection once for all grips
            ser = serial.Serial('COM3', 9600, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset

            # Optional: open gripper before approach
            print("Opening gripper...")
            gripper_set_position(base, 0.0)

            # Move to first two waypoints (approach → lower)
            if not execute_waypoints(base, first_route[:2]):
                ser.close()
                return 1

            # Close gripper with force feedback
            print("Starting force-controlled gripping using Arduino force sensor...")
            FORCE_THRESHOLD = 500
            if gripper_set_position(base, 1.0, force_threshold=FORCE_THRESHOLD, ser=ser):
                gripper_closed = True
            else:
                print("Warning: Gripper close timed out or threshold not reached")

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

            print("Closing gripper to re-pick at place (force feedback)...")
            if gripper_set_position(base, 1.0, force_threshold=FORCE_THRESHOLD, ser=ser):
                gripper_closed = True
            else:
                print("Warning: Gripper close (re-pick) timed out or threshold not reached")

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