import os
import sys
import time
import threading

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

def gripper_set_position(base: BaseClient, position: float):
    # position in [0.0 (open), 1.0 (closed)]
    cmd = Base_pb2.GripperCommand()
    cmd.mode = Base_pb2.GRIPPER_POSITION
    finger = cmd.gripper.finger.add()
    finger.finger_identifier = 1
    finger.value = position
    base.SendGripperCommand(cmd)

    # Wait until position is reached (or timeout)
    req = Base_pb2.GripperRequest()
    req.mode = Base_pb2.GRIPPER_POSITION
    # Heuristics for "close on contact"
    POLL_S = 0.2
    STABLE_EPS = 0.003          # position change considered "no movement"
    STABLE_TIME_S = 1.0         # how long it must be stable to consider contact
    MIN_MOVEMENT_FOR_CONTACT = 0.05  # ensure the gripper actually moved some
    stable_needed = max(1, int(STABLE_TIME_S / POLL_S))
    close_mode = position >= 0.9
    prev_val = None
    first_val = None
    stable_count = 0

    t0 = time.time()
    while time.time() - t0 < 10.0:
        meas = base.GetMeasuredGripperMovement(req)
        if len(meas.finger):
            val = meas.finger[0].value
            if first_val is None:
                first_val = val

            # Success when within tolerance of the target (works for open or close)
            if abs(val - position) < 0.02:
                return True

            # Close-on-contact: if closing and position stops changing for a while,
            # treat as success (object contact likely prevented reaching 1.0)
            if prev_val is not None:
                if abs(val - prev_val) < STABLE_EPS:
                    stable_count += 1
                else:
                    stable_count = 0
            else:
                stable_count = 0
            prev_val = val

            if close_mode:
                moved_enough = abs(val - first_val) > MIN_MOVEMENT_FOR_CONTACT
                if stable_count >= stable_needed and moved_enough:
                    print("Gripper contact detected (plateau); treating as closed.")
                    return True

        time.sleep(POLL_S)
    return False

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
            # Optional: open gripper before approach
            print("Opening gripper...")
            gripper_set_position(base, 0.0)

            # Move to first two waypoints (approach → lower)
            if not execute_waypoints(base, first_route[:2]):
                return 1

            # Close gripper at the lower pose
            print("Closing gripper...")
            if not gripper_set_position(base, 1.0):
                print("Warning: Gripper close timed out")
            else:
                gripper_closed = True

            # Execute the next waypoints (lift → place)
            if not execute_waypoints(base, first_route[2:4]):
                return 1

            # Brief settle before release to ensure stability at place pose
            time.sleep(0.5)

            print("Opening gripper to release...")
            if not gripper_set_position(base, 0.0):
                print("Warning: Gripper open timed out")
            else:
                gripper_closed = False

            # Lift-away after release (waypoint 5) to clear above the placed object
            if not execute_waypoints(base, [first_route[4]]):
                return 1

            # Return to the place pose (waypoint 4) to re-pick the object
            if not execute_waypoints(base, [first_route[3]]):
                return 1

            print("Closing gripper to re-pick at place...")
            if not gripper_set_position(base, 1.0):
                print("Warning: Gripper close (re-pick) timed out")
            else:
                gripper_closed = True

            # Transit back to the original area: go to original lift (waypoint 3)
            if not execute_waypoints(base, [first_route[2]]):
                return 1

            # Lower to the original lower pose (waypoint 2) to re-place at origin
            if not execute_waypoints(base, [first_route[1]]):
                return 1

            # Brief settle before re-release
            time.sleep(0.5)

            print("Opening gripper to release at original location...")
            if not gripper_set_position(base, 0.0):
                print("Warning: Gripper open (re-release) timed out")
            else:
                gripper_closed = False

            # Lift back up to original lift (waypoint 3) to clear
            if not execute_waypoints(base, [first_route[2]]):
                return 1

            # Finally, retract to safe pose (waypoint 6)
            if not execute_waypoints(base, [first_route[5]]):
                return 1

            print("Done.")
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