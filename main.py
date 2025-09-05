#! /usr/bin/env python3
import os
import sys
import time
import threading

# Kortex clients / messages
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

# --- Import modules from Kinova folder ---
from Kinova import utilities
from Kinova.send_cartesian_waypoint_trajectory import (
    example_move_to_home_position as move_to_home,
    populateCartesianCoordinate,
    check_for_end_or_abort,
)
from Kinova.gripper_command import GripperCommandExample

# Future: perception hook for camera-based object detection
# try:
#     from perception import compute_object_pose_base  
#     HAS_PERCEPTION = True
# except Exception:
#     HAS_PERCEPTION = False

# -------- Object pose and movement parameters --------
OBJECT_X, OBJECT_Y, OBJECT_Z = 0.60, 0.00, 0.10
TOOL_TX, TOOL_TY, TOOL_TZ = 90.0, 0.0, 90.0
APPROACH_H, LIFT_H = 0.12, 0.15
BLEND = 0.0
TIMEOUT_S = 45


def _execute_waypoints_via_examples(base: BaseClient, waypoint_defs):
    """
    Build a WaypointList using YOUR populateCartesianCoordinate(), then execute it
    and wait using check_for_end_or_abort() callback. No custom helpers.
    """
    # Ensure servoing mode as in the examples
    servo = Base_pb2.ServoingModeInformation()
    servo.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(servo)

    waypoints = Base_pb2.WaypointList()
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = False

    for i, (x, y, z, tx, ty, tz, blend) in enumerate(waypoint_defs):
        # Helper expects (x, y, z, blending, theta_x, theta_y, theta_z)
        packed = (x, y, z, blend, tx, ty, tz)
        wp = waypoints.waypoints.add()
        wp.name = f"wp_{i}"
        wp.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(packed))

    # Validate then execute
    validation = base.ValidateWaypointList(waypoints)
    if validation.trajectory_error_report.trajectory_error_elements:
        print("Waypoint validation failed:")
        validation.trajectory_error_report.PrintDebugString()
        return False

    done = threading.Event()
    sub = base.OnNotificationActionTopic(check_for_end_or_abort(done), Base_pb2.NotificationOptions())
    print("Executing waypoint trajectory...")
    base.ExecuteWaypointTrajectory(waypoints)
    ok = done.wait(TIMEOUT_S)
    base.Unsubscribe(sub)
    if not ok:
        print("Timeout waiting for trajectory")
    return ok


def main():
    # Parse IP/credentials using utilities
    args = utilities.parseConnectionArguments()

    # Connect using DeviceConnection
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)  # kept for future (force feedback, etc.)

        # 1) Move to Home position
        if not move_to_home(base):
            print("Warning: failed to reach Home; continuing.")

        # 2) Future: add perception.py to get object pose from camera
        # global OBJECT_X, OBJECT_Y, OBJECT_Z
        # if HAS_PERCEPTION:
        #     try:
        #         OBJECT_X, OBJECT_Y, OBJECT_Z = compute_object_pose_base()
        #         print(f"Camera pose: x={OBJECT_X:.3f}, y={OBJECT_Y:.3f}, z={OBJECT_Z:.3f}")
        #     except Exception as e:
        #         print("Perception failed; using predefined pose:", e)
        
        print(f"Using predefined object pose: x={OBJECT_X:.3f}, y={OBJECT_Y:.3f}, z={OBJECT_Z:.3f}")

        # 3) Approach above object and descend
        approach_z = OBJECT_Z + APPROACH_H
        if not _execute_waypoints_via_examples(
            base,
            [
                (OBJECT_X, OBJECT_Y, approach_z, TOOL_TX, TOOL_TY, TOOL_TZ, BLEND),
                (OBJECT_X, OBJECT_Y, OBJECT_Z,  TOOL_TX, TOOL_TY, TOOL_TZ, BLEND),
            ],
        ):
            return 1

        # 4) Close gripper to grasp object (future: add force feedback threshold)
        gripper = GripperCommandExample(router)
        gripper.ExampleSendGripperCommands()

        # 5) Lift object to safe height
        lift_z = OBJECT_Z + LIFT_H
        if not _execute_waypoints_via_examples(
            base,
            [(OBJECT_X, OBJECT_Y, lift_z, TOOL_TX, TOOL_TY, TOOL_TZ, BLEND)],
        ):
            return 1

        print("Pick sequence complete.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
