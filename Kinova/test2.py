import os
import sys
import time
import threading
import serial
from serial.tools import list_ports

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2

TIMEOUT_DURATION = 30.0


def log_pose_and_joints(base: BaseClient, context: str):
    try:
        pose = base.GetMeasuredCartesianPose()
        print(f"{context} pose (base frame): "
              f"x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f}, "
              f"θx={pose.theta_x:.3f}, θy={pose.theta_y:.3f}, θz={pose.theta_z:.3f}")
    except Exception as e:
        print(f"{context} pose unavailable: {e}")

    try:
        joints = base.GetMeasuredJointAngles()
        joint_vals = ", ".join(
            f"J{idx + 1}={angle.value:.3f}"
            for idx, angle in enumerate(joints.joint_angles)
        )
        print(f"{context} joints: {joint_vals}")
    except Exception as e:
        print(f"{context} joints unavailable: {e}")

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
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    wl = Base_pb2.WaypointList()
    wl.duration = 0.0
    wl.use_optimal_blending = False

    for idx, wp_def in enumerate(waypoints_def):
        wp = wl.waypoints.add()
        wp.name = f"waypoint_{idx}"
        wp.cartesian_waypoint.CopyFrom(populate_cartesian(wp_def))

    result = base.ValidateWaypointList(wl)
    if len(result.trajectory_error_report.trajectory_error_elements):
        print("Trajectory validation failed:")
        result.trajectory_error_report.PrintDebugString()
        return False

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
    Fixed gripper control with force feedback.
    For closing (position >= 0.9): use force sensor if available
    For opening: just open normally
    """
    
    # Opening - simple position command
    if position < 0.5:
        print("Opening gripper...")
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = position
        base.SendGripperCommand(cmd)
        time.sleep(2)
        return True
    
    # Closing with force feedback
    if ser is not None:
        print(f"Closing with force monitoring (threshold: {force_threshold})...")
        
        # Use SPEED mode for gradual controlled closing
        # NEGATIVE speed closes, POSITIVE speed opens on this gripper (from official example)
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_SPEED
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = -0.10  # negative closes slowly
        base.SendGripperCommand(cmd)
        
        start_time = time.time()
        last_pos = 0.0
        position_stable_count = 0
        force_approaching = False
        
        # Clear serial buffer to get fresh readings
        ser.reset_input_buffer()
        
        while True:  # Keep closing until force threshold or fully closed
            # Get gripper position
            req = Base_pb2.GripperRequest()
            req.mode = Base_pb2.GRIPPER_POSITION
            meas = base.GetMeasuredGripperMovement(req)
            current_pos = 0.0
            if len(meas.finger):
                current_pos = meas.finger[0].value
                print(f"Gripper pos: {current_pos:.3f}", end="")
                
                # Check if gripper has stopped moving (object detected)
                if abs(current_pos - last_pos) < 0.001:
                    position_stable_count += 1
                    if position_stable_count > 15 and current_pos > 0.1:
                        print(" - Object contacted")
                else:
                    position_stable_count = 0
                last_pos = current_pos
                
                # Stop if fully closed
                if current_pos >= 0.99:
                    print(" - Fully closed")
                    base.Stop()
                    return True
            
            # Read force data continuously
            try:
                line = ser.readline()
                if line:
                    text = line.decode('utf-8', errors='ignore').strip()
                    if text:
                        # Try to extract any number from the line
                        import re
                        match = re.search(r'(\d+(?:\.\d+)?)', text)
                        if match:
                            force = float(match.group(1))
                            print(f" Force: {force}")
                            
                            # Stop immediately if threshold reached
                            if force >= force_threshold:
                                print(f"\n✓ Force threshold reached: {force}")
                                base.Stop()
                                # Small backoff to reduce pressure
                                time.sleep(0.1)
                                cmd2 = Base_pb2.GripperCommand()
                                cmd2.mode = Base_pb2.GRIPPER_SPEED
                                f2 = cmd2.gripper.finger.add()
                                f2.finger_identifier = 1
                                f2.value = 0.05  # positive opens slightly to relieve pressure
                                base.SendGripperCommand(cmd2)
                                time.sleep(0.1)
                                base.Stop()
                                return True
                            
                            # Slow down even more when approaching threshold
                            elif force > force_threshold * 0.7 and not force_approaching:
                                print(f" - Slowing down (force approaching threshold)")
                                force_approaching = True
                                cmd_slow = Base_pb2.GripperCommand()
                                cmd_slow.mode = Base_pb2.GRIPPER_SPEED
                                f_slow = cmd_slow.gripper.finger.add()
                                f_slow.finger_identifier = 1
                                f_slow.value = -0.05  # smaller negative keeps closing slowly
                                base.SendGripperCommand(cmd_slow)
                        else:
                            print(f" [info: {text}]")
                else:
                    print("")  # New line if no serial data
            except Exception as e:
                print(f" [serial error: {e}]")
            
            time.sleep(0.02)  # Faster polling for quicker response
        
        # This should never be reached since we only exit via force threshold or fully closed
        base.Stop()
        return True
    
    # No force sensor - just close normally
    else:
        print("Closing gripper (no force sensor)...")
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = position
        base.SendGripperCommand(cmd)
        time.sleep(3)
        return True

def main():
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)

        # Updated cartesian waypoints (x, y, z, blending, θx, θy, θz)
        # Coordinates tuned for: pick at (0.403, 0.081) and place at (0.391, 0.090)
        ball_action = (
            (0.391, 0.090, 0.113, 0.0, 10.722, 177.924, 82.692),   # 0: Pick approach (high)
            (0.391, 0.090, -0.011, 0.0, 10.748, 177.957, 82.756),  # 1: Pick descend (grip)
            (0.391, 0.090, 0.107, 0.0, 10.644, 177.900, 82.593),   # 2: Pick retreat (lift)
            (0.423, -0.353, 0.102, 0.0, 11.014, 177.982, 82.730),   # 3: Place approach (high)
            (0.428, -0.357, -0.012, 0.0, 10.762, 177.776, 82.769),
            (0.422, -0.352, 0.113, 0.0, 10.975, 177.962, 82.693),
            (0.403, 0.081, 0.117, 0.0, 10.742, 177.961, 82.753),   # 6: Rise for clearance
            (0.403, 0.081, -0.022, 0.0, 10.74, 177.978, 82.745),
            (0.403, 0.081, 0.129, 0.0, 10.733, 177.937, 82.720),  
            (0.13, -0.068, 0.118, 0.0, 10.777, 177.812, 82.762),# 6: Retract 
        )

        gripper_closed = False
        ser = None
        
        try:
            # Open serial connection
            try:
                ser = serial.Serial("COM5", 9600, timeout=0.1)
                ser.reset_input_buffer()
                time.sleep(2)
                print("✓ Serial port opened")
            except Exception as e:
                print(f"⚠ Serial failed: {e} - continuing without force feedback")
                ser = None

            FORCE_THRESHOLD = 500

            # === PICK SEQUENCE ===
            print("\n=== Step 1: Opening gripper ===")
            gripper_set_position(base, 0.0)

            print("\n=== Step 2: Move above pick (waypoint 0) ===")
            log_pose_and_joints(base, "Current before waypoint")
            if not execute_waypoints(base, [ball_action[0]]):
                return 1
            print("✓ Waypoint execution reported ACTION_END")
            log_pose_and_joints(base, "Reached waypoint")

            print("\n=== Step 3: Descend to pick (waypoint 1) ===")
            log_pose_and_joints(base, "Current before waypoint")
            if not execute_waypoints(base, [ball_action[1]]):
                return 1
            print("✓ Waypoint execution reported ACTION_END")
            log_pose_and_joints(base, "Reached waypoint")

            print("\n=== Step 4: Closing gripper with force control ===")
            if gripper_set_position(base, 1.0, force_threshold=FORCE_THRESHOLD, ser=ser):
                gripper_closed = True
                print("✓ Object gripped with force control")
            else:
                print("⚠ Force threshold not reached, continuing with caution")
                gripper_closed = True

            print("\n=== Step 5: Lift from pick (waypoint 2) ===")
            log_pose_and_joints(base, "Current before waypoint")
            if not execute_waypoints(base, [ball_action[2]]):
                return 1
            print("✓ Waypoint execution reported ACTION_END")
            log_pose_and_joints(base, "Reached waypoint")

            # === PLACE SEQUENCE ===
            print("\n=== Step 6: Move above place (waypoint 3) ===")
            log_pose_and_joints(base, "Current before waypoint")
            if not execute_waypoints(base, [ball_action[3]]):
                return 1
            print("✓ Waypoint execution reported ACTION_END")
            log_pose_and_joints(base, "Reached waypoint")

            print("\n=== Step 7: Lower to place (waypoint 4) ===")
            log_pose_and_joints(base, "Current before waypoint")
            if not execute_waypoints(base, [ball_action[4]]):
                return 1
            print("✓ Waypoint execution reported ACTION_END")
            log_pose_and_joints(base, "Reached waypoint")

            time.sleep(0.3)

            print("\n=== Step 8: Opening gripper to release ===")
            gripper_set_position(base, 0.0)
            gripper_closed = False

            print("\n=== Step 9: Retreat from place (waypoint 5) ===")
            log_pose_and_joints(base, "Current before waypoint")
            if not execute_waypoints(base, [ball_action[5]]):
                return 1
            print("✓ Waypoint execution reported ACTION_END")
            log_pose_and_joints(base, "Reached waypoint")

            # === CLEARANCE AND RETRACT ===
            print("\n=== Step 10: Clear path (high) (waypoint 6) ===")
            log_pose_and_joints(base, "Current before waypoint")
            if not execute_waypoints(base, [ball_action[6]]):
                return 1
            print("✓ Waypoint execution reported ACTION_END")
            log_pose_and_joints(base, "Reached waypoint")

            print("\n=== Step 11: Clear path (transition) (waypoint 7) ===")
            log_pose_and_joints(base, "Current before waypoint")
            if not execute_waypoints(base, [ball_action[7]]):
                return 1
            print("✓ Waypoint execution reported ACTION_END")
            log_pose_and_joints(base, "Reached waypoint")

            print("\n=== Step 12: Retract to home (waypoint 8) ===")
            log_pose_and_joints(base, "Current before waypoint")
            if not execute_waypoints(base, [ball_action[8]]):
                return 1
            print("✓ Waypoint execution reported ACTION_END")
            log_pose_and_joints(base, "Reached waypoint")

            print("\n=== Step 13: Final retract to home (waypoint 9) ===")
            log_pose_and_joints(base, "Current before waypoint")
            if not execute_waypoints(base, [ball_action[9]]):
                return 1
            print("✓ Waypoint execution reported ACTION_END")
            log_pose_and_joints(base, "Reached waypoint")

            print("\n✓ Complete")
            return 0
            
        finally:
            if ser:
                ser.close()
            if gripper_closed:
                try:
                    gripper_set_position(base, 0.0)
                except:
                    pass

if __name__ == "__main__":
    sys.exit(main())