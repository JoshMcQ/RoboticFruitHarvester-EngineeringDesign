import os
import sys
import time
import threading
import serial
from serial.tools import list_ports

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

        # Your exact waypoints
        first_route = (
            (0.497, -0.024, 0.263, 0.0, 98.035,  4.639, 123.538),  # First Position test 
            (0.497, -0.024, 0.263, 0.0, 81.9, 178.3, 122.7), # Ball Action 1
            (0.38, 0.087, -0.036, 0.0, 72.7, -179.1, 145.4),  # Ball action 2 (close gripper after)
            (0.471, 0.052, 0.306, 0.0, 88.7, -178.1, 137.1), # Ball Action 3
            (0.447, -0.158, 0.306, 0.0, 88.7, -187.1, 111.4),   # Ball action 4
            (0.384, -0.09, -0.029, 0.0, 73, 177.1, 118.2),  # Ball action 5 
            # Let go
            #ball action 4
            (0.447, -0.157, 0.304, 0.0, 91.5, -8.3, 111.3),   # End Effector twist
            (0.13,  -0.068, 0.118, 0.0, 10.777, 177.812, 82.762),  # 6: Retract
        )

        gripper_closed = False
        ser = None
        
        try:
            # Open serial connection
            try:
                ser = serial.Serial("COM5", 9600, timeout=0.1)
                ser.reset_input_buffer()
                time.sleep(2)  # Arduino boot time
                print("✓ Serial port opened")
            except Exception as e:
                print(f"⚠ Serial failed: {e} - continuing without force feedback")
                ser = None

            # Open gripper
            gripper_set_position(base, 0.0)

            # Move to approach and lower positions
            if not execute_waypoints(base, first_route[:2]):
                return 1

            # Close with force feedback
            FORCE_THRESHOLD = 500
            if gripper_set_position(base, 1.0, force_threshold=FORCE_THRESHOLD, ser=ser):
                gripper_closed = True
                print("✓ Object gripped with force control")
            else:
                print("⚠ Force threshold not reached, but continuing")
                gripper_closed = True

            # Lift and move to place position
            if not execute_waypoints(base, first_route[2:4]):
                return 1

            time.sleep(0.5)

            # Release
            gripper_set_position(base, 0.0)
            gripper_closed = False

            # Lift away
            if not execute_waypoints(base, [first_route[4]]):
                return 1

            # Return to place position to re-pick
            if not execute_waypoints(base, [first_route[3]]):
                return 1

            # Re-grip with force feedback
            if gripper_set_position(base, 1.0, force_threshold=FORCE_THRESHOLD, ser=ser):
                gripper_closed = True
                print("✓ Re-gripped with force control")
            else:
                print("⚠ Force threshold not reached on re-grip")
                gripper_closed = True

            # Move back to original position
            if not execute_waypoints(base, [first_route[2], first_route[1]]):
                return 1

            time.sleep(0.5)

            # Final release
            gripper_set_position(base, 0.0)
            gripper_closed = False

            # Retract
            if not execute_waypoints(base, [first_route[2], first_route[5]]):
                return 1

            print("✓ Complete")
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