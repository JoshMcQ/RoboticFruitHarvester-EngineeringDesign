#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2021 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading
import argparse

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 30

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END:
            print("Completed Successfully")
            e.set()
        elif notification.action_event == Base_pb2.ACTION_ABORT:
            print("Action was aborted")
            e.set()
    return check

def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished

def populateCartesianCoordinate(waypointInformation):
    
    waypoint = Base_pb2.CartesianWaypoint()  
    waypoint.pose.x = waypointInformation[0]
    waypoint.pose.y = waypointInformation[1]
    waypoint.pose.z = waypointInformation[2]
    waypoint.blending_radius = waypointInformation[3]
    waypoint.pose.theta_x = waypointInformation[4]
    waypoint.pose.theta_y = waypointInformation[5]
    waypoint.pose.theta_z = waypointInformation[6] 
    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    
    return waypoint

def example_trajectory(base, base_cyclic):

    # Set servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # Define trajectory waypoints
    kTheta_x = 90.0
    kTheta_y = 0.0
    kTheta_z = 90.0

    waypointsDefinition = (
        (0.4, 0.0, 0.4, 0.0, kTheta_x, kTheta_y, kTheta_z),  # Approach above fruit
        (0.4, 0.0, 0.2, 0.0, kTheta_x, kTheta_y, kTheta_z),  # Lower to fruit
        (0.4, 0.0, 0.4, 0.0, kTheta_x, kTheta_y, kTheta_z),  # Lift fruit
    )

    # Build the waypoint list
    waypoints = Base_pb2.WaypointList()
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = False

    index = 0
    for waypointDefinition in waypointsDefinition:
        waypoint = waypoints.waypoints.add()
        waypoint.name = "waypoint_" + str(index)   
        waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypointDefinition))
        index += 1

    # Executing trajectory
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e), Base_pb2.NotificationOptions())

    print("Moving cartesian trajectory...")
        
    base.ExecuteWaypointTrajectory(waypoints)

    print("Waiting for trajectory to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)
    return finished


### Simulation mocks for testing without hardware ###
class MockAction:
    def __init__(self, name="Home", handle=1234):
        self.name = name
        self.handle = handle

class MockActionList:
    def __init__(self):
        self.action_list = [MockAction()]

class MockBaseClient:
    def __init__(self):
        self.subscribed = False

    def SetServoingMode(self, mode):
        print("[SIM] SetServoingMode called")

    def ReadAllActions(self, action_type):
        print("[SIM] ReadAllActions called")
        return MockActionList()

    def OnNotificationActionTopic(self, callback, options):
        print("[SIM] OnNotificationActionTopic subscribed")
        self.subscribed = True
        # Simulate an action end after 2 seconds
        def notify():
            time.sleep(2)
            class Notification:
                action_event = Base_pb2.ACTION_END
            callback(Notification())
        threading.Thread(target=notify, daemon=True).start()
        return 1  # mock subscription handle

    def ExecuteActionFromReference(self, handle):
        print(f"[SIM] ExecuteActionFromReference called with handle: {handle}")

    def Unsubscribe(self, handle):
        print(f"[SIM] Unsubscribe called for handle: {handle}")
        self.subscribed = False

    def ExecuteWaypointTrajectory(self, waypoints):
        print("[SIM] ExecuteWaypointTrajectory called")
        # No real execution here

class MockBaseCyclicClient:
    pass


def main():
    parser = argparse.ArgumentParser(description="Kinova Kortex Example Script")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode without hardware")
    args, unknown = parser.parse_known_args()

    if args.simulate:
        print("** Running in SIMULATION mode **")
        base = MockBaseClient()
        base_cyclic = MockBaseCyclicClient()
        success = True
        print("Move to home position")
        success &= example_move_to_home_position(base)
        print("Send cartesian waypoint trajectory")    
        success &= example_trajectory(base, base_cyclic)
        return 0 if success else 1

    else:
        # Normal real robot run
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import utilities

        # Parse arguments from utilities (e.g. IP, port)
        args = utilities.parseConnectionArguments()

        # Create connection to the device and get the router
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            # Create required services
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router)

            success = True
            print("Move to home position")
            success &= example_move_to_home_position(base)
            print("Send cartesian waypoint trajectory")    
            success &= example_trajectory(base, base_cyclic)

            return 0 if success else 1


if __name__ == "__main__":
    exit(main())

