#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed under the
# terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2


class GripperCommandExample:
    def __init__(self, router, force_sensor=None, proportional_gain=2.0, force_threshold=5.0):
        self.proportional_gain = proportional_gain
        self.router = router
        self.base = BaseClient(self.router)
        self.force_sensor = force_sensor  # Should be an object with a .read() method
        self.force_threshold = force_threshold


    def ExampleSendGripperCommands(self):
        # ...existing code...
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Close the gripper with position increments, with force feedback
        print("Performing gripper test in position with force feedback...")
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        position = 0.00
        finger.finger_identifier = 1
        while position < 1.0:
            finger.value = position
            print("Going to position {:0.2f}...".format(finger.value))
            self.base.SendGripperCommand(gripper_command)
            position += 0.1
            time.sleep(1)
            # Force feedback integration
            if self.force_sensor:
                force, status = self.force_sensor.read()
                print(f"Force sensor reading: {force}, status: {status}")
                if force > self.force_threshold:
                    print("Gripper stopped: safe force reached")
                    break
                if status == "error":
                    print("Sensor error: gripper stopped for safety.")
                    break

        # Set speed to open gripper
        print ("Opening gripper using speed command...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = 0.1
        self.base.SendGripperCommand(gripper_command)
        gripper_request = Base_pb2.GripperRequest()

        # Wait for reported position to be opened
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                print("Current position is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value < 0.01:
                    break
            else:
                break

        # Set speed to close gripper with force feedback
        print ("Closing gripper using speed command with force feedback...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = -0.1
        self.base.SendGripperCommand(gripper_command)

        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                print("Current speed is : {0}".format(gripper_measure.finger[0].value))
                # Force feedback integration
                if self.force_sensor:
                    force, status = self.force_sensor.read()
                    print(f"Force sensor reading: {force}, status: {status}")
                    if force > self.force_threshold:
                        print("Gripper stopped: safe force reached.")
                        self.base.SendGripperCommand(gripper_command)  # Optionally send stop command
                        break
                    if status == "error":
                        print("Sensor error: gripper stopped for safety.")
                        self.base.SendGripperCommand(gripper_command)
                        break
                if gripper_measure.finger[0].value == 0.0:
                    break
            else:
                break

def main():
    # Import the utilities helper module
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        example = GripperCommandExample(router)
        example.ExampleSendGripperCommands()

if __name__ == "__main__":
    main()