# ...existing code...
BLEND = 0.0
TIMEOUT_S = 45

# --- Force sensor settings (new) ---
FORCE_THRESHOLD_N = 5.0    # Newtons, contact detected when magnitude >= this
FORCE_TIMEOUT_S = 5.0      # seconds to wait for contact before giving up
FORCE_POLL_HZ = 20         # polling frequency for force sensor

def wait_for_contact(base_cyclic: BaseCyclicClient, threshold_n=FORCE_THRESHOLD_N, timeout_s=FORCE_TIMEOUT_S, poll_hz=FORCE_POLL_HZ):
    """
    Poll the cyclic client for measured Cartesian forces and return True when
    |F| >= threshold_n. Returns False on timeout or if force RPC is unavailable.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            # common Kortex API name
            forces = base_cyclic.GetMeasuredCartesianForces()
            fx, fy, fz = forces.x, forces.y, forces.z
        except Exception:
            try:
                # alternative name
                forces = base_cyclic.GetMeasuredCartesianForce()
                fx, fy, fz = forces.x, forces.y, forces.z
            except Exception:
                # If unavailable, bail out gracefully
                print("Force reading not available via BaseCyclicClient; skipping contact wait.")
                return False

        mag = (fx*fx + fy*fy + fz*fz) ** 0.5
        # optional debug; comment out if noisy
        print(f"[force] fx={fx:.2f} N fy={fy:.2f} N fz={fz:.2f} N |F|={mag:.2f} N")
        if mag >= threshold_n:
            print(f"Contact detected: |F|={mag:.2f} N")
            return True

        time.sleep(1.0 / poll_hz)

    print("Contact not detected within timeout")
    return False
# ...existing code...

def main():
    # Parse IP/credentials using utilities
    args = utilities.parseConnectionArguments()

    # Connect using DeviceConnection
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)  # kept for future (force feedback, etc.)

        # Move to Home position
        if not move_to_home(base):
            print("Warning: failed to reach Home; continuing.")

        # Future: add perception.py to get object pose from camera
        # global OBJECT_X, OBJECT_Y, OBJECT_Z
        # if HAS_PERCEPTION:
        #     try:
        #         OBJECT_X, OBJECT_Y, OBJECT_Z = compute_object_pose_base()
        #         print(f"Camera pose: x={OBJECT_X:.3f}, y={OBJECT_Y:.3f}, z={OBJECT_Z:.3f}")
        #     except Exception as e:
        #         print("Perception failed; using predefined pose:", e)
        
        print(f"Using predefined object pose: x={OBJECT_X:.3f}, y={OBJECT_Y:.3f}, z={OBJECT_Z:.3f}")

        # Approach above object and descend
        approach_z = OBJECT_Z + APPROACH_H
        if not _execute_waypoints_via_examples(
            base,
            [
                (OBJECT_X, OBJECT_Y, approach_z, TOOL_TX, TOOL_TY, TOOL_TZ, BLEND),
                (OBJECT_X, OBJECT_Y, OBJECT_Z,  TOOL_TX, TOOL_TY, TOOL_TZ, BLEND),
            ],
        ):
            return 1

        # --- wait for contact using force sensor before closing gripper ---
        contacted = wait_for_contact(base_cyclic, threshold_n=FORCE_THRESHOLD_N, timeout_s=FORCE_TIMEOUT_S)
        if not contacted:
            print("No contact detected. Proceeding based on policy (continuing).")
            # Optionally: return 1  # abort if you prefer
        # --------------------------------------------------------------------

        # Close gripper to grasp object (future: add force feedback threshold)
        gripper = GripperCommandExample(router)
        gripper.ExampleSendGripperCommands()

        # Lift object to safe height
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