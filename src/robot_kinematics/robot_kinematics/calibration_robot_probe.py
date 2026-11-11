#!/usr/bin/env python3

import argparse
import time
import numpy as np
import rclpy

import openmanipulator_fixed_orientation_pick_and_place as robot

parser = argparse.ArgumentParser()
parser.add_argument("--x", type=float, default=0.09524)
parser.add_argument("--y", type=float, default=0.00268)
parser.add_argument("--z", type=float, default=0.160)
parser.add_argument("--duration", type=float, default=8.0)
parser.add_argument("--max-distance-mm", type=float, default=60.0)
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()

node = None

try:
    if args.duration < 5:
        raise ValueError("Duration must be at least 5 seconds.")

    if args.z <= 0 or args.z > 0.30:
        raise ValueError("Target Z outside permitted check range.")

    rclpy.init()
    node = robot.PickPlaceNode()

    # Always start in dry-run mode.
    robot.DRY_RUN = True
    robot.FREQUENCY_HZ = 2.0

    # Read actual robot joints.
    q = robot.validate_q(
        node.get_current_q(), "Measured joint state"
    )

    # Calculate current TCP position using FK.
    start_xyz, _ = robot.fk(q)

    target_xyz = np.array(
        [args.x, args.y, args.z], dtype=float
    )

    distance = np.linalg.norm(target_xyz - start_xyz)

    print("\n===== CALIBRATION PROBE =====")
    print("Current joints:", np.round(q, 5))
    print("Current TCP:", np.round(start_xyz, 5))
    print("Target TCP:", np.round(target_xyz, 5))
    print(f"Distance: {distance * 1000:.2f} mm")
    print("Gripper: NO COMMAND")
    print("Automatic return: DISABLED")

    if distance > args.max_distance_mm / 1000:
        raise RuntimeError(
            f"Movement exceeds {args.max_distance_mm} mm limit."
        )

    # Preserve the current tool pitch.
    robot.TASK_TOOL_ANGLE = float(np.sum(q[1:]))

    # Generate and validate trajectory.
    times, q_path, xyz_path = robot.plan_cartesian_segment(
        start_xyz,
        target_xyz,
        q,
        args.duration
    )

    predicted_xyz, _ = robot.fk(q_path[-1])
    error = np.linalg.norm(predicted_xyz - target_xyz)

    if error > robot.POSITION_TOLERANCE:
        raise RuntimeError("Endpoint FK validation failed.")

    print("\n===== PLAN VALIDATION PASSED =====")
    print("Waypoints:", len(q_path))
    print("Duration:", round(times[-1], 2), "seconds")
    print("Predicted endpoint:", np.round(predicted_xyz, 5))
    print(f"Endpoint error: {error * 1000:.3f} mm")

    if not args.execute:
        print("\nDRY RUN: NO PHYSICAL MOVEMENT.")
        print("Add --execute only after inspecting the path.")
    else:
        print("\nWARNING: SINGLE ARM MOVEMENT ONLY.")
        print("Verify the complete path is clear.")
        print("Verify target Z is above the physical work surface.")
        print("Keep emergency stop accessible.")

        if not node.arm_client.wait_for_server(timeout_sec=5):
            raise RuntimeError("Arm trajectory action server unavailable.")

        answer = input(
            f"Type MOVE to execute this {distance*1000:.1f} mm movement: "
        )

        if answer.strip() != "MOVE":
            print("Cancelled. No movement sent.")
        else:
            robot.DRY_RUN = False
            node.execute_arm(
                q_path, times, "single calibration hover test"
            )

            time.sleep(0.3)

            final_q = robot.validate_q(
                node.get_current_q(), "Final joint feedback"
            )
            final_xyz, _ = robot.fk(final_q)

            print("\n===== MOVEMENT COMPLETE =====")
            print("Final joints:", np.round(final_q, 5))
            print("Final TCP:", np.round(final_xyz, 5))
            print(
                "Distance from requested TCP:",
                round(np.linalg.norm(final_xyz-target_xyz)*1000, 2),
                "mm"
            )

finally:
    if node is not None:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
