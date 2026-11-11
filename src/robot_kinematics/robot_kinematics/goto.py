#!/usr/bin/env python3
"""Move OpenMANIPULATOR-X from its measured joint state to a fixed joint pose.

Default behaviour is a dry plan only. Add --execute to send the trajectory;
the script then requires the operator to type MOVE before commanding the arm.

This is joint-space interpolation, not collision-aware motion planning. It does
not guarantee a collision-free route from every possible starting pose.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from std_srvs.srv import SetBool


# Match the joint names and action endpoint used by the existing controller.
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]
JOINT_STATE_TOPIC = "/joint_states"
ARM_ACTION = "/arm_controller/follow_joint_trajectory"
TORQUE_SERVICE = "/dynamixel_hardware_interface/set_dxl_torque"

# Target configuration supplied by the user (radians).
TARGET_Q = np.array([-0.08744, -0.95720, 0.34515, 0.77159], dtype=float)

# Joint limits from the project's kinematics.py (radians).
JOINT_LIMITS = np.array([
    [-math.pi, math.pi],  # joint1
    [-1.5, 1.5],          # joint2
    [-1.5, 1.4],          # joint3
    [-1.7, 1.97],         # joint4
], dtype=float)

# Conservative initial trajectory limits. These are software limits, not
# manufacturer-certified limits. Increase only after validating the robot.
MAX_JOINT_VELOCITY = np.array([0.35, 0.35, 0.35, 0.35], dtype=float)  # rad/s
MAX_JOINT_ACCELERATION = np.array([0.70, 0.70, 0.70, 0.70], dtype=float)  # rad/s^2
FREQUENCY_HZ = 50.0
MIN_DURATION_S = 4.0
JOINT_STATE_TIMEOUT_S = 10.0
JOINT_STATE_MAX_AGE_S = 0.5
GOAL_TIME_TOLERANCE_S = 5.0
TORQUE_SERVICE_TIMEOUT_S = 5.0
FINAL_STATE_TIMEOUT_S = 8.0
FINAL_JOINT_TOLERANCE_RAD = 0.05


def validate_configuration(q: np.ndarray, label: str) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError(f"{label} must contain four finite joint values; got {q!r}.")
    bad = np.where((q < JOINT_LIMITS[:, 0]) | (q > JOINT_LIMITS[:, 1]))[0]
    if len(bad):
        details = ", ".join(
            f"{JOINT_NAMES[i]}={q[i]:.5f} outside "
            f"[{JOINT_LIMITS[i, 0]:.3f}, {JOINT_LIMITS[i, 1]:.3f}]"
            for i in bad
        )
        raise ValueError(f"{label} violates the configured joint limits: {details}")
    return q


def choose_joint_delta(current_q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
    """Choose a shortest valid joint1 wrap; other joints are non-continuous."""
    delta = target_q - current_q
    candidates = []
    for turns in (-1, 0, 1):
        candidate_target = target_q.copy()
        candidate_target[0] += 2.0 * math.pi * turns
        if JOINT_LIMITS[0, 0] <= candidate_target[0] <= JOINT_LIMITS[0, 1]:
            candidates.append(candidate_target[0] - current_q[0])
    if candidates:
        delta[0] = min(candidates, key=abs)
    return delta


def quintic_profile(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return quintic position scale s(u) and derivative ds/du."""
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    ds_du = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    return s, ds_du


def plan_joint_trajectory(current_q: np.ndarray, target_q: np.ndarray):
    delta = choose_joint_delta(current_q, target_q)

    # Peak |ds/du| for a quintic profile is 1.875. The bound 5.774 is a
    # conservative bound for the peak absolute second derivative.
    duration_v = float(np.max(1.875 * np.abs(delta) / MAX_JOINT_VELOCITY))
    duration_a = float(np.max(np.sqrt(5.774 * np.abs(delta) / MAX_JOINT_ACCELERATION)))
    duration = max(MIN_DURATION_S, duration_v, duration_a)

    n_intervals = max(2, int(math.ceil(duration * FREQUENCY_HZ)))
    sample_times = np.linspace(0.0, duration, n_intervals + 1)[1:]
    u = sample_times / duration
    s, ds_du = quintic_profile(u)

    q_path = current_q[None, :] + s[:, None] * delta[None, :]
    qd_path = (ds_du[:, None] / duration) * delta[None, :]

    # Force the final point to the exact target and exact zero velocity.
    q_path[-1] = current_q + delta
    qd_path[-1] = 0.0

    # Validate sampled joint positions and numerical velocity/acceleration.
    validate_configuration(q_path[-1], "Final target")
    if np.any(q_path < JOINT_LIMITS[:, 0]) or np.any(q_path > JOINT_LIMITS[:, 1]):
        raise ValueError("Interpolated path crosses a configured joint limit.")

    times_with_zero = np.concatenate(([0.0], sample_times))
    q_with_zero = np.vstack((current_q, q_path))
    if len(times_with_zero) >= 3:
        qd_num = np.gradient(q_with_zero, times_with_zero, axis=0, edge_order=2)
        qdd_num = np.gradient(qd_num, times_with_zero, axis=0, edge_order=2)
        max_v = np.max(np.abs(qd_num), axis=0)
        max_a = np.max(np.abs(qdd_num), axis=0)
        # Check with a small numerical tolerance; analytical duration limits
        # include a conservative margin and the exact quintic endpoint speeds.
        if np.any(max_v > MAX_JOINT_VELOCITY * 1.08):
            raise ValueError(f"Sampled velocity cap exceeded: {max_v}")
        if np.any(max_a > MAX_JOINT_ACCELERATION * 1.15):
            raise ValueError(f"Sampled acceleration cap exceeded: {max_a}")

    return sample_times, q_path, qd_path, duration, delta


class MoveToPoseNode(Node):
    def __init__(self):
        super().__init__("move_to_fixed_joint_configuration")
        self.latest_q: Optional[np.ndarray] = None
        self.latest_stamp: Optional[float] = None
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_cb, 10)
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION)
        self.torque_client = self.create_client(SetBool, TORQUE_SERVICE)

    def enable_torque(self) -> None:
        """Request torque ON before planning/execution; fail closed if unavailable."""
        self.get_logger().info(f"Checking torque-enable service: {TORQUE_SERVICE}")
        if not self.torque_client.wait_for_service(timeout_sec=TORQUE_SERVICE_TIMEOUT_S):
            raise RuntimeError(
                f"Torque service unavailable: {TORQUE_SERVICE}. "
                "No trajectory will be sent. Check the Dynamixel hardware interface."
            )

        request = SetBool.Request()
        request.data = True
        future = self.torque_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=TORQUE_SERVICE_TIMEOUT_S)
        if not future.done():
            raise TimeoutError(
                f"Timed out requesting torque enable from {TORQUE_SERVICE}. "
                "No trajectory will be sent."
            )

        response = future.result()
        if response is None:
            raise RuntimeError(
                f"No response from {TORQUE_SERVICE}. No trajectory will be sent."
            )
        if not response.success:
            raise RuntimeError(
                f"Torque enable was rejected: {response.message}. No trajectory will be sent."
            )
        self.get_logger().info(
            "Torque-enable request succeeded: " + (response.message or "torque ON requested")
        )

    def _joint_cb(self, msg: JointState) -> None:
        if len(msg.name) != len(msg.position):
            return
        values: Dict[str, float] = dict(zip(msg.name, msg.position))
        if not all(name in values for name in JOINT_NAMES):
            return
        q = np.array([values[name] for name in JOINT_NAMES], dtype=float)
        if np.all(np.isfinite(q)):
            self.latest_q = q
            self.latest_stamp = time.monotonic()

    def get_fresh_joint_state(self) -> np.ndarray:
        start = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_q is not None and self.latest_stamp is not None:
                age = time.monotonic() - self.latest_stamp
                if age <= JOINT_STATE_MAX_AGE_S:
                    return validate_configuration(self.latest_q.copy(), "Measured current state")
            if time.monotonic() - start > JOINT_STATE_TIMEOUT_S:
                raise TimeoutError(
                    f"No fresh four-joint state received from {JOINT_STATE_TOPIC} "
                    f"within {JOINT_STATE_TIMEOUT_S:.1f} seconds."
                )
        raise RuntimeError("ROS shutdown while waiting for joint states.")

    def send_trajectory(self, sample_times, q_path, qd_path) -> None:
        if not self.arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Trajectory action server unavailable: {ARM_ACTION}")

        trajectory = JointTrajectory()
        trajectory.joint_names = JOINT_NAMES

        for t, q, qd in zip(sample_times, q_path, qd_path):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in q]
            point.velocities = [float(v) for v in qd]
            sec = int(math.floor(float(t)))
            nsec = int(round((float(t) - sec) * 1e9))
            if nsec >= 1_000_000_000:
                sec += 1
                nsec -= 1_000_000_000
            point.time_from_start = Duration(sec=sec, nanosec=nsec)
            trajectory.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = Duration(sec=int(GOAL_TIME_TOLERANCE_S))

        self.get_logger().info("Sending joint-space trajectory to arm controller...")
        send_future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Arm controller rejected the trajectory goal.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError("Arm controller returned no trajectory result.")
        result = wrapped.result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                f"Trajectory failed: error_code={result.error_code}, "
                f"message='{result.error_string}'"
            )
        self.get_logger().info("Target joint configuration reached successfully.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true",
        help="send the trajectory after an explicit MOVE confirmation (default: dry plan only)",
    )
    args = parser.parse_args()

    target_q = validate_configuration(TARGET_Q, "Requested target")
    rclpy.init()
    node = MoveToPoseNode()
    try:
        if args.execute:
            print("\nEXECUTION MODE: requesting Dynamixel torque ON before reading the start pose.")
            print("Keep hands clear. Enabling torque can make the arm hold or resist movement.")
            node.enable_torque()
            # Discard any cached pre-enable sample and wait for fresh feedback.
            node.latest_q = None
            node.latest_stamp = None

        current_q = node.get_fresh_joint_state()
        sample_times, q_path, qd_path, duration, delta = plan_joint_trajectory(
            current_q, target_q
        )

        print("\nCURRENT JOINT CONFIGURATION (rad)")
        for name, value in zip(JOINT_NAMES, current_q):
            print(f"  {name}: {value:+.5f}")
        print("\nTARGET JOINT CONFIGURATION (rad)")
        for name, value in zip(JOINT_NAMES, target_q):
            print(f"  {name}: {value:+.5f}")
        print("\nPLANNED JOINT CHANGE (rad)")
        for name, value in zip(JOINT_NAMES, delta):
            print(f"  {name}: {value:+.5f}")
        print(f"\nPlanned duration: {duration:.2f} s")
        print(f"Trajectory samples: {len(q_path)} at up to {FREQUENCY_HZ:.0f} Hz")
        print("Motion type: quintic joint-space interpolation")
        print("No collision checking is performed by this script.")
        if args.execute:
            print(f"Torque service: {TORQUE_SERVICE} (SetBool data=True requested)")

        if not args.execute:
            print("\nDRY PLAN ONLY. No robot command was sent.")
            print("After inspecting the workspace and planned move, rerun with --execute.")
            print("Torque is not changed in dry-plan mode.")
            return 0

        print("\nBefore execution, verify the arm can move through the entire joint-space route.")
        print("Keep the workspace clear and the emergency stop accessible.")
        if input("Type MOVE to send the trajectory: ").strip() != "MOVE":
            print("Cancelled. No robot command was sent.")
            return 0

        node.send_trajectory(sample_times, q_path, qd_path)
        # Wait for measured joint feedback to converge; an action result alone
        # does not prove that the physical joints reached the requested pose.
        deadline = time.monotonic() + FINAL_STATE_TIMEOUT_S
        final_q = node.get_fresh_joint_state()
        final_error = float(np.max(np.abs(final_q - target_q)))
        while final_error > FINAL_JOINT_TOLERANCE_RAD and time.monotonic() < deadline:
            time.sleep(0.10)
            final_q = node.get_fresh_joint_state()
            final_error = float(np.max(np.abs(final_q - target_q)))

        print("Measured final joints:", np.round(final_q, 5))
        print(f"Maximum final joint error: {final_error:.5f} rad")
        if final_error > FINAL_JOINT_TOLERANCE_RAD:
            raise RuntimeError(
                "The trajectory action completed, but measured joints did not "
                f"reach the target within {FINAL_JOINT_TOLERANCE_RAD:.2f} rad. "
                "Check torque, controller state, motor errors, and joint feedback."
            )
        return 0
    except Exception as exc:  # print a concise failure while ensuring ROS shutdown
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
