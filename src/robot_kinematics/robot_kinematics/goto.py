#!/usr/bin/env python3
"""Move OpenMANIPULATOR-X to the POSE SAMPLE 17 home TCP pose.

Execution:
    python3 goto.py                 # dry-run only
    python3 goto.py --execute       # request torque ON, then confirm with MOVE

Safety behaviour:
- Uses the joint limits in robot_kinematics.kinematics; it does not widen them.
- The exact sampled joint vector has joint4=2.02485 rad, outside the URDF and
  kinematics limit (1.97 rad). It is never commanded.
- The script preserves the recorded HOME TCP XYZ, then searches for the
  closest feasible tool pitch and an IK configuration within the existing
  joint limits. If none is found, it stops without sending motion.
- Torque is requested ON before reading the live start state in --execute mode.
- Final success is checked against fresh /joint_states feedback.

No collision checking is performed. Test unloaded, keep the workspace clear,
and keep the emergency stop accessible.
"""

import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration

from robot_kinematics.kinematics import (
    JOINT_LIMITS,
    forward_kinematics,
    inverse_kinematics,
)


# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]

# POSE SAMPLE 17: recorded joint vector. joint4=2.02485 rad is outside the
# current local JOINT_LIMITS, so this is retained as a reference, not blindly
# commanded. The script will solve for a legal equivalent configuration at
# the same TCP position and tool pitch.
REFERENCE_SAMPLE_Q = np.array(
    [-0.13346, -1.29315, 0.04142, 2.02485], dtype=float
)

# End-effector position reported for POSE SAMPLE 17 (metres).
HOME_XYZ = np.array([0.02445, -0.00167, 0.16441], dtype=float)

# Preserve the tool pitch represented by POSE SAMPLE 17 (q2 + q3 + q4).
HOME_TOOL_PITCH = float(np.sum(REFERENCE_SAMPLE_Q[1:]))

JOINT_STATE_TOPIC = "/joint_states"
ARM_ACTION = "/arm_controller/follow_joint_trajectory"
TORQUE_SERVICE = "/dynamixel_hardware_interface/set_dxl_torque"

FREQUENCY_HZ = 50.0
MIN_DURATION = 5.0
MAX_JOINT_VELOCITY = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)       # rad/s
MAX_JOINT_ACCELERATION = np.array([0.8, 0.8, 0.8, 0.8], dtype=float)  # rad/s^2

JOINT_STATE_TIMEOUT = 10.0
JOINT_STATE_MAX_AGE = 0.5
ACTION_SERVER_TIMEOUT = 10.0
ACTION_RESULT_TIMEOUT = 30.0
FINAL_POSITION_TIMEOUT = 8.0
FINAL_JOINT_TOLERANCE = 0.05  # rad
HOME_POSITION_TOLERANCE = 0.002  # m
HOME_PITCH_TOLERANCE = 0.002  # rad

JOINT_LIMITS = np.asarray(JOINT_LIMITS, dtype=float)
if JOINT_LIMITS.shape != (4, 2):
    raise RuntimeError(f"Expected JOINT_LIMITS shape (4, 2), got {JOINT_LIMITS.shape}.")


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def make_duration(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def validate_joint_vector(q: np.ndarray, label: str) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError(f"{label} must contain four finite joint values.")
    outside = np.where((q < JOINT_LIMITS[:, 0]) | (q > JOINT_LIMITS[:, 1]))[0]
    if len(outside):
        details = ", ".join(
            f"{JOINT_NAMES[i]}={q[i]:.5f} not in "
            f"[{JOINT_LIMITS[i, 0]:.5f}, {JOINT_LIMITS[i, 1]:.5f}]"
            for i in outside
        )
        raise RuntimeError(f"{label} violates robot joint limits: {details}")
    return q


def forward_pose(q: np.ndarray):
    result = forward_kinematics(*np.asarray(q, dtype=float))
    xyz = np.asarray(result[0], dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise RuntimeError("Forward kinematics returned an invalid XYZ position.")
    return xyz


def _candidate_home_solution(pitch: float, seed_q: np.ndarray):
    """Try one pitch/seed pair and return a verified legal IK solution or None."""
    try:
        q = inverse_kinematics(
            x=float(HOME_XYZ[0]),
            y=float(HOME_XYZ[1]),
            z=float(HOME_XYZ[2]),
            angle=float(pitch),
            previous_solution=np.asarray(seed_q, dtype=float),
        )
        if q is None:
            return None
        q = np.asarray(q, dtype=float)
        if q.shape != (4,) or not np.all(np.isfinite(q)):
            return None
        if np.any(q < JOINT_LIMITS[:, 0] - 1e-9) or np.any(
            q > JOINT_LIMITS[:, 1] + 1e-9
        ):
            return None

        xyz = forward_pose(q)
        position_error = float(np.linalg.norm(xyz - HOME_XYZ))
        actual_pitch = float(np.sum(q[1:]))
        pitch_error = abs(wrap_angle(actual_pitch - pitch))
        if position_error > HOME_POSITION_TOLERANCE:
            return None
        if pitch_error > HOME_PITCH_TOLERANCE:
            return None
        return q
    except (ValueError, RuntimeError, FloatingPointError):
        return None


def resolve_legal_home_configuration(seed_q: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Find the closest-pitch in-limit IK solution for the POSE SAMPLE 17 TCP.

    The reference joint vector is outside the confirmed URDF limit on joint4.
    Rather than widening limits, keep HOME_XYZ fixed and relax only the pitch
    preference. Search outward from the recorded pitch, checking both
    directions. A coarse scan finds a feasible interval; a finer scan then
    finds a closer feasible pitch.
    """
    exact_outside = np.where(
        (REFERENCE_SAMPLE_Q < JOINT_LIMITS[:, 0])
        | (REFERENCE_SAMPLE_Q > JOINT_LIMITS[:, 1])
    )[0]

    if len(exact_outside) == 0:
        q = validate_joint_vector(REFERENCE_SAMPLE_Q.copy(), "POSE SAMPLE 17")
        xyz = forward_pose(q)
        pos_error = float(np.linalg.norm(xyz - HOME_XYZ))
        if pos_error <= HOME_POSITION_TOLERANCE:
            return q, HOME_TOOL_PITCH, "exact POSE SAMPLE 17 joint configuration"

    print("\n[HOME IK] Reference joint vector is not directly commandable:")
    for i in exact_outside:
        print(
            f"  {JOINT_NAMES[i]}={REFERENCE_SAMPLE_Q[i]:.5f} rad; URDF/IK limit "
            f"[{JOINT_LIMITS[i, 0]:.5f}, {JOINT_LIMITS[i, 1]:.5f}] rad"
        )
    print("[HOME IK] Keeping HOME XYZ fixed; searching for the nearest feasible tool pitch.")
    print("[HOME IK] Joint limits will not be widened.")

    # Distinct, limit-compliant initial guesses help the IK solver select a
    # valid branch. Clip the reference away from exact boundaries so bounded
    # least-squares solvers do not start exactly on a limit.
    eps = 1e-4
    clipped_reference = np.clip(
        REFERENCE_SAMPLE_Q,
        JOINT_LIMITS[:, 0] + eps,
        JOINT_LIMITS[:, 1] - eps,
    )
    current_seed = np.clip(
        np.asarray(seed_q, dtype=float),
        JOINT_LIMITS[:, 0] + eps,
        JOINT_LIMITS[:, 1] - eps,
    )
    seeds = [current_seed, clipped_reference]

    # Remove duplicate seeds.
    unique_seeds = []
    for candidate_seed in seeds:
        if not any(np.allclose(candidate_seed, prev, atol=1e-8) for prev in unique_seeds):
            unique_seeds.append(candidate_seed)

    max_deviation = np.deg2rad(20.0)
    coarse_step = np.deg2rad(0.5)
    coarse_count = int(round(max_deviation / coarse_step))
    feasible = None

    # Search the recorded pitch first, then alternate +/- offsets in increasing
    # absolute deviation so the first feasible pitch is close to the sample.
    offsets = [0.0]
    for k in range(1, coarse_count + 1):
        offsets.extend((k * coarse_step, -k * coarse_step))

    for offset in offsets:
        pitch = HOME_TOOL_PITCH + offset
        pitch_solutions = []
        for candidate_seed in unique_seeds:
            q = _candidate_home_solution(pitch, candidate_seed)
            if q is not None:
                # Prefer a solution that requires less joint displacement from
                # the current measured pose; both candidates satisfy the same
                # Cartesian position and selected pitch.
                cost = float(np.linalg.norm(q - seed_q))
                pitch_solutions.append((cost, q))
        if pitch_solutions:
            _, best_q = min(pitch_solutions, key=lambda item: item[0])
            feasible = (best_q, pitch)
            break

    if feasible is None:
        raise RuntimeError(
            "No IK solution was found for the POSE SAMPLE 17 HOME XYZ within "
            "the current joint limits, even after searching tool pitch within "
            "+/-20 degrees of the recorded 44.297-degree pitch. No trajectory "
            "was sent. Do not widen the URDF/kinematics limits to force the "
            "out-of-range joint4 sample."
        )

    coarse_q, coarse_pitch = feasible
    coarse_offset = coarse_pitch - HOME_TOOL_PITCH
    if abs(coarse_offset) > 1e-12:
        # Refine between the reference pitch and the first feasible coarse
        # sample, at 0.05-degree resolution, on the same side of the reference.
        fine_step = np.deg2rad(0.05)
        fine_count = max(0, int(np.floor(abs(coarse_offset) / fine_step)) - 1)
        direction = 1.0 if coarse_offset > 0 else -1.0
        best_q, best_pitch = coarse_q, coarse_pitch
        for k in range(1, fine_count + 1):
            pitch = HOME_TOOL_PITCH + direction * k * fine_step
            pitch_solutions = []
            for candidate_seed in unique_seeds:
                q = _candidate_home_solution(pitch, candidate_seed)
                if q is not None:
                    cost = float(np.linalg.norm(q - seed_q))
                    pitch_solutions.append((cost, q))
            if pitch_solutions:
                _, best_q = min(pitch_solutions, key=lambda item: item[0])
                best_pitch = pitch
                break
        feasible = (best_q, best_pitch)

    q_solution, selected_pitch = feasible
    q_solution = validate_joint_vector(q_solution, "Limit-compliant HOME IK solution")
    xyz_solution = forward_pose(q_solution)
    pos_error = float(np.linalg.norm(xyz_solution - HOME_XYZ))
    actual_pitch = float(np.sum(q_solution[1:]))
    pitch_error = abs(wrap_angle(actual_pitch - selected_pitch))

    if pos_error > HOME_POSITION_TOLERANCE:
        raise RuntimeError(
            f"Selected HOME IK solution misses HOME_XYZ by {pos_error * 1000:.2f} mm. "
            "No trajectory was sent."
        )
    if pitch_error > HOME_PITCH_TOLERANCE:
        raise RuntimeError(
            f"Selected HOME IK solution pitch error is {pitch_error:.5f} rad. "
            "No trajectory was sent."
        )

    return (
        q_solution,
        selected_pitch,
        "limit-compliant IK at the exact POSE SAMPLE 17 HOME XYZ "
        "with the closest feasible pitch found",
    )


def quintic_trajectory(q_start: np.ndarray, q_goal: np.ndarray):
    """Return sampled positions, velocities, accelerations, and timestamps."""
    delta = np.asarray(q_goal, dtype=float) - np.asarray(q_start, dtype=float)

    # Peak |ds/du| for quintic blend = 1.875; peak |d2s/du2| = 10/sqrt(3).
    duration_velocity = float(
        np.max(1.875 * np.abs(delta) / MAX_JOINT_VELOCITY)
    )
    duration_acceleration = float(
        np.max(
            np.sqrt(
                (10.0 / np.sqrt(3.0))
                * np.abs(delta)
                / MAX_JOINT_ACCELERATION
            )
        )
    )
    duration = max(MIN_DURATION, duration_velocity, duration_acceleration)

    sample_count = max(3, int(np.ceil(duration * FREQUENCY_HZ)) + 1)
    times = np.linspace(0.0, duration, sample_count)
    u = times / duration

    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    ds_du = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    d2s_du2 = 60.0 * u - 180.0 * u**2 + 120.0 * u**3

    positions = q_start[None, :] + s[:, None] * delta[None, :]
    velocities = (ds_du[:, None] / duration) * delta[None, :]
    accelerations = (d2s_du2[:, None] / duration**2) * delta[None, :]

    velocities[0] = 0.0
    velocities[-1] = 0.0
    accelerations[0] = 0.0
    accelerations[-1] = 0.0

    if np.any(positions < JOINT_LIMITS[:, 0] - 1e-9) or np.any(
        positions > JOINT_LIMITS[:, 1] + 1e-9
    ):
        raise RuntimeError("Planned trajectory crosses a joint limit. No command sent.")
    if np.any(np.abs(velocities) > MAX_JOINT_VELOCITY[None, :] * 1.001):
        raise RuntimeError("Planned trajectory exceeds configured joint velocity limits.")
    if np.any(np.abs(accelerations) > MAX_JOINT_ACCELERATION[None, :] * 1.001):
        raise RuntimeError("Planned trajectory exceeds configured joint acceleration limits.")

    return duration, times, positions, velocities, accelerations


class MoveToTargetNode(Node):
    def __init__(self):
        super().__init__("move_to_pose_sample_17_home")
        self.current_q = None
        self.last_joint_state_time = None
        self.joint_state_count = 0

        self.create_subscription(JointState, JOINT_STATE_TOPIC, self.joint_callback, 10)
        self.torque_client = self.create_client(SetBool, TORQUE_SERVICE)
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION)

    def joint_callback(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            return
        joint_map = dict(zip(msg.name, msg.position))
        if not all(name in joint_map for name in JOINT_NAMES):
            return
        q = np.array([joint_map[name] for name in JOINT_NAMES], dtype=float)
        if np.all(np.isfinite(q)):
            self.current_q = q
            self.last_joint_state_time = time.monotonic()
            self.joint_state_count += 1

    def enable_torque(self):
        self.get_logger().info(f"Requesting torque ON through {TORQUE_SERVICE}")
        if not self.torque_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"Torque service unavailable: {TORQUE_SERVICE}")

        request = SetBool.Request()
        request.data = True
        future = self.torque_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done():
            raise RuntimeError("Timed out waiting for the torque-enable service.")

        response = future.result()
        if response is None:
            raise RuntimeError("Torque-enable service returned no response.")
        if not response.success:
            raise RuntimeError(f"Torque enable failed: {response.message}")

        self.get_logger().info(
            f"Torque-enable request succeeded: {response.message or 'torque enabled'}"
        )

    def get_fresh_joint_positions(self) -> np.ndarray:
        start = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if (
                self.current_q is not None
                and self.last_joint_state_time is not None
                and time.monotonic() - self.last_joint_state_time <= JOINT_STATE_MAX_AGE
            ):
                return validate_joint_vector(self.current_q.copy(), "Measured current joints")
            if time.monotonic() - start > JOINT_STATE_TIMEOUT:
                raise RuntimeError(
                    f"No fresh complete joint state received on {JOINT_STATE_TOPIC}."
                )
        raise RuntimeError("ROS shutdown while waiting for joint states.")

    def wait_until_target(self, target: np.ndarray):
        start = time.monotonic()
        consecutive_good = 0
        last_count = self.joint_state_count

        while rclpy.ok() and time.monotonic() - start < FINAL_POSITION_TIMEOUT:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_q is None or self.last_joint_state_time is None:
                continue
            if time.monotonic() - self.last_joint_state_time > JOINT_STATE_MAX_AGE:
                continue
            if self.joint_state_count == last_count:
                continue
            last_count = self.joint_state_count

            error = np.abs(self.current_q - target)
            if np.all(error <= FINAL_JOINT_TOLERANCE):
                consecutive_good += 1
                if consecutive_good >= 5:
                    return self.current_q.copy(), error
            else:
                consecutive_good = 0

        measured = None if self.current_q is None else self.current_q.copy()
        if measured is None:
            raise RuntimeError("No joint feedback received after trajectory execution.")
        error = np.abs(measured - target)
        raise RuntimeError(
            "Trajectory action completed, but fresh joint feedback did not reach "
            f"the commanded target within {FINAL_POSITION_TIMEOUT:.1f} s. "
            f"Measured={np.round(measured, 5)}, "
            f"absolute error={np.round(error, 5)} rad."
        )

    def send_trajectory(self, times, positions, velocities, accelerations):
        self.get_logger().info(f"Waiting for trajectory action server: {ARM_ACTION}")
        if not self.arm_client.wait_for_server(timeout_sec=ACTION_SERVER_TIMEOUT):
            raise RuntimeError(f"Arm action server unavailable: {ARM_ACTION}")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        for t, q, qd, qdd in zip(times, positions, velocities, accelerations):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in q]
            point.velocities = [float(v) for v in qd]
            point.accelerations = [float(v) for v in qdd]
            point.time_from_start = make_duration(float(t))
            goal.trajectory.points.append(point)

        send_future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        if not send_future.done():
            raise RuntimeError("Timed out while submitting the arm trajectory.")
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Arm controller rejected the trajectory goal.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=ACTION_RESULT_TIMEOUT
        )
        if not result_future.done():
            goal_handle.cancel_goal_async()
            raise RuntimeError(
                "Timed out waiting for trajectory completion; cancellation requested."
            )

        wrapped_result = result_future.result()
        if wrapped_result is None:
            raise RuntimeError("Trajectory action returned no result.")
        result = wrapped_result.result
        if result.error_code != 0:
            raise RuntimeError(
                f"Trajectory action failed: error_code={result.error_code}, "
                f"error_string='{result.error_string}'"
            )
        self.get_logger().info("Trajectory action reports successful completion.")


def print_joint_block(title: str, q: np.ndarray):
    print(f"\n{title}")
    for name, value in zip(JOINT_NAMES, q):
        print(f"  {name}: {value:+.5f} rad ({np.rad2deg(value):+.2f} degrees)")


def main():
    parser = argparse.ArgumentParser(
        description="Move OpenMANIPULATOR-X to the POSE SAMPLE 17 HOME TCP pose."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Request Dynamixel torque ON and send the trajectory after confirmation.",
    )
    args = parser.parse_args()

    print("POSE SAMPLE 17 reference joints:", np.round(REFERENCE_SAMPLE_Q, 5))
    print("POSE SAMPLE 17 HOME XYZ (m):", np.round(HOME_XYZ, 5))
    print(
        f"Reference tool pitch: {HOME_TOOL_PITCH:.6f} rad "
        f"({np.rad2deg(HOME_TOOL_PITCH):.3f} deg)"
    )
    print("Loaded kinematics joint limits:\n", JOINT_LIMITS)

    rclpy.init()
    node = MoveToTargetNode()

    try:
        if args.execute:
            print("\nEXECUTION MODE: requesting Dynamixel torque ON before reading start pose.")
            print("Keep hands clear. Enabling torque can make the arm hold or resist movement.")
            node.enable_torque()
        else:
            print("DRY-RUN MODE: no torque request and no arm movement will be commanded.")

        current = node.get_fresh_joint_positions()
        print_joint_block("CURRENT JOINT CONFIGURATION", current)

        target, selected_pitch, target_description = resolve_legal_home_configuration(current)
        print("\nSelected target:", target_description)
        print(
            f"Selected HOME tool pitch: {selected_pitch:.6f} rad "
            f"({np.rad2deg(selected_pitch):.3f} deg); "
            f"reference was {np.rad2deg(HOME_TOOL_PITCH):.3f} deg"
        )
        print_joint_block("COMMAND TARGET JOINT CONFIGURATION", target)
        print("FK of command target (m):", np.round(forward_pose(target), 5))
        print("Requested HOME XYZ (m):", np.round(HOME_XYZ, 5))
        print("Joint change (rad):", np.round(target - current, 5))

        duration, times, positions, velocities, accelerations = quintic_trajectory(
            current, target
        )
        print(f"\nPlanned duration: {duration:.2f} s")
        print(f"Trajectory samples: {len(times)} at up to {FREQUENCY_HZ:.0f} Hz")
        print("Motion type: quintic joint-space interpolation")
        print("Software velocity limits:", MAX_JOINT_VELOCITY, "rad/s")
        print("Software acceleration limits:", MAX_JOINT_ACCELERATION, "rad/s^2")
        print("No collision checking is performed by this script.")
        print(f"Torque service: {TORQUE_SERVICE}")

        if not args.execute:
            print("DRY RUN: trajectory was planned but not sent.")
            return 0

        print("\nBefore execution, verify that the complete joint-space route is clear.")
        print("Keep the workspace clear and the emergency stop accessible.")
        if input("Type MOVE to send the trajectory: ").strip() != "MOVE":
            print("Cancelled. No trajectory sent.")
            return 0

        print("Sending joint-space trajectory to arm controller...")
        node.send_trajectory(times, positions, velocities, accelerations)

        measured_final, final_error = node.wait_until_target(target)
        print_joint_block("MEASURED FINAL JOINT CONFIGURATION", measured_final)
        print(f"Maximum final joint error: {float(np.max(final_error)):.5f} rad")
        print("Target HOME TCP pose reached successfully.")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("No additional motion will be commanded.")
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
