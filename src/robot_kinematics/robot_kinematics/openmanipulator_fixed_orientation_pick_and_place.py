#!/usr/bin/env python3
"""OpenMANIPULATOR-X fixed-pitch pick-and-place controller.

The four main pick/place phases use at most 15 UNIQUE Cartesian IK waypoints:
  1) descend to pick:       5 points (including shared start/end)
  2) lift to Z=0.20 m:      4 points (3 new points; pick contact is shared)
  3) XY transfer at 0.20 m: 4 points (3 new points; pick-side start is shared)
  4) descend to place:      5 points (4 new points; place-side start is shared)

Unique count = 5 + 3 + 3 + 4 = 15.
The initial move from home to above the pick and the post-release retract are
separate safety segments, not part of the 15-waypoint four-phase transfer.

All Cartesian waypoint IK solutions are chained using the previous solution.
The actual cubic joint interpolation sent to FollowJointTrajectory is sampled
with FK and checked before the trajectory is sent. DRY_RUN remains enabled.

WARNING: This is not a collision-checked or safety-rated motion planner.
Verify the kinematic model, TCP calibration, gripper direction, clearances,
controller interpolation behavior, and emergency-stop operation before use.
"""

import os
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from robot_kinematics.kinematics import (
    forward_kinematics,
    inverse_kinematics,
    JOINT_LIMITS,
)


# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]
JOINT_STATE_TOPIC = "/joint_states"
ARM_ACTION = "/arm_controller/follow_joint_trajectory"
GRIPPER_ACTION = "/gripper_controller/gripper_cmd"

# Primary home pose: measured POSE SAMPLE 17 supplied by the user.
# Joint values are in radians; XYZ is the corresponding measured TCP position.
TARGET_Q = np.array([-0.13346, -1.29315, 0.04142, 2.02485], dtype=float)
HOME_Q = TARGET_Q.copy()
HOME_ANGLE = float(HOME_Q[1] + HOME_Q[2] + HOME_Q[3])
TASK_TOOL_ANGLE = float(np.deg2rad(45.0))

# TCP position recorded for POSE SAMPLE 17 (metres).
HOME_XYZ = np.array([0.02445, -0.00167, 0.16441], dtype=float)

PICK_CONTACT_Z = 0.060
PLACE_CONTACT_Z = 0.060
TRAVEL_Z = 0.200                 # Requested 20 cm lift and XY transfer plane.
MIN_TRANSFER_Z = 0.150            # Hard floor for non-contact travel.

# Four main phases. Counts include both endpoints of each individual phase.
# Since adjoining phases share their common endpoint, the unique count is 15.
PICK_DESCENT_WAYPOINTS = 5
PICK_LIFT_WAYPOINTS = 4
XY_TRANSFER_WAYPOINTS = 4
PLACE_DESCENT_WAYPOINTS = 5
MAIN_UNIQUE_WAYPOINT_COUNT = (
    PICK_DESCENT_WAYPOINTS
    + (PICK_LIFT_WAYPOINTS - 1)
    + (XY_TRANSFER_WAYPOINTS - 1)
    + (PLACE_DESCENT_WAYPOINTS - 1)
)
MAX_MAIN_UNIQUE_WAYPOINTS = 15

# Approach/retract/return paths are separate from the 15-point four-phase path.
APPROACH_WAYPOINTS = 5
POST_PLACE_RETRACT_WAYPOINTS = 5
RETURN_HOME_WAYPOINTS = 7

MAX_HORIZONTAL_Z_DEVIATION = 0.005  # 5 mm maximum height drift during XY transfer.
MAX_VERTICAL_XY_DRIFT = 0.005       # 5 mm maximum lateral drift during vertical moves.
MAX_GENERAL_CARTESIAN_DEVIATION = 0.030  # 30 mm for unconstrained approach/return.
PATH_CHECK_HZ = 50.0

SORT_TARGETS_FILE = Path(os.environ.get(
    "SHAPE_SORT_TARGETS_FILE",
    Path(__file__).resolve().with_name("shape_sort_targets.npz"),
)).expanduser().resolve()

SORT_SHAPES = (
    "green_circle",
    "blue_square",
    "orange_hexagon",
    "yellow_triangle",
)

# GripperCommand positions are in metres. Verify direction with a gripper-only test.
GRIPPER_OPEN_POSITION = 0.000
GRIPPER_MAX_SEPARATION_POSITION = GRIPPER_OPEN_POSITION
GRIPPER_CLOSED_POSITION = 0.010
GRIPPER_HALF_CLOSED_POSITION = (
    GRIPPER_OPEN_POSITION + GRIPPER_CLOSED_POSITION
) / 2.0
GRIPPER_MAX_EFFORT = 20.0

# Keep True until every path has been inspected and validated on the real robot.
DRY_RUN = True
REQUIRE_EXECUTE_CONFIRMATION = True

FREQUENCY_HZ = 50.0
HOME_DURATION = 5.0
CARTESIAN_DURATION = 3.0
VERTICAL_DURATION = 3.0
MAX_JOINT_VELOCITY = np.array([1.5, 1.5, 1.5, 1.5], dtype=float)
MAX_JOINT_ACCELERATION = np.array([3.0, 3.0, 3.0, 3.0], dtype=float)

JOINT_STATE_TIMEOUT = 10.0
JOINT_STATE_MAX_AGE = 0.5
POSITION_TOLERANCE = 0.002
MAX_SEGMENT_START_ERROR = 0.010
ORIENTATION_TOLERANCE = 2e-3
START_ORIENTATION_CORRECTION_TOLERANCE = 0.02
JOINT_GOAL_TOLERANCE = 0.05

STARTUP_MIN_Z = MIN_TRANSFER_Z
STARTUP_MAX_Z = 0.300


# -----------------------------------------------------------------------------
# KINEMATICS AND VALIDATION HELPERS
# -----------------------------------------------------------------------------

def quintic(s):
    s = float(np.clip(s, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def make_times(duration):
    count = int(round(float(duration) * FREQUENCY_HZ)) + 1
    if count < 3:
        raise ValueError("Trajectory requires at least three samples.")
    return np.linspace(0.0, float(duration), count)


def wrap_delta(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def fk(q):
    result = forward_kinematics(*np.asarray(q, dtype=float))
    position = np.asarray(result[0], dtype=float)
    rotation = np.asarray(result[1], dtype=float)
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise RuntimeError("Unexpected forward_kinematics return shape.")
    return position, rotation


def validate_q(q, label):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError(f"{label} must contain four finite joint values.")
    if np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1]):
        raise RuntimeError(f"{label} violates JOINT_LIMITS: {q}")
    return q


def expected_tool_x_axis(q1, angle):
    return np.array([
        np.cos(q1) * np.cos(angle),
        np.sin(q1) * np.cos(angle),
        -np.sin(angle),
    ], dtype=float)


def verify_orientation(rotation, q1, angle, label):
    expected = expected_tool_x_axis(q1, angle)
    actual = np.asarray(rotation, dtype=float)[:, 0]
    error = float(np.linalg.norm(actual - expected))
    if error > ORIENTATION_TOLERANCE:
        raise RuntimeError(
            f"{label}: orientation check failed; axis error={error:.6g}."
        )


def verify_pose(q, xyz, label, tool_angle):
    actual_position, actual_rotation = fk(q)
    position_error = float(np.linalg.norm(actual_position - xyz))
    if position_error > POSITION_TOLERANCE:
        raise RuntimeError(
            f"{label}: FK position error {position_error:.4f} m exceeds "
            f"{POSITION_TOLERANCE:.4f} m."
        )
    pitch_error = abs(wrap_delta(float(q[1] + q[2] + q[3]), tool_angle))
    if pitch_error > ORIENTATION_TOLERANCE:
        raise RuntimeError(f"{label}: pitch constraint error={pitch_error:.6g} rad.")
    verify_orientation(actual_rotation, float(q[0]), tool_angle, label)
    return actual_position, actual_rotation


def solve_ik(xyz, previous_q, tool_angle=None):
    if tool_angle is None:
        tool_angle = TASK_TOOL_ANGLE
    x, y, z = [float(value) for value in xyz]
    q = inverse_kinematics(
        x=x,
        y=y,
        z=z,
        angle=float(tool_angle),
        previous_solution=np.asarray(previous_q, dtype=float),
    )
    if q is None:
        raise RuntimeError(
            f"IK failed at XYZ=({x:.4f}, {y:.4f}, {z:.4f}) m "
            f"for fixed tool pitch={tool_angle:.5f} rad."
        )
    q = validate_q(q, "IK result")
    verify_pose(q, np.array([x, y, z], dtype=float), "IK result", float(tool_angle))
    return q


def derivatives(q_path, times):
    q_path = np.asarray(q_path, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(times) == 2:
        return np.zeros_like(q_path), np.zeros_like(q_path)
    qd = np.gradient(q_path, times, axis=0, edge_order=2)
    qdd = np.gradient(qd, times, axis=0, edge_order=2)
    qd[0] = 0.0
    qd[-1] = 0.0
    return qd, qdd


def validate_joint_path(q_path, times, label):
    q_path = np.asarray(q_path, dtype=float)
    times = np.asarray(times, dtype=float)
    if q_path.ndim != 2 or q_path.shape[1] != 4:
        raise ValueError(f"{label}: expected an N x 4 joint path.")
    if len(q_path) != len(times) or len(times) < 3:
        raise ValueError(f"{label}: invalid path/time dimensions.")
    if np.any(q_path < JOINT_LIMITS[:, 0]) or np.any(q_path > JOINT_LIMITS[:, 1]):
        raise RuntimeError(f"{label}: joint limit violation.")
    qd, qdd = derivatives(q_path, times)
    max_v = np.max(np.abs(qd), axis=0)
    max_a = np.max(np.abs(qdd), axis=0)
    if np.any(max_v > MAX_JOINT_VELOCITY):
        raise RuntimeError(f"{label}: velocity limit exceeded: {max_v}")
    if np.any(max_a > MAX_JOINT_ACCELERATION):
        raise RuntimeError(f"{label}: acceleration limit exceeded: {max_a}")
    return qd, qdd


# -----------------------------------------------------------------------------
# JOINT-TRAJECTORY INTERPOLATION MODEL
# -----------------------------------------------------------------------------

def cubic_hermite_samples(q0, q1, v0, v1, dt, s_values):
    """Reproduce cubic position/velocity interpolation for positions+velocities."""
    s = np.asarray(s_values, dtype=float)[:, None]
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2

    q = h00 * q0 + h10 * dt * v0 + h01 * q1 + h11 * dt * v1

    dh00 = (6.0 * s**2 - 6.0 * s) / dt
    dh10 = 3.0 * s**2 - 4.0 * s + 1.0
    dh01 = (-6.0 * s**2 + 6.0 * s) / dt
    dh11 = 3.0 * s**2 - 2.0 * s
    qd = dh00 * q0 + dh10 * v0 + dh01 * q1 + dh11 * v1

    ddh00 = (12.0 * s - 6.0) / (dt**2)
    ddh10 = (6.0 * s - 4.0) / dt
    ddh01 = (-12.0 * s + 6.0) / (dt**2)
    ddh11 = (6.0 * s - 2.0) / dt
    qdd = ddh00 * q0 + ddh10 * v0 + ddh01 * q1 + ddh11 * v1
    return q, qd, qdd


def sample_ros_trajectory(q_path, times, qd_path):
    """Sample the cubic interpolation that is represented by sent ROS points."""
    all_q = []
    all_qd = []
    all_qdd = []
    all_segment_ids = []
    all_segment_s = []

    for index in range(len(q_path) - 1):
        dt = float(times[index + 1] - times[index])
        if dt <= 0.0:
            raise ValueError("Trajectory times must be strictly increasing.")
        sample_count = max(16, int(np.ceil(dt * PATH_CHECK_HZ)) + 1)
        s_values = np.linspace(0.0, 1.0, sample_count)
        if index > 0:
            s_values = s_values[1:]  # Do not duplicate shared segment endpoints.
        q_samples, qd_samples, qdd_samples = cubic_hermite_samples(
            q_path[index], q_path[index + 1],
            qd_path[index], qd_path[index + 1],
            dt, s_values,
        )
        all_q.append(q_samples)
        all_qd.append(qd_samples)
        all_qdd.append(qdd_samples)
        all_segment_ids.extend([index] * len(s_values))
        all_segment_s.extend(s_values.tolist())

    return (
        np.vstack(all_q),
        np.vstack(all_qd),
        np.vstack(all_qdd),
        np.asarray(all_segment_ids, dtype=int),
        np.asarray(all_segment_s, dtype=float),
    )


# -----------------------------------------------------------------------------
# TRAJECTORY PLANNERS
# -----------------------------------------------------------------------------

def plan_startup_joint_move(q_start, q_goal):
    """Reach a specified joint configuration using a quintic joint-space path."""
    q_start = validate_q(q_start, "Measured starting joints")
    q_goal = validate_q(q_goal, "Joint-space goal")

    duration = HOME_DURATION
    delta = wrap_delta(q_goal, q_start)
    unwrapped_goal = q_start + delta
    if np.any(unwrapped_goal < JOINT_LIMITS[:, 0]) or np.any(unwrapped_goal > JOINT_LIMITS[:, 1]):
        raise RuntimeError("Joint-space startup interpolation crosses a joint limit.")

    original_duration = duration
    for _ in range(15):
        times = make_times(duration)
        path = np.array([
            q_start + quintic(t / duration) * delta for t in times
        ])
        qd, qdd = derivatives(path, times)
        velocity_ratio = float(np.max(np.max(np.abs(qd), axis=0) / MAX_JOINT_VELOCITY))
        acceleration_ratio = float(np.max(np.max(np.abs(qdd), axis=0) / MAX_JOINT_ACCELERATION))
        if velocity_ratio <= 1.0 and acceleration_ratio <= 1.0:
            break
        duration *= max(1.05, 1.05 * velocity_ratio, 1.05 * np.sqrt(acceleration_ratio))
    else:
        raise RuntimeError("Startup move cannot satisfy joint velocity/acceleration limits.")

    if duration > original_duration * 1.001:
        print(f"Joint-space move timing adjusted: {original_duration:.2f}s -> {duration:.2f}s.")

    fk_points = np.asarray([fk(q)[0] for q in path])
    z_min = float(np.min(fk_points[:, 2]))
    z_max = float(np.max(fk_points[:, 2]))
    if z_min < STARTUP_MIN_Z - 1e-6 or z_max > STARTUP_MAX_Z:
        raise RuntimeError(
            f"Joint-space move FK Z range [{z_min:.3f}, {z_max:.3f}] m violates "
            f"[{STARTUP_MIN_Z:.3f}, {STARTUP_MAX_Z:.3f}] m. No command sent."
        )
    validate_joint_path(path, times, "Joint-space move")
    print(f"Joint-space FK Z range: [{z_min:.3f}, {z_max:.3f}] m; height check passed.")
    return times, path


def plan_cartesian_segment(
    start_xyz,
    goal_xyz,
    initial_q,
    duration,
    waypoint_count=5,
    label="Cartesian segment",
    vertical_only=False,
    horizontal_only=False,
    minimum_path_z=None,
):
    """Plan a Cartesian line using chained waypoint IK and validate ROS interpolation."""
    start_xyz = np.asarray(start_xyz, dtype=float)
    goal_xyz = np.asarray(goal_xyz, dtype=float)
    if start_xyz.shape != (3,) or goal_xyz.shape != (3,):
        raise ValueError("Cartesian endpoints must each have three coordinates.")
    if int(waypoint_count) < 3:
        raise ValueError("Cartesian waypoint_count must be at least 3.")

    q_start = validate_q(initial_q, f"Initial joints for {label}")
    actual_start_xyz, _ = fk(q_start)
    start_error = float(np.linalg.norm(actual_start_xyz - start_xyz))
    if start_error > MAX_SEGMENT_START_ERROR:
        raise RuntimeError(
            f"{label}: measured FK differs from requested start by {start_error*1000:.1f} mm; "
            "no trajectory sent. Check tracking and calibration."
        )
    if start_error > POSITION_TOLERANCE:
        print(f"[{label}] Start re-anchored to measured FK ({start_error*1000:.1f} mm).")
    start_xyz = actual_start_xyz.copy()

    start_pitch = float(q_start[1] + q_start[2] + q_start[3])
    pitch_error = abs(wrap_delta(start_pitch, TASK_TOOL_ANGLE))
    if pitch_error > START_ORIENTATION_CORRECTION_TOLERANCE:
        raise RuntimeError(
            f"{label}: starting pitch error {pitch_error:.5f} rad exceeds allowance."
        )

    if vertical_only:
        xy_error = float(np.linalg.norm(start_xyz[:2] - goal_xyz[:2]))
        if xy_error > POSITION_TOLERANCE:
            raise RuntimeError(
                f"{label}: vertical segment endpoints differ in XY by "
                f"{xy_error*1000:.1f} mm; no motion planned."
            )
        goal_xyz[:2] = start_xyz[:2]

    if horizontal_only:
        z_error = abs(float(start_xyz[2] - goal_xyz[2]))
        if z_error > POSITION_TOLERANCE:
            raise RuntimeError(
                f"{label}: horizontal segment endpoints differ in Z by "
                f"{z_error*1000:.1f} mm; no motion planned."
            )
        goal_xyz[2] = start_xyz[2]

    # The start waypoint is the measured configuration; solve every later point
    # with the previous solution to maintain a continuous IK branch.
    cartesian_waypoints = np.linspace(start_xyz, goal_xyz, int(waypoint_count))
    q_waypoints = [q_start.copy()]
    for waypoint_index in range(1, len(cartesian_waypoints)):
        q_next = solve_ik(
            cartesian_waypoints[waypoint_index],
            q_waypoints[-1],
            TASK_TOOL_ANGLE,
        )
        q_waypoints.append(q_next)
    q_path = np.asarray(q_waypoints, dtype=float)

    duration = float(duration)
    original_duration = duration
    for _ in range(15):
        times = np.linspace(0.0, duration, len(q_path))
        qd_path, _ = derivatives(q_path, times)
        q_check, qd_check, qdd_check, _, _ = sample_ros_trajectory(
            q_path, times, qd_path
        )

        velocity_ratio = float(np.max(np.max(np.abs(qd_check), axis=0) / MAX_JOINT_VELOCITY))
        acceleration_ratio = float(np.max(np.max(np.abs(qdd_check), axis=0) / MAX_JOINT_ACCELERATION))
        if velocity_ratio <= 1.0 and acceleration_ratio <= 1.0:
            break
        duration *= max(1.05, 1.05 * velocity_ratio, 1.05 * np.sqrt(acceleration_ratio))
    else:
        raise RuntimeError(f"{label}: cannot satisfy joint velocity/acceleration limits.")

    # Resample after the final duration has been selected.
    times = np.linspace(0.0, duration, len(q_path))
    qd_path, _ = derivatives(q_path, times)
    q_check, qd_check, qdd_check, segment_ids, segment_s = sample_ros_trajectory(
        q_path, times, qd_path
    )

    if np.any(q_check < JOINT_LIMITS[:, 0] - 1e-8) or np.any(q_check > JOINT_LIMITS[:, 1] + 1e-8):
        raise RuntimeError(f"{label}: interpolated trajectory violates a joint limit.")
    if np.any(np.abs(qd_check) > MAX_JOINT_VELOCITY[None, :] + 1e-7):
        raise RuntimeError(f"{label}: interpolated trajectory exceeds joint velocity limits.")
    if np.any(np.abs(qdd_check) > MAX_JOINT_ACCELERATION[None, :] + 1e-6):
        raise RuntimeError(f"{label}: interpolated trajectory exceeds joint acceleration limits.")

    xyz_check = np.asarray([fk(q)[0] for q in q_check], dtype=float)
    # Measure geometric cross-track error from each intended Cartesian line
    # segment. This avoids falsely treating a harmless speed/timing difference
    # along the line as a path deviation.
    path_deviation = np.zeros(len(xyz_check), dtype=float)
    for sample_index, segment_index in enumerate(segment_ids):
        line_start = cartesian_waypoints[segment_index]
        line_vector = cartesian_waypoints[segment_index + 1] - line_start
        line_length_sq = float(np.dot(line_vector, line_vector))
        if line_length_sq < 1e-12:
            projection = line_start
        else:
            progress = float(np.dot(xyz_check[sample_index] - line_start, line_vector) / line_length_sq)
            projection = line_start + np.clip(progress, 0.0, 1.0) * line_vector
        path_deviation[sample_index] = float(np.linalg.norm(xyz_check[sample_index] - projection))
    max_path_deviation = float(np.max(path_deviation))

    if vertical_only:
        lateral_drift = np.linalg.norm(xyz_check[:, :2] - start_xyz[None, :2], axis=1)
        max_lateral_drift = float(np.max(lateral_drift))
        if max_lateral_drift > MAX_VERTICAL_XY_DRIFT:
            raise RuntimeError(
                f"{label}: maximum lateral drift is {max_lateral_drift*1000:.1f} mm; "
                f"limit is {MAX_VERTICAL_XY_DRIFT*1000:.1f} mm. No command sent."
            )
    else:
        max_lateral_drift = None

    if horizontal_only:
        z_deviation = float(np.max(np.abs(xyz_check[:, 2] - start_xyz[2])))
        if z_deviation > MAX_HORIZONTAL_Z_DEVIATION:
            raise RuntimeError(
                f"{label}: maximum Z deviation is {z_deviation*1000:.1f} mm; "
                f"limit is {MAX_HORIZONTAL_Z_DEVIATION*1000:.1f} mm. No command sent."
            )
    else:
        z_deviation = None

    if minimum_path_z is not None:
        minimum_actual_z = float(np.min(xyz_check[:, 2]))
        if minimum_actual_z < float(minimum_path_z) - 1e-6:
            raise RuntimeError(
                f"{label}: interpolated TCP reaches Z={minimum_actual_z:.4f} m, "
                f"below required {float(minimum_path_z):.3f} m. No command sent."
            )

    if not vertical_only and not horizontal_only:
        if max_path_deviation > MAX_GENERAL_CARTESIAN_DEVIATION:
            raise RuntimeError(
                f"{label}: Cartesian path deviation is {max_path_deviation*1000:.1f} mm; "
                f"limit is {MAX_GENERAL_CARTESIAN_DEVIATION*1000:.1f} mm. No command sent."
            )

    pitch_values = q_check[:, 1] + q_check[:, 2] + q_check[:, 3]
    pitch_errors = np.abs(np.arctan2(
        np.sin(pitch_values - TASK_TOOL_ANGLE),
        np.cos(pitch_values - TASK_TOOL_ANGLE),
    ))
    max_pitch_error = float(np.max(pitch_errors))
    if max_pitch_error > START_ORIENTATION_CORRECTION_TOLERANCE:
        raise RuntimeError(
            f"{label}: interpolated tool pitch error is {max_pitch_error:.5f} rad."
        )

    for q in q_check:
        _, rotation = fk(q)
        verify_orientation(rotation, float(q[0]), TASK_TOOL_ANGLE, label)

    if duration > original_duration * 1.001:
        print(f"[{label}] Duration stretched {original_duration:.2f}s -> {duration:.2f}s for joint limits.")

    summary = (
        f"[{label}] {len(q_path)} Cartesian IK waypoints, "
        f"{len(q_check)} FK interpolation samples, "
        f"max path deviation={max_path_deviation*1000:.1f} mm"
    )
    if max_lateral_drift is not None:
        summary += f", lateral drift={max_lateral_drift*1000:.1f} mm"
    if z_deviation is not None:
        summary += f", Z deviation={z_deviation*1000:.1f} mm"
    print(summary + ".")

    return times, q_path, xyz_check


# -----------------------------------------------------------------------------
# ROS NODE
# -----------------------------------------------------------------------------

class PickPlaceNode(Node):
    def __init__(self):
        super().__init__("fixed_pitch_3d_pick_place")
        self.current_q = None
        self.last_state_time = None
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self.joint_callback, 10)
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION)
        self.gripper_client = ActionClient(self, GripperCommand, GRIPPER_ACTION)

    def joint_callback(self, msg):
        if len(msg.name) != len(msg.position):
            return
        joint_map = dict(zip(msg.name, msg.position))
        if not all(name in joint_map for name in JOINT_NAMES):
            return
        q = np.array([joint_map[name] for name in JOINT_NAMES], dtype=float)
        if np.all(np.isfinite(q)):
            self.current_q = q
            self.last_state_time = time.monotonic()

    def get_current_q(self):
        start = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_q is not None and self.last_state_time is not None:
                if time.monotonic() - self.last_state_time <= JOINT_STATE_MAX_AGE:
                    return self.current_q.copy()
            if time.monotonic() - start > JOINT_STATE_TIMEOUT:
                raise RuntimeError("Timed out waiting for fresh /joint_states.")
        raise RuntimeError("ROS shutdown while waiting for joint states.")

    def execute_arm(self, q_path, times, label):
        if DRY_RUN:
            print(f"[DRY RUN] Would execute: {label} ({len(q_path)} trajectory points)")
            return

        if not self.arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Arm action server unavailable: {ARM_ACTION}")

        q_path = np.asarray(q_path, dtype=float)
        times = np.asarray(times, dtype=float)
        qd_path, _ = derivatives(q_path, times)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        for q, velocity, t in zip(q_path, qd_path, times):
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in q]
            point.velocities = [float(value) for value in velocity]
            seconds = int(t)
            nanoseconds = int(round((t - seconds) * 1e9))
            if nanoseconds >= 1_000_000_000:
                seconds += 1
                nanoseconds -= 1_000_000_000
            point.time_from_start = Duration(sec=seconds, nanosec=nanoseconds)
            goal.trajectory.points.append(point)

        send_future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"Arm rejected trajectory: {label}")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped_result = result_future.result()
        if wrapped_result is None or wrapped_result.result.error_code != 0:
            reason = "no result" if wrapped_result is None else wrapped_result.result.error_string
            raise RuntimeError(f"Arm trajectory failed ({label}): {reason}")
        print(f"Completed: {label}")

    def command_gripper(self, position, label):
        if DRY_RUN:
            print(f"[DRY RUN] Would command gripper {label}: {position:.4f} m")
            return

        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Gripper action server unavailable: {GRIPPER_ACTION}")

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(GRIPPER_MAX_EFFORT)
        send_future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"Gripper rejected command: {label}")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped_result = result_future.result()
        if wrapped_result is None:
            raise RuntimeError(f"No gripper result: {label}")
        if not wrapped_result.result.reached_goal:
            raise RuntimeError(
                f"Gripper did not reach requested position for {label}; "
                f"stalled={wrapped_result.result.stalled}, "
                f"position={wrapped_result.result.position:.4f}"
            )
        print(f"Gripper completed: {label}")


# -----------------------------------------------------------------------------
# TARGET FILE AND FULL-SEQUENCE PREFLIGHT
# -----------------------------------------------------------------------------

def load_sort_targets():
    """Load targets as an (N, 2, 2) array: shape x [pick/place] x [X/Y]."""
    if not SORT_TARGETS_FILE.is_file():
        raise FileNotFoundError(
            f"Vision target file not found: {SORT_TARGETS_FILE}. "
            "Run run_shape_sorting.py to generate the averaged targets first."
        )

    with np.load(SORT_TARGETS_FILE, allow_pickle=False) as data:
        if "targets" not in data.files or "labels" not in data.files:
            raise ValueError("shape_sort_targets.npz must contain 'targets' and 'labels'.")
        targets = np.asarray(data["targets"], dtype=float)
        labels = tuple(str(value) for value in data["labels"].tolist())

    if targets.ndim != 3 or targets.shape[1:] != (2, 2):
        raise ValueError(f"Expected targets with shape (N, 2, 2), got {targets.shape}.")
    if len(labels) != targets.shape[0]:
        raise ValueError("Target row count does not match the labels count.")
    if not labels:
        raise ValueError("Vision file contains no complete pick/place pairs.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate shape labels in vision file: {labels}")
    unknown = [name for name in labels if name not in SORT_SHAPES]
    if unknown:
        raise ValueError(f"Unsupported shape labels: {unknown}")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Vision target array contains NaN or infinite values.")
    for index, name in enumerate(labels):
        if np.linalg.norm(targets[index, 0] - targets[index, 1]) < 0.005:
            raise ValueError(f"{name}: pick and place coordinates are less than 5 mm apart.")
    return targets, labels


def preflight_plan_segment(
    simulated_q, goal_xyz, label, duration, waypoint_count,
    vertical_only=False, horizontal_only=False, minimum_path_z=None,
):
    start_xyz, _ = fk(simulated_q)
    times, q_path, _ = plan_cartesian_segment(
        start_xyz=start_xyz,
        goal_xyz=goal_xyz,
        initial_q=simulated_q,
        duration=duration,
        waypoint_count=waypoint_count,
        label=label,
        vertical_only=vertical_only,
        horizontal_only=horizontal_only,
        minimum_path_z=minimum_path_z,
    )
    return q_path[-1].copy()


def preflight_entire_sequence(targets, labels, start_q):
    """Validate all approach, four main phases, retracts and return before motion."""
    simulated_q = validate_q(start_q, "Preflight starting joints").copy()
    print("\nFULL CARTESIAN PRE-FLIGHT")
    print(
        "Main four-phase waypoint budget: "
        f"{PICK_DESCENT_WAYPOINTS} + {PICK_LIFT_WAYPOINTS-1} + "
        f"{XY_TRANSFER_WAYPOINTS-1} + {PLACE_DESCENT_WAYPOINTS-1} "
        f"= {MAIN_UNIQUE_WAYPOINT_COUNT} unique points."
    )
    if MAIN_UNIQUE_WAYPOINT_COUNT > MAX_MAIN_UNIQUE_WAYPOINTS:
        raise RuntimeError("Main pick/place waypoint budget exceeds 15.")

    for index, name in enumerate(labels):
        pick_xy = targets[index, 0]
        place_xy = targets[index, 1]
        pick_above = np.array([pick_xy[0], pick_xy[1], TRAVEL_Z], dtype=float)
        pick_contact = np.array([pick_xy[0], pick_xy[1], PICK_CONTACT_Z], dtype=float)
        place_above = np.array([place_xy[0], place_xy[1], TRAVEL_Z], dtype=float)
        place_contact = np.array([place_xy[0], place_xy[1], PLACE_CONTACT_Z], dtype=float)

        print(f"\n[PRE-FLIGHT] {name}: approach to pick height")
        simulated_q = preflight_plan_segment(
            simulated_q, pick_above, f"{name} approach to pick", CARTESIAN_DURATION,
            APPROACH_WAYPOINTS, minimum_path_z=MIN_TRANSFER_Z,
        )

        print(f"[PRE-FLIGHT] {name}: phase 1/4 - descend to pick")
        simulated_q = preflight_plan_segment(
            simulated_q, pick_contact, f"{name} pick descent", VERTICAL_DURATION,
            PICK_DESCENT_WAYPOINTS, vertical_only=True,
        )

        print(f"[PRE-FLIGHT] {name}: phase 2/4 - lift to 20 cm")
        simulated_q = preflight_plan_segment(
            simulated_q, pick_above, f"{name} lift to 20 cm", VERTICAL_DURATION,
            PICK_LIFT_WAYPOINTS, vertical_only=True,
        )

        print(f"[PRE-FLIGHT] {name}: phase 3/4 - XY transfer at 20 cm")
        simulated_q = preflight_plan_segment(
            simulated_q, place_above, f"{name} XY transfer", CARTESIAN_DURATION,
            XY_TRANSFER_WAYPOINTS, horizontal_only=True,
            minimum_path_z=MIN_TRANSFER_Z,
        )

        print(f"[PRE-FLIGHT] {name}: phase 4/4 - descend to place")
        simulated_q = preflight_plan_segment(
            simulated_q, place_contact, f"{name} place descent", VERTICAL_DURATION,
            PLACE_DESCENT_WAYPOINTS, vertical_only=True,
        )

        print(f"[PRE-FLIGHT] {name}: retract after release")
        simulated_q = preflight_plan_segment(
            simulated_q, place_above, f"{name} post-place retract", VERTICAL_DURATION,
            POST_PLACE_RETRACT_WAYPOINTS, vertical_only=True,
        )
        print(f"[PRE-FLIGHT] {name}: all paths passed.")

    print("\n[PRE-FLIGHT] Return to home TCP")
    simulated_q = preflight_plan_segment(
        simulated_q, HOME_XYZ, "return to HOME XYZ", CARTESIAN_DURATION,
        RETURN_HOME_WAYPOINTS, minimum_path_z=MIN_TRANSFER_Z,
    )
    final_times, _ = plan_startup_joint_move(simulated_q, HOME_Q)
    print(f"[PRE-FLIGHT] Final joint-space home move passed ({len(final_times)} samples).")
    print("FULL CARTESIAN PRE-FLIGHT PASSED. No hardware motion has been commanded yet.")


def plan_and_execute_to(
    node,
    current_q,
    goal_xyz,
    label,
    duration,
    waypoint_count,
    minimum_path_z=None,
    vertical_only=False,
    horizontal_only=False,
):
    current_q = validate_q(current_q, f"Current joints before {label}")
    start_xyz, _ = fk(current_q)
    times, q_path, _ = plan_cartesian_segment(
        start_xyz=start_xyz,
        goal_xyz=np.asarray(goal_xyz, dtype=float),
        initial_q=current_q,
        duration=duration,
        waypoint_count=waypoint_count,
        label=label,
        vertical_only=vertical_only,
        horizontal_only=horizontal_only,
        minimum_path_z=minimum_path_z,
    )
    node.execute_arm(q_path, times, label)

    if DRY_RUN:
        return q_path[-1].copy()
    return validate_q(node.get_current_q(), f"Measured joints after {label}")


# -----------------------------------------------------------------------------
# PICK-AND-PLACE SEQUENCE
# -----------------------------------------------------------------------------

def run_sequence(node):
    global WORK_HOME_Q
    targets, labels = load_sort_targets()

    print("\nVISION PICK/PLACE ARRAY")
    print(f"Array shape: ({len(labels)}, 2, 2) = detected shapes x [pick/place] x [X/Y]")
    for index, name in enumerate(labels):
        print(
            f"{index + 1}. {name}: PICK={targets[index, 0]} m, "
            f"PLACE={targets[index, 1]} m"
        )

    print("Requested home joints:", np.round(HOME_Q, 5))
    print(f"Home pitch: {HOME_ANGLE:.6f} rad ({np.rad2deg(HOME_ANGLE):.3f} deg)")
    print(f"Task pitch: {TASK_TOOL_ANGLE:.6f} rad (45 degrees)")
    print(f"Pick contact Z={PICK_CONTACT_Z:.3f} m; place contact Z={PLACE_CONTACT_Z:.3f} m")
    print(f"Lift and XY transfer height: Z={TRAVEL_Z:.3f} m")
    print("No full 3D orientation lock: only tool pitch is constrained.")
    print("No collision model or safety-rated stop is available in this script.")

    current_q = validate_q(node.get_current_q(), "Current joint feedback")
    current_xyz, _ = fk(current_q)
    print("Current joints:", np.round(current_q, 5))
    print("Current TCP XYZ:", np.round(current_xyz, 5))

    # Validate both startup moves before issuing any command.
    t_home, q_home_path = plan_startup_joint_move(current_q, HOME_Q)
    home_xyz_fk, home_rotation = fk(HOME_Q)
    if np.linalg.norm(home_xyz_fk - HOME_XYZ) > 0.001:
        raise RuntimeError("HOME_Q FK does not match HOME_XYZ; check kinematics.py.")
    verify_orientation(home_rotation, float(HOME_Q[0]), HOME_ANGLE, "HOME_Q")

    WORK_HOME_Q = solve_ik(HOME_XYZ, HOME_Q, TASK_TOOL_ANGLE)
    verify_pose(WORK_HOME_Q, HOME_XYZ, "45-degree working home pose", TASK_TOOL_ANGLE)
    t_reorient, q_reorient = plan_startup_joint_move(HOME_Q, WORK_HOME_Q)

    # All object paths are checked before any robot or gripper action is sent.
    preflight_entire_sequence(targets, labels, WORK_HOME_Q)

    print("\nEXECUTION PLAN")
    print(f"Sorting {len(labels)} detected shape(s): {', '.join(labels)}")
    print("The four main phases use at most 15 unique Cartesian IK waypoints per object.")
    print("Every phase is validated against the interpolated joint trajectory using FK.")

    if DRY_RUN:
        print("\nDRY_RUN=True: no physical robot or gripper motion will be commanded.")
    elif REQUIRE_EXECUTE_CONFIRMATION:
        print("Clear the workspace and keep the emergency stop accessible.")
        if input(f"Type EXECUTE to run {len(labels)} sorting cycle(s): ").strip() != "EXECUTE":
            print("Cancelled. No movement sent.")
            return

    node.execute_arm(q_home_path, t_home, "startup to HOME_Q")
    if DRY_RUN:
        current_q = HOME_Q.copy()
    else:
        current_q = validate_q(node.get_current_q(), "Measured joints at HOME_Q")
        home_error = float(np.max(np.abs(wrap_delta(current_q, HOME_Q))))
        if home_error > JOINT_GOAL_TOLERANCE:
            raise RuntimeError(f"Robot did not reach HOME_Q; joint error={home_error:.4f} rad.")

    measured_home_pitch = float(current_q[1] + current_q[2] + current_q[3])
    if abs(wrap_delta(measured_home_pitch, HOME_ANGLE)) > START_ORIENTATION_CORRECTION_TOLERANCE:
        raise RuntimeError("Measured TARGET_Q pitch differs from HOME_ANGLE.")

    node.execute_arm(q_reorient, t_reorient, "reorient at home to 45-degree working pitch")
    if DRY_RUN:
        current_q = WORK_HOME_Q.copy()
    else:
        current_q = validate_q(node.get_current_q(), "Measured working-home joints")
        work_xyz, _ = fk(current_q)
        work_pitch = float(current_q[1] + current_q[2] + current_q[3])
        if abs(wrap_delta(work_pitch, TASK_TOOL_ANGLE)) > START_ORIENTATION_CORRECTION_TOLERANCE:
            raise RuntimeError("Robot did not reach the 45-degree working orientation.")
        if np.linalg.norm(work_xyz - HOME_XYZ) > MAX_SEGMENT_START_ERROR:
            raise RuntimeError("Working-home TCP drifted beyond the allowed tolerance.")

    for index, name in enumerate(labels):
        pick_xy = targets[index, 0]
        place_xy = targets[index, 1]
        pick_above = np.array([pick_xy[0], pick_xy[1], TRAVEL_Z], dtype=float)
        pick_contact = np.array([pick_xy[0], pick_xy[1], PICK_CONTACT_Z], dtype=float)
        place_above = np.array([place_xy[0], place_xy[1], TRAVEL_Z], dtype=float)
        place_contact = np.array([place_xy[0], place_xy[1], PLACE_CONTACT_Z], dtype=float)

        print(f"\n========== {index + 1}/{len(labels)}: {name} ==========")
        print("Pick XY:", np.round(pick_xy, 5), "Place XY:", np.round(place_xy, 5))

        current_q = plan_and_execute_to(
            node, current_q, pick_above,
            f"{name}: approach above pick", CARTESIAN_DURATION,
            APPROACH_WAYPOINTS, minimum_path_z=MIN_TRANSFER_Z,
        )
        node.command_gripper(
            GRIPPER_MAX_SEPARATION_POSITION,
            f"open before picking {name}",
        )

        # Phase 1/4: five positions from Z=0.20 m down to Z=0.06 m.
        current_q = plan_and_execute_to(
            node, current_q, pick_contact,
            f"{name}: phase 1/4 descend to pick", VERTICAL_DURATION,
            PICK_DESCENT_WAYPOINTS, vertical_only=True,
        )
        node.command_gripper(GRIPPER_HALF_CLOSED_POSITION, f"grip {name}")

        # Phase 2/4: four positions from Z=0.06 m back up to Z=0.20 m.
        # Its first endpoint is shared with the end of phase 1.
        current_q = plan_and_execute_to(
            node, current_q, pick_above,
            f"{name}: phase 2/4 lift to 20 cm", VERTICAL_DURATION,
            PICK_LIFT_WAYPOINTS, vertical_only=True,
        )

        # Phase 3/4: four positions across XY at constant Z=0.20 m.
        # Its first endpoint is shared with the end of phase 2.
        current_q = plan_and_execute_to(
            node, current_q, place_above,
            f"{name}: phase 3/4 XY transfer at 20 cm", CARTESIAN_DURATION,
            XY_TRANSFER_WAYPOINTS, minimum_path_z=MIN_TRANSFER_Z,
            horizontal_only=True,
        )

        # Phase 4/4: five positions from Z=0.20 m down to Z=0.06 m.
        # Its first endpoint is shared with the end of phase 3.
        current_q = plan_and_execute_to(
            node, current_q, place_contact,
            f"{name}: phase 4/4 descend to place", VERTICAL_DURATION,
            PLACE_DESCENT_WAYPOINTS, vertical_only=True,
        )
        node.command_gripper(
            GRIPPER_MAX_SEPARATION_POSITION,
            f"open after placing {name}",
        )

        # Retract is a separate safety segment after the requested four phases.
        current_q = plan_and_execute_to(
            node, current_q, place_above,
            f"{name}: retract after placement", VERTICAL_DURATION,
            POST_PLACE_RETRACT_WAYPOINTS, vertical_only=True,
        )

    current_q = plan_and_execute_to(
        node, current_q, HOME_XYZ,
        "return to HOME XYZ at working pitch", CARTESIAN_DURATION,
        RETURN_HOME_WAYPOINTS, minimum_path_z=MIN_TRANSFER_Z,
    )
    final_times, final_path = plan_startup_joint_move(current_q, HOME_Q)
    node.execute_arm(final_path, final_times, "final move to exact TARGET_Q home pose")

    if not DRY_RUN:
        final_q = validate_q(node.get_current_q(), "Measured final joints")
        final_error = float(np.max(np.abs(wrap_delta(final_q, HOME_Q))))
        if final_error > JOINT_GOAL_TOLERANCE:
            raise RuntimeError(f"Final TARGET_Q verification failed: {final_error:.4f} rad.")

    print(f"\nSorting sequence completed for {len(labels)} detected shape(s).")
    if DRY_RUN:
        print("DRY RUN ONLY: no physical robot or gripper motion was commanded.")


def main():
    if MAIN_UNIQUE_WAYPOINT_COUNT > MAX_MAIN_UNIQUE_WAYPOINTS:
        raise RuntimeError("Configured main waypoint count exceeds the 15-point maximum.")

    home_position, home_rotation = fk(HOME_Q)
    if np.linalg.norm(home_position - HOME_XYZ) > 0.001:
        raise RuntimeError(
            f"HOME_Q FK={home_position}, but HOME_XYZ={HOME_XYZ}. "
            "Do not move until kinematics.py is corrected."
        )
    verify_orientation(home_rotation, float(HOME_Q[0]), HOME_ANGLE, "HOME_Q")

    rclpy.init()
    node = PickPlaceNode()
    try:
        run_sequence(node)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("Sequence stopped. No subsequent motion will be commanded.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
