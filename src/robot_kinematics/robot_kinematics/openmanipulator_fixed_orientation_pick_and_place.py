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
from std_srvs.srv import SetBool
from dynamixel_interfaces.srv import SetDataToDxl
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from robot_kinematics.kinematics import (
    forward_kinematics,
    inverse_kinematics,
    joint_limit_margin,
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

def normalize_z(val, default=0.080):
    """Normalize input Z to metres, safely handling mm (e.g. 80.0) or typo decimetres (e.g. 0.8m -> 0.08m)."""
    try:
        f = float(val)
        if f >= 10.0:  # e.g., 80.0 mm -> 0.080 m
            return f / 1000.0
        elif f >= 0.5:  # e.g., 0.8 m -> 0.080 m (protect against user typo for 0.08m)
            return f / 10.0
        return f
    except (ValueError, TypeError):
        return float(default)


PICK_CONTACT_Z = normalize_z(os.environ.get("PICK_CONTACT_Z", "0.070"), 0.070)   # Lowest pick contact height (0.070 m = 7 cm)

# Hard safety floor: arm/TCP must NEVER descend below 0.070 m (70 mm) for any shape during placing.
MIN_PLACE_Z = normalize_z(os.environ.get("MIN_PLACE_Z", "0.070"), 0.070)         # Hard minimum placing floor (0.070 m)

# Lowest point for vertical descent during placing: 0.070 m (70 mm)
DEFAULT_PLACE_CONTACT_Z = max(
    normalize_z(os.environ.get("PLACE_CONTACT_Z", "0.070"), 0.070),
    MIN_PLACE_Z,
)
PLACE_CONTACT_Z = DEFAULT_PLACE_CONTACT_Z                                        # Default place contact height (0.070 m)

# Optional per-shape place contact heights (all strictly clamped >= MIN_PLACE_Z)
PER_SHAPE_PLACE_Z = {
    "green_circle":    DEFAULT_PLACE_CONTACT_Z,
    "orange_hexagon":  DEFAULT_PLACE_CONTACT_Z,
    "blue_square":     DEFAULT_PLACE_CONTACT_Z,
    "yellow_triangle": DEFAULT_PLACE_CONTACT_Z,
}


def get_shape_place_contact_z(shape_name):
    """Return the place contact height for a shape, strictly enforced >= MIN_PLACE_Z (0.070 m)."""
    z_val = PER_SHAPE_PLACE_Z.get(shape_name, DEFAULT_PLACE_CONTACT_Z)
    return max(float(z_val), MIN_PLACE_Z)


TRAVEL_Z = normalize_z(os.environ.get("TRAVEL_Z", "0.170"), 0.170)                  # 17 cm clearance plane above pick and place
PLACE_RETRACT_Z = normalize_z(os.environ.get("PLACE_RETRACT_Z", "0.170"), 0.170)    # 17 cm vertical retract after placing
MIN_TRANSFER_Z = normalize_z(os.environ.get("MIN_TRANSFER_Z", "0.170"), 0.170)      # Hard floor for non-contact travel (17 cm)

# Four main phases. Counts include both endpoints of each individual phase.
# Since adjoining phases share their common endpoint, the unique count is 17.
PICK_DESCENT_WAYPOINTS = 5
PICK_LIFT_WAYPOINTS = 4
XY_TRANSFER_WAYPOINTS = 6
PLACE_DESCENT_WAYPOINTS = 5
MAIN_UNIQUE_WAYPOINT_COUNT = (
    PICK_DESCENT_WAYPOINTS
    + (PICK_LIFT_WAYPOINTS - 1)
    + (XY_TRANSFER_WAYPOINTS - 1)
    + (PLACE_DESCENT_WAYPOINTS - 1)
)
MAX_MAIN_UNIQUE_WAYPOINTS = 20

# Approach/retract/return paths are separate from the 15-point four-phase path.
APPROACH_WAYPOINTS = 9
POST_PLACE_RETRACT_WAYPOINTS = 5
RETURN_HOME_WAYPOINTS = 9

MAX_HORIZONTAL_Z_DEVIATION = 0.015  # 15 mm maximum height drift during XY transfer.
MAX_VERTICAL_XY_DRIFT = 0.015       # 15 mm maximum lateral drift during vertical moves.
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

# Gripper physical calibration from DYNAMIXEL Wizard 2.0 (in degrees):
#   Open / Home state: 145.5 deg (maps to -0.010 m prismatic)
#   Closed reference:  210.0 deg (maps to +0.019 m prismatic)
# Per-shape measured grip angles:
#   green_circle:     182.0 deg
#   blue_square:      184.6 deg
#   orange_hexagon:   193.0 deg
#   yellow_triangle:  203.8 deg
GRIPPER_HOME_ANGLE_DEG = float(os.environ.get("GRIPPER_HOME_ANGLE_DEG", os.environ.get("GRIPPER_OPEN_DEG", "145.5")))
GRIPPER_OPEN_ANGLE_DEG = GRIPPER_HOME_ANGLE_DEG
GRIPPER_CLOSED_MAX_DEG = float(os.environ.get("GRIPPER_MAX_CLOSE_DEG", "210.0"))

SHAPE_GRIP_ANGLES_DEG = {
    "green_circle":    float(os.environ.get("GRIP_DEG_GREEN_CIRCLE", "182.0")),
    "blue_square":     float(os.environ.get("GRIP_DEG_BLUE_SQUARE", "184.6")),
    "orange_hexagon":  float(os.environ.get("GRIP_DEG_ORANGE_HEXAGON", "193.0")),
    "yellow_triangle": float(os.environ.get("GRIP_DEG_YELLOW_TRIANGLE", "203.8")),
}

# Per-shape grip width adjustments (+0.003 m closes 3 mm tighter / narrower grip width)
SHAPE_GRIP_WIDTH_OFFSET_M = {
    "green_circle":    float(os.environ.get("GRIP_WIDTH_ADJ_GREEN_CIRCLE", "0.003")),   # 3mm tighter
    "blue_square":     float(os.environ.get("GRIP_WIDTH_ADJ_BLUE_SQUARE", "0.000")),
    "orange_hexagon":  float(os.environ.get("GRIP_WIDTH_ADJ_ORANGE_HEXAGON", "0.000")),
    "yellow_triangle": float(os.environ.get("GRIP_WIDTH_ADJ_YELLOW_TRIANGLE", "0.003")), # 3mm tighter
}

def angle_deg_to_prismatic(deg: float) -> float:
    """
    Map physical motor degrees [145.5, 210.0] linearly to ROS 2 prismatic meters [-0.010, 0.019].
    With xacro revolute_min = -0.6021 rad (145.5 deg) and revolute_max = 0.5236 rad (210.0 deg),
    this produces the exact commanded motor angle in hardware.
    """
    fraction = (deg - GRIPPER_HOME_ANGLE_DEG) / (GRIPPER_CLOSED_MAX_DEG - GRIPPER_HOME_ANGLE_DEG)
    return float(-0.010 + fraction * (0.019 - (-0.010)))

def prismatic_to_angle_deg(pris: float) -> float:
    """Map ROS 2 prismatic meters [-0.010, 0.019] linearly to physical motor degrees [145.5, 210.0]."""
    fraction = (pris - (-0.010)) / (0.019 - (-0.010))
    return float(GRIPPER_HOME_ANGLE_DEG + fraction * (GRIPPER_CLOSED_MAX_DEG - GRIPPER_HOME_ANGLE_DEG))

def parse_gripper_target(val) -> tuple:
    """Parse gripper target returning (deg, prismatic_m)."""
    v = float(val)
    if abs(v) > 1.0:
        deg = v
        pris = angle_deg_to_prismatic(deg)
    else:
        pris = v
        deg = prismatic_to_angle_deg(pris)
    return deg, pris

GRIPPER_HOME_POSITION = angle_deg_to_prismatic(GRIPPER_HOME_ANGLE_DEG)  # -0.010 m (145.5 deg)
GRIPPER_OPEN_POSITION = GRIPPER_HOME_POSITION
GRIPPER_MAX_SEPARATION_POSITION = GRIPPER_HOME_POSITION
GRIPPER_CLOSED_POSITION = angle_deg_to_prismatic(203.8)
GRIPPER_DEFAULT_GRIP_POSITION = GRIPPER_CLOSED_POSITION
GRIPPER_MAX_EFFORT = 20.0

def dxl5_to_prismatic(dxl5_rad: float) -> float:
    """Pass-through/helper for compatibility."""
    return float(dxl5_rad)

def prismatic_to_dxl5(prismatic_m: float) -> float:
    """Pass-through/helper for compatibility."""
    return float(prismatic_m)

GRIP_POSITIONS_BY_SHAPE = {
    shape: min(0.019, angle_deg_to_prismatic(deg) + SHAPE_GRIP_WIDTH_OFFSET_M.get(shape, 0.0))
    for shape, deg in SHAPE_GRIP_ANGLES_DEG.items()
}

# Live execution on hardware (DRY_RUN disabled)
DRY_RUN = False
REQUIRE_EXECUTE_CONFIRMATION = False

FREQUENCY_HZ = 50.0
HOME_DURATION = float(os.environ.get("HOME_DURATION", "8.0"))
CARTESIAN_DURATION = float(os.environ.get("CARTESIAN_DURATION", "4.0"))
VERTICAL_DURATION = float(os.environ.get("VERTICAL_DURATION", "4.0"))
MAX_JOINT_VELOCITY = np.array([float(os.environ.get("MAX_JOINT_VEL", "0.5"))] * 4, dtype=float)
MAX_JOINT_ACCELERATION = np.array([float(os.environ.get("MAX_JOINT_ACCEL", "1.0"))] * 4, dtype=float)

JOINT_STATE_TIMEOUT = 10.0
JOINT_STATE_MAX_AGE = 0.5
POSITION_TOLERANCE = 0.002
MAX_SEGMENT_START_ERROR = 0.020
ORIENTATION_TOLERANCE = 0.035
START_ORIENTATION_CORRECTION_TOLERANCE = 0.080
JOINT_GOAL_TOLERANCE = 0.08

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
    prev_sol = np.asarray(previous_q, dtype=float) if previous_q is not None else None
    q = inverse_kinematics(
        x=x,
        y=y,
        z=z,
        angle=float(tool_angle),
        previous_solution=prev_sol,
    )
    if q is None:
        raise RuntimeError(
            f"IK failed at XYZ=({x:.4f}, {y:.4f}, {z:.4f}) m "
            f"for fixed tool pitch={tool_angle:.5f} rad."
        )
    q = validate_q(q, "IK result")
    verify_pose(q, np.array([x, y, z], dtype=float), "IK result", float(tool_angle))
    return q


def find_feasible_pitch_at_z(xy, z, preferred=TASK_TOOL_ANGLE):
    x, y = float(xy[0]), float(xy[1])
    r_target = np.hypot(x - 0.012, y)

    if r_target < 0.080:
        candidates = np.deg2rad(np.arange(45.0, 85.0, 2.0))
    else:
        candidates = [preferred]
        deltas = np.deg2rad(np.arange(2.0, 40.0, 2.0))
        for d in deltas:
            candidates.append(preferred + d)
            candidates.append(preferred - d)

    best_ang = None
    best_margin = -1.0
    for ang in candidates:
        if ang < np.deg2rad(20.0) or ang > np.deg2rad(85.0):
            continue
        sol = inverse_kinematics(x, y, z, angle=ang)
        if sol is not None:
            m = joint_limit_margin(sol)
            if m >= 0.08:
                return float(ang)
            if m > best_margin:
                best_margin = m
                best_ang = ang
    if best_ang is not None:
        return float(best_ang)
    return float(preferred)


def get_pick_place_pitches(xy, z_contact, z_above, preferred=TASK_TOOL_ANGLE):
    """Return (pitch_above, pitch_contact) for vertical travel at (x, y).

    If a single pitch angle satisfies all intermediate heights from z_contact to z_above
    with a good joint margin, that single pitch is used for both (strictly constant pitch).
    If no single pitch can span both heights (which occurs for targets extremely close
    to the robot base), separate feasible pitches are returned and smoothly interpolated.
    """
    x, y = float(xy[0]), float(xy[1])
    candidate_angles = [preferred]
    r_target = np.hypot(x - 0.012, y)
    if r_target < 0.080:
        candidate_angles.extend(np.deg2rad(np.arange(46.0, 85.0, 2.0)))
    else:
        deltas = np.deg2rad(np.arange(2.0, 40.0, 2.0))
        for d in deltas:
            candidate_angles.append(preferred + d)
            candidate_angles.append(preferred - d)

    zs = np.linspace(z_contact, z_above, 5)
    best_single = None
    best_margin = -1.0
    for ang in candidate_angles:
        if ang < np.deg2rad(20.0) or ang > np.deg2rad(85.0):
            continue
        sol = None
        all_ok = True
        margins = []
        for z in zs:
            sol = inverse_kinematics(x, y, z, angle=ang, previous_solution=sol)
            if sol is None:
                all_ok = False
                break
            margins.append(joint_limit_margin(sol))
        if all_ok:
            m = min(margins)
            if m >= 0.08:
                return float(ang), float(ang)
            if m > best_margin:
                best_margin = m
                best_single = ang

    if best_single is not None:
        return float(best_single), float(best_single)

    p_contact = find_feasible_pitch_at_z(xy, z_contact, preferred)
    p_above = find_feasible_pitch_at_z(xy, z_above, preferred)
    return float(p_above), float(p_contact)


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
    z_start = float(fk_points[0, 2])
    z_min = float(np.min(fk_points[:, 2]))
    z_max = float(np.max(fk_points[:, 2]))

    # Permit a starting pose below STARTUP_MIN_Z (e.g. recovering from contact or previous abort),
    # but do not permit the planned trajectory to dip further down.
    allowed_min_z = min(STARTUP_MIN_Z, z_start - 0.005)

    if z_min < allowed_min_z - 1e-6 or z_max > STARTUP_MAX_Z:
        raise RuntimeError(
            f"Joint-space move FK Z range [{z_min:.3f}, {z_max:.3f}] m violates "
            f"[{allowed_min_z:.3f}, {STARTUP_MAX_Z:.3f}] m. No command sent."
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
    start_tool_angle=None,
    goal_tool_angle=None,
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

    # Safety floor enforcement: actively prevent goal or waypoints from going below minimum_path_z
    if minimum_path_z is not None:
        effective_min_z = float(minimum_path_z)
        if goal_xyz[2] < effective_min_z:
            print(f"[{label}] Active prevention: goal Z={goal_xyz[2]:.4f} m clamped to safety floor {effective_min_z:.4f} m.")
            goal_xyz[2] = effective_min_z
        # For ascending motions departing from contact slightly below floor due to servo compliance,
        # allow upward departure without spurious abort
        allowed_min_z = min(effective_min_z, float(start_xyz[2]) - 1e-4)
    else:
        effective_min_z = None
        allowed_min_z = None

    current_pitch = float(q_start[1] + q_start[2] + q_start[3])
    if start_tool_angle is None:
        start_tool_angle = current_pitch
    if goal_tool_angle is None:
        goal_tool_angle = start_tool_angle

    start_pitch = float(start_tool_angle)
    goal_pitch = float(goal_tool_angle)

    pitch_error = abs(wrap_delta(current_pitch, start_pitch))
    if pitch_error > START_ORIENTATION_CORRECTION_TOLERANCE:
        raise RuntimeError(
            f"{label}: starting pitch error {pitch_error:.5f} rad exceeds allowance."
        )

    if vertical_only:
        xy_error = float(np.linalg.norm(start_xyz[:2] - goal_xyz[:2]))
        if xy_error > MAX_SEGMENT_START_ERROR:
            raise RuntimeError(
                f"{label}: vertical segment endpoints differ in XY by "
                f"{xy_error*1000:.1f} mm; no motion planned."
            )

    if horizontal_only:
        z_error = abs(float(start_xyz[2] - goal_xyz[2]))
        if z_error > MAX_SEGMENT_START_ERROR:
            raise RuntimeError(
                f"{label}: horizontal segment endpoints differ in Z by "
                f"{z_error*1000:.1f} mm; no motion planned."
            )

    # Initial Cartesian waypoints and pitch schedule
    cartesian_waypoints = np.linspace(start_xyz, goal_xyz, int(waypoint_count))
    if effective_min_z is not None:
        cartesian_waypoints[:, 2] = np.maximum(cartesian_waypoints[:, 2], effective_min_z)
    target_pitches = np.linspace(start_pitch, goal_pitch, int(waypoint_count))

    duration = float(duration)
    original_duration = duration

    # Active trajectory planning & spline sag compensation loop:
    # If the non-linear joint-space spline sags below effective_min_z,
    # automatically elevate waypoints to eliminate the sag and prevent going below the floor.
    for sag_iteration in range(5):
        q_waypoints = [q_start.copy()]
        for waypoint_index in range(1, len(cartesian_waypoints)):
            q_next = solve_ik(
                cartesian_waypoints[waypoint_index],
                q_waypoints[-1],
                target_pitches[waypoint_index],
            )
            q_waypoints.append(q_next)
        q_path = np.asarray(q_waypoints, dtype=float)

        current_dur = duration
        for _ in range(15):
            times = np.linspace(0.0, current_dur, len(q_path))
            qd_path, _ = derivatives(q_path, times)
            q_check, qd_check, qdd_check, _, _ = sample_ros_trajectory(
                q_path, times, qd_path
            )

            velocity_ratio = float(np.max(np.max(np.abs(qd_check), axis=0) / MAX_JOINT_VELOCITY))
            acceleration_ratio = float(np.max(np.max(np.abs(qdd_check), axis=0) / MAX_JOINT_ACCELERATION))
            if velocity_ratio <= 1.0 and acceleration_ratio <= 1.0:
                break
            current_dur *= max(1.05, 1.05 * velocity_ratio, 1.05 * np.sqrt(acceleration_ratio))
        else:
            raise RuntimeError(f"{label}: cannot satisfy joint velocity/acceleration limits.")

        # Resample after the final duration has been selected.
        times = np.linspace(0.0, current_dur, len(q_path))
        qd_path, _ = derivatives(q_path, times)
        q_check, qd_check, qdd_check, segment_ids, segment_s = sample_ros_trajectory(
            q_path, times, qd_path
        )
        xyz_check = np.asarray([fk(q)[0] for q in q_check], dtype=float)

        if effective_min_z is None:
            duration = current_dur
            break

        min_actual_z = float(np.min(xyz_check[:, 2]))
        if min_actual_z >= allowed_min_z - 1e-6:
            duration = current_dur
            break

        # Sag detected! Elevate the waypoints to completely prevent going below minimum_path_z
        sag = allowed_min_z - min_actual_z
        print(
            f"[{label}] Active prevention: spline sag of {sag*1000:.2f} mm detected "
            f"(min Z={min_actual_z:.4f} m < {allowed_min_z:.4f} m). Elevating waypoints..."
        )
        if horizontal_only or (start_xyz[2] >= allowed_min_z - 1e-4 and goal_xyz[2] >= allowed_min_z - 1e-4):
            # Both endpoints are at or above floor: elevate the sagging interior with a sine bell curve
            # so endpoints stay anchored at their exact target heights without being pushed into unreachable space.
            elevation_profile = np.sin(np.pi * np.linspace(0.0, 1.0, len(cartesian_waypoints)))
        elif start_xyz[2] > goal_xyz[2]:
            # Descending motion towards floor: elevate towards goal endpoint
            elevation_profile = np.linspace(0.0, 1.0, len(cartesian_waypoints))
        else:
            # Ascending motion departing floor: elevate towards start endpoint
            elevation_profile = np.linspace(1.0, 0.0, len(cartesian_waypoints))

        cartesian_waypoints[:, 2] += (sag + 0.0005) * elevation_profile
    else:
        duration = current_dur

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
        if minimum_actual_z < allowed_min_z - 1e-4:
            raise RuntimeError(
                f"{label}: interpolated TCP reaches Z={minimum_actual_z:.4f} m, "
                f"below required {allowed_min_z:.3f} m. No command sent."
            )

    if not vertical_only and not horizontal_only:
        if max_path_deviation > MAX_GENERAL_CARTESIAN_DEVIATION:
            print(
                f"[WARN] {label}: Cartesian path deviation is {max_path_deviation*1000:.1f} mm; "
                f"soft limit is {MAX_GENERAL_CARTESIAN_DEVIATION*1000:.1f} mm. Continuing anyway.",
                flush=True,
            )

    num_segs = len(cartesian_waypoints) - 1
    total_progress = (np.asarray(segment_ids, dtype=float) + np.asarray(segment_s, dtype=float)) / float(num_segs)
    expected_pitches = start_pitch + total_progress * (goal_pitch - start_pitch)

    pitch_values = q_check[:, 1] + q_check[:, 2] + q_check[:, 3]
    pitch_errors = np.abs(wrap_delta(pitch_values, expected_pitches))
    max_pitch_error = float(np.max(pitch_errors))
    if max_pitch_error > START_ORIENTATION_CORRECTION_TOLERANCE:
        raise RuntimeError(
            f"{label}: interpolated tool pitch error is {max_pitch_error:.5f} rad."
        )

    for sample_index, q in enumerate(q_check):
        _, rotation = fk(q)
        verify_orientation(rotation, float(q[0]), expected_pitches[sample_index], label)

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
        self.torque_client = self.create_client(SetBool, "/dynamixel_hardware_interface/set_dxl_torque")
        self.dxl_data_client = self.create_client(SetDataToDxl, "/dynamixel_hardware_interface/set_dxl_data")

    def enable_torque(self):
        # 1. Ensure master torque is enabled via service
        if self.torque_client.wait_for_service(timeout_sec=3.0):
            req = SetBool.Request(data=True)
            future = self.torque_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.done() and future.result() is not None:
                print(f"[INFO] Torque service response: {future.result().message or 'OK'}")
        else:
            print("[WARN] Torque service not available, continuing...")

        # 2. Specifically ensure arm Dynamixels (dxl1-dxl4: IDs 11-14) have Torque Enable = 1
        if self.dxl_data_client.wait_for_service(timeout_sec=3.0):
            for dxl_id in (11, 12, 13, 14):
                req = SetDataToDxl.Request()
                req.id = int(dxl_id)
                req.item_name = "Torque Enable"
                req.item_data = 1
                future = self.dxl_data_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                if future.done() and future.result() is not None and future.result().result:
                    print(f"[INFO] Dynamixel ID {dxl_id} torque explicitly enabled.")
                else:
                    print(f"[WARN] Failed to explicitly enable Dynamixel ID {dxl_id} torque.")

            # Gripper motor (ID 15): enable torque for pick-and-place operations (set ENABLE_GRIPPER_TORQUE=0 to disable)
            grip_torque_val = 0 if os.environ.get("ENABLE_GRIPPER_TORQUE", "1") == "0" else 1
            req = SetDataToDxl.Request()
            req.id = 15
            req.item_name = "Torque Enable"
            req.item_data = grip_torque_val
            future = self.dxl_data_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if grip_torque_val == 0:
                print("[INFO] Gripper Dynamixel (ID 15) torque explicitly DISABLED.")
            else:
                print("[INFO] Gripper Dynamixel (ID 15) torque explicitly ENABLED.")

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
        deg_val, pris_val = parse_gripper_target(position)
        if DRY_RUN:
            print(f"[DRY RUN] Would command gripper {label}: {deg_val:.1f}° (prismatic: {pris_val:.4f} m)")
            return

        if os.environ.get("ENABLE_GRIPPER_TORQUE", "1") == "0":
            print(f"[INFO] Gripper torque disabled: skipping gripper move for {label}")
            return

        print(f"[GRIPPER] Commanding {label}: {deg_val:.1f}° (prismatic: {pris_val:.4f} m)")

        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Gripper action server unavailable: {GRIPPER_ACTION}")

        goal = GripperCommand.Goal()
        goal.command.position = float(pris_val)
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
        if not (wrapped_result.result.reached_goal or wrapped_result.result.stalled):
            raise RuntimeError(
                f"Gripper did not reach requested position for {label}; "
                f"stalled={wrapped_result.result.stalled}, "
                f"position={wrapped_result.result.position:.4f}"
            )
        res_deg, _ = parse_gripper_target(wrapped_result.result.position)
        print(f"Gripper completed: {label} (actual: {res_deg:.1f}°, {wrapped_result.result.position:.4f} m)")

    def destroy_node(self):
        # Guarantee gripper torque is unpowered upon sequence completion or exit
        if hasattr(self, 'dxl_data_client') and self.dxl_data_client.wait_for_service(timeout_sec=1.0):
            try:
                req = SetDataToDxl.Request()
                req.id = 15
                req.item_name = "Torque Enable"
                req.item_data = 0
                future = self.dxl_data_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            except Exception:
                pass
        super().destroy_node()


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

    # Optional controller-level offset overrides (if running controller standalone without re-running vision)
    ctrl_pick_dx = float(os.environ.get("CONTROLLER_PICK_OFFSET_X", "0.000"))
    ctrl_pick_dy = float(os.environ.get("CONTROLLER_PICK_OFFSET_Y", "0.000"))
    if ctrl_pick_dx != 0.0 or ctrl_pick_dy != 0.0:
        targets[:, 0, 0] += ctrl_pick_dx
        targets[:, 0, 1] += ctrl_pick_dy
        print(f"[OFFSET] Applied runtime pick offset: dX={ctrl_pick_dx * 1000:+.1f} mm, dY={ctrl_pick_dy * 1000:+.1f} mm", flush=True)

    ctrl_dx = float(os.environ.get("CONTROLLER_PLACE_OFFSET_X", "0.000"))
    ctrl_dy = float(os.environ.get("CONTROLLER_PLACE_OFFSET_Y", "0.000"))
    if ctrl_dx != 0.0 or ctrl_dy != 0.0:
        targets[:, 1, 0] += ctrl_dx
        targets[:, 1, 1] += ctrl_dy
        print(f"[OFFSET] Applied runtime place offset: dX={ctrl_dx * 1000:+.1f} mm, dY={ctrl_dy * 1000:+.1f} mm", flush=True)

    bottom_dy = float(os.environ.get("CONTROLLER_BOTTOM_SHAPES_PLACE_OFFSET_Y", "0.000"))
    if bottom_dy != 0.0:
        for idx, sname in enumerate(labels):
            if sname in ("green_circle", "orange_hexagon"):
                targets[idx, 1, 1] += bottom_dy
                print(f"[OFFSET] Applied runtime place offset to {sname}: dY={bottom_dy * 1000:+.1f} mm", flush=True)

    ctrl_circle_dy = float(os.environ.get("CONTROLLER_CIRCLE_PLACE_OFFSET_Y", "0.000"))
    if ctrl_circle_dy != 0.0:
        for idx, sname in enumerate(labels):
            if sname == "green_circle":
                targets[idx, 1, 1] += ctrl_circle_dy
                print(f"[OFFSET] Applied runtime circle place offset: dY={ctrl_circle_dy * 1000:+.1f} mm", flush=True)

    # Calculate transfer length (Euclidean distance between pick and place) for each shape
    transfer_lengths = [
        float(np.linalg.norm(targets[i, 1] - targets[i, 0]))
        for i in range(len(labels))
    ]

    # Sort shapes in order of increasing transfer length (closest to destination first)
    sort_order = np.argsort(transfer_lengths)
    targets = targets[sort_order]
    labels = tuple(labels[i] for i in sort_order)
    transfer_lengths = [transfer_lengths[i] for i in sort_order]

    print("\n[ORDER] Shapes sorted by increasing transfer length (closest to destination first):", flush=True)
    for rank, (name, length) in enumerate(zip(labels, transfer_lengths), start=1):
        print(
            f"  {rank}. {name:16s}: transfer length = {length * 1000:6.1f} mm ({length * 100:4.1f} cm)",
            flush=True,
        )

    return targets, labels


def preflight_plan_segment(
    simulated_q, goal_xyz, label, duration, waypoint_count,
    vertical_only=False, horizontal_only=False, minimum_path_z=None,
    start_tool_angle=None, goal_tool_angle=None,
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
        start_tool_angle=start_tool_angle,
        goal_tool_angle=goal_tool_angle,
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
        place_z = get_shape_place_contact_z(name)
        pick_above_pitch, pick_contact_pitch = get_pick_place_pitches(
            pick_xy, PICK_CONTACT_Z, TRAVEL_Z, TASK_TOOL_ANGLE
        )
        place_above_pitch, place_contact_pitch = get_pick_place_pitches(
            place_xy, place_z, TRAVEL_Z, TASK_TOOL_ANGLE
        )

        pick_above = np.array([pick_xy[0], pick_xy[1], TRAVEL_Z], dtype=float)
        pick_contact = np.array([pick_xy[0], pick_xy[1], PICK_CONTACT_Z], dtype=float)
        place_above = np.array([place_xy[0], place_xy[1], TRAVEL_Z], dtype=float)
        place_contact = np.array([place_xy[0], place_xy[1], place_z], dtype=float)

        approach_min_z = min(float(fk(simulated_q)[0][2]), MIN_TRANSFER_Z)
        print(f"\n[PRE-FLIGHT] {name}: approach to pick height (pitch={np.rad2deg(pick_above_pitch):.1f} deg, min_z={approach_min_z:.3f} m)")
        simulated_q = preflight_plan_segment(
            simulated_q, pick_above, f"{name} approach to pick", CARTESIAN_DURATION,
            APPROACH_WAYPOINTS, minimum_path_z=approach_min_z,
            start_tool_angle=float(simulated_q[1] + simulated_q[2] + simulated_q[3]),
            goal_tool_angle=pick_above_pitch,
        )

        print(f"[PRE-FLIGHT] {name}: phase 1/4 - descend to pick (pitch {np.rad2deg(pick_above_pitch):.1f} -> {np.rad2deg(pick_contact_pitch):.1f} deg)")
        simulated_q = preflight_plan_segment(
            simulated_q, pick_contact, f"{name} pick descent", VERTICAL_DURATION,
            PICK_DESCENT_WAYPOINTS, vertical_only=True,
            start_tool_angle=pick_above_pitch,
            goal_tool_angle=pick_contact_pitch,
        )

        print(f"[PRE-FLIGHT] {name}: phase 2/4 - lift to clearance (pitch {np.rad2deg(pick_contact_pitch):.1f} -> {np.rad2deg(pick_above_pitch):.1f} deg)")
        simulated_q = preflight_plan_segment(
            simulated_q, pick_above, f"{name} lift to clearance", VERTICAL_DURATION,
            PICK_LIFT_WAYPOINTS, vertical_only=True,
            start_tool_angle=pick_contact_pitch,
            goal_tool_angle=pick_above_pitch,
        )

        print(f"[PRE-FLIGHT] {name}: phase 3/4 - XY transfer at clearance (pitch {np.rad2deg(pick_above_pitch):.1f} -> {np.rad2deg(place_above_pitch):.1f} deg)")
        simulated_q = preflight_plan_segment(
            simulated_q, place_above, f"{name} XY transfer", CARTESIAN_DURATION,
            XY_TRANSFER_WAYPOINTS, horizontal_only=True,
            minimum_path_z=MIN_TRANSFER_Z,
            start_tool_angle=pick_above_pitch,
            goal_tool_angle=place_above_pitch,
        )

        print(f"[PRE-FLIGHT] {name}: phase 4/4 - descend to place (pitch {np.rad2deg(place_above_pitch):.1f} -> {np.rad2deg(place_contact_pitch):.1f} deg, Z={place_z:.3f} m >= {MIN_PLACE_Z:.3f} m)")
        simulated_q = preflight_plan_segment(
            simulated_q, place_contact, f"{name} place descent", VERTICAL_DURATION,
            PLACE_DESCENT_WAYPOINTS, vertical_only=True,
            minimum_path_z=MIN_PLACE_Z,
            start_tool_angle=place_above_pitch,
            goal_tool_angle=place_contact_pitch,
        )

        place_retract_pitch = find_feasible_pitch_at_z(place_xy, PLACE_RETRACT_Z, preferred=place_above_pitch)
        place_retract = np.array([place_xy[0], place_xy[1], PLACE_RETRACT_Z], dtype=float)
        print(f"[PRE-FLIGHT] {name}: retract after release to {PLACE_RETRACT_Z*100:.0f}cm (pitch {np.rad2deg(place_contact_pitch):.1f} -> {np.rad2deg(place_retract_pitch):.1f} deg)")
        simulated_q = preflight_plan_segment(
            simulated_q, place_retract, f"{name} post-place retract", VERTICAL_DURATION,
            POST_PLACE_RETRACT_WAYPOINTS, vertical_only=True,
            minimum_path_z=MIN_PLACE_Z,
            start_tool_angle=place_contact_pitch,
            goal_tool_angle=place_retract_pitch,
        )
        print(f"[PRE-FLIGHT] {name}: all paths passed.")

    print("\n[PRE-FLIGHT] Return to home TCP")
    home_min_z = min(float(HOME_XYZ[2]), 0.160)
    simulated_q = preflight_plan_segment(
        simulated_q, HOME_XYZ, "return to HOME XYZ", CARTESIAN_DURATION,
        RETURN_HOME_WAYPOINTS, minimum_path_z=home_min_z,
        start_tool_angle=float(simulated_q[1] + simulated_q[2] + simulated_q[3]),
        goal_tool_angle=TASK_TOOL_ANGLE,
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
    start_tool_angle=None,
    goal_tool_angle=None,
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
        start_tool_angle=start_tool_angle,
        goal_tool_angle=goal_tool_angle,
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

    print("\nVISION PICK/PLACE ARRAY (ordered by increasing transfer length)")
    print(f"Array shape: ({len(labels)}, 2, 2) = detected shapes x [pick/place] x [X/Y]")
    for index, name in enumerate(labels):
        dist = float(np.linalg.norm(targets[index, 1] - targets[index, 0]))
        print(
            f"{index + 1}. {name:16s}: PICK={targets[index, 0]} m -> "
            f"PLACE={targets[index, 1]} m | transfer length: {dist * 1000:6.1f} mm ({dist * 100:4.1f} cm)"
        )

    print("Requested home joints:", np.round(HOME_Q, 5))
    print(f"Home pitch: {HOME_ANGLE:.6f} rad ({np.rad2deg(HOME_ANGLE):.3f} deg)")
    print(f"Task pitch: {TASK_TOOL_ANGLE:.6f} rad (45 degrees)")
    print(f"Pick contact Z={PICK_CONTACT_Z:.3f} m; place contact Z={PLACE_CONTACT_Z:.3f} m (hard floor: {MIN_PLACE_Z:.3f} m)")
    print(f"Lift and XY transfer height: Z={TRAVEL_Z:.3f} m")
    print("No full 3D orientation lock: only tool pitch is constrained.")
    print("No collision model or safety-rated stop is available in this script.")

    node.enable_torque()
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

    # Open gripper to primary home position (145.5 deg) before beginning picking cycles
    node.command_gripper(GRIPPER_HOME_POSITION, f"initial open to home position ({GRIPPER_HOME_ANGLE_DEG:.1f}°)")

    for index, name in enumerate(labels):
        try:
            pick_xy = targets[index, 0]
            place_xy = targets[index, 1]
            place_z = get_shape_place_contact_z(name)
            pick_above_pitch, pick_contact_pitch = get_pick_place_pitches(
                pick_xy, PICK_CONTACT_Z, TRAVEL_Z, TASK_TOOL_ANGLE
            )
            place_above_pitch, place_contact_pitch = get_pick_place_pitches(
                place_xy, place_z, TRAVEL_Z, TASK_TOOL_ANGLE
            )

            pick_above = np.array([pick_xy[0], pick_xy[1], TRAVEL_Z], dtype=float)
            pick_contact = np.array([pick_xy[0], pick_xy[1], PICK_CONTACT_Z], dtype=float)
            place_above = np.array([place_xy[0], place_xy[1], TRAVEL_Z], dtype=float)
            place_contact = np.array([place_xy[0], place_xy[1], place_z], dtype=float)

            print(f"\n========== {index + 1}/{len(labels)}: {name} ==========")
            print(
                f"Pick XY: {np.round(pick_xy, 5)} (pitch {np.rad2deg(pick_above_pitch):.1f} -> {np.rad2deg(pick_contact_pitch):.1f} deg), "
                f"Place XY: {np.round(place_xy, 5)} (pitch {np.rad2deg(place_above_pitch):.1f} -> {np.rad2deg(place_contact_pitch):.1f} deg)"
            )

            approach_min_z = min(float(fk(current_q)[0][2]), MIN_TRANSFER_Z)
            current_q = plan_and_execute_to(
                node, current_q, pick_above,
                f"{name}: approach above pick", CARTESIAN_DURATION,
                APPROACH_WAYPOINTS, minimum_path_z=approach_min_z,
                start_tool_angle=float(current_q[1] + current_q[2] + current_q[3]),
                goal_tool_angle=pick_above_pitch,
            )
            node.command_gripper(
                GRIPPER_HOME_POSITION,
                f"open to home before picking {name} ({GRIPPER_HOME_ANGLE_DEG:.1f}°)",
            )

            # Phase 1/4: descend to pick
            current_q = plan_and_execute_to(
                node, current_q, pick_contact,
                f"{name}: phase 1/4 descend to pick", VERTICAL_DURATION,
                PICK_DESCENT_WAYPOINTS, vertical_only=True,
                start_tool_angle=pick_above_pitch,
                goal_tool_angle=pick_contact_pitch,
            )
            grip_pos = GRIP_POSITIONS_BY_SHAPE.get(name, GRIPPER_CLOSED_POSITION)
            grip_deg = prismatic_to_angle_deg(grip_pos)
            node.command_gripper(grip_pos, f"grip {name} ({grip_deg:.1f}° / {grip_pos:.4f} m)")
            time.sleep(0.5)

            # Phase 2/4: lift to clearance
            current_q = plan_and_execute_to(
                node, current_q, pick_above,
                f"{name}: phase 2/4 lift to clearance", VERTICAL_DURATION,
                PICK_LIFT_WAYPOINTS, vertical_only=True,
                start_tool_angle=pick_contact_pitch,
                goal_tool_angle=pick_above_pitch,
            )

            # Phase 3/4: XY transfer
            current_q = plan_and_execute_to(
                node, current_q, place_above,
                f"{name}: phase 3/4 XY transfer", CARTESIAN_DURATION,
                XY_TRANSFER_WAYPOINTS, minimum_path_z=MIN_TRANSFER_Z,
                horizontal_only=True,
                start_tool_angle=pick_above_pitch,
                goal_tool_angle=place_above_pitch,
            )

            # Phase 4/4: descend to place
            current_q = plan_and_execute_to(
                node, current_q, place_contact,
                f"{name}: phase 4/4 descend to place", VERTICAL_DURATION,
                PLACE_DESCENT_WAYPOINTS, vertical_only=True,
                minimum_path_z=MIN_PLACE_Z,
                start_tool_angle=place_above_pitch,
                goal_tool_angle=place_contact_pitch,
            )
            node.command_gripper(
                GRIPPER_HOME_POSITION,
                f"open to home after placing {name} ({GRIPPER_HOME_ANGLE_DEG:.1f}°)",
            )

            # Retract after placement
            place_retract_pitch = find_feasible_pitch_at_z(place_xy, PLACE_RETRACT_Z, preferred=place_above_pitch)
            place_retract = np.array([place_xy[0], place_xy[1], PLACE_RETRACT_Z], dtype=float)
            try:
                current_q = plan_and_execute_to(
                    node, current_q, place_retract,
                    f"{name}: retract after placement to {PLACE_RETRACT_Z*100:.0f}cm", VERTICAL_DURATION,
                    POST_PLACE_RETRACT_WAYPOINTS, vertical_only=True,
                    minimum_path_z=MIN_PLACE_Z,
                    start_tool_angle=place_contact_pitch,
                    goal_tool_angle=place_retract_pitch,
                )
            except Exception as exc:
                print(f"[WARN] Retract planning error after placing {name}: {exc}. Proceeding directly home.", flush=True)
        except Exception as exc:
            print(f"[WARN] {name}: step failed ({exc}). Recovering to home and continuing with the next shape.", flush=True)
            try:
                current_q = validate_q(node.get_current_q(), "Measured joints after failure")
                rt, rp = plan_startup_joint_move(current_q, HOME_Q)
                node.execute_arm(rp, rt, f"recover after {name}")
                current_q = validate_q(node.get_current_q(), "Measured joints after recovery")
                node.command_gripper(GRIPPER_HOME_POSITION, "open gripper after recovery")
            except Exception as exc2:
                print(f"[WARN] Recovery move failed ({exc2}); continuing.", flush=True)

    try:
        home_min_z = min(float(HOME_XYZ[2]), 0.160)
        current_q = plan_and_execute_to(
            node, current_q, HOME_XYZ,
            "return to HOME XYZ at working pitch", CARTESIAN_DURATION,
            RETURN_HOME_WAYPOINTS, minimum_path_z=home_min_z,
            start_tool_angle=float(current_q[1] + current_q[2] + current_q[3]),
            goal_tool_angle=TASK_TOOL_ANGLE,
        )
    except Exception as exc:
        print(f"[WARN] Cartesian return to HOME_XYZ error: {exc}. Using joint-space move directly.", flush=True)

    final_times, final_path = plan_startup_joint_move(current_q, HOME_Q)
    node.execute_arm(final_path, final_times, "final move to exact TARGET_Q home pose")
    node.command_gripper(GRIPPER_HOME_POSITION, f"final home gripper ({GRIPPER_HOME_ANGLE_DEG:.1f}°)")

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
