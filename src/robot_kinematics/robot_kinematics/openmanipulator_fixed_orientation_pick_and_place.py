#!/usr/bin/env python3
"""
OpenMANIPULATOR-X fixed-tool-pitch pick-and-place controller.

Motion policy:
  * Startup: move from measured joint state to HOME_Q.
  * Cartesian paths interpolate X, Y and Z together in 3D.
  * No Cartesian segment forces the TCP to remain at a constant height or
    forces X/Y to remain fixed. TRAVEL_Z is only a clearance waypoint value.
  * The tool pitch angle (q2 + q3 + q4) is held at one task angle during
    Cartesian motion. A small measured pitch error at a segment start is
    smoothly corrected over that segment rather than causing a sudden jump.
  * The full 3D orientation is NOT held fixed when q1 changes; see below.
  * Pick: open -> descend -> close -> ascend.
  * Place: travel -> descend -> open -> ascend.

IMPORTANT:
  1. HOME_ANGLE is calculated from HOME_Q using the supplied kinematics.py:
         angle = q2 + q3 + q4
     This is the orientation represented by your supplied home configuration.
     It is NOT the 90-degree vertically-down orientation in that model.
  2. Moving from an arbitrary starting pose to HOME_Q by joint interpolation
     cannot guarantee constant tool orientation during that initial transition.
  3. This 4-DOF arm and the supplied IK constrain the tool pitch only:
         q2 + q3 + q4 = TASK_TOOL_ANGLE
     As q1 changes, the tool's world-frame yaw changes. Thus this script does
     NOT guarantee an unchanged full 3D orientation.
  4. This code assumes the FK/IK model you supplied is calibrated to the real
     robot. Validate it against measured TCP positions before enabling motion.
  5. DRY_RUN is True by default after changing the Cartesian path policy.
     Inspect and validate the path before enabling hardware execution.
  6. Cartesian segment starts are re-anchored to measured FK for offsets up to
     MAX_SEGMENT_START_ERROR. Larger offsets stop planning and require inspection.
  7. There is no collision model or safety-rated stop in this script.
"""

import time
import os
from pathlib import Path
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory, GripperCommand
from builtin_interfaces.msg import Duration

from robot_kinematics.kinematics import (
    forward_kinematics,
    inverse_kinematics,
    JOINT_LIMITS,
)

# ---------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]
JOINT_STATE_TOPIC = "/joint_states"
ARM_ACTION = "/arm_controller/follow_joint_trajectory"
GRIPPER_ACTION = "/gripper_controller/gripper_cmd"

# Supplied initial/home configuration (rad).
HOME_Q = np.array([0.08283, -0.96181, 0.17181, 1.29008], dtype=float)

# This value follows from your supplied FK model's orientation convention.
HOME_ANGLE = float(HOME_Q[1] + HOME_Q[2] + HOME_Q[3])
# Set once from measured joints after reaching HOME_Q. All Cartesian paths
# retain this same task pitch, even if the measured home pose has a tiny offset.
TASK_TOOL_ANGLE = HOME_ANGLE

# The supplied FK gives this position for HOME_Q:
HOME_XYZ = np.array([0.11822, 0.00882, 0.19707], dtype=float)

TRAVEL_Z = 0.160  # Clearance waypoint, not a path-wide Z constraint.

# One-shot camera output. Keep this file beside this controller.
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
PICK_CONTACT_Z = 0.110
PLACE_CONTACT_Z = 0.110

# GripperCommand action position values in metres.
# User observed the previous mapping behaved in reverse, so the endpoints
# are swapped here. Verify with a gripper-only test before running the arm.
GRIPPER_OPEN_POSITION = 0.000
GRIPPER_CLOSED_POSITION = 0.010
GRIPPER_HALF_CLOSED_POSITION = (
    GRIPPER_OPEN_POSITION + GRIPPER_CLOSED_POSITION
) / 2.0
GRIPPER_MAX_EFFORT = 20.0

# Safe default after changing the path planner. Set False only after reviewing
# the new 3D paths and confirming the workspace is clear. EXECUTE is still required.
DRY_RUN = True

# Explicit confirmation is still required if DRY_RUN is set False.
REQUIRE_EXECUTE_CONFIRMATION = True

# Motion settings. Higher waypoint density and longer segment durations are
# intended as a faster, conservative first hardware test. These are software
# caps, not manufacturer-certified limits; test unloaded at low speed first.
FREQUENCY_HZ = 50.0
HOME_DURATION = 5.0
CARTESIAN_DURATION = 3.0
VERTICAL_DURATION = 3.0
# Conservative initial hardware test caps. These are below the URDF's
# 4.8 rad/s velocity declaration and must be validated on the real arm.
MAX_JOINT_VELOCITY = np.array([1.5, 1.5, 1.5, 1.5])
# Raised from 1.0 rad/s^2 to reduce excessive time scaling. Start unloaded.
MAX_JOINT_ACCELERATION = np.array([3.0, 3.0, 3.0, 3.0])
MAX_IK_STEP = 0.08

JOINT_STATE_TIMEOUT = 10.0
JOINT_STATE_MAX_AGE = 0.5
POSITION_TOLERANCE = 0.002  # IK waypoint accuracy relative to its requested XYZ.
MAX_SEGMENT_START_ERROR = 0.010  # Re-anchor up to 10 mm to measured FK; reject larger discrepancies.
ORIENTATION_TOLERANCE = 2e-3  # Allow small measured-servo deviation after reaching HOME_Q.
START_ORIENTATION_CORRECTION_TOLERANCE = 0.02  # rad; correct small start error smoothly over the segment.
JOINT_GOAL_TOLERANCE = 0.05

# Reject startup joint-space paths if FK height leaves this broad envelope.
# This is not collision detection. Adjust only after physically checking it.
STARTUP_MIN_Z = 0.030
STARTUP_MAX_Z = 0.300


# ---------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------

def quintic(s):
    s = float(np.clip(s, 0.0, 1.0))
    return 10.0*s**3 - 15.0*s**4 + 6.0*s**5


def make_times(duration):
    count = int(round(duration * FREQUENCY_HZ)) + 1
    if count < 3:
        raise ValueError("Trajectory requires at least three samples.")
    return np.linspace(0.0, duration, count)


def wrap_delta(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def fk(q):
    result = forward_kinematics(*np.asarray(q, dtype=float))
    p = np.asarray(result[0], dtype=float)
    R = np.asarray(result[1], dtype=float)
    if p.shape != (3,) or R.shape != (3, 3):
        raise RuntimeError("Unexpected forward_kinematics return shape.")
    return p, R


def validate_q(q, label):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError(f"{label} must be four finite joint values.")
    if np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1]):
        raise RuntimeError(f"{label} violates JOINT_LIMITS: {q}")
    return q


def expected_tool_x_axis(q1, angle):
    # The supplied kinematics rotates the arm's radial plane by q1.
    # Therefore the tool x-axis in the base frame depends on BOTH q1 and angle.
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
    actual_p, actual_R = fk(q)
    pos_error = float(np.linalg.norm(actual_p - xyz))
    if pos_error > POSITION_TOLERANCE:
        raise RuntimeError(
            f"{label}: FK position error {pos_error:.4f} m exceeds tolerance."
        )
    pitch_error = abs(wrap_delta(float(q[1] + q[2] + q[3]), tool_angle))
    if pitch_error > ORIENTATION_TOLERANCE:
        raise RuntimeError(
            f"{label}: pitch constraint error={pitch_error:.6g} rad."
        )
    verify_orientation(actual_R, float(q[0]), tool_angle, label)
    return actual_p, actual_R


def solve_ik(xyz, previous_q, tool_angle=None):
    if tool_angle is None:
        tool_angle = TASK_TOOL_ANGLE
    x, y, z = [float(v) for v in xyz]
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
    verify_pose(q, np.array([x, y, z]), "IK result", float(tool_angle))
    return q


def derivatives(q_path, times):
    qd = np.gradient(q_path, times, axis=0, edge_order=2)
    qdd = np.gradient(qd, times, axis=0, edge_order=2)
    qd[0] = 0.0
    qd[-1] = 0.0
    return qd, qdd


def validate_joint_path(q_path, times, label):
    q_path = np.asarray(q_path, dtype=float)
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


# ---------------------------------------------------------------------
# TRAJECTORY PLANNERS
# ---------------------------------------------------------------------

def plan_startup_joint_move(q_start, q_goal):
    """Reach HOME_Q by joint-space interpolation; no orientation constraint."""

    q_start = validate_q(q_start, "Measured starting joints")
    q_goal = validate_q(q_goal, "HOME_Q")

    duration = HOME_DURATION
    times = make_times(duration)
    delta = wrap_delta(q_goal, q_start)
    q_unwrapped_goal = q_start + delta

    if np.any(q_unwrapped_goal < JOINT_LIMITS[:, 0]) or np.any(
        q_unwrapped_goal > JOINT_LIMITS[:, 1]
    ):
        raise RuntimeError("Startup interpolation crosses a joint limit.")

    path = np.array([
        q_start + quintic(t / duration) * delta for t in times
    ])

    # Automatically lengthen the startup move only as much as required by
    # the configured joint velocity/acceleration caps.
    original_duration = float(duration)
    for _ in range(12):
        qd, qdd = derivatives(path, times)
        max_v = np.max(np.abs(qd), axis=0)
        max_a = np.max(np.abs(qdd), axis=0)
        velocity_ratio = float(np.max(max_v / MAX_JOINT_VELOCITY))
        acceleration_ratio = float(np.max(max_a / MAX_JOINT_ACCELERATION))
        if velocity_ratio <= 1.0 and acceleration_ratio <= 1.0:
            break
        scale = max(velocity_ratio, np.sqrt(acceleration_ratio))
        duration *= max(1.05, 1.05 * scale)
        times = make_times(duration)
        path = np.array([
            q_start + quintic(t / duration) * delta for t in times
        ])
    else:
        raise RuntimeError("Startup move cannot satisfy joint speed/acceleration caps.")

    if duration > original_duration * 1.001:
        print(f"Startup timing adjusted from {original_duration:.2f}s to {duration:.2f}s.")

    # No tool-orientation constraint is applied during this startup move.
    # The arm is commanded only to reach the specified HOME_Q.
    fk_points = np.array([fk(q)[0] for q in path])
    z_start = float(fk_points[0, 2])
    z_min = float(np.min(fk_points[:, 2]))
    z_max = float(np.max(fk_points[:, 2]))

    # Permit a starting pose below STARTUP_MIN_Z, but do not permit the
    # planned path to descend more than 2 mm below its measured start height.
    STARTUP_INITIAL_DROP_TOLERANCE = 0.002
    allowed_min_z = min(STARTUP_MIN_Z, z_start - STARTUP_INITIAL_DROP_TOLERANCE)

    if z_min < allowed_min_z or z_max > STARTUP_MAX_Z:
        raise RuntimeError(
            f"Startup FK Z range [{z_min:.3f}, {z_max:.3f}] m "
            f"exceeds allowed range [{allowed_min_z:.3f}, "
            f"{STARTUP_MAX_Z:.3f}] m. No command sent."
        )

    validate_joint_path(path, times, "Startup-to-home")
    print(f"Startup FK Z range: [{z_min:.3f}, {z_max:.3f}] m")
    print("Startup height check passed.")
    print("Startup orientation is unconstrained; target is HOME_Q.")

    return times, path


def plan_cartesian_segment(start_xyz, goal_xyz, initial_q, duration):
    """Plan a straight Cartesian segment in full XYZ with fixed tool pitch.

    X, Y and Z are all interpolated simultaneously. There is no constraint
    that the end effector remain at a constant height or that either X or Y
    remain fixed. If the measured starting pitch differs slightly from the
    task pitch, the planner blends that small error out smoothly over the
    segment so the first command is continuous with the measured state.
    """
    start_xyz = np.asarray(start_xyz, dtype=float)
    goal_xyz = np.asarray(goal_xyz, dtype=float)

    if start_xyz.shape != (3,) or goal_xyz.shape != (3,):
        raise ValueError("Cartesian endpoints must be 3D.")

    previous_q = validate_q(initial_q, "Cartesian segment initial joints")

    # Re-anchor every new segment at the measured FK position. The previous
    # segment's commanded endpoint can differ from the physical endpoint by a
    # few millimetres because of servo tracking, compliance, and model error.
    # Starting the next path at the old target would create a discontinuity;
    # starting it at measured FK keeps sample 0 exactly at the current joints.
    actual_start_xyz, _ = fk(previous_q)
    planned_start_xyz = start_xyz.copy()
    start_error = float(np.linalg.norm(actual_start_xyz - planned_start_xyz))
    if start_error > MAX_SEGMENT_START_ERROR:
        raise RuntimeError(
            f"Measured FK differs from the previous Cartesian target by "
            f"{start_error:.4f} m, exceeding the {MAX_SEGMENT_START_ERROR:.3f} m "
            "re-anchoring limit. Check for object collision, joint slip, or FK calibration error. "
            "No trajectory sent."
        )
    if start_error > POSITION_TOLERANCE:
        print(
            f"Cartesian start re-anchored to measured FK "
            f"(previous target offset={start_error * 1000:.1f} mm)."
        )
    start_xyz = actual_start_xyz.copy()

    start_pitch = float(previous_q[1] + previous_q[2] + previous_q[3])
    pitch_delta = wrap_delta(TASK_TOOL_ANGLE, start_pitch)
    if abs(pitch_delta) > START_ORIENTATION_CORRECTION_TOLERANCE:
        raise RuntimeError(
            f"Cartesian segment starts with pitch error {pitch_delta:.6f} rad, "
            f"above correction allowance "
            f"{START_ORIENTATION_CORRECTION_TOLERANCE:.6f} rad. "
            "No trajectory sent."
        )

    times = make_times(duration)
    q_path = []
    xyz_path = []
    angle_path = []

    for sample_index, t in enumerate(times):
        s = quintic(t / duration)
        xyz = start_xyz + s * (goal_xyz - start_xyz)
        # Correct any small initial pitch mismatch smoothly; after the blend,
        # every waypoint uses the same TASK_TOOL_ANGLE.
        sample_angle = start_pitch + s * pitch_delta

        if sample_index == 0:
            q = previous_q.copy()
            start_fk, _ = fk(q)
            position_error = float(np.linalg.norm(start_fk - xyz))
            if position_error > POSITION_TOLERANCE:
                raise RuntimeError(
                    "Cartesian initial joint pose does not match its start XYZ; "
                    f"FK error={position_error:.6f} m."
                )
        else:
            q = solve_ik(xyz, previous_q, sample_angle)

        if q_path:
            step = float(np.max(np.abs(wrap_delta(q, previous_q))))
            if step > MAX_IK_STEP:
                raise RuntimeError(
                    f"IK discontinuity in 3D Cartesian segment at sample "
                    f"{len(q_path)}: {step:.3f} rad."
                )

        q_path.append(q)
        xyz_path.append(xyz)
        angle_path.append(sample_angle)
        previous_q = q

    q_path = np.asarray(q_path)
    xyz_path = np.asarray(xyz_path)

    for i, (q, target, angle) in enumerate(zip(q_path, xyz_path, angle_path)):
        if i == 0:
            actual_p, _ = fk(q)
            start_pos_error = float(np.linalg.norm(actual_p - target))
            if start_pos_error > POSITION_TOLERANCE:
                raise RuntimeError(
                    f"3D sample 0: FK position error {start_pos_error:.6f} m."
                )
            measured_pitch = float(q[1] + q[2] + q[3])
            if abs(wrap_delta(measured_pitch, angle)) > ORIENTATION_TOLERANCE:
                raise RuntimeError("Initial measured pitch changed unexpectedly.")
        else:
            verify_pose(q, target, f"3D sample {i}", float(angle))

    # Time-scale the same 3D path to remain inside the configured joint caps.
    original_duration = float(times[-1] - times[0])
    time_scale = 1.0
    for attempt in range(10):
        qd, qdd = derivatives(q_path, times)
        max_v = np.max(np.abs(qd), axis=0)
        max_a = np.max(np.abs(qdd), axis=0)

        velocity_ratio = float(np.max(max_v / MAX_JOINT_VELOCITY))
        acceleration_ratio = float(np.max(max_a / MAX_JOINT_ACCELERATION))

        if velocity_ratio <= 1.0 and acceleration_ratio <= 1.0:
            break

        required_scale = max(velocity_ratio, np.sqrt(acceleration_ratio))
        time_scale *= max(1.10, 1.10 * required_scale)
        times = np.linspace(0.0, original_duration * time_scale, len(q_path))
    else:
        raise RuntimeError(
            "3D Cartesian segment could not satisfy velocity/acceleration "
            f"limits after time scaling. max velocity={max_v}, "
            f"max acceleration={max_a}. No trajectory was sent."
        )

    if time_scale > 1.001:
        print(
            f"3D Cartesian timing stretched from {original_duration:.2f}s "
            f"to {times[-1]:.2f}s to satisfy joint limits."
        )

    return times, q_path, xyz_path


# ---------------------------------------------------------------------
# ROS NODE
# ---------------------------------------------------------------------

class PickPlaceNode(Node):
    def __init__(self):
        super().__init__("fixed_pitch_3d_pick_place")
        self.current_q = None
        self.last_state_time = None

        self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self.joint_callback, 10
        )
        self.arm_client = ActionClient(
            self, FollowJointTrajectory, ARM_ACTION
        )
        self.gripper_client = ActionClient(
            self, GripperCommand, GRIPPER_ACTION
        )

    def joint_callback(self, msg):
        if len(msg.name) != len(msg.position):
            return
        m = dict(zip(msg.name, msg.position))
        if not all(n in m for n in JOINT_NAMES):
            return
        q = np.array([m[n] for n in JOINT_NAMES], dtype=float)
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
            print(f"[DRY RUN] Would execute: {label} ({len(q_path)} samples)")
            return

        if not self.arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Arm action server unavailable: {ARM_ACTION}")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        q_path = np.asarray(q_path, dtype=float)
        times = np.asarray(times, dtype=float)
        qd, _ = derivatives(q_path, times)

        for q, velocity, t in zip(q_path, qd, times):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in q]
            # Supply the planned joint velocities rather than forcing every
            # waypoint velocity to zero. The controller can interpolate the
            # trajectory consistently between these samples.
            p.velocities = [float(v) for v in velocity]
            sec = int(t)
            nsec = int(round((t - sec) * 1e9))
            if nsec >= 1_000_000_000:
                sec += 1
                nsec -= 1_000_000_000
            p.time_from_start = Duration(sec=sec, nanosec=nsec)
            goal.trajectory.points.append(p)

        send = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        handle = send.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"Arm rejected trajectory: {label}")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None or wrapped.result.error_code != 0:
            reason = "no result" if wrapped is None else wrapped.result.error_string
            raise RuntimeError(f"Arm trajectory failed ({label}): {reason}")

        print(f"Completed: {label}")

    def command_gripper(self, position, label):
        if DRY_RUN:
            print(f"[DRY RUN] Would command gripper {label}: {position:.4f} m")
            return

        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(
                f"Gripper action server unavailable: {GRIPPER_ACTION}"
            )

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(GRIPPER_MAX_EFFORT)

        send = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        handle = send.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"Gripper rejected command: {label}")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError(f"No gripper result: {label}")
        if not wrapped.result.reached_goal:
            raise RuntimeError(
                f"Gripper did not reach requested position for {label}; "
                f"stalled={wrapped.result.stalled}, "
                f"position={wrapped.result.position:.4f}"
            )
        print(f"Gripper completed: {label}")


# ---------------------------------------------------------------------
# COMPLETE PICK-AND-PLACE SEQUENCE
# ---------------------------------------------------------------------

def load_sort_targets():
    """Load the N x 2 x 2 array for the detected, available shape classes."""
    if not SORT_TARGETS_FILE.is_file():
        raise FileNotFoundError(
            f"Vision target file not found: {SORT_TARGETS_FILE}. "
            "Run run_shape_sorting.py to generate this run's averaged vision targets first."
        )

    with np.load(SORT_TARGETS_FILE, allow_pickle=False) as data:
        if "targets" not in data.files or "labels" not in data.files:
            raise ValueError(
                "shape_sort_targets.npz must contain 'targets' and 'labels'."
            )
        targets = np.asarray(data["targets"], dtype=float)
        labels = tuple(str(v) for v in data["labels"].tolist())

    if targets.ndim != 3 or targets.shape[1:] != (2, 2):
        raise ValueError(
            f"Expected targets with shape (N, 2, 2), got {targets.shape}."
        )
    if len(labels) != targets.shape[0]:
        raise ValueError(
            f"Target row count ({targets.shape[0]}) does not match labels ({len(labels)})."
        )
    if not labels:
        raise ValueError("Vision file contains no complete pick/place pairs.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate shape labels in vision file: {labels}")
    unknown = [name for name in labels if name not in SORT_SHAPES]
    if unknown:
        raise ValueError(f"Unsupported shape labels in vision file: {unknown}")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Vision target array contains NaN or infinite values.")

    for i, name in enumerate(labels):
        pick_xy = targets[i, 0]
        place_xy = targets[i, 1]
        if np.linalg.norm(pick_xy - place_xy) < 0.005:
            raise ValueError(
                f"{name}: pick and place coordinates are less than 5 mm apart."
            )

    return targets, labels


def preflight_target_ik(targets, labels):
    """Check every approach/contact target for IK reachability before motion.

    This validates target IK and joint-step continuity. Full interpolated path
    checks are still performed immediately before each trajectory is sent.
    """
    simulated_q = HOME_Q.copy()
    print("\nPRE-FLIGHT IK CHECK")
    for i, name in enumerate(labels):
        pick_xy = targets[i, 0]
        place_xy = targets[i, 1]
        points = [
            ("pick approach", np.array([pick_xy[0], pick_xy[1], TRAVEL_Z])),
            ("pick contact", np.array([pick_xy[0], pick_xy[1], PICK_CONTACT_Z])),
            ("pick retract", np.array([pick_xy[0], pick_xy[1], TRAVEL_Z])),
            ("place approach", np.array([place_xy[0], place_xy[1], TRAVEL_Z])),
            ("place contact", np.array([place_xy[0], place_xy[1], PLACE_CONTACT_Z])),
            ("place retract", np.array([place_xy[0], place_xy[1], TRAVEL_Z])),
        ]
        for phase, xyz in points:
            q = solve_ik(xyz, simulated_q, TASK_TOOL_ANGLE)
            # Carry the selected IK branch forward. MAX_IK_STEP is checked
            # later between adjacent interpolated path samples, not between
            # sparse contact/approach endpoints.
            simulated_q = q
        print(f"  {name}: all six pick/place IK targets reachable")


def plan_and_execute_to(node, current_q, goal_xyz, label, duration):
    """Plan from current measured/simulated FK to a 3D goal, then execute."""
    current_q = validate_q(current_q, f"Current joints before {label}")
    start_xyz, _ = fk(current_q)
    times, q_path, _ = plan_cartesian_segment(
        start_xyz, np.asarray(goal_xyz, dtype=float), current_q, duration
    )
    node.execute_arm(q_path, times, label)

    if DRY_RUN:
        # In dry-run, advance the software state to the planned endpoint so all
        # all detected pick/place paths can be checked in one pass.
        return q_path[-1].copy()

    # In hardware mode, always re-read the actual joint feedback after motion.
    measured_q = validate_q(node.get_current_q(), f"Measured joints after {label}")
    return measured_q


def run_sequence(node):
    targets, labels = load_sort_targets()
    print("\nVISION PICK/PLACE ARRAY")
    print(
        f"Array shape: ({len(labels)}, 2, 2) = detected shapes x "
        "[pick/place] x [X/Y]"
    )
    for i, name in enumerate(labels):
        print(
            f"{i + 1}. {name}: PICK={targets[i, 0]} m, "
            f"PLACE={targets[i, 1]} m"
        )

    print(f"Reference tool pitch: {HOME_ANGLE:.6f} rad "
          f"({np.rad2deg(HOME_ANGLE):.3f} deg)")
    print(f"Pick contact Z={PICK_CONTACT_Z:.3f} m; "
          f"place contact Z={PLACE_CONTACT_Z:.3f} m; "
          f"clearance waypoint Z={TRAVEL_Z:.3f} m")
    print("Cartesian trajectories interpolate X, Y and Z together.")
    print("Only tool pitch is constrained; full 3D orientation is not fixed.")
    print("No path collision checking is available.")

    current_q = validate_q(node.get_current_q(), "Current joint feedback")
    current_xyz, _ = fk(current_q)
    print("Current joints:", np.round(current_q, 5))
    print("Current TCP XYZ:", np.round(current_xyz, 5))

    # Prepare startup and validate all target endpoints before any movement.
    t_home, q_home_path = plan_startup_joint_move(current_q, HOME_Q)
    home_xyz_fk, home_R = fk(HOME_Q)
    if np.linalg.norm(home_xyz_fk - HOME_XYZ) > 0.001:
        raise RuntimeError("HOME_Q FK does not match HOME_XYZ; check kinematics.py.")
    verify_orientation(home_R, float(HOME_Q[0]), HOME_ANGLE, "HOME_Q")

    global TASK_TOOL_ANGLE
    TASK_TOOL_ANGLE = HOME_ANGLE
    preflight_target_ik(targets, labels)

    print("\nEXECUTION PLAN")
    print("Vision averaged valid object/hole coordinates over the 5-second window; live preview was accepted.")
    print(f"{len(labels)} detected shape(s) will be sorted in this order:", ", ".join(labels))
    print("Each shape uses its own detected pick centre and matching hole centre.")
    print("IK is solved for every approach/contact waypoint and every Cartesian path sample.")

    if DRY_RUN:
        print(
            f"\nDRY_RUN=True: planning {len(labels)} pick/place cycle(s) "
            "without moving hardware."
        )
    elif REQUIRE_EXECUTE_CONFIRMATION:
        print("\nClear the workspace, keep the emergency stop accessible,")
        print("and ensure the robot is supported during any required setup.")
        if input(
            f"Type EXECUTE to run these {len(labels)} sorting cycle(s): "
        ).strip() != "EXECUTE":
            print("Cancelled. No movement sent.")
            return

    # Startup move to HOME_Q.
    node.execute_arm(q_home_path, t_home, "startup to HOME_Q")
    if DRY_RUN:
        current_q = HOME_Q.copy()
    else:
        current_q = validate_q(node.get_current_q(), "Measured joints at HOME_Q")
        home_joint_error = float(np.max(np.abs(wrap_delta(current_q, HOME_Q))))
        if home_joint_error > JOINT_GOAL_TOLERANCE:
            raise RuntimeError(
                f"Robot did not reach HOME_Q; maximum joint error={home_joint_error:.4f} rad."
            )

    measured_home_pitch = float(current_q[1] + current_q[2] + current_q[3])
    home_pitch_offset = abs(wrap_delta(measured_home_pitch, HOME_ANGLE))
    if home_pitch_offset > START_ORIENTATION_CORRECTION_TOLERANCE:
        raise RuntimeError(
            f"Measured HOME_Q pitch differs from reference by {home_pitch_offset:.5f} rad."
        )
    TASK_TOOL_ANGLE = measured_home_pitch
    print(f"Task tool pitch fixed at {TASK_TOOL_ANGLE:.6f} rad.")

    for i, name in enumerate(labels):
        pick_xy = targets[i, 0]
        place_xy = targets[i, 1]

        pick_above = np.array([pick_xy[0], pick_xy[1], TRAVEL_Z], dtype=float)
        pick_contact = np.array(
            [pick_xy[0], pick_xy[1], PICK_CONTACT_Z], dtype=float
        )
        place_above = np.array([place_xy[0], place_xy[1], TRAVEL_Z], dtype=float)
        place_contact = np.array(
            [place_xy[0], place_xy[1], PLACE_CONTACT_Z], dtype=float
        )

        print(f"\n========== {i + 1}/{len(labels)}: {name} ==========")
        print("Pick XY:", np.round(pick_xy, 5), "Place XY:", np.round(place_xy, 5))

        # Travel from current TCP position to above this coloured piece.
        current_q = plan_and_execute_to(
            node, current_q, pick_above,
            f"{name}: 3D move to pick approach", CARTESIAN_DURATION
        )
        node.command_gripper(GRIPPER_OPEN_POSITION, f"open before {name} pickup")

        # Descend to the coloured shape centre and grip it.
        current_q = plan_and_execute_to(
            node, current_q, pick_contact,
            f"{name}: descend to pick centre", VERTICAL_DURATION
        )
        node.command_gripper(
            GRIPPER_HALF_CLOSED_POSITION, f"grip {name}"
        )

        # Lift clear of the base, transfer to the matching uncoloured hole.
        current_q = plan_and_execute_to(
            node, current_q, pick_above,
            f"{name}: lift from pick", VERTICAL_DURATION
        )
        current_q = plan_and_execute_to(
            node, current_q, place_above,
            f"{name}: 3D transfer to matching hole", CARTESIAN_DURATION
        )

        # Insert at the matching hole centre and release.
        current_q = plan_and_execute_to(
            node, current_q, place_contact,
            f"{name}: descend into matching hole", VERTICAL_DURATION
        )
        node.command_gripper(
            GRIPPER_OPEN_POSITION, f"release {name} into matching hole"
        )
        current_q = plan_and_execute_to(
            node, current_q, place_above,
            f"{name}: retract after placement", VERTICAL_DURATION
        )

    # Return to the known home TCP position after all detected shapes are sorted.
    current_xyz, _ = fk(current_q)
    current_q = plan_and_execute_to(
        node, current_q, HOME_XYZ,
        "return to HOME XYZ", CARTESIAN_DURATION
    )
    if DRY_RUN:
        final_times, final_path = plan_startup_joint_move(current_q, HOME_Q)
        node.execute_arm(final_path, final_times, "final move to exact HOME_Q")
    else:
        final_times, final_path = plan_startup_joint_move(current_q, HOME_Q)
        node.execute_arm(final_path, final_times, "final move to exact HOME_Q")
        final_q = node.get_current_q()
        final_error = float(np.max(np.abs(wrap_delta(final_q, HOME_Q))))
        if final_error > JOINT_GOAL_TOLERANCE:
            raise RuntimeError(
                f"Final HOME_Q verification failed: {final_error:.4f} rad."
            )

    print(f"\nSorting sequence completed for {len(labels)} detected shape(s).")
    if DRY_RUN:
        print("DRY RUN ONLY: no physical robot motion was commanded.")


def main():
    # Verify the supplied home values against the kinematics model before ROS motion.
    home_p, home_R = fk(HOME_Q)
    if np.linalg.norm(home_p - HOME_XYZ) > 0.001:
        raise RuntimeError(
            f"HOME_Q FK={home_p}, but supplied HOME_XYZ={HOME_XYZ}. "
            "Do not move until the kinematics model is corrected."
        )
    verify_orientation(home_R, float(HOME_Q[0]), HOME_ANGLE, "HOME_Q")

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