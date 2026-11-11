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

TRAVEL_Z = 0.160  # Clearance waypoint height, not a path-wide Z constraint.
CONTACT_Z = 0.110  # Pick/place target contact height.
PICK_CONTACT_Z = CONTACT_Z
PLACE_CONTACT_Z = CONTACT_Z

# TEMPORARY TEST COORDINATES in base_link metres.
# Replace these with the vision output later.
PICK_XY = np.array([0.200, 0.000], dtype=float)
PLACE_XY = np.array([0.220, 0.050], dtype=float)

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
DRY_RUN = False

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

def run_sequence(node):
    print(f"Reference tool pitch from HOME_Q: {HOME_ANGLE:.6f} rad "
          f"({np.rad2deg(HOME_ANGLE):.3f} deg)")
    print(f"Travel Z={TRAVEL_Z:.3f} m; shared pick/place contact Z={CONTACT_Z:.3f} m")
    print(f"Motion timing: home={HOME_DURATION:.1f}s, XY={CARTESIAN_DURATION:.1f}s, vertical={VERTICAL_DURATION:.1f}s at {FREQUENCY_HZ:.0f} Hz")
    print("Temporary pick XY:", PICK_XY)
    print("Temporary place XY:", PLACE_XY)
    print(f"Gripper positions: open={GRIPPER_OPEN_POSITION:.4f} m, "
          f"half-close={GRIPPER_HALF_CLOSED_POSITION:.4f} m, "
          f"closed={GRIPPER_CLOSED_POSITION:.4f} m")

    current_q = validate_q(node.get_current_q(), "Current joint feedback")
    current_xyz, _ = fk(current_q)
    print("Current joints:", np.round(current_q, 5))
    print("Current TCP XYZ:", np.round(current_xyz, 5))

    # 1. Startup: move from the measured pose to HOME_Q.
    t_home, q_home_path = plan_startup_joint_move(current_q, HOME_Q)

    home_xyz_fk, home_R = fk(HOME_Q)
    if np.linalg.norm(home_xyz_fk - HOME_XYZ) > 0.001:
        raise RuntimeError(
            "HOME_Q FK does not match HOME_XYZ. Check kinematics.py."
        )
    verify_orientation(home_R, float(HOME_Q[0]), HOME_ANGLE, "HOME_Q")

    # The startup joint path is the only path prepared before motion.
    # All Cartesian segments are planned immediately before they are executed.
    # This avoids computing IK for the entire pick-and-place sequence at startup.

    print("\\nEXECUTION PLAN")
    print("Startup: measured pose -> HOME_Q.")
    print("Cartesian paths will interpolate X, Y and Z together in 3D.")
    print("Only tool pitch is constrained during Cartesian movement.")
    print("TRAVEL_Z is a clearance waypoint, not a constant-height path constraint.")
    print("Pick/place contact Z =", CONTACT_Z, "m for both operations.")
    print("No path collision checking is available.")

    if DRY_RUN:
        print("\\nDRY_RUN=True: no commands will be sent to the robot.")
        print("Only the startup path was generated. Cartesian paths are generated")
        print("just before each segment when DRY_RUN is disabled.")
        return

    if REQUIRE_EXECUTE_CONFIRMATION:
        print("\\nClear the workspace, keep the emergency stop accessible,")
        print("and ensure the robot is supported during any required setup.")
        if input("Type EXECUTE to run the complete sequence: ").strip() != "EXECUTE":
            print("Cancelled. No movement sent.")
            return

    # 1. Move to HOME_Q.
    node.execute_arm(q_home_path, t_home, "startup to HOME_Q")
    current_q = node.get_current_q()
    if np.max(np.abs(wrap_delta(current_q, HOME_Q))) > JOINT_GOAL_TOLERANCE:
        raise RuntimeError("Robot did not reach HOME_Q; sequence aborted.")

    # Anchor one fixed tool pitch from the measured home pose for the whole task.
    global TASK_TOOL_ANGLE
    measured_home_pitch = float(current_q[1] + current_q[2] + current_q[3])
    home_pitch_offset = abs(wrap_delta(measured_home_pitch, HOME_ANGLE))
    if home_pitch_offset > START_ORIENTATION_CORRECTION_TOLERANCE:
        raise RuntimeError(
            f"Measured HOME_Q pitch differs from reference by "
            f"{home_pitch_offset:.5f} rad; sequence aborted."
        )
    TASK_TOOL_ANGLE = measured_home_pitch
    print(f"Task tool pitch fixed at {TASK_TOOL_ANGLE:.6f} rad "
          f"({np.rad2deg(TASK_TOOL_ANGLE):.3f} deg).")

    # 2. Move from HOME to a clearance waypoint above the pickup. All three
    # Cartesian coordinates are interpolated together; no constant-Z path rule.
    pick_above = np.array([PICK_XY[0], PICK_XY[1], TRAVEL_Z])
    t, q, _ = plan_cartesian_segment(
        HOME_XYZ, pick_above, current_q, CARTESIAN_DURATION
    )
    node.execute_arm(q, t, "3D travel from HOME to pickup approach")
    current_q = node.get_current_q()

    # 3. Open before approaching the pickup contact point.
    node.command_gripper(GRIPPER_OPEN_POSITION, "open before pickup approach")
    pick_contact = np.array([PICK_XY[0], PICK_XY[1], PICK_CONTACT_Z])
    t, q, _ = plan_cartesian_segment(
        pick_above, pick_contact, current_q, VERTICAL_DURATION
    )
    node.execute_arm(q, t, "3D approach to pickup contact")
    current_q = node.get_current_q()

    # 4. Grip, lift to a clearance waypoint, then transfer to placement.
    # Each leg still uses the unconstrained full-XYZ planner; the clearance
    # waypoint reduces the chance of dragging the object across the base.
    node.command_gripper(
        GRIPPER_HALF_CLOSED_POSITION, "half-close on picked object"
    )
    t, q, _ = plan_cartesian_segment(
        pick_contact, pick_above, current_q, VERTICAL_DURATION
    )
    node.execute_arm(q, t, "3D lift from pickup")
    current_q = node.get_current_q()

    place_above = np.array([PLACE_XY[0], PLACE_XY[1], TRAVEL_Z])
    t, q, _ = plan_cartesian_segment(
        pick_above, place_above, current_q, CARTESIAN_DURATION
    )
    node.execute_arm(q, t, "3D transfer to placement approach")
    current_q = node.get_current_q()

    # 5. Approach the matching hole. The final target fixes XYZ, while the
    # interpolated path is free to change height smoothly.
    place_contact = np.array([PLACE_XY[0], PLACE_XY[1], PLACE_CONTACT_Z])
    t, q, _ = plan_cartesian_segment(
        place_above, place_contact, current_q, VERTICAL_DURATION
    )
    node.execute_arm(q, t, "3D approach to placement contact")
    current_q = node.get_current_q()
    node.command_gripper(
        GRIPPER_OPEN_POSITION, "open to release at placement contact"
    )

    # 6. Retract to clearance, then return to home in full XYZ.
    t, q, _ = plan_cartesian_segment(
        place_contact, place_above, current_q, VERTICAL_DURATION
    )
    node.execute_arm(q, t, "retract from placement")
    current_q = node.get_current_q()

    t, q, _ = plan_cartesian_segment(
        place_above, HOME_XYZ, current_q, CARTESIAN_DURATION
    )
    node.execute_arm(q, t, "3D return from placement to HOME")
    current_q = node.get_current_q()

    t, q = plan_startup_joint_move(current_q, HOME_Q)
    node.execute_arm(q, t, "final move to exact HOME_Q")

    q_final = node.get_current_q()
    final_error = np.max(np.abs(wrap_delta(q_final, HOME_Q)))
    if final_error > JOINT_GOAL_TOLERANCE:
        raise RuntimeError(
            f"Final HOME_Q verification failed; max joint error={final_error:.4f} rad."
        )

    print("\\nPick-and-place sequence completed. Robot returned to HOME_Q.")



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