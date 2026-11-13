#!/usr/bin/env python3
"""Move OpenMANIPULATOR-X to HOME_XYZ position from its current pose."""

import os
import sys
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory, GripperCommand
from builtin_interfaces.msg import Duration

try:
    from control_msgs.msg import DynamicJointState
except ImportError:
    DynamicJointState = None

try:
    from dynamixel_interfaces.srv import SetDataToDxl
except ImportError:
    SetDataToDxl = None

try:
    from robot_kinematics.kinematics import forward_kinematics
except ImportError:
    from kinematics import forward_kinematics


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]

# End-effector target home position (metres) and corresponding joint angles (rad)
HOME_XYZ = np.array([0.02445, -0.00167, 0.16441], dtype=float)
TARGET_Q = np.array([-0.13346, -1.29315, 0.04142, 2.02485], dtype=float)

# Gripper configuration (dxl5 / ID 15):
# Primary home angle: 145.5 degrees
# Physical calibration from DYNAMIXEL Wizard 2.0:
#   Open / Home angle: 145.5 deg -> maps to -0.010 m prismatic
#   Closed max angle:  210.0 deg -> maps to +0.019 m prismatic
GRIPPER_HOME_ANGLE_DEG = float(os.environ.get("GRIPPER_HOME_ANGLE_DEG", "145.5"))
GRIPPER_CLOSED_MAX_DEG = float(os.environ.get("GRIPPER_MAX_CLOSE_DEG", "210.0"))
GRIPPER_MAX_EFFORT = 20.0


def angle_deg_to_prismatic(deg: float) -> float:
    """Map physical motor degrees [145.5, 210.0] linearly to ROS 2 prismatic metres [-0.010, 0.019]."""
    fraction = (deg - GRIPPER_HOME_ANGLE_DEG) / (GRIPPER_CLOSED_MAX_DEG - GRIPPER_HOME_ANGLE_DEG)
    return float(-0.010 + fraction * (0.019 - (-0.010)))


def prismatic_to_angle_deg(pris: float) -> float:
    """Map ROS 2 prismatic metres [-0.010, 0.019] linearly to physical motor degrees [145.5, 210.0]."""
    fraction = (pris - (-0.010)) / (0.019 - (-0.010))
    return float(GRIPPER_HOME_ANGLE_DEG + fraction * (GRIPPER_CLOSED_MAX_DEG - GRIPPER_HOME_ANGLE_DEG))


def parse_gripper_target(val) -> tuple:
    """Parse gripper target returning (deg, prismatic_m).

    Accepts degrees (e.g. 145.5) or prismatic metres (e.g. -0.010).
    """
    v = float(val)
    if abs(v) > 1.0:  # Value is in degrees (e.g. 145.5 deg)
        deg = v
        pris = angle_deg_to_prismatic(deg)
    else:  # Value is in prismatic metres (e.g. -0.010 m)
        pris = v
        deg = prismatic_to_angle_deg(pris)
    return deg, pris


def dxl5_to_prismatic(dxl5_rad: float) -> float:
    """Pass-through/helper for compatibility."""
    return float(dxl5_rad)


# Target gripper home position:
_env_grip = os.environ.get("GRIPPER_HOME_ANGLE_DEG") or os.environ.get("GRIPPER_OPEN_POS") or "145.5"
TARGET_GRIPPER_DEG, TARGET_GRIPPER_POSITION = parse_gripper_target(_env_grip)

# ROS communication endpoints
JOINT_STATE_TOPIC = "/joint_states"
DYNAMIC_JOINT_STATE_TOPIC = "/dynamic_joint_states"
ARM_ACTION = "/arm_controller/follow_joint_trajectory"
GRIPPER_ACTION = "/gripper_controller/gripper_cmd"
TORQUE_SERVICE = "/dynamixel_hardware_interface/set_dxl_torque"
DXL_DATA_SERVICE = "/dynamixel_hardware_interface/set_dxl_data"

# Motion limits and parameters (configured for gentle, smooth, and safe motion)
FREQUENCY_HZ = 50.0
DEFAULT_MIN_DURATION = float(os.environ.get("GOTO_DURATION", "8.0"))
DEFAULT_MAX_VELOCITY = float(os.environ.get("GOTO_MAX_VEL", "0.20"))       # rad/s (~11.5 deg/s)
DEFAULT_MAX_ACCEL = float(os.environ.get("GOTO_MAX_ACCEL", "0.30"))       # rad/s^2 (~17.2 deg/s^2)

MAX_JOINT_VELOCITY = np.array([DEFAULT_MAX_VELOCITY] * 4, dtype=float)
MAX_JOINT_ACCELERATION = np.array([DEFAULT_MAX_ACCEL] * 4, dtype=float)


def make_duration(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def compute_trajectory(q_start: np.ndarray, q_goal: np.ndarray, min_duration: float = None):
    """Generate a smooth quintic joint-space trajectory with zero start and end velocity/acceleration."""
    min_d = DEFAULT_MIN_DURATION if min_duration is None else float(min_duration)
    delta = np.asarray(q_goal, dtype=float) - np.asarray(q_start, dtype=float)
    t_v = float(np.max(1.875 * np.abs(delta) / MAX_JOINT_VELOCITY))
    t_a = float(np.max(np.sqrt((10.0 / np.sqrt(3.0)) * np.abs(delta) / MAX_JOINT_ACCELERATION)))
    duration = max(min_d, t_v, t_a)

    sample_count = max(3, int(np.ceil(duration * FREQUENCY_HZ)) + 1)
    times = np.linspace(0.0, duration, sample_count)
    u = times / duration

    # Quintic blend: s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0
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

    return duration, times, positions, velocities, accelerations


class GotoHomeNode(Node):
    def __init__(self):
        super().__init__("goto_home")
        self.current_q = None
        self.current_gripper = None
        self.current_dxl5 = None
        self.last_joint_state_time = None

        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_callback, 10)
        if DynamicJointState is not None:
            self.create_subscription(
                DynamicJointState,
                DYNAMIC_JOINT_STATE_TOPIC,
                self._dynamic_joint_callback,
                10,
            )
        self.torque_client = self.create_client(SetBool, TORQUE_SERVICE)
        self.dxl_data_client = (
            self.create_client(SetDataToDxl, DXL_DATA_SERVICE)
            if SetDataToDxl is not None
            else None
        )
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION)
        self.gripper_client = ActionClient(self, GripperCommand, GRIPPER_ACTION)

    def _joint_callback(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            return
        joint_map = dict(zip(msg.name, msg.position))
        if all(name in joint_map for name in JOINT_NAMES):
            self.current_q = np.array([joint_map[name] for name in JOINT_NAMES], dtype=float)
            self.last_joint_state_time = time.monotonic()
        for g_name in ("gripper_left_joint", "gripper", "gripper_right_joint"):
            if g_name in joint_map:
                self.current_gripper = float(joint_map[g_name])
                break

    def _dynamic_joint_callback(self, msg):
        if len(msg.joint_names) != len(msg.interface_values):
            return
        for joint_name, interface_value in zip(msg.joint_names, msg.interface_values):
            if str(joint_name) == "dxl5":
                for name, value in zip(interface_value.interface_names, interface_value.values):
                    if name == "Present Position":
                        try:
                            val = float(value)
                            if np.isfinite(val):
                                self.current_dxl5 = val
                        except (TypeError, ValueError):
                            pass

    def enable_torque(self):
        """Request Dynamixel torque ON if service is available."""
        self.get_logger().info("Ensuring Dynamixel torque is enabled...")
        if not self.torque_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(f"Torque service '{TORQUE_SERVICE}' not available; proceeding...")
        else:
            # Force torque cycle (OFF then ON) to un-stale driver state and guarantee hardware latching
            req_off = SetBool.Request(data=False)
            f_off = self.torque_client.call_async(req_off)
            rclpy.spin_until_future_complete(self, f_off, timeout_sec=2.0)

            request = SetBool.Request(data=True)
            future = self.torque_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.done() and future.result() is not None:
                res = future.result()
                if res.success:
                    self.get_logger().info(f"Torque enabled successfully ({res.message or 'OK'}).")
                else:
                    self.get_logger().warn(f"Torque enable returned false: {res.message}")

        # Optional override for gripper motor (ID 15) if explicitly disabled via env
        enable_grip = os.environ.get("ENABLE_GRIPPER_TORQUE", "1") != "0"
        if not enable_grip and self.dxl_data_client is not None and self.dxl_data_client.wait_for_service(timeout_sec=2.0):
            req = SetDataToDxl.Request()
            req.id = 15
            req.item_name = "Torque Enable"
            req.item_data = 0
            future = self.dxl_data_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            self.get_logger().info("Gripper Dynamixel (ID 15) torque explicitly disabled.")
        elif enable_grip and self.dxl_data_client is not None and self.dxl_data_client.wait_for_service(timeout_sec=2.0):
            # Ensure gripper motor ID 15 uses a gentle profile velocity to prevent snapping
            try:
                req_vel = SetDataToDxl.Request()
                req_vel.id = 15
                req_vel.item_name = "Profile Velocity"
                req_vel.item_data = int(os.environ.get("GRIPPER_PROFILE_VEL", "80"))
                f_vel = self.dxl_data_client.call_async(req_vel)
                rclpy.spin_until_future_complete(self, f_vel, timeout_sec=1.0)
            except Exception:
                pass

    def get_current_joints(self, timeout_sec: float = 10.0) -> np.ndarray:
        """Wait for and return fresh measured joint angles from /joint_states."""
        start = time.monotonic()
        while rclpy.ok() and (time.monotonic() - start < timeout_sec):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_q is not None and (time.monotonic() - self.last_joint_state_time < 1.0):
                return self.current_q.copy()
        raise RuntimeError(f"Timed out waiting for fresh joint states on '{JOINT_STATE_TOPIC}'.")

    def execute_trajectory(self, times, positions, velocities, accelerations):
        """Send trajectory to the FollowJointTrajectory action server and wait for completion."""
        self.get_logger().info(f"Waiting for arm trajectory action server: {ARM_ACTION}")
        if not self.arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Arm action server unavailable: {ARM_ACTION}")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        for t, q, qd in zip(times, positions, velocities):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in q]
            point.velocities = [float(v) for v in qd]
            # Match openmanipulator_fixed_orientation_pick_and_place: omit accelerations
            point.time_from_start = make_duration(float(t))
            goal.trajectory.points.append(point)

        self.get_logger().info(f"Sending trajectory ({len(times)} points, duration: {times[-1]:.2f}s)...")
        send_future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Arm controller rejected the trajectory goal.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=times[-1] + 15.0)
        if not result_future.done():
            goal_handle.cancel_goal_async()
            raise RuntimeError("Timed out waiting for trajectory execution.")

        wrapped_result = result_future.result()
        if wrapped_result is None or wrapped_result.result.error_code != 0:
            err = "No result" if wrapped_result is None else wrapped_result.result.error_string
            raise RuntimeError(f"Trajectory execution failed: {err}")

        self.get_logger().info("Trajectory execution completed successfully.")

    def command_gripper(self, position: float, label: str = "gripper", timeout_sec: float = 10.0):
        """Send target position to the GripperCommand action server and wait for completion."""
        if os.environ.get("ENABLE_GRIPPER_TORQUE", "1") == "0":
            self.get_logger().info(f"Gripper torque disabled: skipping command for {label}.")
            return

        self.get_logger().info(f"Waiting for gripper action server: {GRIPPER_ACTION}")
        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Gripper action server unavailable: {GRIPPER_ACTION}")

        deg_val, pris_val = parse_gripper_target(position)

        goal = GripperCommand.Goal()
        goal.command.position = float(pris_val)
        goal.command.max_effort = float(GRIPPER_MAX_EFFORT)

        self.get_logger().info(
            f"Sending gripper goal: {deg_val:.1f}° ({pris_val:+.5f} m) ({label})..."
        )
        send_future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"Gripper rejected goal: {label}")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)
        if not result_future.done():
            goal_handle.cancel_goal_async()
            raise RuntimeError(f"Timed out waiting for gripper execution ({label}).")

        wrapped_result = result_future.result()
        if wrapped_result is None:
            raise RuntimeError(f"No result returned from gripper controller ({label}).")

        res = wrapped_result.result
        if not (res.reached_goal or res.stalled):
            raise RuntimeError(
                f"Gripper did not reach requested position for {label}; "
                f"stalled={res.stalled}, position={res.position:.5f}"
            )

        res_deg, _ = parse_gripper_target(res.position)
        self.get_logger().info(
            f"Gripper execution completed ({label}): {res_deg:.1f}° (position={res.position:.5f} m)"
        )
        return res.position


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OpenMANIPULATOR-X: Move gently to Home pose.")
    parser.add_argument("--duration", type=float, default=None, help="Target trajectory duration in seconds (default: 8.0-12.0s)")
    parser.add_argument("--speed", choices=["slow", "gentle", "normal", "fast"], default=None, help="Speed preset")
    args, _ = parser.parse_known_args()

    global MAX_JOINT_VELOCITY, MAX_JOINT_ACCELERATION
    min_d = args.duration
    if args.speed == "slow":
        MAX_JOINT_VELOCITY = np.array([0.12] * 4, dtype=float)
        MAX_JOINT_ACCELERATION = np.array([0.20] * 4, dtype=float)
        if min_d is None:
            min_d = 12.0
    elif args.speed == "gentle" or args.speed is None:
        MAX_JOINT_VELOCITY = np.array([0.20] * 4, dtype=float)
        MAX_JOINT_ACCELERATION = np.array([0.30] * 4, dtype=float)
        if min_d is None:
            min_d = DEFAULT_MIN_DURATION
    elif args.speed == "normal":
        MAX_JOINT_VELOCITY = np.array([0.35] * 4, dtype=float)
        MAX_JOINT_ACCELERATION = np.array([0.50] * 4, dtype=float)
        if min_d is None:
            min_d = 5.0
    elif args.speed == "fast":
        MAX_JOINT_VELOCITY = np.array([0.50] * 4, dtype=float)
        MAX_JOINT_ACCELERATION = np.array([0.80] * 4, dtype=float)
        if min_d is None:
            min_d = 4.0

    print("=" * 60)
    print("OpenMANIPULATOR-X: Goto Home (Gentle Motion)")
    print(f"Target HOME_XYZ:        {HOME_XYZ} m")
    print(f"Target Joints:          {np.round(TARGET_Q, 5)} rad")
    print(f"Target Gripper:         {TARGET_GRIPPER_DEG:.1f}° ({TARGET_GRIPPER_POSITION:+.5f} m, home)")
    print(f"Speed Limit:            {np.rad2deg(MAX_JOINT_VELOCITY[0]):.1f}°/s ({MAX_JOINT_VELOCITY[0]:.2f} rad/s)")
    print("=" * 60)

    rclpy.init()
    node = GotoHomeNode()

    try:
        # 1. Enable torque if it wasn't enabled already
        node.enable_torque()

        # 2. Read live starting joint configuration
        current_q = node.get_current_joints()
        print(f"Current Joints:         {np.round(current_q, 5)} rad")
        if node.current_dxl5 is not None:
            cur_deg, _ = parse_gripper_target(node.current_dxl5)
            print(f"Current dxl5 motor:     {cur_deg:.1f}° ({node.current_dxl5:+.5f} raw)")
        elif node.current_gripper is not None:
            cur_deg, _ = parse_gripper_target(node.current_gripper)
            print(f"Current Gripper:        {cur_deg:.1f}° ({node.current_gripper:+.5f} m)")
        cur_xyz, _, _ = forward_kinematics(*current_q)
        print(f"Current XYZ:            {np.round(cur_xyz, 5)} m")

        # 3. Plan smooth quintic trajectory
        duration, times, positions, velocities, accelerations = compute_trajectory(
            current_q, TARGET_Q, min_duration=min_d
        )
        print(f"Planned smooth trajectory: {duration:.2f}s ({len(times)} points, max velocity: {np.rad2deg(MAX_JOINT_VELOCITY[0]):.1f}°/s)")

        # 4. Execute trajectory immediately
        node.execute_trajectory(times, positions, velocities, accelerations)

        # 5. Command gripper joint to primary home angle (145.5 deg -> -0.010 m)
        print(f"\nMoving gripper to home position: {TARGET_GRIPPER_DEG:.1f}° ({TARGET_GRIPPER_POSITION:+.5f} m)...")
        node.command_gripper(TARGET_GRIPPER_POSITION, label=f"home gripper {TARGET_GRIPPER_DEG:.1f}°")

        # 6. Confirm final position
        final_q = node.get_current_joints()
        final_xyz, _, _ = forward_kinematics(*final_q)
        max_error = float(np.max(np.abs(final_q - TARGET_Q)))
        print("\nReached Target Position:")
        print(f"Final Joints:           {np.round(final_q, 5)} rad")
        if node.current_dxl5 is not None:
            cur_deg, _ = parse_gripper_target(node.current_dxl5)
            print(f"Final dxl5 motor:       {cur_deg:.1f}° ({node.current_dxl5:+.5f} raw)")
        elif node.current_gripper is not None:
            cur_deg, _ = parse_gripper_target(node.current_gripper)
            print(f"Final Gripper:          {cur_deg:.1f}° ({node.current_gripper:+.5f} m)")
        print(f"Final XYZ:              {np.round(final_xyz, 5)} m")
        print(f"Joint Error:            {np.round(np.abs(final_q - TARGET_Q), 5)} rad")
        print("=" * 60)

        if max_error > 0.05:
            print(f"\n[WARNING] Robot did not reach target pose! Max joint error: {max_error:.4f} rad.")
            print("[CHECK] If the physical arm did not move at all, verify:")
            print("  1. Physical 12V DC power cable to OpenMANIPULATOR-X is plugged in and switch is ON.")
            print("  2. Dynamixel U2D2 / USB interface cable connection.")
            return 1

        print("Motion completed successfully.")
        return 0

    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
