#!/usr/bin/env python3
"""Move OpenMANIPULATOR-X to HOME_XYZ position from its current pose."""

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

# ROS communication endpoints
JOINT_STATE_TOPIC = "/joint_states"
ARM_ACTION = "/arm_controller/follow_joint_trajectory"
TORQUE_SERVICE = "/dynamixel_hardware_interface/set_dxl_torque"

# Motion limits and parameters
FREQUENCY_HZ = 50.0
MIN_DURATION = 4.0
MAX_JOINT_VELOCITY = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)       # rad/s
MAX_JOINT_ACCELERATION = np.array([0.8, 0.8, 0.8, 0.8], dtype=float)  # rad/s^2


def make_duration(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def compute_trajectory(q_start: np.ndarray, q_goal: np.ndarray):
    """Generate a smooth quintic joint-space trajectory with zero start and end velocity/acceleration."""
    delta = np.asarray(q_goal, dtype=float) - np.asarray(q_start, dtype=float)
    t_v = float(np.max(1.875 * np.abs(delta) / MAX_JOINT_VELOCITY))
    t_a = float(np.max(np.sqrt((10.0 / np.sqrt(3.0)) * np.abs(delta) / MAX_JOINT_ACCELERATION)))
    duration = max(MIN_DURATION, t_v, t_a)

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
        self.last_joint_state_time = None

        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_callback, 10)
        self.torque_client = self.create_client(SetBool, TORQUE_SERVICE)
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION)

    def _joint_callback(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            return
        joint_map = dict(zip(msg.name, msg.position))
        if all(name in joint_map for name in JOINT_NAMES):
            self.current_q = np.array([joint_map[name] for name in JOINT_NAMES], dtype=float)
            self.last_joint_state_time = time.monotonic()

    def enable_torque(self):
        """Request Dynamixel torque ON if service is available."""
        self.get_logger().info("Ensuring Dynamixel torque is enabled...")
        if not self.torque_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(f"Torque service '{TORQUE_SERVICE}' not available; proceeding...")
            return

        request = SetBool.Request(data=True)
        future = self.torque_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.done() and future.result() is not None:
            res = future.result()
            if res.success:
                self.get_logger().info(f"Torque enabled successfully ({res.message or 'OK'}).")
            else:
                self.get_logger().warn(f"Torque enable returned false: {res.message}")

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

        for t, q, qd, qdd in zip(times, positions, velocities, accelerations):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in q]
            point.velocities = [float(v) for v in qd]
            point.accelerations = [float(v) for v in qdd]
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


def main():
    print("=" * 60)
    print("OpenMANIPULATOR-X: Goto Home")
    print(f"Target HOME_XYZ: {HOME_XYZ} m")
    print(f"Target Joints:   {np.round(TARGET_Q, 5)} rad")
    print("=" * 60)

    rclpy.init()
    node = GotoHomeNode()

    try:
        # 1. Enable torque if it wasn't enabled already
        node.enable_torque()

        # 2. Read live starting joint configuration
        current_q = node.get_current_joints()
        print(f"Current Joints:  {np.round(current_q, 5)} rad")
        cur_xyz, _, _ = forward_kinematics(*current_q)
        print(f"Current XYZ:     {np.round(cur_xyz, 5)} m")

        # 3. Plan smooth quintic trajectory
        duration, times, positions, velocities, accelerations = compute_trajectory(
            current_q, TARGET_Q
        )
        print(f"Planned smooth trajectory: {duration:.2f}s ({len(times)} points)")

        # 4. Execute trajectory immediately
        node.execute_trajectory(times, positions, velocities, accelerations)

        # 5. Confirm final position
        final_q = node.get_current_joints()
        final_xyz, _, _ = forward_kinematics(*final_q)
        print("\nReached Target Position:")
        print(f"Final Joints:    {np.round(final_q, 5)} rad")
        print(f"Final XYZ:       {np.round(final_xyz, 5)} m")
        print(f"Joint Error:     {np.round(np.abs(final_q - TARGET_Q), 5)} rad")
        print("=" * 60)
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
