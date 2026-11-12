#!/usr/bin/env python3
"""Record OpenMANIPULATOR-X arm, gripper and gripper-Dynamixel telemetry.

Subscriptions:
  /joint_states          -> four arm joints + two gripper finger coordinates
  /dynamic_joint_states  -> raw hardware interfaces for dxl5 (gripper motor)

End-effector XYZ is calculated from the four arm joints using this project's FK.
The node is read-only: it does not command joints and does not change torque.
"""

import csv
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.msg import DynamicJointState

from robot_kinematics.kinematics import forward_kinematics


ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4")
GRIPPER_FINGER_JOINTS = ("gripper_left_joint", "gripper_right_joint")

JOINT_STATE_TOPIC = "/joint_states"
DYNAMIC_JOINT_STATE_TOPIC = "/dynamic_joint_states"

# In the supplied OpenMANIPULATOR-X dynamic_joint_states output, dxl5 is
# the Dynamixel hardware component associated with the gripper.
GRIPPER_MOTOR_COMPONENT = "dxl5"

SAMPLE_INTERVAL = 0.1       # 10 Hz recording
MAX_FEEDBACK_AGE = 0.5      # Skip a sample if required arm feedback is older
OUTPUT_FILE = Path("robot_pose_samples.csv")

# Interfaces reported for dxl5 in the user's /dynamic_joint_states message.
MOTOR_INTERFACES = (
    "Present Position",
    "Present Velocity",
    "Present Current",
    "Torque Enable",
    "Present Input Voltage",
)


def finite_float(value) -> Optional[float]:
    """Convert a ROS interface value to a finite float, or None."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def format_value(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "not published"
    return f"{value:+.5f}{suffix}"


class JointPoseRecorder(Node):
    def __init__(self):
        super().__init__("joint_pose_recorder")

        self.latest_positions: Dict[str, float] = {}
        self.last_joint_update: Optional[float] = None

        # Shape: {"dxl5": {"Present Position": value, ...}}
        self.latest_dynamic: Dict[str, Dict[str, float]] = {}
        self.last_dynamic_update: Optional[float] = None

        self.sample_number = 0
        self.warned_no_joint_state = False
        self.warned_stale_joint_state = False
        self.warned_missing_gripper_finger = False
        self.warned_missing_dynamic_topic_data = False
        self.warned_stale_dynamic_data = False
        self.warned_missing_motor_interfaces = set()

        self.csv_file = OUTPUT_FILE.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            "sample",
            "unix_time_s",
            "joint1_rad", "joint2_rad", "joint3_rad", "joint4_rad",
            "joint1_deg", "joint2_deg", "joint3_deg", "joint4_deg",
            "gripper_left_position_m", "gripper_left_position_mm",
            "gripper_right_position_m", "gripper_right_position_mm",
            "x_m", "y_m", "z_m",
            "dxl5_present_position_hw_units",
            "dxl5_present_velocity_hw_units",
            "dxl5_present_current_hw_units",
            "dxl5_torque_enable",
            "dxl5_present_input_voltage_V",
            "joint_state_age_s",
            "dynamic_joint_state_age_s",
        ])
        self.csv_file.flush()

        self.joint_subscription = self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self.joint_callback,
            50,
        )
        self.dynamic_subscription = self.create_subscription(
            DynamicJointState,
            DYNAMIC_JOINT_STATE_TOPIC,
            self.dynamic_joint_callback,
            50,
        )
        self.timer = self.create_timer(SAMPLE_INTERVAL, self.record_pose)

        self.get_logger().info(
            f"Recording arm joints {ARM_JOINTS} and gripper fingers "
            f"{GRIPPER_FINGER_JOINTS} from {JOINT_STATE_TOPIC}."
        )
        self.get_logger().info(
            f"Recording gripper motor interfaces from "
            f"{DYNAMIC_JOINT_STATE_TOPIC}, component={GRIPPER_MOTOR_COMPONENT}."
        )
        self.get_logger().info(
            f"Sampling at {1.0 / SAMPLE_INTERVAL:.1f} Hz; CSV: {OUTPUT_FILE.resolve()}"
        )
        self.get_logger().info(
            "Read-only recorder: it does not command joints or change Dynamixel torque."
        )

    def joint_callback(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            self.get_logger().warning(
                "Ignoring malformed JointState: name/position lengths differ."
            )
            return

        positions = {}
        for name, value in zip(msg.name, msg.position):
            parsed = finite_float(value)
            if parsed is not None:
                positions[str(name)] = parsed

        # Require all four arm joints in the same message, but retain gripper
        # finger positions from that same message when available.
        if all(name in positions for name in ARM_JOINTS):
            self.latest_positions = positions
            self.last_joint_update = time.monotonic()
            self.warned_no_joint_state = False

    def dynamic_joint_callback(self, msg: DynamicJointState):
        if len(msg.joint_names) != len(msg.interface_values):
            self.get_logger().warning(
                "Ignoring malformed DynamicJointState: joint_names and "
                "interface_values lengths differ."
            )
            return

        components: Dict[str, Dict[str, float]] = {}
        for joint_name, interface_value in zip(msg.joint_names, msg.interface_values):
            names = list(interface_value.interface_names)
            values = list(interface_value.values)
            if len(names) != len(values):
                continue

            interface_map = {}
            for interface_name, value in zip(names, values):
                parsed = finite_float(value)
                if parsed is not None:
                    interface_map[str(interface_name)] = parsed
            components[str(joint_name)] = interface_map

        self.latest_dynamic = components
        self.last_dynamic_update = time.monotonic()
        self.warned_missing_dynamic_topic_data = False

    def get_fresh_motor_data(self):
        """Return dxl5 interface data and age; missing/stale data remains None."""
        if self.last_dynamic_update is None:
            if not self.warned_missing_dynamic_topic_data:
                self.get_logger().warning(
                    f"No messages received from {DYNAMIC_JOINT_STATE_TOPIC}; "
                    "gripper motor telemetry will be blank until it publishes."
                )
                self.warned_missing_dynamic_topic_data = True
            return {}, None

        age = time.monotonic() - self.last_dynamic_update
        if age > MAX_FEEDBACK_AGE:
            if not self.warned_stale_dynamic_data:
                self.get_logger().warning(
                    f"{DYNAMIC_JOINT_STATE_TOPIC} is stale ({age:.2f} s); "
                    "gripper motor telemetry will be blank for this sample."
                )
                self.warned_stale_dynamic_data = True
            return {}, age

        self.warned_stale_dynamic_data = False
        motor_data = self.latest_dynamic.get(GRIPPER_MOTOR_COMPONENT, {})
        if not motor_data and not self.warned_missing_dynamic_topic_data:
            self.get_logger().warning(
                f"Component '{GRIPPER_MOTOR_COMPONENT}' was not present in "
                f"{DYNAMIC_JOINT_STATE_TOPIC}; check the component name."
            )
            self.warned_missing_dynamic_topic_data = True
        return motor_data, age

    def record_pose(self):
        if self.last_joint_update is None:
            if not self.warned_no_joint_state:
                self.get_logger().warning(
                    f"Waiting for complete arm feedback on {JOINT_STATE_TOPIC}."
                )
                self.warned_no_joint_state = True
            return

        joint_age = time.monotonic() - self.last_joint_update
        if joint_age > MAX_FEEDBACK_AGE:
            if not self.warned_stale_joint_state:
                self.get_logger().warning(
                    f"{JOINT_STATE_TOPIC} is stale ({joint_age:.2f} s); "
                    "skipping pose samples until messages resume."
                )
                self.warned_stale_joint_state = True
            return
        self.warned_stale_joint_state = False

        missing_fingers = [
            name for name in GRIPPER_FINGER_JOINTS if name not in self.latest_positions
        ]
        if missing_fingers and not self.warned_missing_gripper_finger:
            self.get_logger().warning(
                "Gripper finger joint(s) absent from /joint_states: "
                + ", ".join(missing_fingers)
                + ". Their CSV values will be blank."
            )
            self.warned_missing_gripper_finger = True
        elif not missing_fingers:
            self.warned_missing_gripper_finger = False

        q = np.array([self.latest_positions[name] for name in ARM_JOINTS], dtype=float)
        q_deg = np.degrees(q)

        gripper_left = self.latest_positions.get("gripper_left_joint")
        gripper_right = self.latest_positions.get("gripper_right_joint")

        try:
            fk_result = forward_kinematics(*q)
            xyz = np.asarray(
                fk_result[0] if isinstance(fk_result, tuple) else fk_result,
                dtype=float,
            )
            if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
                raise ValueError(f"Expected finite XYZ with shape (3,), got {xyz!r}")
            x, y, z = (float(value) for value in xyz)
        except Exception as exc:
            self.get_logger().error(f"FK calculation failed; sample skipped: {exc}")
            return

        motor_data, dynamic_age = self.get_fresh_motor_data()
        motor_position = motor_data.get("Present Position")
        motor_velocity = motor_data.get("Present Velocity")
        motor_current = motor_data.get("Present Current")
        motor_torque = motor_data.get("Torque Enable")
        motor_voltage = motor_data.get("Present Input Voltage")

        missing_interfaces = [
            name for name in MOTOR_INTERFACES if name not in motor_data
        ]
        for name in missing_interfaces:
            if name not in self.warned_missing_motor_interfaces:
                self.get_logger().warning(
                    f"Gripper motor component '{GRIPPER_MOTOR_COMPONENT}' "
                    f"does not publish interface '{name}'."
                )
                self.warned_missing_motor_interfaces.add(name)
        for name in MOTOR_INTERFACES:
            if name in motor_data:
                self.warned_missing_motor_interfaces.discard(name)

        self.sample_number += 1
        unix_time = time.time()

        print("\n" + "=" * 50, flush=True)
        print(f"SAMPLE {self.sample_number}", flush=True)
        print("=" * 50, flush=True)
        print(f"  TCP position (m):   X={x:+.4f} m, Y={y:+.4f} m, Z={z:+.4f} m", flush=True)
        print(f"  TCP position (mm):  X={x * 1000:+.1f}, Y={y * 1000:+.1f}, Z={z * 1000:+.1f} mm", flush=True)
        if gripper_left is None:
            print("  gripper_left_joint: not published", flush=True)
        else:
            print(
                f"  gripper position:   {gripper_left:+.5f} m ({gripper_left * 1000:+.2f} mm)",
                flush=True,
            )

        if motor_position is not None:
            print(f"  motor (dxl5):       {motor_position:+.5f} (raw interface units)", flush=True)
        print("=" * 50, flush=True)

        self.writer.writerow([
            self.sample_number,
            f"{unix_time:.6f}",
            *[f"{value:.8f}" for value in q],
            *[f"{value:.5f}" for value in q_deg],
            "" if gripper_left is None else f"{gripper_left:.8f}",
            "" if gripper_left is None else f"{gripper_left * 1000:.5f}",
            "" if gripper_right is None else f"{gripper_right:.8f}",
            "" if gripper_right is None else f"{gripper_right * 1000:.5f}",
            f"{x:.8f}", f"{y:.8f}", f"{z:.8f}",
            "" if motor_position is None else f"{motor_position:.8f}",
            "" if motor_velocity is None else f"{motor_velocity:.8f}",
            "" if motor_current is None else f"{motor_current:.8f}",
            "" if motor_torque is None else f"{motor_torque:.0f}",
            "" if motor_voltage is None else f"{motor_voltage:.5f}",
            f"{joint_age:.4f}",
            "" if dynamic_age is None else f"{dynamic_age:.4f}",
        ])
        self.csv_file.flush()

    def destroy_node(self):
        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = JointPoseRecorder()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nRecording stopped.", flush=True)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()