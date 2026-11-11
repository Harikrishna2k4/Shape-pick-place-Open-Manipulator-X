
#!/usr/bin/env python3

import csv
import time
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import JointState

# Use your project's forward kinematics
from robot_kinematics.kinematics import forward_kinematics


ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4"
]

SAMPLE_INTERVAL = 2.0


class JointPoseRecorder(Node):

    def __init__(self):
        super().__init__("joint_pose_recorder")

        self.latest_positions = None
        self.last_update = None
        self.sample_number = 0

        self.subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
            10
        )

        self.timer = self.create_timer(
            SAMPLE_INTERVAL,
            self.record_pose
        )

        self.csv_file = open(
            "robot_pose_samples.csv",
            "w",
            newline=""
        )

        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            "sample",
            "time",
            "joint1_rad",
            "joint2_rad",
            "joint3_rad",
            "joint4_rad",
            "joint1_deg",
            "joint2_deg",
            "joint3_deg",
            "joint4_deg",
            "x_m",
            "y_m",
            "z_m"
        ])

        self.get_logger().info(
            "Live joint pose recorder started. "
            "Sampling every 2 seconds."
        )

    def joint_callback(self, msg):

        joint_map = dict(zip(msg.name, msg.position))

        if not all(name in joint_map for name in ARM_JOINTS):
            return

        self.latest_positions = np.array(
            [joint_map[name] for name in ARM_JOINTS],
            dtype=float
        )

        self.last_update = time.monotonic()

    def record_pose(self):

        if self.latest_positions is None:
            self.get_logger().warning(
                "Waiting for /joint_states..."
            )
            return

        if time.monotonic() - self.last_update > 0.5:
            self.get_logger().warning(
                "Joint-state feedback is stale. Skipping sample."
            )
            return

        q = self.latest_positions.copy()
        q_deg = np.degrees(q)

        # Calculate end-effector position using project FK
        try:
            fk_result = forward_kinematics(*q)

            # Support either position-only or (position, rotation, ...)
            if isinstance(fk_result, tuple):
                xyz = np.asarray(fk_result[0], dtype=float)
            else:
                xyz = np.asarray(fk_result, dtype=float)

            if xyz.shape != (3,):
                raise ValueError("FK position must contain X, Y, Z")

            x, y, z = xyz

        except Exception as error:
            self.get_logger().warning(
                f"FK calculation failed: {error}"
            )
            x = y = z = float("nan")

        self.sample_number += 1

        print("\n" + "=" * 60)
        print(f"POSE SAMPLE {self.sample_number}")
        print("=" * 60)

        print("\nJoint configuration:")

        for name, rad, deg in zip(ARM_JOINTS, q, q_deg):
            print(
                f"{name}: {rad:+.5f} rad "
                f"({deg:+.2f} degrees)"
            )

        print("\nEnd-effector position:")
        print(f"X = {x:.5f} m")
        print(f"Y = {y:.5f} m")
        print(f"Z = {z:.5f} m")

        print("=" * 60)

        self.writer.writerow([
            self.sample_number,
            time.time(),
            *q.tolist(),
            *q_deg.tolist(),
            x, y, z
        ])

        self.csv_file.flush()

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main():

    rclpy.init()

    node = JointPoseRecorder()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nRecording stopped.")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
