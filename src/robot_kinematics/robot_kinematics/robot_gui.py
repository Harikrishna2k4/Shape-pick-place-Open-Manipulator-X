#!/usr/bin/env python3

import math
import time
import threading
import queue
import tkinter as tk

from tkinter import ttk, messagebox

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from control_msgs.action import (
    FollowJointTrajectory,
    GripperCommand
)


# ============================================================
# ROBOT CONFIGURATION
# ============================================================

ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4"
]

JOINT_LIMITS = {
    "joint1": (-math.pi, math.pi),
    "joint2": (-1.5, 1.5),
    "joint3": (-1.5, 1.4),
    "joint4": (-1.7, 1.97)
}

ARM_ACTION = "/arm_controller/follow_joint_trajectory"
GRIPPER_ACTION = "/gripper_controller/gripper_cmd"

TRAJECTORY_DURATION = 2.0

GRIPPER_OPEN = 0.01
GRIPPER_CLOSE = -0.01

GRIPPER_MAX_EFFORT = 10.0


# ============================================================
# ROS NODE
# ============================================================

class RobotNode(Node):

    def __init__(self):

        super().__init__("robot_gui_node")

        self.lock = threading.RLock()

        self.joint_positions = {}
        self.last_joint_update = 0.0

        self.arm_busy = False
        self.gripper_busy = False

        self.active_arm_goal = None

        self.events = queue.Queue()

        # Joint feedback

        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
            10
        )

        # Arm trajectory action

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            ARM_ACTION
        )

        # Gripper action

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            GRIPPER_ACTION
        )

    # --------------------------------------------------------
    # JOINT FEEDBACK
    # --------------------------------------------------------

    def joint_callback(self, msg):

        with self.lock:

            for name, position in zip(
                msg.name,
                msg.position
            ):

                self.joint_positions[name] = position

            self.last_joint_update = time.monotonic()

    def get_positions(self):

        with self.lock:

            return dict(self.joint_positions)

    def feedback_is_fresh(self):

        with self.lock:

            return (
                time.monotonic() - self.last_joint_update
                < 0.5
            )

    def get_status(self):

        with self.lock:

            return self.arm_busy, self.gripper_busy

    # --------------------------------------------------------
    # ARM MOVEMENT
    # --------------------------------------------------------

    def move_joint(self, joint_name, direction, step_deg):

        with self.lock:

            if self.arm_busy:

                self.events.put(
                    ("error", "Arm is executing a trajectory.")
                )

                return

            if not self.feedback_is_fresh():

                self.events.put(
                    ("error", "Joint feedback unavailable.")
                )

                return

            if joint_name not in self.joint_positions:

                self.events.put(
                    ("error", f"Missing feedback: {joint_name}")
                )

                return

            positions = dict(self.joint_positions)

            for joint in ARM_JOINTS:

                if joint not in positions:

                    self.events.put(
                        ("error", f"Missing feedback: {joint}")
                    )

                    return

            current = positions[joint_name]

            step = math.radians(step_deg) * direction

            target = current + step

            lower, upper = JOINT_LIMITS[joint_name]

            if not lower <= target <= upper:

                self.events.put(
                    ("error", f"{joint_name}: joint limit reached.")
                )

                return

            target_positions = [
                target if joint == joint_name
                else positions[joint]
                for joint in ARM_JOINTS
            ]

            self.send_trajectory(target_positions)

    # --------------------------------------------------------
    # TRAJECTORY ACTION
    # --------------------------------------------------------

    def send_trajectory(self, positions):

        if not self.arm_client.server_is_ready():

            self.events.put(
                ("error", "Arm action server unavailable.")
            )

            return

        goal = FollowJointTrajectory.Goal()

        goal.trajectory.joint_names = ARM_JOINTS

        point = JointTrajectoryPoint()

        point.positions = positions

        point.time_from_start.sec = int(
            TRAJECTORY_DURATION
        )

        point.time_from_start.nanosec = int(
            (TRAJECTORY_DURATION % 1) * 1e9
        )

        goal.trajectory.points = [point]

        with self.lock:

            self.arm_busy = True

        self.events.put(
            ("command", dict(zip(ARM_JOINTS, positions)))
        )

        try:

            future = self.arm_client.send_goal_async(goal)

            future.add_done_callback(
                self.arm_goal_response
            )

        except Exception as exc:

            with self.lock:
                self.arm_busy = False

            self.events.put(
                ("error", str(exc))
            )

    def arm_goal_response(self, future):

        try:

            goal_handle = future.result()

            if not goal_handle.accepted:

                with self.lock:
                    self.arm_busy = False

                self.events.put(
                    ("error", "Trajectory rejected.")
                )

                return

            with self.lock:
                self.active_arm_goal = goal_handle

            result_future = goal_handle.get_result_async()

            result_future.add_done_callback(
                self.arm_result_callback
            )

        except Exception as exc:

            with self.lock:
                self.arm_busy = False

            self.events.put(
                ("error", str(exc))
            )

    def arm_result_callback(self, future):

        try:

            wrapped_result = future.result()

            result = wrapped_result.result

            code = result.error_code

            error_text = result.error_string

            with self.lock:

                self.arm_busy = False
                self.active_arm_goal = None

            if code == 0:

                self.events.put(
                    ("success", "Arm trajectory completed.")
                )

            else:

                self.events.put(
                    (
                        "error",
                        f"Trajectory error {code}: {error_text}"
                    )
                )

        except Exception as exc:

            with self.lock:
                self.arm_busy = False
                self.active_arm_goal = None

            self.events.put(
                ("error", str(exc))
            )

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def stop_arm(self):

        with self.lock:

            goal = self.active_arm_goal

        if goal is None:

            self.events.put(
                ("info", "No active trajectory.")
            )

            return

        try:

            future = goal.cancel_goal_async()

            future.add_done_callback(
                lambda f: self.events.put(
                    ("info", "Cancellation requested.")
                )
            )

        except Exception as exc:

            self.events.put(
                ("error", str(exc))
            )

    # --------------------------------------------------------
    # GRIPPER
    # --------------------------------------------------------

    def command_gripper(self, position):

        with self.lock:

            if self.gripper_busy:

                self.events.put(
                    ("error", "Gripper command already active.")
                )

                return

            if not self.gripper_client.server_is_ready():

                self.events.put(
                    ("error", "Gripper action server unavailable.")
                )

                return

            self.gripper_busy = True

        goal = GripperCommand.Goal()

        goal.command.position = float(position)

        goal.command.max_effort = GRIPPER_MAX_EFFORT

        try:

            future = self.gripper_client.send_goal_async(goal)

            future.add_done_callback(
                self.gripper_goal_response
            )

        except Exception as exc:

            with self.lock:
                self.gripper_busy = False

            self.events.put(
                ("error", str(exc))
            )

    def gripper_goal_response(self, future):

        try:

            handle = future.result()

            if not handle.accepted:

                with self.lock:
                    self.gripper_busy = False

                self.events.put(
                    ("error", "Gripper command rejected.")
                )

                return

            result_future = handle.get_result_async()

            result_future.add_done_callback(
                self.gripper_result_callback
            )

        except Exception as exc:

            with self.lock:
                self.gripper_busy = False

            self.events.put(
                ("error", str(exc))
            )

    def gripper_result_callback(self, future):

        try:

            result = future.result().result

            with self.lock:
                self.gripper_busy = False

            self.events.put(
                (
                    "success",
                    f"Gripper reached position: {result.position:.4f} m"
                )
            )

        except Exception as exc:

            with self.lock:
                self.gripper_busy = False

            self.events.put(
                ("error", str(exc))
            )


# ============================================================
# TKINTER GUI
# ============================================================

class RobotGUI:

    def __init__(self, root, node):

        self.root = root
        self.node = node

        self.root.title(
            "OpenMANIPULATOR-X Control Panel"
        )

        self.root.geometry("850x720")

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        self.step_deg = tk.IntVar(value=1)

        self.status = tk.StringVar(
            value="Connecting to ROS..."
        )

        self.commanded_positions = {}

        self.measured_labels = {}
        self.command_labels = {}

        self.buttons = []

        self.create_gui()

        self.update_gui()

    # --------------------------------------------------------
    # GUI CREATION
    # --------------------------------------------------------

    def create_gui(self):

        main = ttk.Frame(
            self.root,
            padding=20
        )

        main.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="OPENMANIPULATOR-X",
            font=("Arial", 23, "bold")
        ).pack(pady=5)

        ttk.Label(
            main,
            text="Joint Position Control",
            font=("Arial", 12)
        ).pack(pady=5)

        ttk.Separator(
            main
        ).pack(fill="x", pady=15)

        # STATUS

        self.status_label = ttk.Label(
            main,
            textvariable=self.status,
            font=("Arial", 11, "bold")
        )

        self.status_label.pack(pady=5)

        # MOVEMENT INCREMENT

        step_frame = ttk.LabelFrame(
            main,
            text="Movement Increment",
            padding=12
        )

        step_frame.pack(fill="x", pady=10)

        for value in [1, 2, 5]:

            ttk.Radiobutton(
                step_frame,
                text=f"{value}°",
                variable=self.step_deg,
                value=value
            ).pack(side="left", padx=20)

        # JOINT CONTROL

        joint_frame = ttk.LabelFrame(
            main,
            text="Arm Joint Control",
            padding=12
        )

        joint_frame.pack(fill="x", pady=10)

        headers = ttk.Frame(joint_frame)

        headers.pack(fill="x", pady=5)

        ttk.Label(
            headers,
            text="Joint",
            width=15
        ).pack(side="left")

        ttk.Label(
            headers,
            text="Control",
            width=17
        ).pack(side="left")

        ttk.Label(
            headers,
            text="Measured",
            width=15
        ).pack(side="left")

        ttk.Label(
            headers,
            text="Commanded",
            width=15
        ).pack(side="left")

        for joint in ARM_JOINTS:

            row = ttk.Frame(joint_frame)

            row.pack(fill="x", pady=9)

            ttk.Label(
                row,
                text=joint,
                width=15
            ).pack(side="left")

            minus = ttk.Button(
                row,
                text="−",
                width=5,
                command=lambda j=joint:
                    self.move_joint(j, -1)
            )

            minus.pack(side="left", padx=3)

            plus = ttk.Button(
                row,
                text="+",
                width=5,
                command=lambda j=joint:
                    self.move_joint(j, 1)
            )

            plus.pack(side="left", padx=3)

            self.buttons.extend([minus, plus])

            measured = ttk.Label(
                row,
                text="--",
                width=15
            )

            measured.pack(side="left")

            commanded = ttk.Label(
                row,
                text="--",
                width=15
            )

            commanded.pack(side="left")

            self.measured_labels[joint] = measured

            self.command_labels[joint] = commanded

        # GRIPPER

        grip_frame = ttk.LabelFrame(
            main,
            text="Gripper Control",
            padding=12
        )

        grip_frame.pack(fill="x", pady=10)

        self.open_button = ttk.Button(
            grip_frame,
            text="OPEN",
            command=lambda:
                self.node.command_gripper(GRIPPER_OPEN)
        )

        self.close_button = ttk.Button(
            grip_frame,
            text="CLOSE",
            command=lambda:
                self.node.command_gripper(GRIPPER_CLOSE)
        )

        self.open_button.pack(
            side="left",
            padx=25
        )

        self.close_button.pack(
            side="left",
            padx=25
        )

        # STOP

        stop_frame = ttk.Frame(main)

        stop_frame.pack(fill="x", pady=15)

        self.stop_button = ttk.Button(
            stop_frame,
            text="STOP ACTIVE TRAJECTORY",
            command=self.node.stop_arm
        )

        self.stop_button.pack(
            fill="x",
            ipady=8
        )

        ttk.Separator(
            main
        ).pack(fill="x", pady=10)

        ttk.Label(
            main,
            text=(
                "Measured = actual joint feedback from ROS.\n"
                "Commanded = last requested joint target.\n"
                "Stop requests trajectory cancellation."
            ),
            justify="center"
        ).pack(pady=5)

    # --------------------------------------------------------
    # JOINT COMMAND
    # --------------------------------------------------------

    def move_joint(self, joint, direction):

        if not self.node.feedback_is_fresh():

            messagebox.showerror(
                "ROS Feedback",
                "Joint feedback is unavailable."
            )

            return

        step = self.step_deg.get()

        self.node.move_joint(
            joint,
            direction,
            step
        )

    # --------------------------------------------------------
    # UPDATE GUI
    # --------------------------------------------------------

    def update_gui(self):

        positions = self.node.get_positions()

        arm_busy, gripper_busy = self.node.get_status()

        fresh = self.node.feedback_is_fresh()

        # Measured positions

        for joint in ARM_JOINTS:

            if joint in positions:

                degrees = math.degrees(
                    positions[joint]
                )

                self.measured_labels[joint].config(
                    text=f"{degrees:.2f}°"
                )

            if joint in self.commanded_positions:

                degrees = math.degrees(
                    self.commanded_positions[joint]
                )

                self.command_labels[joint].config(
                    text=f"{degrees:.2f}°"
                )

        # Button locking

        arm_state = (
            "disabled"
            if arm_busy or not fresh
            else "normal"
        )

        for button in self.buttons:

            button.config(state=arm_state)

        grip_state = (
            "disabled"
            if gripper_busy
            else "normal"
        )

        self.open_button.config(state=grip_state)
        self.close_button.config(state=grip_state)

        # Status

        if not fresh:

            self.status.set(
                "DISCONNECTED — No fresh joint feedback"
            )

            self.status_label.config(
                foreground="red"
            )

        elif arm_busy:

            self.status.set(
                "ARM MOVING — Trajectory executing"
            )

            self.status_label.config(
                foreground="blue"
            )

        elif gripper_busy:

            self.status.set(
                "GRIPPER MOVING"
            )

            self.status_label.config(
                foreground="blue"
            )

        else:

            self.status.set(
                "CONNECTED — Robot ready"
            )

            self.status_label.config(
                foreground="green"
            )

        # Process asynchronous ROS events

        self.process_events()

        self.root.after(
            100,
            self.update_gui
        )

    # --------------------------------------------------------
    # ROS EVENTS
    # --------------------------------------------------------

    def process_events(self):

        while True:

            try:

                event, message = self.node.events.get_nowait()

            except queue.Empty:

                break

            if event == "command":

                self.commanded_positions = message

            elif event == "error":

                self.status.set(message)

                print("[ERROR]", message)

            elif event == "success":

                print("[SUCCESS]", message)

            else:

                print("[INFO]", message)

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(self):

        self.node.stop_arm()

        self.root.after(
            300,
            self.root.destroy
        )


# ============================================================
# MAIN
# ============================================================

def main():

    rclpy.init()

    node = RobotNode()

    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )

    ros_thread.start()

    root = tk.Tk()

    RobotGUI(root, node)

    try:

        root.mainloop()

    finally:

        node.destroy_node()

        rclpy.shutdown()

        ros_thread.join(timeout=2)


if __name__ == "__main__":

    main()