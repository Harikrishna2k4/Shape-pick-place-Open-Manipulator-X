#!/usr/bin/env python3
"""Continuous object detection and camera-pixel to OpenMANIPULATOR target mapping.

Required alongside this file:
    camera_to_robot.npz

The saved homography maps image pixels to robot-base XY on the calibration
plane. This script continuously detects the lime-yellow circular object,
computes an end-effector TARGET position, writes the latest result to CSV,
and (when run in a sourced ROS 2 environment) publishes ROS topics.

It DOES NOT command robot motion. The robot controller must subscribe to the
vision topic, validate detections/timeouts, and plan a safe trajectory.
"""

from pathlib import Path
import csv
import os
import time

import cv2
import numpy as np

# ---------------- CAMERA CONFIGURATION ----------------
CAMERA_DEVICE = "/dev/video6"
WIDTH = 640
HEIGHT = 480
FPS = 30
USE_YUYV = True

# ---------------- CALIBRATION / OUTPUT ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_FILE = SCRIPT_DIR / "camera_to_robot.npz"
LATEST_TARGET_CSV = SCRIPT_DIR / "object_target_latest.csv"

# The previous pick-and-place controller used CONTACT_Z = 0.110 m.
# This is the desired EE TCP height, not measured from the camera image.
TARGET_EE_Z_M = 0.110

# Add any measured TCP-to-object-centre correction here. Leave zero if the
# gripper TCP should be directly above the detected object centre.
EE_OFFSET_X_M = 0.0
EE_OFFSET_Y_M = 0.0

# Root/base link in the OpenManipulator-X URDF used in this project.
BASE_FRAME_ID = "link1"

# Write the latest state at up to 10 Hz, not on every 30 FPS camera frame.
OUTPUT_WRITE_PERIOD_S = 0.10
STATUS_PRINT_PERIOD_S = 1.0

# ---------------- COLOUR DETECTION --------------------
LOWER_COLOR = np.array([20, 80, 100], dtype=np.uint8)
UPPER_COLOR = np.array([45, 255, 255], dtype=np.uint8)
MIN_AREA = 300

# ---------------- CIRCLE DETECTION --------------------
MIN_RADIUS = 15
MAX_RADIUS = 55
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 20
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 30
ROI_PADDING = 25

# Optional ROS 2 publishing. If ROS 2 Python packages are unavailable, the
# camera and CSV output still run normally.
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PointStamped
    from std_msgs.msg import Bool
    ROS_AVAILABLE = True
except ImportError:
    rclpy = None
    Node = object
    PointStamped = None
    Bool = None
    ROS_AVAILABLE = False


class VisionRosPublisher:
    """Publishes current EE target and detection-validity state."""

    def __init__(self):
        self.enabled = False
        self.node = None
        self.target_pub = None
        self.detected_pub = None

        if not ROS_AVAILABLE:
            print("ROS 2 Python packages not found; CSV output only.")
            return

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.node = Node("overhead_object_vision")
            self.target_pub = self.node.create_publisher(
                PointStamped, "/vision/object_target", 10
            )
            self.detected_pub = self.node.create_publisher(
                Bool, "/vision/object_detected", 10
            )
            self.enabled = True
            print("ROS 2 publishing enabled:")
            print("  /vision/object_target  [geometry_msgs/msg/PointStamped]")
            print("  /vision/object_detected [std_msgs/msg/Bool]")
        except Exception as exc:
            print(f"ROS 2 publishing unavailable ({exc}); CSV output only.")
            self.enabled = False

    def publish(self, target_xyz=None):
        if not self.enabled:
            return

        try:
            detected_msg = Bool()
            detected_msg.data = target_xyz is not None
            self.detected_pub.publish(detected_msg)

            if target_xyz is not None:
                msg = PointStamped()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.header.frame_id = BASE_FRAME_ID
                msg.point.x = float(target_xyz[0])
                msg.point.y = float(target_xyz[1])
                msg.point.z = float(target_xyz[2])
                self.target_pub.publish(msg)

            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            print(f"ROS publish error: {exc}")

    def close(self):
        if self.node is not None:
            self.node.destroy_node()
        if self.enabled and rclpy is not None and rclpy.ok():
            rclpy.shutdown()


def load_calibration():
    if not CALIBRATION_FILE.is_file():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_FILE}. "
            "Keep camera_to_robot.npz beside this script."
        )

    with np.load(CALIBRATION_FILE) as data:
        required = {"H", "image_width", "image_height"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Calibration file is missing: {sorted(missing)}")
        homography = np.asarray(data["H"], dtype=np.float64)
        calib_w = int(data["image_width"])
        calib_h = int(data["image_height"])

    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        raise ValueError("Invalid homography matrix in calibration file.")
    if abs(np.linalg.det(homography)) < 1e-12:
        raise ValueError("Homography matrix is singular.")
    return homography, calib_w, calib_h


def pixel_to_robot_xy(pixel_x, pixel_y, homography):
    point = np.array(
        [[[float(pixel_x), float(pixel_y)]]], dtype=np.float32
    )
    mapped = cv2.perspectiveTransform(point, homography)[0, 0]
    x, y = float(mapped[0]), float(mapped[1])
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError("Pixel-to-robot mapping returned non-finite values.")
    return x, y


def detect_held_object(frame):
    """Return circle centre/radius in full-frame pixel coordinates."""
    frame_h, frame_w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_AREA:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    x1 = max(0, x - ROI_PADDING)
    y1 = max(0, y - ROI_PADDING)
    x2 = min(frame_w, x + w + ROI_PADDING)
    y2 = min(frame_h, y + h + ROI_PADDING)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )
    if circles is None:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    color_cx = moments["m10"] / moments["m00"]
    color_cy = moments["m01"] / moments["m00"]

    best = None
    best_dist = float("inf")
    for local_cx, local_cy, radius in np.round(circles[0]).astype(int):
        global_cx = int(local_cx + x1)
        global_cy = int(local_cy + y1)
        dist = float(np.hypot(global_cx - color_cx, global_cy - color_cy))
        if dist < best_dist:
            best_dist = dist
            best = (global_cx, global_cy, int(radius))

    if best is None:
        return None
    return {"center": (best[0], best[1]), "radius": best[2]}


def write_latest_target(valid, pixel_xy=None, target_xyz=None, radius_px=None):
    """Atomically overwrite one-row CSV, preventing consumers reading a partial file."""
    tmp_file = LATEST_TARGET_CSV.with_suffix(".tmp")
    with tmp_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_unix", "valid", "pixel_x", "pixel_y",
            "ee_target_x_m", "ee_target_y_m", "ee_target_z_m",
            "radius_px", "frame_id",
        ])
        if valid and pixel_xy is not None and target_xyz is not None:
            writer.writerow([
                f"{time.time():.6f}", 1,
                pixel_xy[0], pixel_xy[1],
                f"{target_xyz[0]:.6f}",
                f"{target_xyz[1]:.6f}",
                f"{target_xyz[2]:.6f}",
                radius_px, BASE_FRAME_ID,
            ])
        else:
            writer.writerow([
                f"{time.time():.6f}", 0, "", "", "", "", "", "", BASE_FRAME_ID
            ])
    os.replace(tmp_file, LATEST_TARGET_CSV)


def main():
    try:
        homography, calibration_w, calibration_h = load_calibration()
    except Exception as exc:
        print(f"CALIBRATION ERROR: {exc}")
        return 1

    camera = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not camera.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_DEVICE}")
        return 1

    if USE_YUYV:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)

    ros_pub = VisionRosPublisher()
    print(f"Camera: {CAMERA_DEVICE}")
    print(f"Calibration: {CALIBRATION_FILE}")
    print(f"Output (auto-updated): {LATEST_TARGET_CSV}")
    print(f"EE target Z: {TARGET_EE_Z_M:.3f} m")
    print("Automatic detection and coordinate conversion are active.")
    print("No key press is required. Press ESC to exit the preview.")
    print("This program publishes target coordinates; it does NOT move the robot.")

    last_write = 0.0
    last_print = 0.0
    resolution_checked = False

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("ERROR: Frame capture failed")
                break

            frame_h, frame_w = frame.shape[:2]
            if not resolution_checked:
                print(f"Camera frame resolution: {frame_w} x {frame_h}")
                if (frame_w, frame_h) != (calibration_w, calibration_h):
                    raise RuntimeError(
                        "Camera frame resolution does not match calibration "
                        f"({calibration_w}x{calibration_h}). Actual: "
                        f"{frame_w}x{frame_h}. Set the camera resolution "
                        "to the calibration resolution or recalibrate."
                    )
                resolution_checked = True

            obj = detect_held_object(frame)
            display = frame.copy()
            target_xyz = None
            pixel_xy = None
            radius_px = None

            if obj is not None:
                cx, cy = obj["center"]
                radius_px = obj["radius"]
                try:
                    base_x, base_y = pixel_to_robot_xy(cx, cy, homography)
                    ee_x = base_x + EE_OFFSET_X_M
                    ee_y = base_y + EE_OFFSET_Y_M
                    target_xyz = (ee_x, ee_y, TARGET_EE_Z_M)
                    pixel_xy = (cx, cy)

                    cv2.circle(display, (cx, cy), radius_px, (0, 255, 0), 2)
                    cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(
                        display, f"Pixel: ({cx}, {cy})", (12, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                    )
                    cv2.putText(
                        display,
                        f"EE target ({BASE_FRAME_ID}): X={ee_x:.4f} Y={ee_y:.4f} Z={TARGET_EE_Z_M:.3f} m",
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (0, 255, 0), 2
                    )
                    cv2.putText(
                        display, f"Radius: {radius_px}px | AUTO", (12, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                    )
                except Exception as exc:
                    cv2.putText(
                        display, f"Coordinate error: {exc}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1
                    )
                    target_xyz = None
                    pixel_xy = None
            else:
                cv2.putText(
                    display, "OBJECT NOT DETECTED", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

            now = time.monotonic()
            if now - last_write >= OUTPUT_WRITE_PERIOD_S:
                write_latest_target(
                    target_xyz is not None,
                    pixel_xy=pixel_xy,
                    target_xyz=target_xyz,
                    radius_px=radius_px,
                )
                last_write = now

            # Publish every camera frame. Detection=false is published when
            # object tracking is lost; consumers should also implement a timeout.
            ros_pub.publish(target_xyz)

            if target_xyz is not None and now - last_print >= STATUS_PRINT_PERIOD_S:
                print(
                    f"EE target [{BASE_FRAME_ID}]: "
                    f"X={target_xyz[0]:.4f}, Y={target_xyz[1]:.4f}, "
                    f"Z={target_xyz[2]:.3f} m"
                )
                last_print = now

            cv2.imshow("Automatic Object Detection - EE Target", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        # Mark target invalid on shutdown so a consumer does not assume the
        # last coordinate is still current.
        try:
            write_latest_target(False)
            ros_pub.publish(None)
        except Exception:
            pass
        camera.release()
        cv2.destroyAllWindows()
        ros_pub.close()

    print(f"Latest target file: {LATEST_TARGET_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
