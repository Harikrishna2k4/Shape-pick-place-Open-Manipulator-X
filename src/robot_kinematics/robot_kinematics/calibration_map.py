#!/usr/bin/env python3
"""Single-ArUco planar pixel-to-robot calibration for a fixed overhead camera.

Assumptions:
  * Camera is fixed above a flat board.
  * Marker lies flat on the same plane used for object XY measurements.
  * Marker top edge is parallel to robot +X.
  * Robot +X points to image-right along the marker top edge.
  * Robot +Y points toward image-top.
  * The measured marker top-right outer corner is (0.082, 0.108) m.

The script uses the four detected OUTER marker corners to compute a homography.
It does not estimate or correct lens distortion.
"""

from pathlib import Path
import sys

import cv2
import numpy as np

# ---------------- USER CONFIGURATION ----------------
CAMERA_DEVICE = "/dev/video6"
WIDTH = 640
HEIGHT = 480
FPS = 30

# Assumed dictionary: 4x4 data cells. Change this if the marker was generated
# from another dictionary (for example DICT_5X5_50 or DICT_APRILTAG_36h11).
ARUCO_DICT_NAME = "DICT_4X4_50"

# Set an integer after identifying the marker ID, or leave None to use the
# largest marker detected from the selected dictionary.
TARGET_MARKER_ID = None

# Measure the complete OUTER black square, from outer edge to outer edge.
MARKER_SIDE_M = 0.037

# Robot-base coordinates of the marker's TOP-RIGHT outer corner.
MARKER_TOP_RIGHT_X_M = 0.082
MARKER_TOP_RIGHT_Y_M = 0.108

# The calibration is stored beside this script.
OUTPUT_FILE = Path(__file__).resolve().with_name("camera_to_robot.npz")
# ----------------------------------------------------


def make_detector():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is unavailable. Install the OpenCV contrib package, "
            "for example: python3 -m pip install opencv-contrib-python"
        )

    aruco = cv2.aruco
    if not hasattr(aruco, ARUCO_DICT_NAME):
        raise RuntimeError(
            f"Dictionary {ARUCO_DICT_NAME} is not available in this OpenCV build."
        )

    dictionary = aruco.getPredefinedDictionary(
        getattr(aruco, ARUCO_DICT_NAME)
    )

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    # Refine detected corners to sub-pixel precision when supported.
    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        return detector, dictionary, parameters

    return None, dictionary, parameters


def detect_markers(gray, detector, dictionary, parameters):
    if detector is not None:
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(
        gray, dictionary, parameters=parameters
    )


def robot_marker_corners():
    """Return robot XY for image-order corners TL, TR, BR, BL."""
    x_tr = MARKER_TOP_RIGHT_X_M
    y_tr = MARKER_TOP_RIGHT_Y_M
    side = MARKER_SIDE_M

    x_tl = x_tr - side
    y_tl = y_tr
    x_br = x_tr
    y_br = y_tr - side
    x_bl = x_tr - side
    y_bl = y_tr - side

    return np.array(
        [
            [x_tl, y_tl],  # top-left
            [x_tr, y_tr],  # top-right
            [x_br, y_br],  # bottom-right
            [x_bl, y_bl],  # bottom-left
        ],
        dtype=np.float32,
    )


def pixel_to_robot_xy(pixel_xy, homography):
    point = np.asarray(pixel_xy, dtype=np.float32).reshape(1, 1, 2)
    mapped = cv2.perspectiveTransform(point, homography)
    return mapped[0, 0]


def main():
    try:
        detector, dictionary, parameters = make_detector()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    camera = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not camera.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_DEVICE}")
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)

    print(f"Camera opened: {CAMERA_DEVICE}")
    print(f"ArUco dictionary: {ARUCO_DICT_NAME}")
    print(f"Target marker ID: {TARGET_MARKER_ID if TARGET_MARKER_ID is not None else 'largest detected'}")
    print(f"Marker side: {MARKER_SIDE_M * 1000:.1f} mm")
    print(
        "Robot coordinates of marker top-right: "
        f"({MARKER_TOP_RIGHT_X_M:.3f}, {MARKER_TOP_RIGHT_Y_M:.3f}) m"
    )
    print("Press S to save calibration; ESC to exit.")

    robot_corners = robot_marker_corners()
    saved = False

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("ERROR: Frame capture failed")
                break

            image_h, image_w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detect_markers(
                gray, detector, dictionary, parameters
            )

            display = frame.copy()
            H = None
            selected_id = None
            image_corners = None

            if ids is not None and len(ids) > 0:
                flat_ids = ids.flatten()
                candidate_indices = [
                    i for i, marker_id in enumerate(flat_ids)
                    if TARGET_MARKER_ID is None or int(marker_id) == TARGET_MARKER_ID
                ]

                if candidate_indices:
                    # If several markers are visible and no ID was specified,
                    # use the largest detected square.
                    chosen = max(
                        candidate_indices,
                        key=lambda i: cv2.contourArea(
                            corners[i].reshape(4, 2).astype(np.float32)
                        ),
                    )
                    selected_id = int(flat_ids[chosen])
                    image_corners = corners[chosen].reshape(4, 2).astype(np.float32)

                    # OpenCV ArUco order: TL, TR, BR, BL (clockwise), provided
                    # the printed marker is physically oriented as described.
                    H = cv2.getPerspectiveTransform(
                        image_corners,
                        robot_corners,
                    )

                    cv2.aruco.drawDetectedMarkers(
                        display,
                        [corners[chosen]],
                        np.array([[selected_id]], dtype=np.int32),
                    )

                    labels = ["TL", "TR", "BR", "BL"]
                    for i, (px, py) in enumerate(image_corners):
                        rx, ry = robot_corners[i]
                        cv2.circle(display, (int(px), int(py)), 5, (0, 255, 0), -1)
                        cv2.putText(
                            display,
                            f"{labels[i]} ({rx:.3f},{ry:.3f})",
                            (int(px) + 6, int(py) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                    # Show where the image centre maps on the calibrated plane.
                    image_centre = (image_w / 2.0, image_h / 2.0)
                    centre_xy = pixel_to_robot_xy(image_centre, H)
                    status = (
                        f"ID {selected_id} | image centre -> "
                        f"X={centre_xy[0]:.3f} Y={centre_xy[1]:.3f} m"
                    )
                    cv2.putText(
                        display, status, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (0, 255, 255), 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        display, "S: save calibration", (12, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA
                    )
                else:
                    cv2.putText(
                        display,
                        f"Marker found, but ID {TARGET_MARKER_ID} not detected",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2
                    )
            else:
                cv2.putText(
                    display,
                    f"No marker detected ({ARUCO_DICT_NAME})",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 255), 2
                )

            cv2.imshow("Single ArUco - Camera to Robot Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if H is None or image_corners is None:
                    print("Cannot save: no target marker detected.")
                    continue

                # Check that all four marker corners map to the intended
                # robot coordinates within numerical precision.
                mapped_corners = cv2.perspectiveTransform(
                    image_corners.reshape(1, 4, 2), H
                )[0]
                residual = np.linalg.norm(mapped_corners - robot_corners, axis=1)
                max_error = float(np.max(residual))

                np.savez(
                    OUTPUT_FILE,
                    H=H,
                    image_width=np.int32(image_w),
                    image_height=np.int32(image_h),
                    marker_id=np.int32(selected_id),
                    dictionary=np.array(ARUCO_DICT_NAME),
                    marker_side_m=np.float64(MARKER_SIDE_M),
                    marker_top_right_xy_m=np.array(
                        [MARKER_TOP_RIGHT_X_M, MARKER_TOP_RIGHT_Y_M],
                        dtype=np.float64,
                    ),
                    image_corners=image_corners,
                    robot_corners=robot_corners,
                )
                saved = True
                print(f"Calibration saved: {OUTPUT_FILE}")
                print("Homography (pixel -> robot XY in metres):")
                print(H)
                print(f"Four-corner reprojection residual: {max_error:.9f} m")
                print(
                    "NOTE: This residual is only the four points used to fit H; "
                    "it is not an independent accuracy test."
                )

            elif key == 27:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not saved:
        print("No calibration file was saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
