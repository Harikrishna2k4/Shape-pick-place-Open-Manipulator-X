#!/usr/bin/env python3

import cv2
import numpy as np
import time

# =====================================================
# CAMERA CONFIGURATION
# =====================================================

CAMERA_DEVICE = "/dev/video6"

WIDTH = 640
HEIGHT = 480
FPS = 30

# =====================================================
# ARUCO CONFIGURATION
# =====================================================

ARUCO_DICT_NAME = "DICT_4X4_50"

# None = detect largest marker
TARGET_MARKER_ID = None

# =====================================================
# ROBOT COORDINATES OF FOUR MARKER CORNERS
# Measurements obtained using robot TCP + FK.
# Units: metres
#
# Order: TL, TR, BR, BL
# =====================================================

ROBOT_CORNERS = np.array([
    [0.121, 0.131],  # Top-left
    [0.117, 0.092],  # Top-right
    [0.079, 0.084],  # Bottom-right
    [0.083, 0.120]   # Bottom-left
], dtype=np.float32)

CORNER_LABELS = ["TL", "TR", "BR", "BL"]

# =====================================================
# PIXEL TO CONVERT
# =====================================================

TARGET_PIXEL = (158, 293)


# =====================================================
# CREATE ARUCO DETECTOR
# =====================================================

def create_detector():

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco unavailable. Install opencv-contrib-python."
        )

    aruco = cv2.aruco

    dictionary = aruco.getPredefinedDictionary(
        getattr(aruco, ARUCO_DICT_NAME)
    )

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    parameters.cornerRefinementMethod = (
        aruco.CORNER_REFINE_SUBPIX
    )

    if hasattr(aruco, "ArucoDetector"):

        detector = aruco.ArucoDetector(
            dictionary,
            parameters
        )

    else:
        detector = None

    return detector, dictionary, parameters


# =====================================================
# DETECT MARKER
# =====================================================

def detect_markers(gray, detector, dictionary, parameters):

    if detector is not None:

        return detector.detectMarkers(gray)

    return cv2.aruco.detectMarkers(
        gray,
        dictionary,
        parameters=parameters
    )


# =====================================================
# PIXEL TO ROBOT COORDINATE CONVERSION
# =====================================================

def pixel_to_robot(pixel, H):

    point = np.asarray(
        pixel,
        dtype=np.float32
    ).reshape(1, 1, 2)

    result = cv2.perspectiveTransform(
        point,
        H
    )

    return result[0, 0]


# =====================================================
# MAIN
# =====================================================

def main():

    detector, dictionary, parameters = create_detector()

    camera = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    if not camera.isOpened():

        print("ERROR: Cannot open camera")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)

    print("\n======================================")
    print(" FOUR-CORNER ARUCO CALIBRATION")
    print("======================================")

    print("\nRobot calibration coordinates:")

    for label, point in zip(
        CORNER_LABELS,
        ROBOT_CORNERS
    ):

        print(
            f"{label}: X={point[0]:.5f}, "
            f"Y={point[1]:.5f} m"
        )

    print(f"\nTarget pixel: {TARGET_PIXEL}")

    print("\nDetecting marker...")
    print("Press ESC to exit.\n")

    last_print_time = 0

    try:

        while True:

            ret, frame = camera.read()

            if not ret:

                print("ERROR: Camera frame failed")
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            corners, ids, rejected = detect_markers(
                gray,
                detector,
                dictionary,
                parameters
            )

            display = frame.copy()

            if ids is not None:

                flat_ids = ids.flatten()

                candidates = [

                    i for i, marker_id in enumerate(flat_ids)

                    if TARGET_MARKER_ID is None
                    or int(marker_id) == TARGET_MARKER_ID

                ]

                if candidates:

                    # Select largest matching marker

                    chosen = max(
                        candidates,
                        key=lambda i: cv2.contourArea(
                            corners[i].reshape(4, 2)
                        )
                    )

                    marker_id = int(flat_ids[chosen])

                    # Detected image corners:
                    # TL, TR, BR, BL

                    image_corners = corners[chosen].reshape(
                        4, 2
                    ).astype(np.float32)

                    # =====================================
                    # CALCULATE HOMOGRAPHY
                    # =====================================

                    H = cv2.getPerspectiveTransform(
                        image_corners,
                        ROBOT_CORNERS
                    )

                    if not np.all(np.isfinite(H)):
                        continue

                    # =====================================
                    # CONVERT TARGET PIXEL
                    # =====================================

                    robot_xy = pixel_to_robot(
                        TARGET_PIXEL,
                        H
                    )

                    robot_x, robot_y = robot_xy

                    if not np.all(np.isfinite(robot_xy)):
                        continue

                    # =====================================
                    # DRAW DETECTED MARKER
                    # =====================================

                    cv2.aruco.drawDetectedMarkers(
                        display,
                        [corners[chosen]],
                        np.array([[marker_id]], dtype=np.int32)
                    )

                    # Display corner labels

                    for i, point in enumerate(image_corners):

                        px, py = point.astype(int)

                        cv2.circle(
                            display,
                            (px, py),
                            6,
                            (0, 255, 0),
                            -1
                        )

                        cv2.putText(
                            display,
                            CORNER_LABELS[i],
                            (px + 7, py - 7),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2
                        )

                    # =====================================
                    # DRAW TARGET PIXEL
                    # =====================================

                    cv2.circle(
                        display,
                        TARGET_PIXEL,
                        8,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        display,
                        f"Target: {TARGET_PIXEL}",
                        (TARGET_PIXEL[0] + 10,
                         TARGET_PIXEL[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

                    # =====================================
                    # DISPLAY ROBOT COORDINATES
                    # =====================================

                    cv2.putText(
                        display,
                        f"Robot X = {robot_x:.5f} m",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 0),
                        2
                    )

                    cv2.putText(
                        display,
                        f"Robot Y = {robot_y:.5f} m",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 0),
                        2
                    )

                    # =====================================
                    # TERMINAL OUTPUT
                    # =====================================

                    current_time = time.time()

                    if current_time - last_print_time > 1:

                        print(
                            f"Pixel {TARGET_PIXEL} -> "
                            f"Robot X={robot_x:.5f} m, "
                            f"Robot Y={robot_y:.5f} m"
                        )

                        last_print_time = current_time

            else:

                cv2.putText(
                    display,
                    "ArUco marker not detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Four Corner ArUco Calibration",
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()