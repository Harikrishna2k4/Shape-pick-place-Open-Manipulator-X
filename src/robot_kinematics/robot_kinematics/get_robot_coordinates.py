
#!/usr/bin/env python3

import cv2
import numpy as np

# ---------------- CAMERA CONFIGURATION ----------------

CAMERA_DEVICE = "/dev/video6"
WIDTH = 640
HEIGHT = 480
FPS = 30

# ---------------- ARUCO CONFIGURATION ----------------

ARUCO_DICT_NAME = "DICT_4X4_50"
TARGET_MARKER_ID = None  # None = largest detected marker

# ---------------- ROBOT CALIBRATION ----------------

MARKER_SIDE_M = 0.037

# TOP-RIGHT OUTER CORNER robot coordinates
MARKER_TOP_RIGHT_X_M = 0.117
MARKER_TOP_RIGHT_Y_M = 0.124

# Pixel to convert
TARGET_PIXEL = (158, 293)


def make_detector():

    aruco = cv2.aruco

    dictionary = aruco.getPredefinedDictionary(
        getattr(aruco, ARUCO_DICT_NAME)
    )

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
    else:
        detector = None

    return detector, dictionary, parameters


def get_robot_marker_corners():

    x = MARKER_TOP_RIGHT_X_M
    y = MARKER_TOP_RIGHT_Y_M
    s = MARKER_SIDE_M

    # Corresponding to image corners TL, TR, BR, BL

    return np.array([
        [x - s, y],       # Top-left
        [x, y],           # Top-right
        [x, y - s],       # Bottom-right
        [x - s, y - s]    # Bottom-left
    ], dtype=np.float32)


def pixel_to_robot(pixel, H):

    point = np.asarray(
        pixel, dtype=np.float32
    ).reshape(1, 1, 2)

    result = cv2.perspectiveTransform(point, H)

    return result[0, 0]


def main():

    detector, dictionary, parameters = make_detector()

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

    robot_corners = get_robot_marker_corners()

    print("\n=== PIXEL TO ROBOT COORDINATES ===")
    print(f"Target pixel: {TARGET_PIXEL}")
    print("ArUco top-right reference: "
          f"({MARKER_TOP_RIGHT_X_M:.3f}, "
          f"{MARKER_TOP_RIGHT_Y_M:.3f}) m")
    print("Press ESC to exit.\n")

    try:

        while True:

            ret, frame = camera.read()

            if not ret:
                print("Camera frame failed")
                break

            gray = cv2.cvtColor(
                frame, cv2.COLOR_BGR2GRAY
            )

            if detector is not None:
                corners, ids, rejected = detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray,
                    dictionary,
                    parameters=parameters
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

                    chosen = max(
                        candidates,
                        key=lambda i: cv2.contourArea(
                            corners[i].reshape(4, 2)
                        )
                    )

                    marker_id = int(flat_ids[chosen])

                    # Detected image corners: TL, TR, BR, BL
                    image_corners = corners[chosen].reshape(
                        4, 2
                    ).astype(np.float32)

                    # Calculate pixel-to-robot homography
                    H = cv2.getPerspectiveTransform(
                        image_corners,
                        robot_corners
                    )

                    # Convert target pixel
                    robot_xy = pixel_to_robot(
                        TARGET_PIXEL, H
                    )

                    x, y = robot_xy

                    # Draw detected marker
                    cv2.aruco.drawDetectedMarkers(
                        display,
                        [corners[chosen]],
                        np.array([[marker_id]], dtype=np.int32)
                    )

                    # Draw target pixel
                    cv2.circle(
                        display,
                        TARGET_PIXEL,
                        7,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        display,
                        "TARGET (158,293)",
                        (TARGET_PIXEL[0] + 10,
                         TARGET_PIXEL[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

                    # Display robot coordinates
                    cv2.putText(
                        display,
                        f"Robot X: {x:.5f} m",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        display,
                        f"Robot Y: {y:.5f} m",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    print(
                        f"\rPixel {TARGET_PIXEL} -> "
                        f"Robot X = {x:.5f} m, "
                        f"Robot Y = {y:.5f} m",
                        end="",
                        flush=True
                    )

                else:

                    cv2.putText(
                        display,
                        "Target ArUco ID not detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

            else:

                cv2.putText(
                    display,
                    "No ArUco marker detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Pixel to Robot Coordinates",
                display
            )

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()