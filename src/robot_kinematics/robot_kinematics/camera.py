import cv2
import numpy as np
import csv

CAMERA_DEVICE = "/dev/video6"

WIDTH = 640
HEIGHT = 480
FPS = 30

OUTPUT_FILE = "held_object.csv"

# Lime-yellow object
LOWER_COLOR = np.array([20, 80, 100])
UPPER_COLOR = np.array([45, 255, 255])

MIN_AREA = 300

# Circle detection parameters
MIN_RADIUS = 15
MAX_RADIUS = 55

HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 20




def detect_held_object(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        LOWER_COLOR,
        UPPER_COLOR
    )

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    if cv2.contourArea(contour) < MIN_AREA:
        return None

    # Find region containing coloured object
    x, y, w, h = cv2.boundingRect(contour)

    padding = 25

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(WIDTH, x + w + padding)
    y2 = min(HEIGHT, y + h + padding)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    # Detect circular boundary in ROI
    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        1.5
    )

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # Select circle closest to colour-region centroid
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None

    color_cx = M["m10"] / M["m00"]
    color_cy = M["m01"] / M["m00"]

    best_circle = None
    best_distance = float("inf")

    for cx, cy, radius in circles:

        global_cx = cx + x1
        global_cy = cy + y1

        distance = np.hypot(
            global_cx - color_cx,
            global_cy - color_cy
        )

        if distance < best_distance:

            best_distance = distance

            best_circle = (
                global_cx,
                global_cy,
                radius
            )

    if best_circle is None:
        return None

    cx, cy, radius = best_circle

    return {
        "center": (cx, cy),
        "radius": radius
    }


def main():

    camera = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"YUYV")
    )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)

    if not camera.isOpened():
        print("ERROR: Cannot open RealSense RGB camera")
        return

    print("RealSense RGB camera connected")
    print("Detecting circular object boundary")
    print("S = Save coordinates")
    print("ESC = Exit")

    saved_points = []

    try:

        while True:

            ret, frame = camera.read()

            if not ret:
                print("Frame capture failed")
                break

            obj = detect_held_object(frame)

            display = frame.copy()

            if obj is not None:

                cx, cy = obj["center"]
                radius = obj["radius"]

                # Draw detected circular boundary
                cv2.circle(
                    display,
                    (cx, cy),
                    radius,
                    (0, 255, 0),
                    2
                )

                # Draw centre
                cv2.circle(
                    display,
                    (cx, cy),
                    5,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    display,
                    f"Center: ({cx}, {cy})",
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"Radius: {radius}px",
                    (cx + 10, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    display,
                    "Circle not detected",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "RealSense RGB - Circle Boundary Detection",
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") and obj is not None:

                cx, cy = obj["center"]

                saved_points.append([
                    cx,
                    cy,
                    obj["radius"]
                ])

                print(
                    f"Saved: ({cx}, {cy}), "
                    f"radius={obj['radius']}"
                )

            elif key == 27:
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()

    if saved_points:

        with open(OUTPUT_FILE, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                "pixel_x",
                "pixel_y",
                "radius"
            ])

            writer.writerows(saved_points)

        print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()