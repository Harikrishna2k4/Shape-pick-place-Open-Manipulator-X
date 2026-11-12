#!/usr/bin/env python3
"""Stabilised five-second shape sorting using live four-corner ArUco homography.

No user selection is needed anywhere:
  1) The cardboard task board is found automatically.
  2) Strict colour + geometry detection runs for 5 s; paired pick/place
     coordinates are converted to robot-base XY and averaged. Motion (e.g. a hand) restarts
     the timer.
  3) Coordinates are FROZEN; the preview stays live. Q/ESC confirms and writes
     the NPZ/CSV targets. X aborts. (M toggles the colour-mask debug view.)

Shapes are verified with rotation-invariant tests (solidity, fill ratios,
corner fits, Fourier harmonics of the outline), so hands and clutter are
rejected. No still image or video is captured or saved.
"""

from pathlib import Path
from datetime import datetime
import os
import csv
import sys
import time
from collections import deque
from itertools import permutations

os.environ.setdefault("QT_X11_NO_MITSHM", "1")   # avoids MIT-SHM BadAccess with Qt-based OpenCV

import cv2
import numpy as np

# ---------------------------- CAMERA -----------------------------
CAMERA_DEVICE = "/dev/video6"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FPS = 30
USE_YUYV = True
AVERAGING_SECONDS = 5.0
# No-window mode: set SHAPE_SORT_HEADLESS=1 (also used automatically when DISPLAY is unset).
# Detection, averaging and file output are identical; targets are accepted automatically.
HEADLESS = (os.environ.get("SHAPE_SORT_HEADLESS") == "1"
            or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")))
MIN_VALID_SAMPLES = 10
MIN_DETECTION_RATE = 0.70      # fraction of averaging frames with a valid object/hole pair
MIN_HOLE_DETECTION_RATE = 0.70  # all four fixed cardboard holes must be seen this often
NO_COLOURED_OBJECTS_EXIT = 4   # valid vision run, but there is nothing to sort
MAX_POSITION_STD_M = 0.004     # reject targets that jitter more than 4 mm (1 sigma)

# -------------------------- OUTPUT FILES --------------------------
SCRIPT_DIR = Path(__file__).resolve().parent


def make_run_dir():
    configured = os.environ.get("SHAPE_SORT_RUN_DIR")
    if configured:
        run_dir = Path(configured).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        run_dir = SCRIPT_DIR / "shape_sort_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------- LIVE ARUCO -> ROBOT CALIBRATION ----------------
# One ArUco marker is detected in every camera frame. Its four OUTER corners
# are mapped to the four robot-base XY positions measured with TCP + FK.
ARUCO_DICT_NAME = "DICT_4X4_50"
TARGET_MARKER_ID = None       # None = use the largest detected marker
MARKER_SIDE_M = 0.037         # physical outer-edge-to-outer-edge side length (m)

# Corner order MUST match OpenCV's detected ArUco order: TL, TR, BR, BL.
# These are measured TCP/FK robot base_link coordinates, in metres.
ROBOT_MARKER_CORNERS = np.array([
    [0.121, 0.131],  # Top-left
    [0.117, 0.092],  # Top-right
    [0.079, 0.084],  # Bottom-right
    [0.083, 0.120],  # Bottom-left
], dtype=np.float32)

ARUCO_CORNER_LABELS = ("TL", "TR", "BR", "BL")
ARUCO_MIN_AREA_PX = 80.0
ARUCO_CORNER_SMOOTHING_ALPHA = 0.35
ARUCO_MAX_CORNER_JUMP_PX = 25.0


# ------------------------- OBJECT CLASSES -------------------------
SHAPE_ORDER = ("green_circle", "blue_square", "orange_hexagon", "yellow_triangle")
# One expected instance of each configured type on the board and at most
# one matching coloured object of each class. This project currently defines
# circle, square, hexagon and triangle (not a separate rectangle class).
EXPECTED_GEOMETRY = {
    "green_circle": "circle",
    "blue_square": "square",
    "orange_hexagon": "hexagon",
    "yellow_triangle": "triangle",
}
GEOMETRIES = ("circle", "square", "hexagon", "triangle")
MAX_HOLES_PER_GEOMETRY = 1
MAX_OBJECTS_PER_CLASS = 1
MAX_CROSS_CLASS_OBJECT_OVERLAP = 0.50
DISPLAY_COLOURS = {
    "green_circle": (0, 255, 0),
    "blue_square": (255, 100, 0),
    "orange_hexagon": (0, 140, 255),
    "yellow_triangle": (0, 255, 255),
}

# Default HSV ranges (OpenCV hue 0..179). Skin sits around H 5-20, S 40-170, so
# orange/yellow need a high saturation floor. If your blocks are less saturated
# than this, lower the S minimum of that colour in COLOR_RANGES.
COLOR_RANGES = {
    "green_circle":    (np.array([38, 65, 65], np.uint8),  np.array([55, 255, 255], np.uint8)),
    "blue_square":     (np.array([99, 150, 60], np.uint8), np.array([120, 255, 230], np.uint8)),
    "orange_hexagon":  (np.array([5, 150, 90], np.uint8),  np.array([19, 255, 255], np.uint8)),
    "yellow_triangle": (np.array([20, 120, 90], np.uint8), np.array([37, 255, 255], np.uint8)),
}

MIN_COLOR_AREA_PX = 200.0
MAX_COLOR_AREA_PX = 18000.0
MAX_ROI_FRACTION = 0.12
COLOR_OPEN_KERNEL = 5
COLOR_CLOSE_KERNEL = 7
MIN_FILL_OF_MASK = 0.92         # coloured pixels / filled contour area
MAX_INNER_HUE_STD = 10.0        # tolerate mild illumination gradients on coloured block faces
# The green circular object is a 3-D/reflective part: its side wall and top face
# can have different saturation/value. Use a more tolerant colour/shape gate
# only for this class; the other coloured shapes retain the stricter defaults.
GREEN_MIN_COLOR_FILL = 0.76
GREEN_MAX_INNER_HUE_STD = 15.0
GREEN_OPEN_KERNEL = 3
GREEN_CLOSE_KERNEL = 9
GREEN_MIN_SOLIDITY = 0.72
GREEN_MIN_CIRCULARITY = 0.72
GREEN_MIN_ELLIPSE_RATIO = 0.55
GREEN_MAX_ELLIPSE_RESIDUAL = 0.18
SIZE_RATIO_MAX = 3.0            # accepted blocks must be of broadly similar size

# Coloured-object candidates are searched across the full camera ROI.
# No fixed overall workspace polygon is imposed. The cardboard mask is still
# excluded, and each candidate must pass its own colour, shape, size and
# temporal-stability checks.
OBJECT_CONFIRM_FRAMES = 5
OBJECT_TRACK_MAX_JUMP_PX = 14.0
OBJECT_TRACK_MAX_MISSES = 2

# Per-class physical-size sanity checks at 640x480. These reject hand-held
# oversized items, broad coloured regions and tiny reflections.
OBJECT_MIN_AREA_PX = {
    "green_circle": 450.0,
    "blue_square": 300.0,
    "orange_hexagon": 280.0,
    "yellow_triangle": 250.0,
}
OBJECT_MAX_AREA_PX = {
    "green_circle": 12000.0,
    "blue_square": 16000.0,
    "orange_hexagon": 14000.0,
    "yellow_triangle": 14000.0,
}
OBJECT_MAX_SIDE_PX = {
    "green_circle": 135,
    "blue_square": 155,
    "orange_hexagon": 150,
    "yellow_triangle": 150,
}

# --------------------------- SHAPE TESTS --------------------------
SOLIDITY_MIN = 0.85             # tolerate mild perspective/shadow irregularity
TRIANGLE_SOLIDITY_MIN = 0.75
TRIANGLE_MIN_MATCH_SCORE = 0.18
HOLE_TRIANGLE_MAX_MATCH_SCORE = 0.10
CIRCLE_MIN_CIRCULARITY = 0.88
CIRCLE_MIN_ELLIPSE_RATIO = 0.80
CIRCLE_MAX_HARMONIC = 0.030     # max amplitude of radial harmonics 3..8
SQUARE_MIN_RECT_FILL = 0.72
SQUARE_MIN_ASPECT = 0.65
SQUARE_MIN_HARMONIC4 = 0.015
HEX_MIN_HARMONIC6 = 0.020
HEX_RECT_FILL = (0.64, 0.86)
HEX_CIRCULARITY = (0.80, 0.975)
HEX_MAX_SIDE_RATIO = 2.2
TRI_MIN_HARMONIC3 = 0.08
TRI_MIN_FILL = 0.58

# ---------------------------- MOTION ------------------------------
MOTION_PIXEL_DIFF = 30
MOTION_FRACTION_MAX = 0.006     # fraction of ROI pixels changing => "hand present"

# --------------------------- HOLE RULES ---------------------------
# Non-coloured shapes are the cut-outs in the tan cardboard strip. The strip is
# found automatically by its colour; every enclosed opening in it is a hole
# candidate (whatever colour/brightness shows through), then shape-verified.
CARDBOARD_BROAD_LO = (4, 30, 90)      # OpenCV HSV, broad tan/brown search
CARDBOARD_BROAD_HI = (30, 170, 255)
CARDBOARD_TOL = np.array([9, 45, 65])  # tight range around the measured strip colour
MIN_CARDBOARD_FRACTION = 0.02          # strip must cover >= 2% of the frame
# The cardboard strip is in the central-left task area in the current camera
# setup. Restricting the search to this region prevents table/floor colours
# outside the task area from becoming a false cardboard component.
CARDBOARD_SEARCH_X = (0.30, 0.72)       # current board spans the centre-right of this camera view
CARDBOARD_SEARCH_Y = (0.00, 1.00)       # fractions of frame height
CARDBOARD_RECT_ASPECT = (0.12, 0.62)    # short side / long side of the strip
CARDBOARD_CENTER_X_RANGE = (0.42, 0.60) # expected strip centre, as frame-width fractions

# Temporal stabilisation for the board and detected centres.
BOARD_FRAME_HISTORY = 7          # frames averaged before each board update
BOARD_REFRESH_EVERY = 5          # update board geometry every 5 camera frames
BOARD_SMOOTHING_ALPHA = 0.30     # weight of the new board estimate (lower = steadier)
BOARD_MAX_MISSES = 12            # retain prior board through brief segmentation dropouts
HOLE_CENTER_ALPHA = 0.25         # EMA update for detected hole centres
HOLE_TRACK_MAX_JUMP_PX = 22.0
MIN_HOLE_AREA_PX = 110.0
MAX_HOLE_AREA_PX = 16000.0
HOLE_MORPH_KERNEL = 3
HOLE_CLOSE_KERNEL = 5
MIN_HOLE_TO_OBJECT_AREA_RATIO = 0.55
MAX_HOLE_TO_OBJECT_AREA_RATIO = 2.50

WINDOW_NAME = "Shape sorting - live detection / averaged targets"
MASK_WINDOW_NAME = "Colour masks (M to toggle)"


# ============================ UTILITIES ============================
def make_aruco_detector():
    """Create an OpenCV ArUco detector, supporting old and new OpenCV APIs."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is unavailable. Install an OpenCV contrib build, e.g. "
            "python3 -m pip install opencv-contrib-python."
        )

    aruco = cv2.aruco
    if not hasattr(aruco, ARUCO_DICT_NAME):
        raise RuntimeError(
            f"ArUco dictionary {ARUCO_DICT_NAME} is not available in this OpenCV build."
        )

    dictionary = aruco.getPredefinedDictionary(
        getattr(aruco, ARUCO_DICT_NAME)
    )

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
    else:
        detector = None

    return detector, dictionary, parameters


def detect_aruco_homography(frame, detector, dictionary, parameters):
    """Return (H, detected_marker_corners, marker_id) for the current frame.

    H maps camera pixels directly to robot base_link XY in metres. All four
    measured robot corner coordinates are used; no axis-alignment assumption
    or external calibration .npz file is used.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if detector is not None:
        corners, ids, _rejected = detector.detectMarkers(gray)
    else:
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )

    if ids is None or len(ids) == 0:
        return None, None, None

    flat_ids = ids.flatten()
    candidates = [
        i for i, marker_id in enumerate(flat_ids)
        if TARGET_MARKER_ID is None or int(marker_id) == TARGET_MARKER_ID
    ]
    if not candidates:
        return None, None, None

    chosen = max(
        candidates,
        key=lambda i: cv2.contourArea(
            corners[i].reshape(4, 2).astype(np.float32)
        ),
    )

    image_corners = corners[chosen].reshape(4, 2).astype(np.float32)
    marker_area = abs(float(cv2.contourArea(image_corners)))
    if marker_area < ARUCO_MIN_AREA_PX:
        return None, None, None

    if not np.all(np.isfinite(image_corners)):
        return None, None, None

    # OpenCV ArUco corner sequence: TL, TR, BR, BL.
    H = cv2.getPerspectiveTransform(
        image_corners,
        ROBOT_MARKER_CORNERS,
    )

    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return None, None, None
    if abs(float(np.linalg.det(H))) < 1e-12:
        return None, None, None

    return H, corners[chosen], int(flat_ids[chosen])


def open_camera():
    camera = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Cannot open camera {CAMERA_DEVICE}.")
    if USE_YUYV:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)
    for _ in range(5):
        camera.read()
    return camera


def cardboard_mask(hsv, lo, hi):
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))



def smooth_aruco_corners(detected_corners, marker_id, state):
    """EMA-smooth the four image corners while preserving the current marker ID."""
    if detected_corners is None or marker_id is None:
        state["misses"] = state.get("misses", 0) + 1
        if state["misses"] > 3:
            state["corners"] = None
            state["marker_id"] = None
        return None, None

    current = np.asarray(detected_corners, dtype=np.float64).reshape(4, 2)
    previous = state.get("corners")
    previous_id = state.get("marker_id")
    if previous is None or previous_id != marker_id:
        stable = current
    elif float(np.max(np.linalg.norm(current - previous, axis=1))) > ARUCO_MAX_CORNER_JUMP_PX:
        # A large jump is likely a genuine board/camera movement or a marker switch.
        stable = current
    else:
        alpha = float(ARUCO_CORNER_SMOOTHING_ALPHA)
        stable = (1.0 - alpha) * previous + alpha * current

    state["corners"] = stable.copy()
    state["marker_id"] = int(marker_id)
    state["misses"] = 0
    stable = stable.astype(np.float32)
    H = cv2.getPerspectiveTransform(stable, ROBOT_MARKER_CORNERS)
    if not np.all(np.isfinite(H)) or abs(float(np.linalg.det(H))) < 1e-12:
        return None, None
    return H, stable.reshape(1, 4, 2)


def detect_cardboard(frames, announce=True):
    """Detect only the task cardboard strip, independently of coloured blocks.

    A colour threshold alone is unsafe in this scene because the table/floor
    contains similar brown/grey pixels. Candidate contours are therefore
    searched only in the known central-left cardboard zone and must also have
    the long, narrow rectangular geometry of the cardboard strip.
    """
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    x1 = int(round(CARDBOARD_SEARCH_X[0] * w))
    x2 = int(round(CARDBOARD_SEARCH_X[1] * w))
    y1 = int(round(CARDBOARD_SEARCH_Y[0] * h))
    y2 = int(round(CARDBOARD_SEARCH_Y[1] * h))
    if x2 <= x1 or y2 <= y1:
        return None

    avg = np.mean(np.stack([f.astype(np.float32) for f in frames]), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(cv2.GaussianBlur(avg, (5, 5), 0), cv2.COLOR_BGR2HSV)
    hsv_roi = hsv[y1:y2, x1:x2]

    # Keep saturation above the near-grey background range. Two close value
    # ranges accommodate illumination changes without accepting almost every
    # neutral surface in the image.
    ranges = [
        (np.array((3, 28, 65), np.uint8), np.array((32, 190, 245), np.uint8)),
        (np.array((3, 35, 45), np.uint8), np.array((32, 220, 235), np.uint8)),
    ]
    candidates = []
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for lo_broad, hi_broad in ranges:
        local_mask = cv2.inRange(hsv_roi, lo_broad, hi_broad)
        local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel_close)
        local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, kernel_open)
        contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for local_contour in contours:
            area = float(cv2.contourArea(local_contour))
            if area < MIN_CARDBOARD_FRACTION * w * h:
                continue
            if area > 0.38 * w * h:
                continue

            # Convert the candidate back into full-frame coordinates.
            contour = local_contour.copy()
            contour[:, 0, 0] += x1
            contour[:, 0, 1] += y1
            (rw, rh) = cv2.minAreaRect(contour)[1]
            long_side = max(float(rw), float(rh), 1.0)
            short_side = max(min(float(rw), float(rh)), 1.0)
            aspect = short_side / long_side
            if not (CARDBOARD_RECT_ASPECT[0] <= aspect <= CARDBOARD_RECT_ASPECT[1]):
                continue

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            if hull_area <= 1.0:
                continue
            solidity = area / hull_area
            rect_area = max(float(rw) * float(rh), 1.0)
            rect_fill = area / rect_area
            approx = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
            vertices = len(approx)

            bx, by, bw, bh = cv2.boundingRect(contour)
            centre_x_fraction = (bx + 0.5 * bw) / float(w)
            # Do not accept similarly coloured floor/table patches at the left
            # edge. The cardboard strip is expected near the centre-left of
            # this camera view; the location prior is combined with geometry.
            if not (CARDBOARD_CENTER_X_RANGE[0] <= centre_x_fraction <= CARDBOARD_CENTER_X_RANGE[1]):
                continue
            if bx <= x1 + 2 or bx + bw >= x2 - 2:
                continue
            # Perspective, holes and printed markings lower the solidity and
            # rectangle fill of the real cardboard. Keep these checks tolerant
            # but still require a long, narrow, roughly rectangular component.
            if solidity < 0.42 or rect_fill < 0.35 or not (4 <= vertices <= 16):
                continue

            vertex_score = 1.0 if vertices <= 6 else max(0.55, 1.0 - 0.06 * (vertices - 6))
            aspect_score = 1.0 - abs(aspect - 0.38)
            centre_score = 1.0 - abs(centre_x_fraction - 0.51) / 0.12
            centre_score = max(0.05, centre_score)
            score = area * solidity * rect_fill * vertex_score * aspect_score * centre_score
            full_mask = np.zeros((h, w), np.uint8)
            full_mask[y1:y2, x1:x2] = local_mask
            candidates.append((score, area, contour, full_mask, solidity,
                               rect_fill, vertices, aspect))

    if not candidates:
        if announce:
            print("[CARDBOARD] NOT FOUND: no long rectangular tan component inside "
                  f"search zone x={x1}:{x2}, y={y1}:{y2}.", flush=True)
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, area, best, broad, solidity, rect_fill, vertices, aspect = candidates[0]

    component = np.zeros((h, w), np.uint8)
    cv2.drawContours(component, [best], -1, 255, cv2.FILLED)
    selected_pixels = (broad > 0) & (component > 0)
    if np.count_nonzero(selected_pixels) < MIN_CARDBOARD_FRACTION * w * h:
        if announce:
            print("[CARDBOARD] Candidate rejected: too few cardboard-coloured pixels.", flush=True)
        return None

    # Estimate the cardboard colour only from the selected strip, never from
    # the entire image. Then refine the strip boundary using only pixels close
    # to this measured colour. This rejects similarly coloured background
    # regions that contaminated the broad first-pass contour.
    med = np.median(hsv[selected_pixels], axis=0)
    tol = np.array([7, 35, 60], dtype=np.float32)
    lo = np.clip(med.astype(np.float32) - tol, 0, 255).astype(np.uint8)
    hi = np.clip(med.astype(np.float32) + tol, 0, 255).astype(np.uint8)

    tight = cv2.inRange(hsv, lo, hi)
    tight[:y1, :] = 0
    tight[y2:, :] = 0
    tight[:, :x1] = 0
    tight[:, x2:] = 0
    tight = cv2.morphologyEx(
        tight, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # The cardboard may be broken into several colour components by its four
    # cut-outs, shadows and printed markings. Combine only components whose
    # centres lie in the expected cardboard-strip zone. A floor/table patch
    # elsewhere in the image is not included in the fitted boundary.
    n_labels, component_labels, stats, centroids = cv2.connectedComponentsWithStats(
        tight, connectivity=8)
    point_groups = []
    for label in range(1, n_labels):
        comp_area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]
        cx_fraction = float(cx) / float(w)
        if comp_area < 50:
            continue
        if not (CARDBOARD_CENTER_X_RANGE[0] <= cx_fraction <= CARDBOARD_CENTER_X_RANGE[1]):
            continue
        yy, xx = np.where(component_labels == label)
        if len(xx):
            point_groups.append(np.column_stack((xx, yy)).astype(np.float32))

    if not point_groups:
        if announce:
            print("[CARDBOARD] Broad candidate found, but no matching cardboard-colour "
                  "components remained after background rejection.", flush=True)
        return None

    refined_points = np.concatenate(point_groups, axis=0).reshape(-1, 1, 2)
    refined_rect = cv2.minAreaRect(refined_points)
    refined_rw, refined_rh = refined_rect[1]
    refined_aspect = min(refined_rw, refined_rh) / max(refined_rw, refined_rh, 1.0)
    refined_cx = float(refined_rect[0][0]) / float(w)
    if (not (CARDBOARD_RECT_ASPECT[0] <= refined_aspect <= CARDBOARD_RECT_ASPECT[1])
            or not (CARDBOARD_CENTER_X_RANGE[0] <= refined_cx <= CARDBOARD_CENTER_X_RANGE[1])):
        if announce:
            print(f"[CARDBOARD] Refined candidate rejected: centre={refined_cx:.2f}, "
                  f"aspect={refined_aspect:.2f}.", flush=True)
        return None

    board_polygon = cv2.boxPoints(refined_rect).astype(np.float32)
    polygon_centre = np.mean(board_polygon, axis=0)
    board_polygon = polygon_centre + 0.98 * (board_polygon - polygon_centre)
    board_polygon_i = np.round(board_polygon).astype(np.int32)
    board_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(board_mask, board_polygon_i, 255)
    # Inset the hole-search region from the board edge to ensure that the
    # outer table/floor can never be considered a hole.
    hole_search_mask = cv2.erode(
        board_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), iterations=1
    )
    exclusion_mask = cv2.dilate(board_mask, np.ones((9, 9), np.uint8), 1)

    bx, by, bw, bh = cv2.boundingRect(board_polygon_i.reshape(-1, 1, 2))
    pad = 3
    bbox = (max(bx - pad, 0), max(by - pad, 0), min(bx + bw + pad, w), min(by + bh + pad, h))
    clipped_edges = []
    if bx <= 1: clipped_edges.append("left")
    if by <= 1: clipped_edges.append("top")
    if bx + bw >= w - 1: clipped_edges.append("right")
    if by + bh >= h - 1: clipped_edges.append("bottom")

    result = {
        "lo": lo, "hi": hi, "bbox": bbox, "contour": best,
        "board_polygon": board_polygon_i.reshape(-1, 1, 2),
        "board_mask": board_mask, "hole_search_mask": hole_search_mask,
        "mask": exclusion_mask, "clipped_edges": clipped_edges,
    }
    if announce:
        print(f"[CARDBOARD] FOUND task strip only: bbox={bbox}, area={area:.0f}px, "
              f"aspect={aspect:.2f}, solidity={solidity:.2f}, "
              f"rectangularity={rect_fill:.2f}, HSV median={med.astype(int).tolist()}",
              flush=True)
    return result



def _rebuild_strip_masks(strip, polygon):
    """Rebuild board masks/bounds from a smoothed four-corner polygon."""
    h, w = strip["mask"].shape[:2]
    polygon = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
    poly_i = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
    board_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(board_mask, poly_i, 255)
    hole_search_mask = cv2.erode(
        board_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=1,
    )
    exclusion_mask = cv2.dilate(board_mask, np.ones((9, 9), np.uint8), 1)
    bx, by, bw, bh = cv2.boundingRect(poly_i)
    pad = 3
    strip["bbox"] = (
        max(bx - pad, 0), max(by - pad, 0),
        min(bx + bw + pad, w), min(by + bh + pad, h),
    )
    strip["board_polygon"] = poly_i
    strip["board_mask"] = board_mask
    strip["hole_search_mask"] = hole_search_mask
    strip["mask"] = exclusion_mask
    strip["clipped_edges"] = []
    if bx <= 1: strip["clipped_edges"].append("left")
    if by <= 1: strip["clipped_edges"].append("top")
    if bx + bw >= w - 1: strip["clipped_edges"].append("right")
    if by + bh >= h - 1: strip["clipped_edges"].append("bottom")
    return strip


def smooth_cardboard_detection(previous, candidate):
    """EMA-smooth board corners and colour thresholds, matching corner order."""
    if candidate is None:
        return previous
    if previous is None:
        return candidate

    prev_poly = np.asarray(previous["board_polygon"], dtype=np.float64).reshape(4, 2)
    cand_poly = np.asarray(candidate["board_polygon"], dtype=np.float64).reshape(4, 2)
    # cv2.boxPoints may rotate its start corner; choose the permutation that
    # minimises correspondence error before smoothing.
    best = min(
        permutations(range(4)),
        key=lambda order: float(np.sum((cand_poly[list(order)] - prev_poly) ** 2)),
    )
    cand_poly = cand_poly[list(best)]
    alpha = float(BOARD_SMOOTHING_ALPHA)
    smoothed_poly = (1.0 - alpha) * prev_poly + alpha * cand_poly

    result = dict(candidate)
    for key in ("lo", "hi"):
        if key in previous and key in candidate:
            result[key] = np.clip(
                (1.0 - alpha) * np.asarray(previous[key], dtype=np.float32)
                + alpha * np.asarray(candidate[key], dtype=np.float32),
                0, 255,
            ).astype(np.uint8)
    result = _rebuild_strip_masks(result, smoothed_poly)
    return result


def smooth_hole_centres(holes, tracks):
    """Stabilise hole-centre coordinates without changing their contours/classes."""
    stable = {}
    for name, hole in holes.items():
        centre = np.asarray(hole["center_px"], dtype=np.float64)
        state = tracks.setdefault(name, {"center": None})
        old = state["center"]
        if old is None or float(np.linalg.norm(centre - old)) > HOLE_TRACK_MAX_JUMP_PX:
            state["center"] = centre.copy()
        else:
            a = float(HOLE_CENTER_ALPHA)
            state["center"] = (1.0 - a) * old + a * centre
        item = dict(hole)
        item["center_px"] = state["center"].copy()
        stable[name] = item
    return stable


def mask_to_roi(mask, roi):
    x1, y1, x2, y2 = roi
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def pixel_to_robot_xy(pixel_xy, H):
    p = np.asarray(pixel_xy, dtype=np.float32).reshape(1, 1, 2)
    mapped = cv2.perspectiveTransform(p, H)[0, 0]
    result = np.asarray(mapped, dtype=np.float64)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Invalid homography result for pixel point {pixel_xy}.")
    return result


# ============================ GEOMETRY ============================
def polygon_internal_angles(poly):
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    angles = []
    for i in range(len(pts)):
        a = pts[(i - 1) % len(pts)] - pts[i]
        b = pts[(i + 1) % len(pts)] - pts[i]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 1e-9:
            return []
        angles.append(float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1, 1)))))
    return angles


def polygon_side_ratio(poly):
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    lengths = np.array([np.linalg.norm(pts[(i + 1) % len(pts)] - pts[i]) for i in range(len(pts))])
    if np.min(lengths) <= 1e-6:
        return float("inf")
    return float(np.max(lengths) / np.min(lengths))


def fit_polygon(hull, perimeter, n):
    """Return an n-vertex polygon from the hull if a stable fit exists."""
    found = []
    for eps in np.linspace(0.010, 0.080, 36):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == n:
            found.append(approx.reshape(-1, 2))
        elif len(approx) < n:
            break
    if len(found) < 3:   # must be stable over several tolerance levels
        return None
    return found[len(found) // 2]


def fit_triangle(hull, perimeter):
    """Fit a triangle across a wider tolerance range than square/hexagon.

    The cardboard cut-outs have soft edges, perspective distortion and small
    mask gaps. Their convex hull can still have a stable triangular outline,
    but the old 0.01-0.08 epsilon window was too narrow for that contour.
    """
    found = []
    for eps in np.linspace(0.025, 0.180, 63):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 3:
            found.append(approx.reshape(-1, 2))
    if len(found) < 4:
        return None
    return found[len(found) // 2]


def radial_harmonics(contour, center, bins=128):
    """Amplitudes of the Fourier series of normalised radius vs angle."""
    pts = contour.reshape(-1, 2).astype(np.float64) - np.asarray(center, dtype=np.float64)
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    rad = np.hypot(pts[:, 0], pts[:, 1])
    order = np.argsort(ang)
    ang, rad = ang[order], rad[order]
    grid = np.linspace(-np.pi, np.pi, bins, endpoint=False)
    prof = np.interp(grid, ang, rad, period=2 * np.pi)
    mean = float(np.mean(prof))
    if mean <= 1e-6:
        return np.zeros(bins // 2 + 1)
    prof = prof / mean
    return 2.0 * np.abs(np.fft.rfft(prof)) / bins


def _template(n, rot):
    t = np.array([[100 * np.cos(rot + 2 * np.pi * k / n), 100 * np.sin(rot + 2 * np.pi * k / n)]
                  for k in range(n)], dtype=np.float32)
    return t.reshape(-1, 1, 2)


TEMPLATES = {
    "circle": _template(120, 0.0),
    "square": _template(4, np.pi / 4),
    "hexagon": _template(6, 0.0),
    "triangle": _template(3, -np.pi / 2),
}


def contour_metrics(contour):
    """Rotation-invariant measurements of a filled blob (needs dense contour)."""
    area = float(cv2.contourArea(contour))
    if area <= 20.0 or len(contour) < 8:
        return None
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    perim = float(cv2.arcLength(hull, True))
    if hull_area <= 1e-6 or perim <= 1e-6:
        return None
    mom = cv2.moments(hull)
    if abs(mom["m00"]) < 1e-9:
        return None
    center = (float(mom["m10"] / mom["m00"]), float(mom["m01"] / mom["m00"]))
    (rw, rh) = cv2.minAreaRect(hull)[1]
    long_s, short_s = max(rw, rh, 1e-6), max(min(rw, rh), 1e-6)
    ellipse_ratio = 0.0
    if len(hull) >= 5:
        (_, _), (ea, eb), _ = cv2.fitEllipse(hull)
        ellipse_ratio = float(min(ea, eb) / max(ea, eb, 1e-6))
    try:
        tri_area = float(cv2.minEnclosingTriangle(hull)[0])
    except cv2.error:
        tri_area = 0.0
    return {
        "area": area, "hull": hull, "hull_area": hull_area, "perimeter": perim,
        "solidity": area / hull_area,
        "circularity": float(4 * np.pi * area / (perim * perim)),
        "rect_fill": float(area / (long_s * short_s)),
        "aspect": float(short_s / long_s),
        "ellipse_ratio": ellipse_ratio,
        "tri_fill": float(area / tri_area) if tri_area > 1e-6 else 0.0,
        "center": center,
        "amp": radial_harmonics(contour, center),
        "bbox": cv2.boundingRect(hull),
        "polygon": None,
    }


def fit_hole_triangle(metrics):
    """Recognise a triangular cardboard cut-out from its convex-hull outline.

    The binary missing-cardboard mask can be ragged because reflections,
    shadows and texture interrupt the cut-out interior. This lowers contour
    solidity even when the physical opening is visibly triangular. For holes,
    require a stable 3-corner hull fit and plausible interior angles instead
    of requiring a high filled-contour solidity/fill ratio.
    """
    hull = metrics["hull"]
    perimeter = metrics["perimeter"]
    triangle = fit_triangle(hull, perimeter)
    if triangle is None:
        return None, float("inf")
    angles = polygon_internal_angles(triangle)
    if len(angles) != 3 or min(angles) < 18.0 or max(angles) > 150.0:
        return None, float("inf")
    # Score the hull, not the ragged source contour.
    score = float(cv2.matchShapes(
        hull, TEMPLATES["triangle"], cv2.CONTOURS_MATCH_I1, 0.0))
    if score > HOLE_TRIANGLE_MAX_MATCH_SCORE:
        return None, score
    return triangle, score


def fit_hole_square(metrics, contour_square_score):
    """Recognise a perspective-distorted square cut-out from its stable hull.

    This fallback is restricted to cardboard holes. It accepts a slightly
    incomplete/soft-edged square only when the convex hull has a stable
    four-corner fit, plausible corner angles, near-square side proportions,
    and a good square-template score. Coloured-object classification is
    unchanged.
    """
    solidity = float(metrics["solidity"])
    rect_fill = float(metrics["rect_fill"])
    aspect = float(metrics["aspect"])
    if solidity < 0.72:
        return None, float("inf")
    if rect_fill < 0.60 or rect_fill > 0.90:
        return None, float("inf")
    if aspect < 0.68:
        return None, float("inf")
    score = float(contour_square_score)
    if score > 0.10:
        return None, score

    hull = metrics["hull"]
    perimeter = float(metrics["perimeter"])
    quad = fit_polygon(hull, perimeter, 4)
    if quad is None:
        return None, score
    angles = polygon_internal_angles(quad)
    if len(angles) != 4 or min(angles) < 60.0 or max(angles) > 120.0:
        return None, score
    if polygon_side_ratio(quad) > 2.0:
        return None, score
    return quad, score


def fit_hole_hexagon(metrics, contour_hex_score):
    """Recognise the cardboard hexagonal cut-out when strict corner fitting fails.

    The detected hexagonal opening in this camera view has rounded/soft edges,
    so the strict six-vertex approximation can fail even when its contour has
    an excellent hexagon shape-match score. This fallback is hole-only; it is
    never used to relax coloured-block classification.

    The circularity, rotated-rectangle aspect, rectangular fill and solidity
    gates distinguish the observed hexagonal opening from the adjacent circle
    and square holes.
    """
    solidity = float(metrics["solidity"])
    circularity = float(metrics["circularity"])
    rect_fill = float(metrics["rect_fill"])
    aspect = float(metrics["aspect"])

    if solidity < 0.80:
        return None, float("inf")
    if not (0.88 <= circularity <= 0.955):
        return None, float("inf")
    if not (0.76 <= rect_fill <= 0.90):
        return None, float("inf")
    if aspect < 0.95:
        return None, float("inf")

    # Reuse the contour-vs-template score already calculated by shape_fit.
    # For the measured cut-out this score was 0.0006 (excellent match).
    score = float(contour_hex_score)
    if score > 0.02:
        return None, score

    # Return the best available six-corner approximation for drawing. The
    # signature gates above remain sufficient when the noisy edge has no
    # stable six-corner polygon at any single approximation tolerance.
    hull = metrics["hull"]
    perimeter = float(metrics["perimeter"])
    six_corner_fits = []
    for eps in np.linspace(0.01, 0.24, 80):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) != 6:
            continue
        poly = approx.reshape(-1, 2)
        angles = polygon_internal_angles(poly)
        if len(angles) != 6 or min(angles) < 45 or max(angles) > 165:
            continue
        if polygon_side_ratio(poly) > 4.0:
            continue
        six_corner_fits.append(poly)

    polygon = six_corner_fits[len(six_corner_fits) // 2] if six_corner_fits else None
    return polygon, score


def shape_fit(contour):
    """Return (metrics, fits). fits[geometry] = (ok, score, reason); lower score is better."""
    m = contour_metrics(contour)
    if m is None:
        return None, None
    amp = m["amp"]
    band = amp[3:9]                                  # harmonics 3..8
    dom = int(np.argmax(band)) + 3
    hull, perim = m["hull"], m["perimeter"]
    sol = m["solidity"]
    fits = {}

    def verdict(name, checks):
        failed = [txt for ok, txt in checks if not ok]
        score = float(cv2.matchShapes(contour, TEMPLATES[name], cv2.CONTOURS_MATCH_I1, 0.0))
        fits[name] = (not failed, score, failed[0] if failed else "")

    verdict("circle", [
        (sol >= SOLIDITY_MIN, f"solidity {sol:.2f}"),
        (m["circularity"] >= CIRCLE_MIN_CIRCULARITY, f"circ {m['circularity']:.2f}"),
        (m["ellipse_ratio"] >= CIRCLE_MIN_ELLIPSE_RATIO, f"ellipse {m['ellipse_ratio']:.2f}"),
        (float(np.max(band)) <= CIRCLE_MAX_HARMONIC, f"corners h{dom}={np.max(band):.3f}"),
    ])

    quad = fit_polygon(hull, perim, 4)
    qa = polygon_internal_angles(quad) if quad is not None else []
    verdict("square", [
        (sol >= SOLIDITY_MIN, f"solidity {sol:.2f}"),
        (quad is not None and len(qa) == 4, "no stable 4 corners"),
        (bool(qa) and 65 <= min(qa) and max(qa) <= 115, "corner angles"),
        (m["rect_fill"] >= SQUARE_MIN_RECT_FILL, f"rect fill {m['rect_fill']:.2f}"),
        (m["aspect"] >= SQUARE_MIN_ASPECT, f"aspect {m['aspect']:.2f}"),
        (dom in (4, 5) and amp[4] >= SQUARE_MIN_HARMONIC4, f"h4={amp[4]:.3f} dom={dom}"),
    ])
    if quad is not None and fits["square"][0]:
        m["polygon_square"] = quad

    hexa = fit_polygon(hull, perim, 6)
    ha = polygon_internal_angles(hexa) if hexa is not None else []
    verdict("hexagon", [
        (sol >= SOLIDITY_MIN, f"solidity {sol:.2f}"),
        (hexa is not None and len(ha) == 6, "no stable 6 corners"),
        (bool(ha) and 90 <= min(ha) and max(ha) <= 150, "corner angles"),
        (hexa is not None and polygon_side_ratio(hexa) <= HEX_MAX_SIDE_RATIO, "side ratio"),
        (HEX_RECT_FILL[0] <= m["rect_fill"] <= HEX_RECT_FILL[1], f"rect fill {m['rect_fill']:.2f}"),
        (HEX_CIRCULARITY[0] <= m["circularity"] <= HEX_CIRCULARITY[1], f"circ {m['circularity']:.2f}"),
        (amp[6] >= HEX_MIN_HARMONIC6 and amp[6] >= 0.60 * float(np.max(band)),
         f"h6={amp[6]:.3f} vs max={np.max(band):.3f}" ),
    ])
    if hexa is not None and fits["hexagon"][0]:
        m["polygon_hexagon"] = hexa

    tri = fit_triangle(hull, perim)
    ta = polygon_internal_angles(tri) if tri is not None else []
    triangle_score = float(cv2.matchShapes(
        contour, TEMPLATES["triangle"], cv2.CONTOURS_MATCH_I1, 0.0))
    verdict("triangle", [
        # Hole boundaries can contain small gaps caused by cardboard texture
        # and lighting. Use the stable convex-hull triangle and match score,
        # rather than demanding an almost-perfect filled silhouette.
        (sol >= TRIANGLE_SOLIDITY_MIN, f"solidity {sol:.2f}"),
        (tri is not None and len(ta) == 3, "no stable 3 corners"),
        (bool(ta) and 20 <= min(ta) and max(ta) <= 145, "corner angles"),
        (m["tri_fill"] >= TRI_MIN_FILL, f"tri fill {m['tri_fill']:.2f}"),
        (triangle_score <= TRIANGLE_MIN_MATCH_SCORE,
         f"triangle match {triangle_score:.3f}"),
    ])
    if tri is not None and fits["triangle"][0]:
        m["polygon_triangle"] = tri
    return m, fits


# ============================ MOTION GUARD ============================
def motion_fraction(prev_gray, gray, roi):
    if prev_gray is None:
        return 0.0
    x1, y1, x2, y2 = roi
    diff = cv2.absdiff(prev_gray[y1:y2, x1:x2], gray[y1:y2, x1:x2])
    moving = (diff > MOTION_PIXEL_DIFF).astype(np.uint8)
    moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return float(np.count_nonzero(moving)) / max(moving.size, 1)


# ===================== 3D OBJECT TOP-FACE ESTIMATION =====================
def fit_top_face_ellipse(contour, upper_fraction=0.62):
    """Estimate the projected centre of a circular top face on a 3-D object.

    A cylinder silhouette includes the top rim plus its side wall. Fitting one
    ellipse to a fixed 62% of that silhouette is brittle under perspective,
    partial occlusion and shadows. Try several upper-silhouette bands and retain
    the best geometrically plausible ellipse. This routine is used for the
    green circular object only.

    Returns (ellipse, centre, projected_area, aspect_ratio, residual).
    """
    pts = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
    if len(pts) < 12:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    if w < 8 or h < 8:
        return None

    candidates = []
    # Different fractions accommodate both a shallow disk and a taller cylinder.
    for fraction in (0.34, 0.42, 0.50, 0.58, float(upper_fraction), 0.70):
        cutoff = y + fraction * h
        upper = pts[pts[:, 1] <= cutoff]
        if len(upper) < 12:
            continue
        try:
            ellipse = cv2.fitEllipse(upper.reshape(-1, 1, 2))
        except cv2.error:
            continue

        (cx, cy), (axis_a, axis_b), angle_deg = ellipse
        major = max(float(axis_a), float(axis_b))
        minor = min(float(axis_a), float(axis_b))
        if major < 8.0 or minor < 5.0:
            continue
        aspect = minor / major
        if not (GREEN_MIN_ELLIPSE_RATIO <= aspect <= 1.02):
            continue
        # The fitted centre must lie near the top of the complete silhouette.
        if not (x - 0.12 * w <= cx <= x + 1.12 * w):
            continue
        if not (y - 0.12 * h <= cy <= y + 0.58 * h):
            continue

        theta = np.deg2rad(float(angle_deg))
        dx = upper[:, 0].astype(np.float64) - float(cx)
        dy = upper[:, 1].astype(np.float64) - float(cy)
        xr = dx * np.cos(theta) + dy * np.sin(theta)
        yr = -dx * np.sin(theta) + dy * np.cos(theta)
        radial = np.sqrt(
            (xr / max(float(axis_a) / 2.0, 1e-6)) ** 2 +
            (yr / max(float(axis_b) / 2.0, 1e-6)) ** 2
        )
        errors = np.abs(radial - 1.0)
        residual = float(np.median(errors))
        p80 = float(np.percentile(errors, 80))
        if residual > GREEN_MAX_ELLIPSE_RESIDUAL or p80 > 0.40:
            continue

        projected_area = float(np.pi * major * minor / 4.0)
        if projected_area < MIN_COLOR_AREA_PX * 0.40:
            continue
        # Prefer low residuals; discourage implausibly flattened ellipses.
        score = residual + 0.05 * max(0.0, 0.68 - aspect)
        candidates.append((score, ellipse, (float(cx), float(cy)),
                           projected_area, aspect, residual))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, ellipse, centre, area, aspect, residual = candidates[0]
    return ellipse, centre, area, aspect, residual


def extract_bright_top_face(hsv, raw_mask, contour):
    """Try to separate a brighter planar top face from a darker 3D side wall.

    This is used for square/hexagon/triangle prisms. It is deliberately
    conservative: it returns a face contour only when a coherent bright region
    reaches the top of the object's silhouette and occupies a meaningful area.
    """
    x, y, w, h = cv2.boundingRect(contour)
    if w < 10 or h < 10:
        return None
    local_contour = np.asarray(contour, dtype=np.int32).copy()
    local_contour[:, 0, 0] -= x
    local_contour[:, 0, 1] -= y
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [local_contour], -1, 255, cv2.FILLED)
    raw_roi = raw_mask[y:y + h, x:x + w]
    valid = (filled > 0) & (raw_roi > 0)
    values = hsv[y:y + h, x:x + w, 2][valid]
    if values.size < 100:
        return None
    p10, p90 = np.percentile(values, [10, 90])
    if p90 - p10 < 16:
        return None
    otsu, _ = cv2.threshold(values.reshape(-1, 1), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Keep Otsu from selecting only isolated specular highlights.
    otsu = max(float(otsu), float(np.percentile(values, 42)))
    v_roi = hsv[y:y + h, x:x + w, 2]
    bright = np.zeros((h, w), np.uint8)
    bright[valid & (v_roi > otsu)] = 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    full_area = max(float(cv2.contourArea(contour)), 1.0)
    options = []
    for candidate in cnts:
        a = float(cv2.contourArea(candidate))
        bx, by, bw, bh = cv2.boundingRect(candidate)
        # A top face must touch/approach the top of the full object, not a
        # bright patch halfway down its side wall.
        if by > max(3, int(0.18 * h)):
            continue
        if a < max(90.0, 0.14 * full_area) or a > 0.95 * full_area:
            continue
        shifted = candidate.copy()
        shifted[:, 0, 0] += x
        shifted[:, 0, 1] += y
        options.append((a, shifted))
    if not options:
        return None
    options.sort(key=lambda item: item[0], reverse=True)
    return options[0][1]


def fit_top_face_polygon(contour, expected):
    """Perspective-tolerant polygon fit for the visible cap of a 3D prism."""
    n_by_shape = {"square": 4, "hexagon": 6, "triangle": 3}
    n = n_by_shape.get(expected)
    if n is None:
        return None, float("inf")
    hull = cv2.convexHull(contour)
    perimeter = float(cv2.arcLength(hull, True))
    if perimeter <= 1e-6:
        return None, float("inf")
    found = []
    for eps in np.linspace(0.012, 0.20, 76):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == n:
            found.append(approx.reshape(-1, 2))
    if len(found) < 3:
        return None, float("inf")
    poly = found[len(found) // 2]
    angles = polygon_internal_angles(poly)
    if len(angles) != n:
        return None, float("inf")
    side_ratio = polygon_side_ratio(poly)
    score = float(cv2.matchShapes(hull, TEMPLATES[expected],
                                   cv2.CONTOURS_MATCH_I1, 0.0))
    if expected == "square":
        valid = min(angles) >= 22 and max(angles) <= 158 and side_ratio <= 5.0 and score <= 0.85
    elif expected == "hexagon":
        valid = min(angles) >= 35 and max(angles) <= 165 and side_ratio <= 4.5 and score <= 0.42
    else:
        valid = min(angles) >= 12 and max(angles) <= 158 and score <= 0.85
    return (poly, score) if valid else (None, score)


def analyse_top_face(name, hsv, raw_mask, contour, full_metrics, full_fits):
    """Return a top-face based centre/area when a 3D shape cap can be isolated."""
    expected = EXPECTED_GEOMETRY[name]
    if expected == "circle":
        fitted = fit_top_face_ellipse(contour)
        if fitted is not None:
            ellipse, centre, area, aspect, residual = fitted
            return {
                "center": centre, "area": area, "contour": None,
                "ellipse": ellipse, "metrics": full_metrics,
                "score": residual, "mode": "upper-arc ellipse",
                "note": f"top ellipse AR={aspect:.2f} residual={residual:.3f}",
            }

    face_contour = extract_bright_top_face(hsv, raw_mask, contour)
    if face_contour is not None:
        face_metrics, face_fits = shape_fit(face_contour)
        if face_metrics is not None:
            ok, score, _ = face_fits[expected]
            polygon = None
            if not ok and expected in ("square", "hexagon", "triangle"):
                polygon, polygon_score = fit_top_face_polygon(face_contour, expected)
                if polygon is not None:
                    ok, score = True, polygon_score
                    face_metrics["polygon_" + expected] = polygon
            if ok:
                return {
                    "center": face_metrics["center"],
                    "area": float(cv2.contourArea(face_contour)),
                    "contour": face_contour, "ellipse": None,
                    "metrics": face_metrics, "score": float(score),
                    "mode": "bright top-face contour",
                    "note": "top face separated from side wall",
                }
    # Green-circle fallback: accept a coherent elliptical/circular silhouette
    # even when the full 3-D side-wall contour is less circular than its top.
    # This fallback remains constrained by broad green HSV, fill, hue and area.
    if expected == "circle" and full_metrics is not None:
        m = full_metrics
        green_circle_ok = (
            m["solidity"] >= GREEN_MIN_SOLIDITY and
            m["circularity"] >= GREEN_MIN_CIRCULARITY and
            m["ellipse_ratio"] >= GREEN_MIN_ELLIPSE_RATIO and
            float(np.max(m["amp"][3:9])) <= 0.11
        )
        if green_circle_ok:
            return {
                "center": m["center"], "area": float(cv2.contourArea(contour)),
                "contour": contour, "ellipse": None, "metrics": m,
                "score": float(1.0 - m["circularity"]),
                "mode": "green circular silhouette fallback",
                "note": f"circularity={m['circularity']:.2f} solidity={m['solidity']:.2f}",
            }

    # Flat/2D objects without a visible separate side wall retain the original
    # whole-contour result. For 3D objects, the top-face path above takes priority.
    if full_metrics is not None and full_fits[expected][0]:
        return {
            "center": full_metrics["center"],
            "area": float(cv2.contourArea(contour)),
            "contour": contour, "ellipse": None,
            "metrics": full_metrics, "score": float(full_fits[expected][1]),
            "mode": "whole contour fallback",
            "note": "no separate top face found",
        }
    return None


# ======================= TEMPORAL OBJECT TRACKING =======================
def new_object_track():
    return {"center": None, "hits": 0, "misses": 0, "last": None}


def confirm_stationary_objects(objects, object_debug, tracks):
    """Only publish a colour/shape candidate after it stays in place for N frames."""
    confirmed = {}
    for name in SHAPE_ORDER:
        state = tracks.setdefault(name, new_object_track())
        candidate = objects.get(name)

        if candidate is None:
            state["misses"] += 1
            if state["hits"] >= OBJECT_CONFIRM_FRAMES and state["misses"] <= OBJECT_TRACK_MAX_MISSES:
                if state["last"] is not None:
                    confirmed[name] = state["last"]
            elif state["misses"] > OBJECT_TRACK_MAX_MISSES:
                tracks[name] = new_object_track()
            continue

        center = np.asarray(candidate["center_px"], dtype=np.float64)
        if state["center"] is None:
            state["center"] = center
            state["hits"] = 1
        elif float(np.linalg.norm(center - state["center"])) <= OBJECT_TRACK_MAX_JUMP_PX:
            state["center"] = 0.65 * state["center"] + 0.35 * center
            state["hits"] += 1
        else:
            state["center"] = center
            state["hits"] = 1

        state["misses"] = 0
        # Publish the EMA-smoothed centre. Previously the tracker updated its
        # internal centre but still returned the raw per-frame centre, so the
        # overlay and coordinate samples continued to jump.
        smoothed_candidate = dict(candidate)
        smoothed_candidate["center_px"] = state["center"].copy()
        state["last"] = smoothed_candidate

        if state["hits"] >= OBJECT_CONFIRM_FRAMES:
            confirmed[name] = smoothed_candidate
        else:
            for entry in object_debug.get(name, []):
                if entry["contour"] is candidate["contour"]:
                    entry["reason"] = (
                        f"waiting for stable position ({state['hits']}/{OBJECT_CONFIRM_FRAMES})"
                    )
                    break

    return confirmed


# ============================ OBJECT DETECTION ============================
def find_coloured_objects(frame, roi, exclude_mask=None):
    """Detect at most one object per colour/shape across the full camera ROI.

    The cardboard is excluded with exclude_mask; there is no fixed global
    object-workspace polygon. Colour, geometry, dimensions and tracking are
    used to reject unrelated scene regions.
    """
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    objects = {}
    debug = {name: [] for name in SHAPE_ORDER}
    object_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    raw_masks = np.zeros(frame.shape, dtype=np.uint8)
    any_colour = np.zeros(frame.shape[:2], dtype=np.uint8)
    open_k = np.ones((COLOR_OPEN_KERNEL, COLOR_OPEN_KERNEL), np.uint8)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (COLOR_CLOSE_KERNEL,) * 2)
    green_open_k = np.ones((GREEN_OPEN_KERNEL, GREEN_OPEN_KERNEL), np.uint8)
    green_close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GREEN_CLOSE_KERNEL,) * 2)
    roi_area = max((roi[2] - roi[0]) * (roi[3] - roi[1]), 1)
    max_area = min(MAX_COLOR_AREA_PX, MAX_ROI_FRACTION * roi_area)
    for name in SHAPE_ORDER:
        lower, upper = COLOR_RANGES[name]
        raw = cv2.inRange(hsv, lower, upper)
        raw = mask_to_roi(raw, roi)
        if exclude_mask is not None:
            raw[exclude_mask > 0] = 0
        raw_masks[raw > 0] = DISPLAY_COLOURS[name]
        if name == "green_circle":
            # Preserve smaller green rim segments, then join small lighting gaps.
            mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, green_open_k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, green_close_k)
        else:
            mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        any_colour |= mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        accepted = []

        for contour in contours:
            full_area = float(cv2.contourArea(contour))
            if full_area < max(MIN_COLOR_AREA_PX, OBJECT_MIN_AREA_PX[name]):
                continue
            full_metrics, full_fits = shape_fit(contour)
            entry = {
                "contour": contour, "area": full_area,
                "metrics": full_metrics, "geometry": "unknown", "reason": "",
                "face_contour": None, "face_ellipse": None, "face_mode": "",
            }
            debug[name].append(entry)
            if full_metrics is None:
                entry["reason"] = "degenerate"
                continue
            class_max_area = min(max_area, OBJECT_MAX_AREA_PX[name])
            if full_area > class_max_area:
                entry["reason"] = f"too large ({full_area:.0f}px > {class_max_area:.0f}px)"
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if max(w, h) > OBJECT_MAX_SIDE_PX[name]:
                entry["reason"] = f"too large in image ({max(w, h)}px side)"
                continue
            if min(w, h) < 12:
                entry["reason"] = "too small in image"
                continue
            if x <= roi[0] + 2 or y <= roi[1] + 2 or x + w >= roi[2] - 2 or y + h >= roi[3] - 2:
                entry["reason"] = "touches ROI edge"
                continue

            # Check colour coverage/hue on the complete silhouette, but use
            # top-face geometry for classification and the robot pick centre.
            filled = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
            fill = float(np.count_nonzero(raw[filled > 0])) / max(np.count_nonzero(filled), 1)
            min_fill = GREEN_MIN_COLOR_FILL if name == "green_circle" else MIN_FILL_OF_MASK
            if fill < min_fill:
                entry["reason"] = f"colour fill {fill:.2f} < {min_fill:.2f}"
                continue
            inner_kernel = np.ones((3, 3), np.uint8) if name == "green_circle" else np.ones((5, 5), np.uint8)
            inner = cv2.erode(filled, inner_kernel)
            hues = hsv[..., 0][inner > 0].astype(np.float32)
            max_hue_std = GREEN_MAX_INNER_HUE_STD if name == "green_circle" else MAX_INNER_HUE_STD
            if hues.size >= 20 and float(np.std(hues)) > max_hue_std:
                entry["reason"] = f"hue spread {np.std(hues):.1f} > {max_hue_std:.1f}"
                continue

            top = analyse_top_face(name, hsv, raw, contour, full_metrics, full_fits)
            if top is None:
                why = full_fits[EXPECTED_GEOMETRY[name]][2]
                entry["reason"] = f"top face not verified: {why}"
                continue

            entry["geometry"] = EXPECTED_GEOMETRY[name]
            entry["reason"] = ""
            entry["face_contour"] = top["contour"]
            entry["face_ellipse"] = top["ellipse"]
            entry["face_mode"] = top["mode"]
            entry["face_note"] = top["note"]
            entry["metrics"] = top["metrics"]
            # score drives best-shape selection; top ellipse residual is the
            # score for cylinders, shape-match score for polygonal caps.
            accepted.append({
                "full_area": full_area, "contour": contour,
                "center": top["center"], "area": top["area"],
                "metrics": top["metrics"], "score": float(top["score"]),
                "entry": entry, "face_contour": top["contour"],
                "face_ellipse": top["ellipse"], "face_mode": top["mode"],
            })

        if not accepted:
            continue
        accepted.sort(key=lambda item: item["score"])
        if len(accepted) > 1:
            for item in accepted:
                item["entry"]["reason"] = (
                    "multiple same-colour, same-shape candidates; refusing to guess"
                )
            continue
        accepted = accepted[:MAX_OBJECTS_PER_CLASS]
        best = accepted[0]
        objects[name] = {
            "center_px": best["center"],
            "area_px": best["area"],
            "contour": best["contour"],             # complete silhouette for exclusion
            "face_contour": best["face_contour"],
            "face_ellipse": best["face_ellipse"],
            "face_mode": best["face_mode"],
            "metrics": best["metrics"],
        }

    # Cross-class uniqueness: two labels must not describe the same physical
    # coloured blob. Compare filled-contour overlap and reject both on ambiguity.
    overlapping_names = set()
    if len(objects) > 1:
        names = list(objects)
        contour_masks = {}
        for name in names:
            cm = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(cm, [objects[name]["contour"]], -1, 255, cv2.FILLED)
            contour_masks[name] = cm > 0
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                a = contour_masks[name_a]
                b = contour_masks[name_b]
                intersection = int(np.count_nonzero(a & b))
                union = int(np.count_nonzero(a | b))
                iou = intersection / max(union, 1)
                if iou >= MAX_CROSS_CLASS_OBJECT_OVERLAP:
                    # Green circle is the priority class. If it overlaps a
                    # second colour label, retain green and reject the other;
                    # all non-green overlaps remain ambiguous and reject both.
                    if name_a == "green_circle" and name_b != "green_circle":
                        overlapping_names.add(name_b)
                    elif name_b == "green_circle" and name_a != "green_circle":
                        overlapping_names.add(name_a)
                    else:
                        overlapping_names.update((name_a, name_b))

    for name in overlapping_names:
        for entry in debug[name]:
            if entry["contour"] is objects[name]["contour"]:
                entry["reason"] = "overlaps another colour class; duplicate object rejected"
        del objects[name]

    if len(objects) > len(SHAPE_ORDER):
        raise RuntimeError("Coloured-object detector exceeded configured class cardinality.")

    # Cross-object size sanity is now based on visible top-face areas rather
    # than the larger silhouette areas that include side walls/extensions.
    if len(objects) >= 3:
        med = float(np.median([o["area_px"] for o in objects.values()]))
        for name in list(objects):
            ratio = objects[name]["area_px"] / max(med, 1.0)
            if ratio > SIZE_RATIO_MAX or ratio < 1.0 / SIZE_RATIO_MAX:
                for e in debug[name]:
                    if e["contour"] is objects[name]["contour"]:
                        e["reason"] = "top-face size outlier"
                del objects[name]

    # The public object map is keyed by the four configured colour/shape names,
    # and every key is limited to one selected physical object.
    if len(objects) > len(SHAPE_ORDER) or len(set(objects.keys())) != len(objects):
        raise RuntimeError("Coloured-object detector violated one-object-per-class rule.")

    object_mask |= any_colour
    for obj in objects.values():
        cv2.drawContours(object_mask, [obj["contour"]], -1, 255, cv2.FILLED)
    object_mask = cv2.dilate(object_mask, np.ones((7, 7), np.uint8), 1)
    return objects, object_mask, debug, raw_masks


# ============================ HOLE DETECTION ============================
def find_holes(frame, strip, roi):
    """Find cut-outs as non-cardboard regions enclosed by the board footprint.

    At most one hole per configured geometry is selected.

    This is independent of all coloured-object detections. The clean fitted
    board rectangle defines the search interior; pixels that do not match the
    cardboard material colour become hole candidates. Components connected to
    the inset board boundary are rejected as background leakage.
    """
    full_mask = np.zeros(frame.shape[:2], np.uint8)
    if strip is None:
        return {}, full_mask, []

    x1, y1, x2, y2 = strip["bbox"]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_BGR2HSV)
    material = cardboard_mask(hsv[y1:y2, x1:x2], strip["lo"], strip["hi"])
    inside = strip["hole_search_mask"][y1:y2, x1:x2]
    material[inside == 0] = 0

    # The difference between the fitted board interior and the detected tan
    # material is the cut-out/background mask. Small defects are removed.
    missing = np.zeros_like(material)
    missing[(inside > 0) & (material == 0)] = 255
    missing = cv2.morphologyEx(missing, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    missing = cv2.morphologyEx(missing, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    full_mask[y1:y2, x1:x2] = missing

    contours, _ = cv2.findContours(missing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
                                    offset=(x1, y1))
    candidates = []
    max_area = min(MAX_HOLE_AREA_PX, 0.10 * (roi[2] - roi[0]) * (roi[3] - roi[1]))
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_HOLE_AREA_PX or area > max_area:
            continue
        # Contours are generated only from missing material inside the inset
        # cardboard polygon. Geometry/area tests below reject residual border
        # gaps and background fragments.
        metrics, fits = shape_fit(contour)
        if metrics is None:
            continue

        # Hole-only fallback: the non-cardboard mask inside a cut-out can be
        # fragmented by shadows/reflections, lowering raw contour solidity.
        # The convex hull can still provide a stable, three-corner triangle.
        # This does not relax coloured-block classification.
        hole_triangle, hole_triangle_score = fit_hole_triangle(metrics)
        if hole_triangle is not None and not fits["triangle"][0]:
            fits["triangle"] = (True, hole_triangle_score, "")
            metrics["polygon_triangle"] = hole_triangle
            metrics["hole_triangle_hull_fit"] = True

        # Cardboard-only fallback for the perspective-distorted square opening.
        # The screenshot's square cut-out has a stable four-corner hull but its
        # raw filled contour solidity is reduced by reflections/shadows.
        hole_square, hole_square_score = fit_hole_square(metrics, fits["square"][1])
        if hole_square is not None and not fits["square"][0]:
            fits["square"] = (True, hole_square_score, "")
            metrics["polygon_square"] = hole_square
            metrics["hole_square_hull_fit"] = True

        # Cardboard-only fallback for the rounded/soft-edged hexagonal opening.
        hole_hexagon, hole_hexagon_score = fit_hole_hexagon(metrics, fits["hexagon"][1])
        if np.isfinite(hole_hexagon_score):
            if not fits["hexagon"][0] or hole_hexagon_score < fits["hexagon"][1]:
                fits["hexagon"] = (True, hole_hexagon_score, "")
            if hole_hexagon is not None:
                metrics["polygon_hexagon"] = hole_hexagon
            metrics["hole_hexagon_signature_fit"] = True
            metrics["hole_hexagon_signature_score"] = hole_hexagon_score

        passing = [g for g in GEOMETRIES if fits[g][0]]
        reason = "" if passing else min((fits[g][2] for g in GEOMETRIES), key=len)

        # If the dedicated cardboard-hexagon signature passed, it is the
        # authoritative class for this hole. Do not let a simultaneous square
        # template pass relabel this same candidate as a square.
        if metrics.get("hole_hexagon_signature_fit", False):
            geometry = "hexagon"
        else:
            geometry = min(passing, key=lambda g: fits[g][1]) if passing else "unknown"
        candidates.append({
            "center_px": metrics["center"], "area_px": area,
            "geometry": geometry, "fits": fits, "contour": contour,
            "metrics": metrics, "ok": bool(passing), "reason": reason,
        })


    # HARD-CODED TASK-BOARD INVENTORY / LAYOUT:
    # There is exactly one circle, one hexagon, one square and one triangle.
    # The circular cut-out is at one end of the strip. The cut-out immediately
    # adjacent to the circle along the board's long axis is ALWAYS the hexagon.
    # This positional rule overrides strict hexagon contour tests and any
    # square-template pass for that same hole. The remaining two holes are
    # assigned by their own square/triangle fits.
    # No candidate can be reused for two shape classes.

    selected = {}
    used_ids = set()
    reserved_ids = set()

    # Derive the cardboard strip's longitudinal axis from its fitted polygon.
    board_pts = np.asarray(strip["board_polygon"], dtype=np.float64).reshape(-1, 2)
    board_centre = np.mean(board_pts, axis=0)
    centred = board_pts - board_centre
    covariance = centred.T @ centred
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    # Fix the sign for reproducible diagnostics; distances below are absolute.
    if long_axis[1] < 0:
        long_axis = -long_axis

    def longitudinal_position(candidate):
        centre = np.asarray(candidate["center_px"], dtype=np.float64)
        return float(np.dot(centre - board_centre, long_axis))

    # 1) Find the unique circle using the circle-specific contour checks.
    circle_options = [
        c for c in candidates
        if c["fits"]["circle"][0]
        and c["metrics"]["circularity"] >= 0.90
        and c["metrics"]["ellipse_ratio"] >= CIRCLE_MIN_ELLIPSE_RATIO
    ]
    circle_options.sort(key=lambda c: float(c["fits"]["circle"][1]))
    circle_candidate = circle_options[0] if circle_options else None
    # Any contour with a strong circle fit is reserved from square/triangle
    # reassignment, even if it is not the winning circle candidate.
    reserved_ids.update(id(c) for c in circle_options)

    if circle_candidate is not None:
        circle_candidate["geometry"] = "circle"
        circle_candidate["ok"] = True
        circle_candidate["reason"] = ""
        circle_candidate["layout_rule"] = "circle end of cardboard strip"
        selected["green_circle"] = circle_candidate
        used_ids.add(id(circle_candidate))

        # 2) The nearest distinct hole along the board's long axis is the
        # hexagon slot. Here the known physical board layout is the deciding
        # constraint: the hexagonal cut-out is immediately adjacent to the
        # circular cut-out. Do NOT require a stable six-corner contour, strict
        # circularity, or a hexagon-template score. The real opening has soft,
        # rounded edges and can look almost circular/square in this camera view.
        # The candidate is already constrained to be a sufficiently large
        # non-cardboard component enclosed by the fitted cardboard footprint.
        other_candidates = [
            c for c in candidates
            if id(c) not in used_ids and id(c) not in reserved_ids
        ]
        other_candidates.sort(
            key=lambda c: abs(longitudinal_position(c) - longitudinal_position(circle_candidate))
        )
        adjacent = other_candidates[0] if other_candidates else None

        if adjacent is not None:
            reserved_ids.add(id(adjacent))
            raw_hex_score = float(adjacent["fits"]["hexagon"][1])
            adjacent["geometry"] = "hexagon"
            adjacent["ok"] = True
            adjacent["reason"] = ""
            adjacent["fits"]["hexagon"] = (
                True, raw_hex_score,
                "assigned by fixed board layout: immediately adjacent to circle"
            )
            adjacent["layout_rule"] = "immediately adjacent to circle = hexagon"
            adjacent["metrics"]["layout_hexagon_override"] = True
            adjacent["metrics"]["hexagon_shape_score_before_layout"] = raw_hex_score
            selected["orange_hexagon"] = adjacent
            used_ids.add(id(adjacent))
            print(
                "[HOLE LAYOUT] Assigned orange_hexagon to the nearest distinct "
                f"hole next to green_circle (raw shape score={raw_hex_score:.4f}; "
                "shape gate bypassed by the fixed layout rule).",
                flush=True,
            )
        else:
            print(
                "[HOLE LAYOUT] Circle detected, but no second hole candidate was "
                "available beside it; orange_hexagon remains unassigned.",
                flush=True,
            )

    # 3) Assign the one remaining square candidate, excluding circle and hexagon.
    square_options = [
        c for c in candidates
        if id(c) not in reserved_ids and id(c) not in used_ids
        and c["fits"]["square"][0]
    ]
    square_options.sort(key=lambda c: float(c["fits"]["square"][1]))
    if square_options:
        square_candidate = square_options[0]
        square_candidate["geometry"] = "square"
        square_candidate["ok"] = True
        square_candidate["reason"] = ""
        square_candidate["layout_rule"] = "remaining square-shaped cardboard opening"
        selected["blue_square"] = square_candidate
        used_ids.add(id(square_candidate))
        reserved_ids.update(id(c) for c in square_options)

    # 4) Assign the one remaining triangle candidate.
    triangle_options = [
        c for c in candidates
        if id(c) not in reserved_ids and id(c) not in used_ids
        and c["fits"]["triangle"][0]
    ]
    triangle_options.sort(key=lambda c: float(c["fits"]["triangle"][1]))
    if triangle_options:
        triangle_candidate = triangle_options[0]
        triangle_candidate["geometry"] = "triangle"
        triangle_candidate["ok"] = True
        triangle_candidate["reason"] = ""
        triangle_candidate["layout_rule"] = "remaining triangular cardboard opening"
        selected["yellow_triangle"] = triangle_candidate
        used_ids.add(id(triangle_candidate))

    # Reject all extras and guarantee that each physical hole is assigned once.
    for candidate in candidates:
        if id(candidate) not in used_ids:
            candidate["ok"] = False
            candidate["reason"] = "not one of the four unique cardboard holes"

    if len(selected) > 4 or len({id(c) for c in selected.values()}) != len(selected):
        raise RuntimeError("Cardboard hole assignment violated the four-unique-hole rule.")

    # One-time diagnostics for cardboard-hole geometry classification.
    if candidates and len(candidates) >= 3 and not getattr(find_holes, "_diagnostics_printed", False):
        print("\\n========== HOLE GEOMETRY DIAGNOSTICS ==========", flush=True)

        for i, c in enumerate(candidates):
            m = c["metrics"]
            print(
                f"Candidate {i + 1}: centre={np.round(c['center_px'], 1)}, "
                f"area={c['area_px']:.0f}, assigned={c['geometry']}, "
                f"solidity={m['solidity']:.3f}, "
                f"circularity={m['circularity']:.3f}, "
                f"rect_fill={m['rect_fill']:.3f}, "
                f"aspect={m['aspect']:.3f}, "
                f"tri_fill={m['tri_fill']:.3f}, "
                f"h6={m['amp'][6]:.3f}, "
                f"triangle_hull_fit={m.get('hole_triangle_hull_fit', False)}, "
                f"hexagon_hull_fit={m.get('hole_hexagon_signature_fit', False)}, "
                f"layout_rule={c.get('layout_rule', 'unassigned')}",
                flush=True
            )

            for geometry in GEOMETRIES:
                ok, score, reason = c["fits"][geometry]
                print(
                    f"    {geometry}: {'PASS' if ok else 'FAIL'} "
                    f"score={score:.4f} "
                    f"{'' if ok else 'reason=' + reason}",
                    flush=True
                )

        print("Selected cardboard holes:", flush=True)
        for name, hole in selected.items():
            print(
                f"    {name}: centre={np.round(hole['center_px'], 1)}, "
                f"candidate_geometry={hole['geometry']}",
                flush=True
            )

        print("================================================\\n", flush=True)
        find_holes._diagnostics_printed = True



    return selected, full_mask, candidates


def hole_geometries(candidates):
    """Which shapes of cut-out are visible, independent of any block."""
    return {c["geometry"] for c in candidates if c["ok"]}


# ============================ DRAWING ============================
def draw_cross(image, point, colour, label, radius=10):
    x, y = int(round(point[0])), int(round(point[1]))
    cv2.drawMarker(image, (x, y), colour, cv2.MARKER_CROSS, radius, 2)
    cv2.putText(image, label, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 2, cv2.LINE_AA)


def _put_readable_text(image, text, origin, scale=0.38, colour=(255, 255, 255), thickness=1):
    x, y = origin
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thickness, cv2.LINE_AA)


def draw_overlay(frame, roi, strip, objects, object_debug, holes, hole_candidates,
                 sample_counts, elapsed, averaged=None, averaging_done=False, motion=0.0):
    """Draw a low-clutter preview using only accepted, temporally smoothed centres."""
    display = frame.copy()

    # Single board outline only. Avoid drawing both polygon and axis-aligned bbox.
    if strip is not None:
        cv2.polylines(display, [strip["board_polygon"]], True,
                      (0, 220, 255), 2, cv2.LINE_AA)
        bx, by, _, _ = strip["bbox"]
        _put_readable_text(display, "BOARD", (max(6, bx), max(18, by - 5)),
                           0.34, (0, 220, 255), 1)
    else:
        _put_readable_text(display, "BOARD NOT DETECTED", (8, 165),
                           0.42, (0, 0, 255), 2)

    # Draw only the selected holes; do not draw the same hole once as a
    # candidate and again as a selected result.
    for name, hole in holes.items():
        cv2.drawContours(display, [hole["contour"]], -1, (255, 0, 255), 1, cv2.LINE_AA)
        draw_cross(display, hole["center_px"], (255, 255, 255),
                   f"HOLE {name.split('_')[0]}", 7)

    # Draw only accepted object detections. Rejected contour/reason labels are
    # available through the M colour-mask view and are omitted from the main view.
    for name, obj in objects.items():
        colour = DISPLAY_COLOURS[name]
        cand = next((c for c in object_debug.get(name, [])
                     if c["contour"] is obj["contour"]), None)
        if cand is not None:
            if cand.get("face_ellipse") is not None:
                cv2.ellipse(display, cand["face_ellipse"], colour, 2, cv2.LINE_AA)
            elif cand.get("face_contour") is not None:
                cv2.drawContours(display, [cand["face_contour"]], -1, colour, 2, cv2.LINE_AA)
            else:
                cv2.drawContours(display, [obj["contour"]], -1, colour, 2, cv2.LINE_AA)
        draw_cross(display, obj["center_px"], colour,
                   f"PICK {name.split('_')[0]}", 8)
        if name in holes:
            a = tuple(map(int, np.round(obj["center_px"])))
            b = tuple(map(int, np.round(holes[name]["center_px"])))
            cv2.line(display, a, b, colour, 1, cv2.LINE_AA)

    # Keep frozen averaged targets visible as small dots; live centres above
    # are smoothed and may continue moving slightly with the objects.
    if averaged:
        for name, item in averaged.items():
            for key, colour in (("pick_pixel", (0, 0, 255)),
                                ("place_pixel", (255, 255, 255))):
                point = tuple(map(int, np.round(item[key])))
                cv2.circle(display, point, 4, colour, 1, cv2.LINE_AA)

    if motion > MOTION_FRACTION_MAX:
        status = "MOTION DETECTED - HOLD STILL"
        status_colour = (0, 0, 255)
    elif not averaging_done:
        status = f"AVERAGING {elapsed:.1f}/{AVERAGING_SECONDS:.1f}s"
        status_colour = (0, 255, 255)
    else:
        status = "TARGETS FROZEN - LIVE PREVIEW"
        status_colour = (0, 255, 255)

    valid_hole_count = len(holes)
    pair_names = [name for name in SHAPE_ORDER if name in objects and name in holes]
    _put_readable_text(display, status, (8, 20), 0.48, status_colour, 2)
    _put_readable_text(display, "Q/ESC: accept | X: abort | M: colour masks",
                       (8, 42), 0.34, (255, 255, 255), 1)
    _put_readable_text(
        display,
        f"BLOCKS {len(objects)}/4   HOLES {valid_hole_count}/4   PAIRS {len(pair_names)}",
        (8, 64), 0.38, (255, 255, 255), 1,
    )

    short_names = {
        "green_circle": "GREEN CIRCLE",
        "blue_square": "BLUE SQUARE",
        "orange_hexagon": "ORANGE HEX",
        "yellow_triangle": "YELLOW TRI",
    }
    for i, name in enumerate(SHAPE_ORDER):
        block_ok = name in objects
        hole_ok = name in holes
        samples = int(sample_counts.get(name, 0))
        line = (f"{short_names[name]}  block:{'OK' if block_ok else '--'}  "
                f"hole:{'OK' if hole_ok else '--'}  samples:{samples}")
        _put_readable_text(display, line, (8, 88 + 19 * i), 0.32,
                           DISPLAY_COLOURS[name], 1)

    return display


# ============================ AVERAGING / OUTPUT ============================
def robust_mean(points):
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
        raise ValueError("No coordinate samples to average.")
    median = np.median(arr, axis=0)
    abs_dev = np.abs(arr - median)
    mad = np.median(abs_dev, axis=0)
    limit = np.maximum(3.5 * 1.4826 * mad, np.array([0.002, 0.002]))
    keep = np.all(abs_dev <= limit, axis=1)
    if int(np.count_nonzero(keep)) < MIN_VALID_SAMPLES:
        keep = np.ones(len(arr), dtype=bool)
    kept = arr[keep]
    return np.mean(kept, axis=0), int(len(kept)), float(np.max(np.std(kept, axis=0)))


def write_targets(paths, labels, targets, pick_pixels, place_pixels, sample_counts):
    targets_file, targets_csv = paths
    np.savez_compressed(
        targets_file,
        targets=np.asarray(targets, dtype=np.float64),
        labels=np.asarray(labels),
        pick_pixels=np.asarray(pick_pixels, dtype=np.float64),
        place_pixels=np.asarray(place_pixels, dtype=np.float64),
        sample_counts=np.asarray(sample_counts, dtype=np.int32),
        averaging_seconds=np.float64(AVERAGING_SECONDS),
        image_width=np.int32(IMAGE_WIDTH),
        image_height=np.int32(IMAGE_HEIGHT),
        aruco_dictionary=np.array(ARUCO_DICT_NAME),
        aruco_marker_side_m=np.float64(MARKER_SIDE_M),
        aruco_robot_corners=np.asarray(ROBOT_MARKER_CORNERS, dtype=np.float64),
    )
    with targets_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shape", "paired_samples", "pick_pixel_x", "pick_pixel_y",
                         "pick_robot_x_m", "pick_robot_y_m", "place_pixel_x", "place_pixel_y",
                         "place_robot_x_m", "place_robot_y_m"])
        for i, name in enumerate(labels):
            writer.writerow([name, sample_counts[i], *pick_pixels[i], *targets[i, 0],
                             *place_pixels[i], *targets[i, 1]])


def _close_windows():
    if HEADLESS:
        return
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def new_sample_store():
    return {name: {"pick_robot": [], "place_robot": [], "pick_pixel": [], "place_pixel": []}
            for name in SHAPE_ORDER}


def main():
    print("Starting 5-second shape sorting vision...", flush=True)
    print("Rule: all four cardboard holes are mandatory; coloured blocks are optional (0-4).", flush=True)
    print(f"DISPLAY={os.environ.get('DISPLAY', '<not set>')}  headless={HEADLESS}", flush=True)
    run_dir = make_run_dir()
    paths = (
        Path(os.environ.get("SHAPE_SORT_TARGETS_FILE", run_dir / "shape_sort_targets.npz")
             ).expanduser().resolve(),
        run_dir / "shape_sort_targets.csv",
    )
    print(f"Run folder: {run_dir}", flush=True)
    detector, dictionary, aruco_parameters = make_aruco_detector()
    print(f"ArUco dictionary: {ARUCO_DICT_NAME}; target ID: "
          f"{TARGET_MARKER_ID if TARGET_MARKER_ID is not None else 'largest detected'}", flush=True)
    print(f"ArUco outer side length: {MARKER_SIDE_M * 1000:.1f} mm", flush=True)
    print("Robot marker corners (base_link metres; order TL, TR, BR, BL):", flush=True)
    for corner_name, corner_xy in zip(ARUCO_CORNER_LABELS, ROBOT_MARKER_CORNERS):
        print(f"  {corner_name}: X={corner_xy[0]:.5f}, Y={corner_xy[1]:.5f}", flush=True)
    edge_pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    measured_edges_mm = [
        1000.0 * float(np.linalg.norm(ROBOT_MARKER_CORNERS[a] - ROBOT_MARKER_CORNERS[b]))
        for a, b in edge_pairs
    ]
    print(
        "Measured robot-coordinate marker edge lengths (mm): "
        + ", ".join(f"{v:.1f}" for v in measured_edges_mm)
        + f" (physical side is {MARKER_SIDE_M * 1000:.1f} mm)",
        flush=True,
    )
    print("Object pick pixels AND cardboard-hole place pixels use the live four-corner "
          "ArUco homography.", flush=True)
    camera = open_camera()
    try:
        ok, first_frame = camera.read()
        if not ok or first_frame is None:
            raise RuntimeError("Could not read initial camera frame.")
        fh, fw = first_frame.shape[:2]
        print(f"Camera frame: {fw}x{fh}", flush=True)
        warm = [first_frame] + [camera.read()[1] for _ in range(9)]
        warm = [f for f in warm if f is not None]
        roi = (3, 3, fw - 3, fh - 3)
        strip = detect_cardboard(warm)
        if strip is None:
            print("[WARN] Cardboard not found at startup; the detector will retry on live frames.", flush=True)
        elif strip.get("clipped_edges"):
            print("[WARN] Cardboard boundary touches the camera image edge. Check whether all four cut-outs are fully in frame.", flush=True)
        print("[CARDBOARD] Detection uses rolling-frame averaging and smoothed board geometry.", flush=True)
        print("Keep the board and any coloured blocks stationary during the 5-second average.", flush=True)

        data = new_sample_store()
        object_tracks = {n: new_object_track() for n in SHAPE_ORDER}
        print(
            "COLOURED OBJECTS: colour + expected geometry + size limits + full camera ROI "
            f"+ {OBJECT_CONFIRM_FRAMES}-frame position confirmation.",
            flush=True
        )
        seen_block = {n: 0 for n in SHAPE_ORDER}
        seen_hole = {n: 0 for n in SHAPE_ORDER}
        hole_pixel_samples = {n: [] for n in SHAPE_ORDER}
        start = time.monotonic()
        frames_seen = 0
        averaging_done = aborted = False
        show_masks = False
        averaged = {}
        last_counts = {name: 0 for name in SHAPE_ORDER}
        labels, targets, pick_pixels, place_pixels, sample_counts = [], [], [], [], []
        prev_gray = None
        motion = 0.0
        frame_index = 0
        aruco_smoothing_state = {"corners": None, "marker_id": None, "misses": 0}
        board_frame_history = deque(maxlen=BOARD_FRAME_HISTORY)
        board_misses = 0
        hole_tracks = {n: {"center": None} for n in SHAPE_ORDER}
        hole_validation_ok = False
        missing_holes = list(SHAPE_ORDER)
        no_coloured_shapes = False

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Camera frame capture failed.")
            frame_index += 1

            # Detect the ArUco marker in this exact frame and build the live
            # pixel -> robot XY map from its four measured outer corners.
            marker_H, marker_corners_cv, marker_id = detect_aruco_homography(
                frame, detector, dictionary, aruco_parameters
            )
            if marker_H is not None:
                raw_marker_corners = marker_corners_cv.reshape(4, 2)
                marker_H, marker_corners_cv = smooth_aruco_corners(
                    raw_marker_corners, marker_id, aruco_smoothing_state
                )
            else:
                # Keep the previous marker state briefly for reacquisition, but
                # do not use stale transforms to create coordinate samples.
                smooth_aruco_corners(None, None, aruco_smoothing_state)

            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
            motion = motion_fraction(prev_gray, gray, roi)
            prev_gray = gray

            # Detect the board from a rolling multi-frame average, not a single
            # frame. Retain the last valid estimate through brief threshold
            # dropouts and EMA-smooth its four corners to prevent mask flicker.
            board_frame_history.append(frame.copy())
            refresh_board = (
                len(board_frame_history) >= min(5, BOARD_FRAME_HISTORY)
                and (frame_index % BOARD_REFRESH_EVERY == 0
                     or (strip is None and frame_index % 2 == 0))
            )
            if refresh_board:
                board_candidate = detect_cardboard(list(board_frame_history), announce=False)
                if board_candidate is not None:
                    strip = smooth_cardboard_detection(strip, board_candidate)
                    board_misses = 0
                else:
                    board_misses += 1
                    if board_misses >= BOARD_MAX_MISSES:
                        strip = None

            objects, object_mask, object_debug, raw_masks = find_coloured_objects(
                frame, roi, None if strip is None else strip["mask"])
            if motion > MOTION_FRACTION_MAX:
                object_tracks = {n: new_object_track() for n in SHAPE_ORDER}
                objects = {}
                for name in SHAPE_ORDER:
                    for entry in object_debug.get(name, []):
                        if not entry.get("reason"):
                            entry["reason"] = "motion in scene; waiting for stationary view"
            else:
                objects = confirm_stationary_objects(objects, object_debug, object_tracks)
            holes, hole_mask, hole_candidates = find_holes(frame, strip, roi)
            holes = smooth_hole_centres(holes, hole_tracks)

            if not averaging_done:
                if motion > MOTION_FRACTION_MAX:
                    data = new_sample_store()
                    seen_block = {n: 0 for n in SHAPE_ORDER}
                    seen_hole = {n: 0 for n in SHAPE_ORDER}
                    hole_pixel_samples = {n: [] for n in SHAPE_ORDER}
                    frames_seen = 0
                    start = time.monotonic()
                elif marker_H is None:
                    # Without a detected marker there is no current pixel-to-
                    # robot transform, so pause coordinate sampling this frame.
                    pass
                else:
                    frames_seen += 1
                    for name in SHAPE_ORDER:
                        seen_block[name] += int(name in objects)
                        seen_hole[name] += int(name in holes)
                        if name in holes:
                            hole_pixel_samples[name].append(
                                np.asarray(holes[name]["center_px"], float)
                            )
                        # Only create a pair when the matching block and hole
                        # exist. Both centres use the same live four-corner
                        # ArUco homography and become robot base_link XY.
                        if name in objects and name in holes:
                            pp = objects[name]["center_px"]
                            hp = holes[name]["center_px"]
                            data[name]["pick_pixel"].append(np.asarray(pp, float))
                            data[name]["place_pixel"].append(np.asarray(hp, float))
                            data[name]["pick_robot"].append(
                                pixel_to_robot_xy(pp, marker_H)
                            )
                            data[name]["place_robot"].append(
                                pixel_to_robot_xy(hp, marker_H)
                            )
                last_counts = {n: len(data[n]["pick_robot"]) for n in SHAPE_ORDER}
                elapsed = time.monotonic() - start

                if elapsed >= AVERAGING_SECONDS and frames_seen >= MIN_VALID_SAMPLES:
                    averaging_done = True
                    hole_rates = {n: seen_hole[n] / max(frames_seen, 1) for n in SHAPE_ORDER}
                    missing_holes = [n for n in SHAPE_ORDER
                                     if hole_rates[n] < MIN_HOLE_DETECTION_RATE]
                    hole_validation_ok = not missing_holes
                    print("\nFIVE-SECOND AVERAGING COMPLETE. Coordinates are now frozen.", flush=True)
                    print("INDEPENDENT CARDBOARD-HOLE CHECK:", flush=True)
                    for name in SHAPE_ORDER:
                        centre_samples = hole_pixel_samples[name]
                        if centre_samples:
                            centre = np.mean(np.vstack(centre_samples), axis=0)
                            print(f"  {name}: detected {100 * hole_rates[name]:.0f}% of frames; "
                                  f"mean centre pixel=({centre[0]:.1f}, {centre[1]:.1f})", flush=True)
                        else:
                            print(f"  {name}: NOT DETECTED", flush=True)
                    if missing_holes:
                        print("[ERROR] All four cardboard holes are required. Missing/unstable: "
                              + ", ".join(missing_holes), flush=True)
                        print("No controller targets will be written. Reposition the board so all four "
                              "hole boundaries are fully inside the camera frame.", flush=True)

                    no_coloured_shapes = sum(seen_block.values()) == 0
                    for name in SHAPE_ORDER:
                        count = len(data[name]["pick_robot"])
                        rate = count / max(frames_seen, 1)
                        if count < MIN_VALID_SAMPLES or rate < MIN_DETECTION_RATE:
                            if seen_block[name] > 0:
                                print(f"[SKIP] {name}: coloured block was seen in "
                                      f"{100 * seen_block[name] / max(frames_seen, 1):.0f}% of frames, "
                                      f"but a stable matching pair was available in {100 * rate:.0f}%.", flush=True)
                            continue
                        pick_mean, used_pick, std_pick = robust_mean(data[name]["pick_robot"])
                        place_mean, used_place, std_place = robust_mean(data[name]["place_robot"])
                        if max(std_pick, std_place) > MAX_POSITION_STD_M:
                            print(f"[SKIP] {name}: unstable position "
                                  f"(std {1000 * max(std_pick, std_place):.1f} mm).", flush=True)
                            continue
                        if min(used_pick, used_place) < MIN_VALID_SAMPLES:
                            print(f"[SKIP] {name}: too few stable coordinate samples.", flush=True)
                            continue
                        if np.linalg.norm(pick_mean - place_mean) < 0.005:
                            print(f"[SKIP] {name}: pick/place centres are less than 5 mm apart.", flush=True)
                            continue
                        pick_px = np.mean(np.vstack(data[name]["pick_pixel"]), axis=0)
                        place_px = np.mean(np.vstack(data[name]["place_pixel"]), axis=0)
                        labels.append(name)
                        targets.append([pick_mean, place_mean])
                        pick_pixels.append(pick_px)
                        place_pixels.append(place_px)
                        sample_counts.append(min(used_pick, used_place))
                        averaged[name] = {"pick_robot": pick_mean, "place_robot": place_mean,
                                          "pick_pixel": pick_px, "place_pixel": place_px}

                    if labels:
                        print("MAPPED COLOURED OBJECTS:", flush=True)
                        for name in labels:
                            print(f"  {name}: PICK={averaged[name]['pick_robot'].round(5)} "
                                  f"PLACE={averaged[name]['place_robot'].round(5)} m", flush=True)
                    elif no_coloured_shapes:
                        print("No coloured shapes were present during averaging. This is valid; "
                              "the four holes were still detected, and there is nothing to sort.", flush=True)
                    else:
                        print("No stable coloured-object/cardboard-hole pairs were found.", flush=True)

            if HEADLESS:
                if averaging_done:
                    print("Headless mode: accepting averaged result automatically.", flush=True)
                    break
                continue

            display = draw_overlay(
                frame, roi, strip, objects, object_debug, holes, hole_candidates, last_counts,
                min(time.monotonic() - start, AVERAGING_SECONDS), averaged,
                averaging_done=averaging_done, motion=motion)

            if marker_H is None:
                cv2.putText(
                    display,
                    "ARUCO NOT DETECTED - coordinate samples paused",
                    (8, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (0, 0, 255), 2, cv2.LINE_AA
                )
            else:
                cv2.aruco.drawDetectedMarkers(
                    display,
                    [marker_corners_cv],
                    np.array([[marker_id]], dtype=np.int32)
                )
                marker_image_corners = marker_corners_cv.reshape(4, 2)
                for i, (px, py) in enumerate(marker_image_corners):
                    cv2.circle(
                        display, (int(round(px)), int(round(py))),
                        4, (0, 255, 255), -1
                    )
                    cv2.putText(
                        display,
                        ARUCO_CORNER_LABELS[i],
                        (int(round(px)) + 4, int(round(py)) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0, 255, 255), 1, cv2.LINE_AA
                    )
                cv2.putText(
                    display,
                    f"ARUCO ID {marker_id} | live pixel-to-robot map active",
                    (8, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (0, 255, 255), 1, cv2.LINE_AA
                )

            cv2.imshow(WINDOW_NAME, display)
            if show_masks:
                cv2.imshow(MASK_WINDOW_NAME, raw_masks)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("m"), ord("M")):
                show_masks = not show_masks
                if not show_masks:
                    cv2.destroyWindow(MASK_WINDOW_NAME)
            if key in (ord("x"), ord("X")):
                aborted = True
                print("Vision cancelled by user.", flush=True)
                break
            if key in (ord("q"), ord("Q"), 27):
                if averaging_done:
                    print("Preview accepted; continuing to target validation.", flush=True)
                else:
                    aborted = True
                    print("Vision cancelled before averaging completed.", flush=True)
                break

        _close_windows()
        if aborted or not averaging_done:
            return 2
        if not hole_validation_ok:
            return 1
        if not labels:
            if no_coloured_shapes:
                print("VISION COMPLETE: all four holes were detected, but no coloured objects were present.", flush=True)
                print("No robot controller will be started.", flush=True)
                return NO_COLOURED_OBJECTS_EXIT
            print("VISION ERROR: coloured objects were detected but no stable matching pairs passed validation.", flush=True)
            return 1

        write_targets(paths, labels, np.asarray(targets, dtype=np.float64),
                      pick_pixels, place_pixels, sample_counts)
        print(f"\nSaved controller targets: {paths[0]}", flush=True)
        print(f"Saved coordinate CSV: {paths[1]}", flush=True)
        print(f"Target array shape: ({len(labels)}, 2, 2) — only available coloured objects.", flush=True)
        print("No still image or video files were captured or saved.", flush=True)
        return 0
    finally:
        camera.release()
        _close_windows()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nVISION ERROR: {exc}", flush=True)
        print("Robot controller will NOT be started.", flush=True)
        raise