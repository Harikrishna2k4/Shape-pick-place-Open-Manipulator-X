#!/usr/bin/env python3
"""Run ArUco-based vision averaging, then execute available shape pairs."""

from datetime import datetime
from pathlib import Path
import os
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
VISION_SCRIPT = SCRIPT_DIR / "shape_sort_vision.py"
CONTROLLER_CANDIDATES = (
    SCRIPT_DIR / "openmanipulator_fixed_orientation_pick_and_place.py",
    SCRIPT_DIR / "openmanipulator_fixed_orientation_pick_place.py",
)
RUNS_ROOT = SCRIPT_DIR / "shape_sort_runs"
NO_COLOURED_OBJECTS_EXIT = 4


def find_controller():
    return next((p for p in CONTROLLER_CANDIDATES if p.is_file()), None)


def create_run_dir():
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    path = RUNS_ROOT / stamp
    path.mkdir(parents=False, exist_ok=False)
    return path


def validate_targets_file(path):
    """Check the vision/controller hand-off before starting robot software."""
    if not path.is_file():
        raise FileNotFoundError(f"Targets file was not created: {path}")

    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        required = {"targets", "labels"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Vision output is missing fields: {sorted(missing)}")

        targets = np.asarray(data["targets"], dtype=float)
        labels = np.asarray(data["labels"]).astype(str)

    if targets.ndim != 3 or targets.shape[1:] != (2, 2):
        raise ValueError(
            f"Expected target array shape (N, 2, 2), got {targets.shape}."
        )
    if len(labels) != len(targets) or len(labels) == 0:
        raise ValueError("Vision output has no complete pick/place pairs or mismatched labels.")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Vision output contains non-finite robot coordinates.")

    print(f"Validated {len(labels)} pick/place pair(s):", flush=True)
    for label, pair in zip(labels, targets):
        pick_xy, place_xy = pair
        print(
            f"  {label}: PICK=({pick_xy[0]:.5f}, {pick_xy[1]:.5f}) m | "
            f"PLACE=({place_xy[0]:.5f}, {place_xy[1]:.5f}) m",
            flush=True,
        )


def main():
    print("=== ARUCO-BASED SHAPE SORTING ===", flush=True)

    controller = find_controller()
    if not VISION_SCRIPT.is_file() or controller is None:
        print("ERROR: Required vision/controller source file is missing.", flush=True)
        print(f"Vision expected: {VISION_SCRIPT}", flush=True)
        for p in CONTROLLER_CANDIDATES:
            print(f"  {p} {'[FOUND]' if p.is_file() else '[MISSING]'}", flush=True)
        return 2

    run_dir = create_run_dir()
    targets_file = run_dir / "shape_sort_targets.npz"

    env = os.environ.copy()
    env["SHAPE_SORT_RUN_DIR"] = str(run_dir)
    env["SHAPE_SORT_TARGETS_FILE"] = str(targets_file)

    print(f"Run folder: {run_dir}", flush=True)
    print("STEP 1: Detect the ArUco marker in each camera frame.", flush=True)
    print("STEP 2: Use all four measured marker corners to map image pixels to robot base_link XY.", flush=True)
    print("STEP 3: Detect coloured-object centres and matching cardboard-hole centres.", flush=True)
    print("STEP 4: Average the mapped PICK and PLACE coordinates over five seconds.", flush=True)
    print("Only frames with a valid ArUco homography contribute coordinate samples.", flush=True)
    print("After averaging, press Q/ESC in the preview to accept, or X to abort.", flush=True)

    result = subprocess.run(
        [sys.executable, "-u", str(VISION_SCRIPT)],
        cwd=str(SCRIPT_DIR),
        env=env,
        check=False,
    )

    if result.returncode == NO_COLOURED_OBJECTS_EXIT:
        print(
            "All four cardboard holes were detected, but no coloured objects were present.",
            flush=True,
        )
        print("Nothing to sort. Robot controller will NOT be started.", flush=True)
        return 0

    if result.returncode != 0:
        print(
            f"Vision stopped with code {result.returncode}; robot controller will NOT run.",
            flush=True,
        )
        print(f"Review run folder: {run_dir}", flush=True)
        return int(result.returncode)

    try:
        validate_targets_file(targets_file)
    except Exception as exc:
        print(f"ERROR: Invalid vision output: {exc}", flush=True)
        print("Robot controller will NOT be started.", flush=True)
        return 3

    print("\nSTEP 5: Starting the existing IK / pick-and-place controller.", flush=True)
    print(f"Controller: {controller.name}", flush=True)
    print(f"Targets: {targets_file}", flush=True)

    robot = subprocess.run(
        [sys.executable, "-u", str(controller)],
        cwd=str(SCRIPT_DIR),
        env=env,
        check=False,
    )
    return int(robot.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
