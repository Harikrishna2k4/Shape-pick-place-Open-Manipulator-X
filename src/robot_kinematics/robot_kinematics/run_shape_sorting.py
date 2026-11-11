#!/usr/bin/env python3
"""Run five-second vision averaging, then sort only available object/hole pairs."""

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


def main():
    print("=== SHAPE SORTING ===", flush=True)
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
    print("STEP 1: Detect all four cardboard holes independently of coloured objects.", flush=True)
    print("STEP 2: Average matching coloured-object/hole pairs for five seconds.", flush=True)
    print("Zero to four coloured objects are allowed; only detected matching pairs are sent to IK.", flush=True)
    print("After averaging, press Q/ESC in the preview to accept, or X to abort.", flush=True)

    result = subprocess.run(
        [sys.executable, "-u", str(VISION_SCRIPT)],
        cwd=str(SCRIPT_DIR), env=env, check=False,
    )

    if result.returncode == NO_COLOURED_OBJECTS_EXIT:
        print("All four cardboard holes were detected, but no coloured objects were present.", flush=True)
        print("Nothing to sort. Robot controller will NOT be started.", flush=True)
        return 0

    if result.returncode != 0:
        print(f"Vision stopped with code {result.returncode}; robot controller will NOT run.", flush=True)
        print(f"Review run folder: {run_dir}", flush=True)
        return int(result.returncode)

    if not targets_file.is_file():
        print(f"ERROR: Targets file was not created: {targets_file}", flush=True)
        return 3

    print("STEP 3: Starting IK / pick-and-place controller.", flush=True)
    print(f"Controller: {controller.name}", flush=True)
    print(f"Targets: {targets_file}", flush=True)
    robot = subprocess.run(
        [sys.executable, "-u", str(controller)],
        cwd=str(SCRIPT_DIR), env=env, check=False,
    )
    return int(robot.returncode)


if __name__ == "__main__":
    raise SystemExit(main())