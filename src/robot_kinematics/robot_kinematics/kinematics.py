#!/usr/bin/env python3

import numpy as np
from scipy.optimize import least_squares

# ============================================================
# ROBOT CONFIGURATION
# ============================================================

JOINT_LIMITS = np.array([
    [-np.pi, np.pi],       # joint1
    [-1.5, 1.5],           # joint2
    [-1.5, 1.4],           # joint3
    [-1.7, 1.97]           # joint4
])

# Angle measured from horizontal:
# 90 degrees = vertically downward
# 60 degrees = 30 degrees tilted from vertical
# 45 degrees = diagonally downward

APPROACH_ANGLE = np.deg2rad(60.0)

POSITION_TOLERANCE = 1e-4

# ============================================================
# TRANSFORMATION MATRICES
# ============================================================

def rot_y(q):
    c, s = np.cos(q), np.sin(q)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rot_z(q):
    c, s = np.cos(q), np.sin(q)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def transform(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p

    return T


# ============================================================
# FORWARD KINEMATICS
# ============================================================

def forward_kinematics(q1, q2, q3, q4):

    T1 = transform(
        rot_z(q1),
        [0.012, 0.0, 0.017]
    )

    T2 = transform(
        rot_y(q2),
        [0.0, 0.0, 0.0595]
    )

    T3 = transform(
        rot_y(q3),
        [0.024, 0.0, 0.128]
    )

    T4 = transform(
        rot_y(q4),
        [0.124, 0.0, 0.0]
    )

    T5 = transform(
        np.eye(3),
        [0.126, 0.0, 0.0]
    )

    T = T1 @ T2 @ T3 @ T4 @ T5

    position = T[:3, 3]
    rotation = T[:3, :3]

    return position, rotation, T


# ============================================================
# FIXED APPROACH ANGLE
# ============================================================

def forward_kinematics_fixed_angle(q1, q2, q3, angle):

    # q2 + q3 + q4 = angle

    q4 = angle - q2 - q3

    return forward_kinematics(q1, q2, q3, q4)


# ============================================================
# ORIENTATION CHECK
# ============================================================

def check_approach_angle(rotation, angle, tolerance=1e-3):
    """
    Check the tool pitch angle independently of joint1.

    The first column of the rotation matrix is the tool's local x-axis
    expressed in the base frame. Since joint1 rotates the arm around base Z,
    the horizontal direction of this axis changes with joint1. Its horizontal
    magnitude and vertical component, however, are determined by `angle`.

    This check therefore validates:
        hypot(R[0, 0], R[1, 0]) = cos(angle)
        R[2, 0]                 = -sin(angle)
    """

    actual_x = np.asarray(rotation, dtype=float)[:, 0]

    expected_horizontal = np.cos(angle)
    expected_vertical = -np.sin(angle)

    actual_horizontal = np.hypot(actual_x[0], actual_x[1])
    actual_vertical = actual_x[2]

    horizontal_error = abs(actual_horizontal - expected_horizontal)
    vertical_error = abs(actual_vertical - expected_vertical)

    return max(horizontal_error, vertical_error) < tolerance


# ============================================================
# JOINT LIMIT CHECK
# ============================================================

def check_joint_limits(q):

    for i in range(4):

        if q[i] < JOINT_LIMITS[i, 0]:
            return False

        if q[i] > JOINT_LIMITS[i, 1]:
            return False

    return True


def joint_limit_margin(q):

    margins = np.minimum(
        q - JOINT_LIMITS[:, 0],
        JOINT_LIMITS[:, 1] - q
    )

    return np.min(margins)


# ============================================================
# IK ERROR FUNCTION
# ============================================================

def ik_error(q23, target_r, target_z, angle):

    q2, q3 = q23

    q4 = angle - q2 - q3

    position, _, _ = forward_kinematics(
        0.0, q2, q3, q4
    )

    # Radial distance from joint1 axis
    r = np.sqrt(
        (position[0] - 0.012)**2 +
        position[1]**2
    )

    return np.array([
        r - target_r,
        position[2] - target_z
    ])


# ============================================================
# INVERSE KINEMATICS
# ============================================================

def inverse_kinematics(
    x,
    y,
    z,
    angle=APPROACH_ANGLE,
    previous_solution=None
):

    # Joint1 rotates the arm around the base Z axis.

    dx = x - 0.012
    dy = y

    rho = np.sqrt(dx**2 + dy**2)

    q1_candidates = [
        np.arctan2(dy, dx),
        np.arctan2(dy, dx) + np.pi
    ]

    radial_candidates = [rho, -rho]

    seeds = [
        [0.0, 0.0],
        [0.5, -0.5],
        [-0.5, 0.5],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.2, 0.0],
        [-1.2, 0.0]
    ]

    solutions = []

    for q1, target_r in zip(
        q1_candidates,
        radial_candidates
    ):

        q1 = (q1 + np.pi) % (2 * np.pi) - np.pi

        if not (
            JOINT_LIMITS[0, 0] <= q1 <= JOINT_LIMITS[0, 1]
        ):
            continue

        for seed in seeds:

            result = least_squares(
                ik_error,
                seed,
                bounds=(
                    [-1.5, -1.5],
                    [1.5, 1.4]
                ),
                args=(target_r, z, angle),
                max_nfev=2000,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12
            )

            q2, q3 = result.x

            q4 = angle - q2 - q3

            q = np.array([q1, q2, q3, q4])

            if not check_joint_limits(q):
                continue

            position, rotation, _ = forward_kinematics(
                *q
            )

            position_error = np.linalg.norm(
                position - np.array([x, y, z])
            )

            orientation_valid = check_approach_angle(
                rotation,
                angle
            )

            if position_error > POSITION_TOLERANCE:
                continue

            if not orientation_valid:
                continue

            if previous_solution is None:
                continuity_cost = 0.0
            else:
                continuity_cost = np.linalg.norm(
                    q - previous_solution
                )

            margin = joint_limit_margin(q)

            # Prefer solutions away from joint limits.
            score = continuity_cost + 0.05 / (margin + 1e-4)

            solutions.append((score, q))

    if not solutions:
        return None

    solutions.sort(key=lambda item: item[0])

    return solutions[0][1]


# ============================================================
# QUINTIC TRAJECTORY GENERATION
# ============================================================

def generate_trajectory(
    q_start,
    q_end,
    duration=3.0,
    dt=0.02
):

    q_start = np.asarray(q_start)
    q_end = np.asarray(q_end)

    steps = int(duration / dt) + 1

    trajectory = []

    for i in range(steps):

        t = i * dt / duration

        # Quintic time scaling
        s = (
            10 * t**3
            - 15 * t**4
            + 6 * t**5
        )

        q = q_start + s * (q_end - q_start)

        trajectory.append(q)

    return np.array(trajectory)


# ============================================================
# TRAJECTORY VALIDATION
# ============================================================

def validate_trajectory(trajectory, dt=0.02):

    velocity = np.diff(
        trajectory,
        axis=0
    ) / dt

    acceleration = np.diff(
        velocity,
        axis=0
    ) / dt

    max_velocity = np.max(
        np.abs(velocity),
        axis=0
    )

    max_acceleration = np.max(
        np.abs(acceleration),
        axis=0
    )

    # Conservative software limits for initial planning.
    velocity_limits = np.array([
        0.5, 0.5, 0.5, 0.5
    ])

    acceleration_limits = np.array([
        1.0, 1.0, 1.0, 1.0
    ])

    positions_valid = all(
        check_joint_limits(q)
        for q in trajectory
    )

    velocity_valid = np.all(
        max_velocity <= velocity_limits
    )

    acceleration_valid = np.all(
        max_acceleration <= acceleration_limits
    )

    print("\nTRAJECTORY VALIDATION")
    print("---------------------")

    print("Maximum velocity:", max_velocity)
    print("Maximum acceleration:", max_acceleration)

    print("Position limits:", positions_valid)
    print("Velocity limits:", velocity_valid)
    print("Acceleration limits:", acceleration_valid)

    return (
        positions_valid
        and velocity_valid
        and acceleration_valid
    )


# ============================================================
# PICK AND PLACE PLANNER
# ============================================================

def plan_pick_and_place(
    current_q,
    object_position,
    hole_position,
    travel_height=0.08,
    pickup_height=0.015,
    approach_angle=APPROACH_ANGLE
):

    ox, oy, oz = object_position
    hx, hy, hz = hole_position

    # --------------------------------------------------------
    # 1. Above object
    # --------------------------------------------------------

    object_above = (
        ox, oy, oz + travel_height
    )

    # --------------------------------------------------------
    # 2. Pickup position
    # --------------------------------------------------------

    object_pick = (
        ox, oy, oz + pickup_height
    )

    # --------------------------------------------------------
    # 3. Above target hole
    # --------------------------------------------------------

    hole_above = (
        hx, hy, hz + travel_height
    )

    # --------------------------------------------------------
    # 4. Find IK for each waypoint
    # --------------------------------------------------------

    q_above_object = inverse_kinematics(
        *object_above,
        angle=approach_angle,
        previous_solution=current_q
    )

    if q_above_object is None:
        raise ValueError("Object approach position is unreachable.")

    q_pick = inverse_kinematics(
        *object_pick,
        angle=approach_angle,
        previous_solution=q_above_object
    )

    if q_pick is None:
        raise ValueError("Pickup position is unreachable.")

    q_above_hole = inverse_kinematics(
        *hole_above,
        angle=approach_angle,
        previous_solution=q_pick
    )

    if q_above_hole is None:
        raise ValueError("Hole approach position is unreachable.")

    # --------------------------------------------------------
    # 5. Generate trajectories
    # --------------------------------------------------------

    segments = [
        generate_trajectory(current_q, q_above_object),
        generate_trajectory(q_above_object, q_pick),
        generate_trajectory(q_pick, q_above_object),
        generate_trajectory(q_above_object, q_above_hole)
    ]

    return {
        "above_object": q_above_object,
        "pick": q_pick,
        "above_hole": q_above_hole,
        "segments": segments
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("OpenMANIPULATOR-X IK TEST")
    print("=========================")

    print(
        "Approach angle:",
        np.rad2deg(APPROACH_ANGLE),
        "degrees"
    )

    # Example target. This is a test coordinate,
    # not a measured object position.

    target = [0.20, 0.0, 0.08]

    q = inverse_kinematics(
        *target,
        angle=APPROACH_ANGLE
    )

    if q is None:

        print("Target is unreachable.")

    else:

        print("\nJoint solution:")
        print(q)

        position, rotation, _ = forward_kinematics(*q)

        print("\nCalculated position:")
        print(position)

        print("\nRotation matrix:")
        print(rotation)

        print("\nPosition error:")
        print(np.linalg.norm(
            position - np.array(target)
        ))

        print("\nJoint limit margin:")
        print(joint_limit_margin(q))


for angle_deg in [45, 60, 75, 90]:

    angle = np.deg2rad(angle_deg)

    q = inverse_kinematics(
        0.20, 0.0, 0.08,
        angle=angle
    )

    print(f"\nAngle: {angle_deg} degrees")

    if q is None:
        print("Unreachable")
        continue

    position, rotation, _ = forward_kinematics(*q)

    print("Joint angles:", q)
    print("Position:", position)
    print("Joint margin:", joint_limit_margin(q))