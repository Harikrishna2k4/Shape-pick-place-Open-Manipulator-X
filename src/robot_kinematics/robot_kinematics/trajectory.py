import numpy as np
import csv

from robot_kinematics.kinematics import (
    inverse_kinematics_fixed_height,
    TRAVEL_HEIGHT,
    JOINT_LIMITS,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DURATION = 3.0
CONTROL_FREQUENCY = 50.0

MAX_JOINT_STEP = 0.08

# Software validation limits.
# Replace these with verified limits for your controller
# and actual operating configuration.

MAX_JOINT_VELOCITY = np.array([
    0.5, 0.5, 0.5, 0.5
])  # rad/s

MAX_JOINT_ACCELERATION = np.array([
    1.0, 1.0, 1.0, 1.0
])  # rad/s^2


# ============================================================
# QUINTIC TIME SCALING
# ============================================================

def quintic_time_scaling(t, duration):

    s = t / duration

    position = (
        10*s**3
        - 15*s**4
        + 6*s**5
    )

    velocity = (
        30*s**2
        - 60*s**3
        + 30*s**4
    ) / duration

    acceleration = (
        60*s
        - 180*s**2
        + 120*s**3
    ) / duration**2

    return position, velocity, acceleration


# ============================================================
# CARTESIAN TRAJECTORY
# ============================================================

def generate_cartesian_path(
    start_xy,
    goal_xy,
    duration=DEFAULT_DURATION,
    frequency=CONTROL_FREQUENCY,
    height=TRAVEL_HEIGHT
):

    start_xy = np.asarray(start_xy, dtype=float)
    goal_xy = np.asarray(goal_xy, dtype=float)

    if duration <= 0 or frequency <= 0:
        raise ValueError("Duration and frequency must be positive.")

    times = np.linspace(
        0,
        duration,
        int(round(duration * frequency)) + 1
    )

    positions = []
    velocities = []
    accelerations = []

    displacement = goal_xy - start_xy

    for t in times:

        s, sd, sdd = quintic_time_scaling(
            t,
            duration
        )

        xy = start_xy + s * displacement

        xy_velocity = sd * displacement

        xy_acceleration = sdd * displacement

        positions.append([
            xy[0],
            xy[1],
            height
        ])

        velocities.append([
            xy_velocity[0],
            xy_velocity[1],
            0.0
        ])

        accelerations.append([
            xy_acceleration[0],
            xy_acceleration[1],
            0.0
        ])

    return (
        times,
        np.asarray(positions),
        np.asarray(velocities),
        np.asarray(accelerations)
    )


# ============================================================
# CARTESIAN TO JOINT TRAJECTORY
# ============================================================

def cartesian_to_joint_trajectory(
    positions,
    times,
    initial_joint_angles=None
):

    joint_positions = []

    previous_solution = initial_joint_angles

    for i, point in enumerate(positions):

        result = inverse_kinematics_fixed_height(
            x=point[0],
            y=point[1],
            z=point[2],
            previous_solution=previous_solution
        )

        if result is None:

            raise RuntimeError(
                f"IK failed at sample {i}: {point}"
            )

        q = result["joint_angles"]

        # Joint-limit validation.
        if np.any(q < JOINT_LIMITS[:, 0]) or np.any(
            q > JOINT_LIMITS[:, 1]
        ):

            raise RuntimeError(
                f"Joint limit violation at sample {i}"
            )

        # Joint continuity validation.
        if previous_solution is not None:

            difference = np.arctan2(
                np.sin(q - previous_solution),
                np.cos(q - previous_solution)
            )

            jump = np.max(np.abs(difference))

            if jump > MAX_JOINT_STEP:

                raise RuntimeError(
                    f"Joint discontinuity at sample {i}: "
                    f"{jump:.4f} rad"
                )

        joint_positions.append(q)

        previous_solution = q

    joint_positions = np.asarray(joint_positions)

    # Numerical derivatives.
    joint_velocities = np.gradient(
        joint_positions,
        times,
        axis=0,
        edge_order=2
    )

    joint_accelerations = np.gradient(
        joint_velocities,
        times,
        axis=0,
        edge_order=2
    )

    # Quintic trajectory has zero endpoint velocity.
    joint_velocities[0] = 0.0
    joint_velocities[-1] = 0.0

    return (
        joint_positions,
        joint_velocities,
        joint_accelerations
    )


# ============================================================
# TRAJECTORY VALIDATION
# ============================================================

def validate_trajectory(q, qd, qdd):

    max_velocity = np.max(
        np.abs(qd),
        axis=0
    )

    max_acceleration = np.max(
        np.abs(qdd),
        axis=0
    )

    print("\n========== TRAJECTORY VALIDATION ==========")

    print("\nMaximum joint velocity:")
    print(max_velocity)

    print("\nAllowed joint velocity:")
    print(MAX_JOINT_VELOCITY)

    print("\nMaximum joint acceleration:")
    print(max_acceleration)

    print("\nAllowed joint acceleration:")
    print(MAX_JOINT_ACCELERATION)

    velocity_valid = np.all(
        max_velocity <= MAX_JOINT_VELOCITY
    )

    acceleration_valid = np.all(
        max_acceleration <= MAX_JOINT_ACCELERATION
    )

    position_valid = np.all(
        (q >= JOINT_LIMITS[:, 0]) &
        (q <= JOINT_LIMITS[:, 1])
    )

    print("\nVelocity valid:", velocity_valid)
    print("Acceleration valid:", acceleration_valid)
    print("Joint positions valid:", position_valid)

    if not (
        velocity_valid
        and acceleration_valid
        and position_valid
    ):

        raise RuntimeError(
            "Trajectory validation failed. "
            "Increase duration or modify the path."
        )

    print("\nTRAJECTORY VALIDATED.")

    return True


# ============================================================
# COMPLETE TRAJECTORY GENERATOR
# ============================================================

def generate_trajectory(
    start_xy,
    goal_xy,
    initial_joint_angles=None,
    duration=DEFAULT_DURATION,
    frequency=CONTROL_FREQUENCY,
    height=TRAVEL_HEIGHT
):

    times, positions, cart_velocities, cart_accelerations = (
        generate_cartesian_path(
            start_xy,
            goal_xy,
            duration,
            frequency,
            height
        )
    )

    q, qd, qdd = cartesian_to_joint_trajectory(
        positions,
        times,
        initial_joint_angles
    )

    validate_trajectory(q, qd, qdd)

    return {
        "time": times,
        "cartesian_positions": positions,
        "cartesian_velocities": cart_velocities,
        "cartesian_accelerations": cart_accelerations,
        "joint_positions": q,
        "joint_velocities": qd,
        "joint_accelerations": qdd
    }


# ============================================================
# CSV EXPORT
# ============================================================

def save_trajectory(
    trajectory,
    filename="trajectory.csv"
):

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "time",
            "x", "y", "z",
            "q1", "q2", "q3", "q4",
            "qd1", "qd2", "qd3", "qd4",
            "qdd1", "qdd2", "qdd3", "qdd4"
        ])

        for i in range(len(trajectory["time"])):

            writer.writerow([
                trajectory["time"][i],
                *trajectory["cartesian_positions"][i],
                *trajectory["joint_positions"][i],
                *trajectory["joint_velocities"][i],
                *trajectory["joint_accelerations"][i]
            ])

    print("\nSaved:", filename)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n========== TRAJECTORY GENERATION TEST ==========")

    start = [0.20, -0.05]
    goal = [0.22, 0.05]

    initial_result = inverse_kinematics_fixed_height(
        start[0],
        start[1],
        TRAVEL_HEIGHT
    )

    if initial_result is None:
        raise RuntimeError("Starting position is unreachable.")

    trajectory = generate_trajectory(
        start_xy=start,
        goal_xy=goal,
        initial_joint_angles=initial_result["joint_angles"],
        duration=3.0
    )

    print("\nNumber of samples:")
    print(len(trajectory["time"]))

    print("\nStarting joint angles:")
    print(trajectory["joint_positions"][0])

    print("\nFinal joint angles:")
    print(trajectory["joint_positions"][-1])

    save_trajectory(trajectory)