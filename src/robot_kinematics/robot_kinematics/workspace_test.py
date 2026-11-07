import numpy as np

from kinematics import (
    inverse_kinematics,
    joint_limit_margin,
    forward_kinematics
)

ANGLES = [45, 60, 75, 90]

X_RANGE = np.linspace(0.12, 0.28, 9)
Y_RANGE = np.linspace(-0.12, 0.12, 9)

Z = 0.08

for angle_deg in ANGLES:

    angle = np.deg2rad(angle_deg)

    successful = 0
    total = 0
    margins = []

    for x in X_RANGE:
        for y in Y_RANGE:

            total += 1

            q = inverse_kinematics(
                x, y, Z,
                angle=angle
            )

            if q is None:
                continue

            successful += 1
            margins.append(joint_limit_margin(q))

    percentage = 100 * successful / total

    print("\nApproach angle:", angle_deg)
    print("Reachable positions:", successful, "/", total)
    print("Workspace coverage:", percentage, "%")

    if margins:
        print("Minimum joint margin:", min(margins))
        print("Average joint margin:", np.mean(margins))