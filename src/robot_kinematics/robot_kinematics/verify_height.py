import numpy as np

from robot_kinematics.kinematics import forward_kinematics


# Initial measured joint positions
q_initial = [
    -2.94677709,
    -0.38042724,
     0.42031074,
     1.41586427
]

# Final measured joint positions
q_final = [
    -2.94831107,
    -0.36201947,
     0.56603891,
     1.24866036
]


# Calculate forward kinematics
p_initial, R_initial, T_initial = forward_kinematics(*q_initial)

p_final, R_final, T_final = forward_kinematics(*q_final)


print("\nINITIAL END-EFFECTOR POSITION")
print(p_initial)

print("\nFINAL END-EFFECTOR POSITION")
print(p_final)


# Calculate Cartesian displacement
displacement = p_final - p_initial

print("\nCARTESIAN DISPLACEMENT")
print(displacement)

print("\nHEIGHT CHANGE")
print(f"{displacement[2] * 100:.4f} cm")

print("\nHORIZONTAL DISPLACEMENT")
print(f"{np.linalg.norm(displacement[:2]) * 100:.4f} cm")