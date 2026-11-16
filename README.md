We choose Task 1.
We implemented pick and place of various shapes from random positions to the desired hole on the cardboard. 

IMPLEMENTATION:
The camera detects the edges of the shapes and calculates its center point. based on color, it seperates the shapes based on two categories, for picking and placing.  Colorfull shapes are for picking, plain shapes are for placing. Then it maps each pick shape with a corresponding placing shape.
If there are multiple shapes at the same time, It starts picking fro the shape that has least distance to its placing point to avoid collision with other shapes while placing current shape.
The camera will give all the pick and place pixel numbers.

We use AruCu marker to calibrate the robot's base coordinate with camera's pixel coordinate. We also added a little artificial distorsion to account for the cameras natural distorsion so that as we go to the side of the cameras field of view, we are still getting correct calibration.
Two aruCo markers were used to calibrate.

For robot movement, we decided to decouple robot's movements into two seperate movements. one is movement horizontally along x-y plane and movement vertically along z - direction only for pick and place.
Two seperate movements helps us for the end effector to more hit any other shapes while picking or placing the current shape.Since we seperated our movements. we need less way points to move, thus making it faster and effecient.
We used OpenCV for computer vision, python for scripts, ROS2 for robot control

SETUP & RUN INSTRUCTIONS
1. Connect the camera and Open manipulator X. Check proper connection and motor functioning using Dynamixel Wizard.
2. Go inside docker. COMMAND: docker exec -it open_manipulator-ros2-1 bash
3. Launch the hardware. COMMAND: ros2 launch open_manipulator_x_bringup hardware.launch.py
4. run goto.py to make the robot go to its home position
5. run run_shape_sorting.py
6. The camera window will record the center points for 5 seconds to take average value, after 5 sec, press Q on keyboard

7. Link: https://youtu.be/-zTurtpzdBI?si=-TrgkYYPM_XaWvDG
