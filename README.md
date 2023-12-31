## Object Recognition Using OpenCV and Robotic Manipulation and Grasping

The goal of this project is to enable the PincherX 100 robot arm to grab a pen that's held in front of it. The tools used to complete this task are OpenCV, Numpy, a Linux FIFO pipe, Python, an Intel RealSense D435i, and a PincherX 100 robot arm.

### Computer Vision

First, I create a color mask using HSV values that looks for the color of the pen in my hand: purple. Next, I calculate the geometric centroid of the shape revealed by the color mask after finding its contours. Finally, I access the value at the position of the centroid from the depth generated by the RealSense camera to figure out the location of the pen in 3D space.

### Robot Manipulations

After finding the 3D position of the pen in the camera's frame, I think translate this position to be in the frame of the robot. Once this point has been calculated, I call the set_ee_cartesian_trajectory() from the PincherX's library to start planning and executing a trajectory to the location of the pen.

Once this trajectory is finished, the gripper will close, and the robot will have grasped the pen.

### Future Work

I completed this project before I knew anything about ROS or basic robotics math like transformation matrices. I plan to go back and redo the project at some point in the near future to improve upon it, and to learn more about computer vision in the process.

## Gallery

**Computer Vision**
[![Pen Vision](https://img.youtube.com/vi/dudBlyBsvok/0.jpg)](https://youtube.com/shorts/dudBlyBsvok?si=K0l7IiKLe3kxetTm "Pen Vision")

**Project Demo**
[![project demo](https://img.youtube.com/vi/xdqtf6kgfiU/0.jpg)](https://youtube.com/shorts/xdqtf6kgfiU?si=K0l7IiKLe3kxetTm "project demo")