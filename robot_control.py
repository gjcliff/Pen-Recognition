#!/usr/bin/python

# from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
# import time
# # The robot object is what you use to control the robot

# robot_joints = ["base_link", "ee_arm_link", "ee_gripper_link", "fingers_link", "forearms_link", "gripper_bar_link", "gripper_link", "gripper_prop_link", "left_finger_link", "right_finger_link", "shoulder_link", "upper_arm_link"]

# robot = InterbotixManipulatorXS("px100", "arm", "gripper")
# mode = 'h'
# # Let the user select the position
# # while mode != 'q':
# #     mode=input("[h]ome, [s]leep, [g] joint commands, [q]uit ")
# #     if mode == "h":
# #         robot.arm.go_to_home_pose()
# #     elif mode == "s":
# #         robot.arm.go_to_sleep_pose()
# #     else:
# #         joint_command = robot.arm.get_ee_pose()
# #         print(f"current joint angles: {joint_command}")

# # robot.arm.get_joint_commands()
# print(robot.arm.get_ee_pose())

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
import sys

# This script makes the end-effector perform pick, pour, and place tasks
# Note that this script may not work for every arm as it was designed for the wx250
# Make sure to adjust commanded joint positions and poses as necessary
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250'
# Then change to this directory and type 'python bartender.py  # python3 bartender.py if using ROS Noetic'

def main():
    bot = InterbotixManipulatorXS("px100", "arm", "gripper")

    bot.arm.go_to_home_pose()

    bot.gripper.grasp(2.0)
    bot.gripper.set_pressure(1.0)

    bot.arm.set_ee_pose_components(x=-0.1, y=-0.1, z=0.25)
    
    bot.gripper.release(2.0)
    # bot.arm.go_to_home_pose()
    # bot.arm.go_to_sleep_pose()

if __name__=='__main__':
    main()
