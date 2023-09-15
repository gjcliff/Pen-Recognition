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

# This script makes the end-effector perform pick, pour, and place tasks
# Note that this script may not work for every arm as it was designed for the wx250
# Make sure to adjust commanded joint positions and poses as necessary
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250'
# Then change to this directory and type 'python bartender.py  # python3 bartender.py if using ROS Noetic'

bot = InterbotixManipulatorXS("px100", "arm", "gripper")
ee_origin = [160, 412, -90] # from camera perspective: (y,x,z)

def camera_coor_to_robot_coor(camera_point):
    robot_point = [-float(camera_point[1]),float(camera_point[0]),float(camera_point[2])]
    return robot_point

def main():
    f = open("/tmp/cv_fifo", "r")
    camera_point = []

    for i in range(3):
        camera_point.append(f.readline())
        print(camera_point[i])

    # ROUTINE
    robot_point = camera_coor_to_robot_coor(camera_point)
    print(f"origin_point: {ee_origin}")
    print(f"robot_point: {robot_point}")
    print(f"x move: {ee_origin[0] - robot_point[0]}, y move: {ee_origin[1] - robot_point[1]}, z move: {ee_origin[2] - robot_point[2]}")
    # bot.arm.go_to_home_pose()robot_point

    # Differences
    xDiff = ee_origin[0] - robot_point[0]
    yDiff = ee_origin[1] - robot_point[1]
    zDiff = ee_origin[2] - robot_point[2]

    # direction factors
    xDir = xDiff/abs(xDiff)
    yDir = yDiff/abs(yDiff)
    zDir = zDiff/abs(zDiff)

    xD = (xDiff * xDir)/1000
    # if ee_origin[0] > robot_point[0]: xD *= -1

    yD = (yDiff * yDir)/1000
    # if ee_origin[1] > robot_point[1]: yD *= -1

    zD = (zDiff * zDir)/1000

    print(f"xD: {xD}, yD: {yD}, zd: {zD}")

    # bot.gripper.grasp(2.0)
    # bot.gripper.set_pressure(1.0)

    try:
        bot.arm.set_ee_cartesian_trajectory(x=xD,z=zD)
        bot.gripper.grasp(1)
        bot.arm.go_to_sleep_pose()
        bot.gripper.release(1)
    except:
        pass


if __name__=='__main__':
    main()
