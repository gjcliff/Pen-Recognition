#!/usr/bin/python

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import time
# The robot object is what you use to control the robot

robot_joints = ["base_link", "ee_arm_link", "ee_gripper_link", "fingers_link", "forearms_link", "gripper_bar_link", "gripper_link", "gripper_prop_link", "left_finger_link", "right_finger_link", "shoulder_link", "upper_arm_link"]

robot = InterbotixManipulatorXS("px100", "arm", "gripper")
mode = 'h'
# Let the user select the position
# while mode != 'q':
#     mode=input("[h]ome, [s]leep, [g] joint commands, [q]uit ")
#     if mode == "h":
#         robot.arm.go_to_home_pose()
#     elif mode == "s":
#         robot.arm.go_to_sleep_pose()
#     else:
#         joint_command = robot.arm.get_ee_pose()
#         print(f"current joint angles: {joint_command}")

# robot.arm.get_joint_commands()
for joint in robot_joints:
    robot.arm.get_single_joint_command(joint_name=joint)
exit()