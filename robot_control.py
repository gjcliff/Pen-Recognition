from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
# The robot object is what you use to control the robot
robot = InterbotixManipulatorXS("px100", "arm", "gripper")
mode = 'h'
# Let the user select the position
while mode != 'q':
    mode=input("[h]ome, [s]leep, [q]uit ")
    if mode == "h":
        robot.arm.go_to_home_pose()
    elif mode == "s":
        robot.arm.go_to_sleep_pose()
    elif mode == "g":
        robot.arm.