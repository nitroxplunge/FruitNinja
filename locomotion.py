from pyrobot import Robot
import numpy as np
from pyrobot.utils.util import try_cv2_import
from time import sleep

robot = Robot('locobot')

# robot.camera.reset()
# robot.arm.go_home()

robot.gripper.open()

robot.camera.set_pan(0.0, wait=False)
robot.camera.set_tilt(1.0, wait=False)

cv2 = try_cv2_import()
im = robot.camera.get_depth()
cv2.imwrite('depth.png', im)


# PIXEL POSITIONS RETURNED BY CHATGPT SELECTION
# For demonstration purposes, these are hardcoded into robot waypoints.
# This NUC is not powerful enough to run Segment Anything, so these numbers
# were generated on a seperate device and then copied here.
# One could trivially connect the two halves given a powerful enough edge computer.

# strawberry: 375 229
# grape 1: 262 266
# grape 2: 418 170
# carrot: 340 169
# blackberry 1: 226 213
# blackberry 2: 68 244
# blackberry 3: 452 218
# INVERT THE NUMBERS IN THE FUNCTIONS BELOW

# Pixel space to world space conversion
loc1, trash = robot.camera.pix_to_3dpt([213], [226])
loc2, trash = robot.camera.pix_to_3dpt([244], [68])
loc3, trash = robot.camera.pix_to_3dpt([218], [452])

# Bias waypoints to account for underpowered stepper motors
way1 = loc1[0]
way1[0] = way1[0] + 0.005
way1[2] = way1[2] + 0.13

way2 = loc2[0]
way2[0] = way2[0] + 0.005
way2[2] = way2[2] + 0.13

way3 = loc3[0]
way3[0] = way3[0] + 0.005
way3[2] = way3[2] + 0.13

# Execute motions. For demonstration purposes, this is hardcoded

target_pose = { 'position':    way1, # forward/back, left/right, height
                'orientation': np.array([0.0, np.pi/2.0, 0.0])}
robot.arm.set_ee_pose(**target_pose)

robot.gripper.close()

robot.arm.go_home()

sleep(1)

robot.gripper.open()

sleep(1)

##

target_pose = { 'position':    way2, # forward/back, left/right, height
                'orientation': np.array([0.0, np.pi/2.0, 0.0])}
robot.arm.set_ee_pose(**target_pose)

robot.gripper.close()

robot.arm.go_home()

sleep(1)

robot.gripper.open()

sleep(1)

##

target_pose = { 'position':    way3, # forward/back, left/right, height
                'orientation': np.array([0.0, np.pi/2.0, 0.0])}
robot.arm.set_ee_pose(**target_pose)

robot.gripper.close()

robot.arm.go_home()

sleep(1)

robot.gripper.open()

sleep(1)