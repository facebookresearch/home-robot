import time

import numpy as np
import pybullet as pb
import pybullet_data

client = pb.connect(pb.GUI)
# pb.setAdditionalSearchPath('.')
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, -9.8)

# Basic stuff here
plane_id = pb.loadURDF("plane.urdf")
start_pos = [0, 0, 0.1]
start_orn = pb.getQuaternionFromEuler([0, 0, -1.5607])

# Load the stretch
# robot_id = pb.loadURDF('./urdf/stretch.urdf', start_pos, start_orn)
# robot_id = pb.loadURDF('./urdf/planner_stretch_dex_wrist_simplified.urdf', start_pos, start_orn)
robot_id = pb.loadURDF(
    "./urdf/planner_calibrated_manipulation_mode.urdf", start_pos, start_orn
)

pos, orn = pb.getBasePositionAndOrientation(robot_id)

print(pb.getJointInfo(robot_id, 0))

for _ in range(1):
    for i in range(-100, 100):
        # Try
        for j in range(-100, 100, 10):
            # innermost loop is increasing y
            for k in range(-100, 100, 25):
                # set first few joints of the robot
                pb.resetBasePositionAndOrientation(robot_id, [0, 0, 0.1], [0, 0, 0, 1])
                pb.resetJointState(robot_id, 2, i * np.pi / 100)
                pb.resetJointState(robot_id, 0, j / 100.0)
                pb.resetJointState(robot_id, 1, k / 100.0)
                time.sleep(0.01)

input("press enter to terminate")
pb.disconnect()
