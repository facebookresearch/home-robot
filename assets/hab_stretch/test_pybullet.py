import time

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
# robot_id = pb.loadURDF('./urdf/hab_stretch_dex_wrist_simplified.urdf', start_pos, start_orn)
# robot_id = pb.loadURDF('./habitat_stretch/urdf/hab_stretch_arm_heavy.urdf', start_pos, start_orn)
robot_id = pb.loadURDF("./habitat_stretch/urdf/hab_stretch.urdf", start_pos, start_orn)

for i in range(10000):
    pb.stepSimulation()
    time.sleep(1.0 / 240.0)
    pos, orn = pb.getBasePositionAndOrientation(robot_id)
    # Debug pose information as the robot falls over
    print("position =", pos, "orientation =", orn)

# With wheels - base is at:
# position = (-0.00213652343264473, 0.10666150396655555, 0.09140726544649008)
input("press enter to terminate")
pb.disconnect()
