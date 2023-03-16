import numpy as np

from home_robot.utils.pose import to_pos_quat
from home_robot_hw.remote import StretchClient

robot = StretchClient(init_node=True)

pose_mat = np.array(
    [
        [0.26931483, -0.96272663, -0.02503896, 0.22509192],
        [-0.96215804, -0.26785372, -0.05006283, -0.45297395],
        [0.04149004, 0.0375741, -0.99843215, 0.98808103],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
pos_grasp, quat_grasp = to_pos_quat(pose_mat)

pos_standoff = pos_grasp + np.array([0, 0, 0.08])

robot.switch_to_manipulation_mode()
robot.manip.home()

pos_curr, quat_curr = robot.manip.get_ee_pose()
print(f"Pose 0: pos={pos_curr}, quat={quat_curr}")
print(f"Pose desired: pos={pos_grasp}, quat={quat_grasp}")

robot.manip.goto_ee_pose((pos_curr + pos_standoff) / 2.0, quat_curr)
robot.manip.goto_ee_pose(pos_standoff, quat_grasp)

pos_curr, quat_curr = robot.manip.get_ee_pose()
print(f"Pose 1 (standoff): pos={pos_curr}, quat={quat_curr}")

robot.manip.goto_ee_pose(pos_grasp, quat_grasp)

pos_curr, quat_curr = robot.manip.get_ee_pose()
print(f"Pose 2 (grasp): pos={pos_curr}, quat={quat_curr}")

robot.manip.goto_ee_pose(pos_standoff, quat_grasp)

pos_curr, quat_curr = robot.manip.get_ee_pose()
print(f"Pose 3 (standoff): pos={pos_curr}, quat={quat_curr}")
