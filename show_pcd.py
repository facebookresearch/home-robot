import numpy as np
import trimesh

import home_robot.utils.transformations as tra
from home_robot.motion.stretch import HelloStretchKinematics

# Create bullet client
from home_robot.utils.bullet import PbArticulatedObject, PbClient, PbObject
from home_robot.utils.point_cloud import show_point_cloud

"""
python projects/habitat_ovmm/eval_baselines_agent.py --baseline_config_path projects/habitat_ovmm/configs/agent/pick_skill_only.yaml --env_config_path projects/habitat_ovmm/configs/env/hssd_demo_gt.yaml habitat.task.pick_init=True habitat.task.episode_init=False habitat.environment.max_episode_steps=10  habitat.task.actions.arm_action.grasp_thresh_dist=0.1 habitat.dataset.episode_ids="[0]"
"""

# Debug pointclouds
show_pcds = False

data = np.load("test.npz")
all_xyz = data["all_xyz"]
xyz = data["xyz"]
rgb = data["rgb"]
pt = data["pt"]

all_xyz = all_xyz.reshape(-1, 3)
all_rgb = rgb.reshape(-1, 3)

print("Show observations from habitat")
if show_pcds:
    show_point_cloud(all_xyz, rgb / 255, orig=np.zeros(3))
rot = tra.euler_matrix(0, 0, np.pi)
all_xyz = trimesh.transform_points(all_xyz, rot)

# Compute it in the correct frame
print("Show corrected point cloud (from the previous observations from habitat)")
pt = trimesh.transform_points(pt[None], rot)[0]
if show_pcds:
    show_point_cloud(all_xyz, rgb / 255, orig=pt)
    # show_point_cloud(xyz, rgb / 255, orig=pt)

client = PbClient(visualize=True)
red_block = PbObject(
    "red_block", "./assets/red_block.urdf", start_pos=pt, client=client.id
)
PLANNER_STRETCH_URDF = "assets/hab_stretch/urdf/planner_calibrated.urdf"
MANIP_STRETCH_URDF = "assets/hab_stretch/urdf/stretch_manip_mode.urdf"
# Load a robot model here
robot = client.add_articulated_object("robot", MANIP_STRETCH_URDF)

# Inverse kinematics
grasp_orientation = tra.quaternion_from_euler(0, np.pi, 0)

model = HelloStretchKinematics()
cfg, res, info = model.manip_ik((pt, grasp_orientation))
print("cfg =", cfg)
print("res =", res)
print("inf =", info)

model.set_articulated_object_positions(robot, cfg)
breakpoint()

input("press enter when done")
