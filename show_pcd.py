import click
import numpy as np
import trimesh

import home_robot.utils.transformations as tra
from home_robot.motion.stretch import STRETCH_STANDOFF_DISTANCE, HelloStretchKinematics

# Create bullet client
from home_robot.utils.bullet import PbArticulatedObject, PbClient, PbObject
from home_robot.utils.point_cloud import show_point_cloud

"""
python projects/habitat_ovmm/eval_baselines_agent.py --baseline_config_path projects/habitat_ovmm/configs/agent/pick_skill_only.yaml --env_config_path projects/habitat_ovmm/configs/env/hssd_demo_gt.yaml habitat.task.pick_init=True habitat.task.episode_init=False habitat.environment.max_episode_steps=10  habitat.task.actions.arm_action.grasp_thresh_dist=0.1 habitat.dataset.episode_ids="[0]"
"""


def main(show_pcds=False):
    # Debug pointclouds

    data = np.load("test-sriram.npz")
    all_xyz = data["all_xyz"]
    # Object XYZ
    # xyz = data["xyz"]
    rgb = data["rgb"]
    pt = data["pt"]

    all_xyz = all_xyz.reshape(-1, 3)
    all_rgb = rgb.reshape(-1, 3)

    print("Show observations from habitat")
    if show_pcds:
        show_point_cloud(all_xyz, all_rgb / 255, orig=np.zeros(3))
    rot = tra.euler_matrix(0, 0, np.pi)
    all_xyz = trimesh.transform_points(all_xyz, rot)

    # Compute it in the correct frame
    print("Show corrected point cloud (from the previous observations from habitat)")
    pt = trimesh.transform_points(pt[None], rot)[0]
    if show_pcds:
        show_point_cloud(all_xyz, all_rgb / 255, orig=pt)
        # show_point_cloud(xyz, rgb / 255, orig=pt)

    client = PbClient(visualize=True)
    red_block = PbObject(
        "red_block", "./assets/red_block.urdf", start_pos=pt, client=client.id
    )
    blue_block = PbObject("blue_block", "./assets/blue_block.urdf", client=client.id)
    print("Created object to track target pt:", red_block)
    print("Created object to track grasp pt:", blue_block)
    # PLANNER_STRETCH_URDF = "assets/hab_stretch/urdf/planner_calibrated.urdf"
    MANIP_STRETCH_URDF = "assets/hab_stretch/urdf/stretch_manip_mode.urdf"
    # Load a robot model here
    robot = client.add_articulated_object("robot", MANIP_STRETCH_URDF)

    # Inverse kinematics
    #
    R = tra.euler_matrix(0, np.pi, 0)
    # R_x = tra.euler_matrix(-np.pi/4, 0, 0)
    # Top down grasp
    pos_top = pt.copy()
    pos_top[2] += STRETCH_STANDOFF_DISTANCE
    rot_top = tra.quaternion_from_matrix(R)
    # Side grasp
    pos_side = pt.copy()
    pos_side[1] += STRETCH_STANDOFF_DISTANCE
    rot_side = tra.quaternion_from_euler(np.pi / 2, 0, 0)

    model = HelloStretchKinematics()

    print("Target point to grasp:", pt)
    print("Raising the point by the size of the stretch gripper before doing IK...")
    pt[1] += STRETCH_STANDOFF_DISTANCE
    blue_block.set_pose(pt, (0, 0, 0, 1))
    print("Target point to grasp:", pt)

    cfg, res, info = model.manip_ik((pos_top, rot_top))
    print("--- TOP GRASP SOLUTION ---")
    print("cfg =", cfg)
    print("res =", res)
    print("inf =", info)

    if res is not True:
        print("Inverse kinematics failed! Trying from the side...")
        cfg, res, info = model.manip_ik((pos_side, rot_side))
        print("--- SIDE GRASP SOLUTION ---")
        print("cfg =", cfg)
        print("res =", res)
        print("inf =", info)

        if res is not True:
            print("--- SIDE GRASP SOLUTION ---")
            print("Still failed!")
            input("press enter when done")
            return

    cfg = model._to_manip_format(cfg)
    print("Fixed cfg =", cfg)

    model.set_articulated_object_positions(robot, cfg)

    print()
    print(
        "After switching to manipulation mode, then you can move the robot like this:"
    )
    print("base x motion:", cfg[0])
    print("lift:         ", cfg[1])
    print("arm extension:", cfg[2] + cfg[3] + cfg[4] + cfg[5])
    print("wrist yaw:    ", cfg[6])
    print("wrist pitch:  ", cfg[7])
    print("wrist roll:   ", cfg[8])
    input("press enter when done")


if __name__ == "__main__":
    main()
