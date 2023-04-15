import argparse
import copy
import os
from pathlib import Path

import numpy as np
import trimesh.transformations as tra
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.const import colors
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

# Create all the tasks for data collection
from rlbench.tasks import (
    BlockPyramid,
    CloseDrawer,
    EmptyContainer,
    InsertOntoSquarePeg,
    OpenDoor,
    OpenDrawer,
    PickAndLift,
    PourFromCupToCup,
    PutPlateInColoredDishRack,
    ReachAndDrag,
    ReachTarget,
    TakeLidOffSaucepan,
)
from rlbench_continual.utils import demo_loading_utils
from tqdm import tqdm

from home_robot.utils.data_tools.writer import DataWriter

tasks = {
    "open_door": (OpenDoor, None),
    "reach_target": (ReachTarget, None),
    "open_drawer": (OpenDrawer, None),
    "close_drawer": (CloseDrawer, None),
    "reach_target_1": (ReachTarget, 0),
    "reach_target_2": (ReachTarget, 1),
    "reach_target_3": (ReachTarget, 2),
    "reach_target_4": (ReachTarget, 3),
    "reach_target_5": (ReachTarget, 4),
    "open_drawer_btm": (OpenDrawer, 0),
    "open_drawer_mid": (OpenDrawer, 1),
    "open_drawer_top": (OpenDrawer, 2),
    "close_drawer_btm": (CloseDrawer, 0),
    "close_drawer_mid": (CloseDrawer, 1),
    "close_drawer_top": (CloseDrawer, 2),
    "make a pyramid out of blocks": (BlockPyramid, 0),
    "take_lid_off_saucepan": (TakeLidOffSaucepan, None),
}

color_tasks = {
    "pick_and_lift": PickAndLift,
    "reach_and_drag": ReachAndDrag,
    "insert_onto_square_peg": InsertOntoSquarePeg,
    "put_plate_in_colored_dish_rack": PutPlateInColoredDishRack,
    "empty_container": EmptyContainer,
    "pour_from_cup_to_cup": PourFromCupToCup,
}

for task_key, task_type in color_tasks.items():
    for color_id in range(len(colors)):
        tasks[f"{task_key}_{color_id}"] = (task_type, color_id)

reach = ["reach_target_%d" % (i + 1) for i in range(5)]
open_drawer = ["open_drawer_%s" % i for i in ["btm", "mid", "top"]]
close_drawer = ["close_drawer_%s" % i for i in ["btm", "mid", "top"]]


def overhead(obs):
    return "overhead", obs.overhead_depth, obs.overhead_point_cloud, obs.overhead_rgb


def front(obs):
    return "front", obs.front_depth, obs.front_point_cloud, obs.front_rgb


def left_side(obs):
    return (
        "left",
        obs.left_shoulder_depth,
        obs.left_shoulder_point_cloud,
        obs.left_shoulder_rgb,
    )


def right_side(obs):
    return (
        "right",
        obs.right_shoulder_depth,
        obs.right_shoulder_point_cloud,
        obs.right_shoulder_rgb,
    )


def wrist(obs):
    return (
        "wrist",
        obs.wrist_depth,
        obs.wrist_point_cloud,
        obs.wrist_rgb,
    )


def get_xyz_rgb(obs, num_pts=30000):
    xyzs, rgbs = [], []
    for view in overhead, front, left_side, right_side, wrist:
        name, d, xyz, rgb = view(obs)
        H, W = d.shape
        d = d.reshape(-1)
        xyz, rgb = xyz.reshape(-1, 3), rgb.reshape(-1, 3)
        mask = np.bitwise_and(d > 0.1, d < 3.0)
        mask2 = np.bitwise_and(xyz[:, 0] > -0.5, xyz[:, 2] > 0.5)
        # import matplotlib.pyplot as plt
        # plt.subplot(131); plt.imshow(mask2.reshape(H, W))
        # plt.subplot(132); plt.imshow(xyz.reshape(H, W, 3))
        # plt.subplot(133); plt.imshow(rgb.reshape(H, W, 3))
        # plt.show()
        xyzs.append(xyz[mask2])
        rgbs.append(rgb[mask2])
    xyz = np.concatenate(xyzs, axis=0)
    rgb = np.concatenate(rgbs, axis=0)

    # TODO: remove debug code
    # from data_tools.point_cloud import show_point_cloud
    # show_point_cloud(xyz, rgb, orig=np.zeros(3))

    # Save only 20k points
    idx = np.arange(xyz.shape[0])
    np.random.shuffle(idx)
    xyz, rgb = xyz[idx[:num_pts]], rgb[idx[:num_pts]]
    return xyz.astype(np.float32), rgb.astype(np.float32)


def collect_data(args, task_names, headless):
    # To use 'saved' demos, set the path below, and set live_demos=False
    live_demos = True
    acc_tol = 0.01
    kp_max_acc = 0.2
    num_demos = args.num_train_demos + args.num_valid_demos
    obs_config = ObservationConfig(gripper_joint_positions=True)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=obs_config,
        headless=headless,
    )
    env.launch()
    Path(args.train_dir).mkdir(parents=True, exist_ok=True)
    Path(args.valid_dir).mkdir(parents=True, exist_ok=True)
    for task_name in task_names:
        train_writer = DataWriter(
            args.filename_root + "_" + str(task_name) + ".h5", args.train_dir
        )
        valid_writer = DataWriter(
            args.filename_root + "_" + str(task_name) + ".h5", args.valid_dir
        )
        task_class, task_variant = tasks[task_name]
        task = env.get_task(task_class)
        if task_variant is not None:
            task.set_variation(task_variant)

        current_demo_id = train_writer.num_trials + valid_writer.num_trials

        # Collect and write 1 at a time in case something goes wrong
        for i in tqdm(range(current_demo_id, num_demos), ncols=50):
            if i < args.num_train_demos:
                writer = train_writer
            else:
                writer = valid_writer

            # Demo setup
            desc, _ = task.reset()
            print("Generating data for task =", task_name, desc)
            demo = task.get_demos(1, live_demos=True)[0]

            # Add some waypoints in case we have a use for them
            for waypoint in task._task.get_waypoints():
                pos = waypoint._waypoint.get_position()
                rot = waypoint._waypoint.get_orientation()
                writer.add_frame(waypoint_pos=pos, waypoint_rot=rot)

            rlbench_keypoints = demo_loading_utils.keypoint_discovery(demo)

            writer.add_config(iter=i)
            writer.add_config(cmd=task_name)
            writer.add_config(descriptions=",".join(desc))

            # Automatically annotate subgoals
            keypoints = np.zeros(len(demo))
            all_keypoints = []
            for j in range(len(demo)):
                if j == len(demo) - 1:
                    keypoints[j] = 1
                    all_keypoints.append(j)
                    # print("- s/e kp at", j)
                    continue
                elif j == 0:
                    continue
                elif keypoints[j - 1] > 0:
                    continue
                elif demo[j - 1].gripper_open != demo[j].gripper_open:
                    keypoints[j] = 1
                    all_keypoints.append(j)
                    # print("- grip kp at", j)
                    continue
                if j < 2:
                    continue
                if j > len(demo) - 3:
                    continue
                acc0 = np.linalg.norm(demo[j - 2].joint_velocities)
                acc1 = np.linalg.norm(demo[j].joint_velocities)
                acc2 = np.linalg.norm(demo[j + 2].joint_velocities)
                # print(j, acc1)
                # print(j, acc0 - acc1, acc2 - acc1)
                if (
                    acc0 > acc1 + acc_tol
                    and acc2 > acc1 + acc_tol
                    and acc1 < kp_max_acc
                ):
                    keypoints[j] = 1
                    all_keypoints.append(j)
                    # print("- acc kp at", j)
                    continue

            # Compute the next thing we should predict
            next_keypoint = np.zeros(len(demo))
            prev_next_keypoint = None
            for j in range(len(demo)):
                idx = len(demo) - 1 - j
                if keypoints[idx] > 0:
                    next_keypoint[idx] = idx
                    prev_next_keypoint = idx
                else:
                    next_keypoint[idx] = prev_next_keypoint

            # Save both my keypoints...
            writer.add_config(new_keypoints=all_keypoints)
            # And the ones from RLBench/ARM
            writer.add_config(keypoints=rlbench_keypoints)
            if task_variant is not None:
                writer.add_config(task_variant=task_variant)
            else:
                writer.add_config(task_variant=-1)

            kpi = 0
            rlb_next_keypoint = rlbench_keypoints[kpi]
            episode_length = 10
            for j, obs in enumerate(demo):
                ee_xyz = obs.gripper_pose[:3]
                ee_rot = obs.gripper_pose[3:]
                q, dq = obs.joint_positions, obs.joint_velocities
                dq_dist = np.linalg.norm(dq)
                g = obs.gripper_open
                x, y, z, w = ee_rot
                T = tra.quaternion_matrix([w, x, y, z])  # [:3, :3]
                T[:3, 3] = ee_xyz
                if obs.gripper_joint_positions is not None:
                    obs.gripper_joint_positions = np.clip(
                        obs.gripper_joint_positions, 0.0, 0.04
                    )
                ts = (
                    1.0 - (j / float(episode_length - 1))
                ) * 2.0 - 1.0  # so many questions, pulled from peract_colab.rlbench.backend.utils
                low_dim_state = np.array(
                    [obs.gripper_open, *obs.gripper_joint_positions, ts],
                    dtype=np.float32,
                )

                # xyz, rgb = get_xyz_rgb(obs)
                imgs = {}
                for view in overhead, front, left_side, right_side, wrist:
                    name, depth, xyz, rgb = view(obs)
                    imgs[name + "_depth"] = depth
                    imgs[name + "_xyz"] = xyz
                    imgs[name + "_rgb"] = rgb
                writer.add_frame(**imgs)

                # TODO: remove debug code
                # print(j, "/", len(demo), "=", rlb_next_keypoint, rlbench_keypoints)
                if j >= rlb_next_keypoint:
                    kpi += 1
                    # Next keypoint, but it can't be more than the last one
                    rlb_next_keypoint = rlbench_keypoints[
                        min(kpi, len(rlbench_keypoints) - 1)
                    ]

                writer.add_frame(
                    ee_xyz=ee_xyz,
                    ee_rot=ee_rot,
                    # xyz=xyz, rgb=rgb,
                    q=q,
                    dq=dq,
                    gripper=g,
                    next_keypoint=rlb_next_keypoint,
                    new_next_keypoint=next_keypoint[j],
                    low_dim_state=low_dim_state,
                )
            writer.write_trial(str(task_name) + str(i))

    env.shutdown()


def parse_args():
    parser = argparse.ArgumentParser("rlbench_data")
    parser.add_argument("--num_train_demos", type=int, default=10)
    parser.add_argument("--num_valid_demos", type=int, default=3)
    parser.add_argument("--filename_root", default="data")
    parser.add_argument("--train_dir", default="data/rlbench/train_roc_pan")
    parser.add_argument("--valid_dir", default="data/rlbench/valid_roc_pan")
    parser.add_argument(
        "--tasks",
        default="open_door,reach_target,open_drawer,close_drawer,"
        "take_lid_off_saucepan",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="visualize RL bench while collecting data",
    )
    parser.add_argument("--separate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    # used to write out data files for RL bench
    args = parse_args()
    # For testing
    # tasks_to_gen = ["reach_target_1"]
    # tasks_to_gen = reach + open_drawer + close_drawer
    # tasks_to_gen += ["take_lid_off_saucepan"]
    if args.tasks == "colors":
        tasks_to_gen = []
        for task_key in color_tasks.keys():
            # red, green, blue, yellow, orange, violet
            for color_id in [0, 3, 4, 6, 11, 16]:
                tasks_to_gen.append(f"{task_key}_{color_id}")
    else:
        tasks_to_gen = args.tasks.split(",")
    # tasks_to_gen = ["take_lid_off_saucepan"]
    # tasks_to_gen += reach + open_drawer + close_drawer

    if args.separate:
        for task_name in tasks_to_gen:
            task_args = copy.deepcopy(args)
            task_args.train_dir = os.path.join(task_args.train_dir, task_name)
            task_args.valid_dir = os.path.join(task_args.valid_dir, task_name)
            collect_data(args, [task_name], headless=(not args.visualize))
    else:
        collect_data(args, tasks_to_gen, headless=(not args.visualize))
