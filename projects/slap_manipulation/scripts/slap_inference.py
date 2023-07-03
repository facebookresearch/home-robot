# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Tuple

import hydra
import numpy as np
import ros_numpy
import rospy
import tf

# import tf2_geometry_msgs
import tf2_ros
import torch
import trimesh
import yaml
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation as R
from slap_manipulation.policy.action_prediction_module import ActionPredictionModule
from slap_manipulation.policy.interaction_prediction_module import (
    InteractionPredictionModule,
)
from slap_manipulation.utils.input_preprocessing import (
    get_local_action_prediction_problem,
)

from home_robot.motion.stretch import STRETCH_GRASP_OFFSET
from home_robot.utils.point_cloud import show_point_cloud
from home_robot.utils.pose import to_matrix, to_pos_quat
from home_robot_hw.env.stretch_manipulation_env import StretchManipulationEnv

# TODO: move this to a constants file
STRETCH_GRIPPER_MAX = 0.6


def get_proprio(raw_data, time=0.0):
    """create a proprio vector from gripper-width
    proprio_vector: (gripper-action, gripper-width, time in task)
        gripper-action: 1.0 if gripper is closing, 0.0 if gripper is opening
        gripper-width: actual gripper width  TODO: normalize this
        time: normalized time in task (-1.0 is start of task, 1.0 is the end)
    """
    proprio = raw_data["gripper_state"].astype(np.float64)
    if proprio < 0.8 * STRETCH_GRIPPER_MAX:
        proprio = np.array([1.0, proprio, time])
    else:
        proprio = np.array([0.0, proprio, time])

    return proprio


def create_action_prediction_input(
    cfg,
    raw_data: Dict[str, Any],
    feat: np.ndarray,
    xyz: np.ndarray,
    p_i: np.ndarray,
    time: float = 0.0,
    debug: bool = False,
):
    """takes raw data from stretch_manipulation_env and converts it into input batch
    for Action Prediction Module by cropping around predicted p_i: interaction_point"""
    cropped_feat, cropped_xyz, status = get_local_action_prediction_problem(
        cfg, feat, xyz, p_i
    )
    if np.any(cropped_feat > 1.0):
        cropped_feat = cropped_feat / 255.0
    if not status:
        raise RuntimeError(
            "Interaction Prediction Module predicted an interaction point with no tractable local problem around it"
        )
    proprio = get_proprio(raw_data, time=time)
    if debug:
        print("create_action_prediction_input")
        show_point_cloud(cropped_xyz, cropped_feat)
    return (cropped_feat, cropped_xyz, proprio)


def create_interaction_prediction_input(
    raw_data: dict, lang: list[str], filter_depth=False, debug=False, num_pts=8000
):
    """takes raw data from stretch_manipulation_env, and language command from user.
    Converts it into input batch used in Interaction Prediction Module
    Return: obs_vector = ((rgb), xyz, proprio, lang)
        obs_vector[0]: tuple of features per point in PCD; each element is expected to be Nxfeat_dim
        obs_vector[1]: xyz coordinates of PCD points; N x 3
        obs_vector[2]: proprioceptive state of robot; 3-dim vector: [gripper-state, gripper-width, time] # probably do not need time for IPM training
        obs_vector[3]: language command; list of 1 string # should this be a list? only 1 string.
    """
    input_vector = ()
    depth = raw_data["depth"]
    rgb = raw_data["rgb"].astype(np.float64)
    xyz = raw_data["xyz"].astype(np.float64)
    camera_pose = raw_data["camera_pose"]
    proprio = get_proprio(raw_data, time=-1.0)

    depth = depth.reshape(-1)
    rgb = rgb.reshape(-1, 3)
    cam_xyz = xyz.reshape(-1, 3)
    xyz = trimesh.transform_points(cam_xyz, camera_pose)

    # apply depth and z-filter for comparative distribution to training data
    if filter_depth:
        valid_depth = np.bitwise_and(depth > 0.1, depth < 1.0)
        rgb = rgb[valid_depth, :]
        xyz = xyz[valid_depth, :]
        z_mask = xyz[:, 2] > 0.7
        rgb = rgb[z_mask, :]
        xyz = xyz[z_mask, :]
    og_xyz = np.copy(xyz)
    og_rgb = np.copy(rgb)

    # get 8k points for tractable learning
    downsample_mask = np.arange(rgb.shape[0])
    np.random.shuffle(downsample_mask)
    if num_pts != -1:
        downsample_mask = downsample_mask[:num_pts]
    rgb = rgb[downsample_mask]
    xyz = xyz[downsample_mask]

    # mean-center the point cloud
    mean = xyz.mean(axis=0)
    xyz -= mean
    og_xyz -= mean

    if np.any(rgb > 1.0):
        rgb = rgb / 255.0
    if debug:
        print("create_action_prediction_input")
        show_point_cloud(xyz, rgb, orig=np.zeros(3))

    input_vector = (rgb, xyz, proprio, lang, mean)
    return input_vector, og_rgb, og_xyz


def get_in_base_frame(robot, mean: np.ndarray) -> np.ndarray:
    """get transform between base and map; use this to transform input into base_frame"""
    map_to_base = robot.get_pose("base_link", "map")
    mean_h = np.concatenate((mean, np.array([1])))
    new_mean = map_to_base @ mean_h
    return new_mean[:-1]


@hydra.main(version_base=None, config_path="./conf", config_name="slap_inference")
def main(cfg):
    rospy.init_node("slap_inference")

    # create tf2 buffer + listener
    # tf_buffer = tf2_ros.Buffer(rospy.Duration(10))
    # tf_listener = tf2_ros.TransformListener(tf_buffer)
    # base2map = tf_buffer.lookup_transform("map", "base_link", rospy.Duration(5))
    # create the robot object
    robot = StretchManipulationEnv(init_cameras=True)
    # create IPM object
    interaction_predictor = InteractionPredictionModule(dry_run=cfg.dry_run)
    interaction_predictor.to(interaction_predictor.device)
    # create APM objects
    action_predictors = []
    for _ in range(cfg.num_actions):
        action_predictors.append(ActionPredictionModule(cfg))
    # load model-weights
    if cfg.interaction_weights:
        interaction_predictor.load_state_dict(torch.load(cfg.interaction_weights))
    if cfg.action_weights:
        for i in range(cfg.num_actions):
            action_predictors[i].load_state_dict(torch.load(cfg.action_weights[i]))
            action_predictors[i].to(action_predictors[i].device)

    print("Loaded models successfully")
    cmds = [
        "pick up the bottle",
        "open top drawer",
        "open bottom drawer",
        "close the drawers",
        "place in the drawer",
        "pick up lemon from basket",
        "place lemon in bowl",
        "place in basket",
    ]
    experiment_running = True
    ros_pub = tf2_ros.TransformBroadcaster()
    actions = []
    world_actions = []
    while experiment_running:
        for i, cmd in enumerate(cmds):
            print(f"{i+1}. {cmd}")
        task_id = int(input("which task to solve, enter integer: "))
        input_cmd = [cmds[task_id - 1]]
        print(f"Executing {input_cmd}")
        # get from the robot: pcd=(xyz, rgb), gripper-state,
        # construct input vector from raw data
        raw_observations = robot.get_observation()
        ipm_input_vector, og_rgb, og_xyz = create_interaction_prediction_input(
            raw_observations, input_cmd, filter_depth=True, debug=False
        )
        print(f"PCD Mean: {ipm_input_vector[-1]}")
        # run inference on sensor data for IPM
        (
            interaction_point_idx,
            interaction_scores,
            ipm_feat,
            input_down_pcd,
        ) = interaction_predictor.predict(*ipm_input_vector[:-1])
        interaction_point = input_down_pcd[0][interaction_point_idx]
        print(f"Interaction point is {interaction_point}")
        experiment_running = True
        # ask if ok to run APM inference
        actions = []
        if cfg.execution.predict_action:
            current_time = [-1.0, 0.0, 1.0]
            for i in range(cfg.num_actions):
                # run APM inference on sensor
                raw_observations = robot.get_observation()
                apm_input_vector = create_action_prediction_input(
                    cfg,
                    raw_observations,
                    og_rgb,
                    og_xyz,
                    interaction_point,
                    time=current_time[i],
                    debug=False,
                )
                apm_input_vector += (
                    input_cmd,
                    interaction_point,
                    input_down_pcd[1],
                    input_down_pcd[0],
                )
                predicted_action = action_predictors[i].predict(*apm_input_vector)
                ori = R.from_matrix(predicted_action["predicted_ori"])
                # mean_base_frame = get_in_base_frame(
                #     robot, ipm_input_vector[-1]
                # ).reshape(-1)
                action = {
                    "pos": predicted_action["predicted_pos"].reshape(-1)
                    + ipm_input_vector[-1].reshape(-1),
                    "ori": ori.as_quat(),
                    "gripper": int(predicted_action["gripper_act"]),
                }
                # ask if ok to execute
                pos = action["pos"]
                rot = action["ori"]
                pose_message = TransformStamped()
                pose_stamped = PoseStamped()
                pose_message.header.stamp = rospy.Time.now()
                pose_message.header.frame_id = "base_link"
                pose_stamped.header.stamp = rospy.Time.now()
                pose_stamped.header.frame_id = "base_link"

                pose_message.child_frame_id = f"prediction_{i}"
                pose_message.transform.translation.x = pos[0]
                pose_message.transform.translation.y = pos[1]
                pose_message.transform.translation.z = pos[2]
                pose_stamped.pose.position.x = pos[0]
                pose_stamped.pose.position.y = pos[1]
                pose_stamped.pose.position.z = pos[2]

                pose_message.transform.rotation.x = rot[0]
                pose_message.transform.rotation.y = rot[1]
                pose_message.transform.rotation.z = rot[2]
                pose_message.transform.rotation.w = rot[3]
                pose_stamped.pose.orientation.x = rot[0]
                pose_stamped.pose.orientation.y = rot[1]
                pose_stamped.pose.orientation.z = rot[2]
                pose_stamped.pose.orientation.w = rot[3]

                ros_pub.sendTransform(pose_message)
                # pose_stamped = tf2_geometry_msgs.do_transform_pose(
                #     pose_stamped, base2map
                # )
                offset = STRETCH_GRASP_OFFSET.copy()
                gripper_pose_mat = to_matrix(action["pos"], action["ori"]) @ offset
                action["pos"], _ = to_pos_quat(gripper_pose_mat)
                actions.append(action)
                # input("Press enter to continue")
                # res = input("Execute the output? (y/n)")
                # if res == "y":
                #     robot.apply_action(action)
                # pass
    # dump the actions list into a YAML file
    yaml.dump(actions, open("actions.yaml", "w"))


if __name__ == "__main__":
    main()
