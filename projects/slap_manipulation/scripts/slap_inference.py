from typing import Any, Dict, Tuple

import hydra
import numpy as np
import rospy
import torch
import trimesh
from slap_manipulation.policy.action_prediction_module import ActionPredictionModule
from slap_manipulation.policy.interaction_prediction_module import (
    InteractionPredictionModule,
)
from slap_manipulation.utils.input_preprocessing import (
    get_local_action_prediction_problem,
)

from home_robot.utils.point_cloud import show_point_cloud
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
    ipm_input_vector: Tuple,
    p_i: np.ndarray,
    time: float = 0.0,
):
    """takes raw data from stretch_manipulation_env and converts it into input batch
    for Action Prediction Module by cropping around predicted p_i: interaction_point"""
    feat = np.concatenate(ipm_input_vector[0], axis=-1)
    xyz = ipm_input_vector[1]
    cropped_feat, cropped_xyz, status = get_local_action_prediction_problem(
        cfg, feat, xyz, p_i
    )
    if not status:
        raise RuntimeError(
            "Interaction Prediction Module predicted an interaction point with no tractable local problem around it"
        )
    proprio = get_proprio(raw_data, time=time)
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

    # get 8k points for tractable learning
    downsample_mask = np.arange(rgb.shape[0])
    np.random.shuffle(downsample_mask)
    if num_pts != -1:
        downsample_mask = downsample_mask[:num_pts]
    rgb = rgb[downsample_mask]
    xyz = xyz[downsample_mask]

    # mean-center the point cloud
    xyz -= xyz.mean(axis=0)

    if np.any(rgb > 1.0):
        rgb = rgb / 255.0
    if debug:
        show_point_cloud(xyz, rgb, orig=np.zeros(3))

    input_vector = (rgb, xyz, proprio, lang)
    return input_vector


@hydra.main(version_base=None, config_path="./conf", config_name="test")
def main(cfg):
    rospy.init_node("slap_inference")
    # create the robot object
    robot = StretchManipulationEnv(init_cameras=True)
    # create IPM object
    interaction_predictor = InteractionPredictionModule(dry_run=cfg.dry_run)
    interaction_predictor.to(interaction_predictor.device)
    # create APM object
    action_predictor = ActionPredictionModule(dry_run=cfg.dry_run)
    # load model-weights
    if cfg.interaction_weights:
        interaction_predictor.load_state_dict(torch.load(cfg.interaction_weights))
    if cfg.action_weights:
        action_predictor.load_state_dict(cfg.action_weights)

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
    while experiment_running:
        for i, cmd in enumerate(cmds):
            print(f"{i+1}. {cmd}")
        task_id = int(input("which task to solve, enter integer: "))
        input_cmd = [cmds[task_id - 1]]
        print(f"Executing {input_cmd}")
        # get from the robot: pcd=(xyz, rgb), gripper-state,
        # construct input vector from raw data
        raw_observations = robot.get_observation()
        ipm_input_vector = create_interaction_prediction_input(
            raw_observations, input_cmd, filter_depth=True, debug=True
        )
        # run inference on sensor data for IPM
        (
            interaction_point,
            interaction_scores,
            ipm_feat,
            input_down_pcd,
        ) = interaction_predictor.predict(*ipm_input_vector)
        print(f"Interaction point is {interaction_point}")
        experiment_running = False
        # ask if ok to run APM inference
        if cfg.execution.predict_action:
            current_time = [-1.0, 0.0, 1.0]
            for i in range(cfg.num_keypoints):
                # run APM inference on sensor
                raw_observations = robot.get_observation()
                apm_input_vector = create_action_prediction_input(
                    cfg,
                    raw_observations,
                    ipm_input_vector,
                    interaction_point,
                    time=current_time[i],
                )
                apm_input_vector += (input_cmd,)
                action = action_predictor.predict(*apm_input_vector)
                # ask if ok to execute
                res = input("Execute the output? (y/n)")
                if res == "y":
                    robot.apply_action(action)
                pass


if __name__ == "__main__":
    main()
