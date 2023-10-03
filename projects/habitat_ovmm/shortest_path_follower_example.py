#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import magnum as mn
import numpy as np
from habitat.core.utils import try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from utils.config_utils import (
    create_agent_config,
    create_env_config,
    get_habitat_config,
    get_omega_config,
)
from utils.env_utils import create_ovmm_env_fn

from home_robot.core.interfaces import DiscreteNavigationAction

cv2 = try_cv2_import()

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# def _quat_to_xy_heading(quat):
#     direction_vector = np.array([0, 0, -1])

#     heading_vector = quaternion_rotate_vector(quat, direction_vector)

#     phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
#     return np.array([phi], dtype=np.float32)

DISCRETE_ACTION_MAP = {
    HabitatSimActions.stop: DiscreteNavigationAction.STOP,
    HabitatSimActions.move_forward: DiscreteNavigationAction.MOVE_FORWARD,
    HabitatSimActions.turn_left: DiscreteNavigationAction.TURN_LEFT,
    HabitatSimActions.turn_right: DiscreteNavigationAction.TURN_RIGHT,
}


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], output_size)


def shortest_path_example(config):
    """
    Example script for performing oracle navigation to object in OVMM episodes.
    Utilizes ShortestPathFollower to output discrete actions in HabitatOpenVocabManipEnv.
    Note: HabitatOpenVocabManipEnv internally takes care of converting discrete actions to continuous actions.
    Note: The environment heirarchy above is as follows:
    ovmm_env
        HabitatOpenVocabManipEnv
    ovmm_env.habitat_env
        GymHabitatEnv<HabGymWrapper instance>
    ovmm_env.habitat_env.env
        HabGymWrapper instance
    ovmm_env.habitat_env.env._env
        RLTaskEnv instance
    ovmm_env.habitat_env.env._env._env
        habitat.core.env.Env
    ovmm_env.habitat_env.env._env._env.sim
        OVMMSim
    """
    ovmm_env = create_ovmm_env_fn(config)
    print(f"Total number of episodes in env: {ovmm_env.number_of_episodes}")

    # Keep a default minimum goal radius is 0.1, but increase it if robot step size is too large
    goal_radius = max(0.1, getattr(config.habitat.simulator, "forward_step_size", 0.1))

    follower = ShortestPathFollower(
        ovmm_env.habitat_env.env._env.habitat_env.sim, goal_radius, False
    )

    for _ in range(ovmm_env.number_of_episodes):
        ovmm_env.reset()
        episode_id = ovmm_env.get_current_episode().episode_id
        dirname = os.path.join(
            IMAGE_DIR,
            "shortest_path_example_ovmm",
            f"{episode_id}",
        )
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        log_file_name = f"logs_ep_{episode_id}.txt"

        with open(os.path.join(os.getcwd(), dirname, log_file_name), "w") as f:
            f.write("Environment creation successful\n")
            f.write("Agent stepping around inside environment.\n")
            images_third_person = []
            steps, max_steps = 0, 1000
            info = None
            object_pos = ovmm_env.habitat_env.env._env.habitat_env.current_episode.candidate_objects[
                0
            ].position
            goal_pos = ovmm_env.habitat_env.env._env.habitat_env.current_episode.candidate_goal_receps[
                0
            ].position

            goal_pos_viewpoints = ovmm_env.habitat_env.env._env.habitat_env.current_episode.candidate_goal_receps[
                0
            ].view_points
            ious = [viewpoint.iou for viewpoint in goal_pos_viewpoints]
            max_iou_idx = ious.index(max(ious))
            goal_pos_max_iou_viewpoint = goal_pos_viewpoints[
                max_iou_idx
            ].agent_state.position

            # for idx, viewpoint in enumerate(goal_pos_viewpoints):
            #     print(f"\nViewpoint {idx}")
            #     print(f"Viewpoint position: {viewpoint.agent_state.position}")
            #     print(f"Viewpoint rotation: {viewpoint.agent_state.rotation}")
            #     print(f"Viewpoint IOU: {viewpoint.iou}")

            # print(f"\nMax IOU viewpoint {max_iou_idx}")
            # print(f"Max IOU viewpoint position: {goal_pos_max_iou_viewpoint}")
            goal = mn.Vector3(goal_pos_max_iou_viewpoint)
            goal_orientation = quaternion_from_coeff(
                goal_pos_viewpoints[max_iou_idx].agent_state.rotation
            )
            while (
                not ovmm_env.habitat_env.env._env.habitat_env.episode_over
                and steps < max_steps
            ):
                if steps != 0:
                    # curr_quat = follower._sim.robot.sim_obj.rotation
                    # curr_rotation = [
                    #     curr_quat.vector.x,
                    #     curr_quat.vector.y,
                    #     curr_quat.vector.z,
                    #     curr_quat.scalar,
                    # ]
                    # curr_quat = quaternion_from_coeff(
                    # curr_rotation
                    # )
                    # # get heading angle
                    # rot = _quat_to_xy_heading(
                    #     curr_quat.inverse()
                    # )
                    # rot = rot - np.pi / 2
                    # # convert back to quaternion
                    # ang_pos = rot[0]
                    # curr_rot = mn.Quaternion(
                    #     mn.Vector3(0, np.sin(ang_pos / 2), 0), np.cos(ang_pos / 2)
                    # )

                    f.write(
                        f"Current agent location:\t{follower._sim.robot.base_pos}\n"
                        # f"Current agent orientation:\t{curr_quat}\n"
                        # f"Current agent orientation:\t{curr_rot}\n"
                        f"Navigation goal location:\t{goal}\n"
                        f"Navigation goal orientation:\t{goal_orientation}\n"
                        # f"Difference between orientations:\t{curr_quat * goal_orientation.inverse()}\n"
                        f"info['ovmm_dist_to_pick_goal']:\t{info['ovmm_dist_to_pick_goal']}\n"
                        f"info['ovmm_dist_to_keep_goal']:\t{info['ovmm_dist_to_place_goal']}\n"
                    )

                f.write(f"\nTimestep: {steps}\n")
                print(f"Timestep: {steps}")
                best_action = DISCRETE_ACTION_MAP[follower.get_next_action(goal)]
                f.write(f"Agent action taken: {best_action}\n")
                if best_action is None:
                    break

                observations, done, info = ovmm_env.apply_action(best_action, info)
                steps += 1
                info["timestep"] = steps
                if config.PRINT_IMAGES and config.GROUND_TRUTH_SEMANTICS:
                    images_third_person.append(observations.third_person_image)

            if len(images_third_person):
                images_to_video(images_third_person, dirname, "trajectory_third_person")
            if steps >= max_steps:
                f.write("Max steps reached! Aborting episode...")
            else:
                f.write("Episode finished succesfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_type",
        type=str,
        choices=["local", "local_vectorized", "remote"],
        default="local",
    )
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/env/hssd_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "overrides",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    # get habitat config
    habitat_config, _ = get_habitat_config(
        args.habitat_config_path,
        overrides=args.overrides
        + [
            "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map"
        ],
    )

    # get env config
    env_config = get_omega_config(args.env_config_path)

    # merge habitat and env config to create env config
    env_config = create_env_config(
        habitat_config, env_config, evaluation_type=args.evaluation_type
    )

    shortest_path_example(env_config)
