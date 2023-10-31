#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pickle
from concurrent import futures
from datetime import datetime
from typing import Optional, Tuple

import click
import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import rospy

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv
from home_robot_hw.utils.config import load_config
from habitat.core.dataset import BaseEpisode

def grpc_dumps(entity):
    return pickle.dumps(entity)


def grpc_loads(entity):
    return pickle.loads(entity)


class Environment(evaluation_pb2_grpc.EnvironmentServicer):
    """RPC version of the environment."""

    def __init__(
        self,
        test_pick=False,
        reset_nav=False,
        pick_object="cup",
        start_recep="table",
        goal_recep="chair",
        dry_run=False,
        visualize_maps=False,
        visualize_grasping=False,
        test_place=False,
        cat_map_file=None,
        max_num_steps=200,
        config_path="projects/real_world_ovmm/configs/agent/eval.yaml",
    ) -> None:
        super().__init__()

        self.test_pick = test_pick
        self.reset_nav = reset_nav
        self.pick_object = pick_object
        self.start_recep = start_recep
        self.goal_recep = goal_recep
        self.dry_run = dry_run
        self.visualize_maps = visualize_maps
        self.visualize_grasping = visualize_grasping
        self.test_place = test_place
        self.cat_map_file = cat_map_file
        self.max_num_steps = max_num_steps
        self.config_path = config_path

        self._env = None
        self._env_number_of_episodes = None
        self._episode_metrics = {}
        self._current_episode_key = None
        self._current_episode_metrics = {}

        self._t = 0

    def init_env(self, request, context):
        """Initialize robot environment"""


        print("- Loading configuration")
        config = load_config(
            config_path=self.config_path, visualize=self.visualize_maps
        )

        print("- Creating environment")
        self._env = StretchPickandPlaceEnv(
            config=config,
            test_grasping=self.test_pick,
            dry_run=self.dry_run,
            cat_map_file=self.cat_map_file,
            visualize_grasping=self.visualize_grasping,
        )
        if self._env.number_of_episodes is not None:
            self._env_number_of_episodes = self._env.number_of_episodes
        else:
            self._env_number_of_episodes = 10000

        self._robot = self._env.get_robot()

        if self.reset_nav:
            print("- Sending the robot to [0, 0, 0]")
            # Send it back to origin position to make testing a bit easier
            self._robot.nav.navigate_to([0, 0, 0])

        self._t = 0

        return evaluation_pb2.Package()

    def number_of_episodes(self, request, context):
        """Return number of episodes"""
        ## does real world have episodes yet? Only looks like one episode
        return evaluation_pb2.Package(
            SerializedEntity=grpc_dumps(self._env_number_of_episodes)
        )

    def reset(self, request, context):
        """Start a new episode"""
        ## real world robot doesn't seem to have reset so this only works for first episode
        observations = self._env.get_observation()
        return evaluation_pb2.Package(SerializedEntity=grpc_dumps(observations))

    def get_current_episode(self, request, context):
        """Return current episode id"""
        current_episode = BaseEpisode(
                  episode_id='743', 
                  scene_id='data/hssd-han/scenes-uncluttered/104348328_171513363.scene_instance.json'
            ) # real world doesn't seem to have episodes yet
        return evaluation_pb2.Package(SerializedEntity=grpc_dumps(current_episode))

    def apply_action(self, request, context):
        """Recieve action from the agent and execute action on the robot.
        Return the observations, done boolean and hab_info"""
        self._t += 1
        action, info = grpc_loads(request.SerializedEntity)
        done = self._env.apply_action(action, info=info)  # is prev_obs required?

        hab_info = {}

        if "skill_done" in info and info["skill_done"] != "":
            metrics = self._extract_scalars_from_info(hab_info)
            metrics_at_skill_end = {
                f"{info['skill_done']}." + k: v for k, v in metrics.items()
            }
            self._current_episode_metrics = {
                **metrics_at_skill_end,
                **self._current_episode_metrics,
            }
            if "goal_name" in info:
                self._current_episode_metrics["goal_name"] = info["goal_name"]

        if done:
            metrics = self._extract_scalars_from_info(hab_info)
            metrics_at_episode_end = {"END." + k: v for k, v in metrics.items()}
            self._current_episode_metrics = {
                **metrics_at_episode_end,
                **self._current_episode_metrics,
            }
            if "goal_name" in info:
                self._current_episode_metrics["goal_name"] = info["goal_name"]

            self._episode_metrics[
                self._current_episode_key
            ] = self._current_episode_metrics
            self._current_episode_metrics = {}

        observations = self._env.get_observation()
        hab_info = None  # not sure what hab_info should be for real world
        return evaluation_pb2.Package(
            SerializedEntity=grpc_dumps((observations, done, hab_info))
        )

    def evalai_update_submission(self, request, context):
        return evaluation_pb2.Package()

    def close(self, request, context):
        """Close environment"""
        self._env.close()
        return evaluation_pb2.Package()


@click.command()
@click.option("--test-pick", default=False, is_flag=True)
@click.option("--test-gaze", default=False, is_flag=True)
@click.option("--test-place", default=False, is_flag=True)
@click.option("--skip-gaze", default=True, is_flag=True)
@click.option("--reset-nav", default=False, is_flag=True)
@click.option("--dry-run", default=False, is_flag=True)
@click.option("--pick-object", default="cup")
@click.option("--start-recep", default="table")
@click.option("--goal-recep", default="chair")
@click.option(
    "--cat-map-file", default="projects/real_world_ovmm/configs/example_cat_map.json"
)
@click.option("--max-num-steps", default=200)
@click.option("--visualize-maps", default=False, is_flag=True)
@click.option("--visualize-grasping", default=False, is_flag=True)
@click.option(
    "--debug",
    default=False,
    is_flag=True,
    help="Add pauses for debugging manipulation behavior.",
)
@click.option("--port", default=8085)
def main(
    test_pick=False,
    test_gaze=False,
    reset_nav=False,
    pick_object="cup",
    start_recep="table",
    goal_recep="chair",
    dry_run=False,
    visualize_maps=False,
    visualize_grasping=False,
    test_place=False,
    skip_gaze=False,
    cat_map_file=None,
    max_num_steps=200,
    config_path="projects/real_world_ovmm/configs/agent/eval.yaml",
    debug=False,
    port=8085,
):

    print("- Starting ROS node")
    rospy.init_node("eval_episode_stretch_objectnav")

    server = grpc.server(
        thread_pool=futures.ThreadPoolExecutor(max_workers=1),
        compression=grpc.Compression.Gzip,
        options=[
            (
                "grpc.max_receive_message_length",
                -1,
            )  # Unlimited message length that the channel can receive
        ],
    )
    evaluation_pb2_grpc.add_EnvironmentServicer_to_server(
        servicer=Environment(
            test_pick,
            reset_nav,
            pick_object,
            start_recep,
            goal_recep,
            dry_run,
            visualize_maps,
            visualize_grasping,
            test_place,
            cat_map_file,
            max_num_steps,
            config_path,
        ),
        server=server,
    )
    print(f"Starting server. Listening on port {port}.")
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    print("---- Starting real-world evaluation ----")
    main()
    print("==================================")
    print("Done real world evaluation.")
