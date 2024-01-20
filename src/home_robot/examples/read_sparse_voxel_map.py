# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pickle
from pathlib import Path

import click

from home_robot.agent.multitask import get_parameters
from home_robot.agent.multitask.robot_agent import RobotAgent
from home_robot.mapping import SparseVoxelMap, SparseVoxelMapNavigationSpace
from home_robot.utils.dummy_stretch_client import DummyStretchClient


@click.command()
@click.option(
    "--input-path",
    "-i",
    type=click.Path(),
    default="output.pkl",
    help="Input path with default value 'output.npy'",
)
@click.option(
    "--config-path",
    "-c",
    type=click.Path(),
    default="src/home_robot_hw/configs/default.yaml",
    help="Path to planner config.",
)
@click.option(
    "--frame",
    "-f",
    type=int,
    default=-1,
    help="number of frames to read",
)
def main(
    input_path,
    config_path,
    voxel_size: float = 0.01,
    show_maps: bool = True,
    pkl_is_svm: bool = True,
    frame: int = -1,
):
    """Simple script to load a voxel map"""
    input_path = Path(input_path)
    print("Loading:", input_path)
    if len(config_path) > 0:
        print("- Load parameters")
        parameters = get_parameters("src/home_robot_hw/configs/default.yaml")
        print(parameters)
        dummy_robot = DummyStretchClient()
        agent = RobotAgent(
            dummy_robot, None, parameters, rpc_stub=None, grasp_client=None
        )
    else:
        agent = None
    if pkl_is_svm:
        with input_path.open("rb") as f:
            voxel_map = pickle.load(f)
    else:
        voxel_map = SparseVoxelMap(resolution=voxel_size)
        voxel_map.read_from_pickle(input_path, num_frames=frame)

    if agent is not None:
        print(agent)
        # Display with agent overlay
        voxel_map.show(instances=True)
        voxel_map.get_2d_map(debug=False)


if __name__ == "__main__":
    """run the test script."""
    main()
