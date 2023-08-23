# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import timeit
from pathlib import Path
from typing import Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import open3d
import rospy

import home_robot.utils.depth as du
from home_robot.agent.ovmm_agent import (
    OvmmPerception,
    build_vocab_from_category_map,
    read_category_map_file,
)
from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.motion.stretch import STRETCH_NAVIGATION_Q, HelloStretchKinematics
from home_robot.utils.image import Camera
from home_robot.utils.point_cloud import numpy_to_pcd, pcd_to_numpy, show_point_cloud
from home_robot.utils.pose import to_pos_quat
from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv
from home_robot_hw.remote import StretchClient
from home_robot_hw.utils.config import load_config


class RosMapDataCollector(object):
    """Simple class to collect RGB, Depth, and Pose information for building 3d spatial-semantic
    maps for the robot. Needs to subscribe to:
    - color images
    - depth images
    - camera info
    - joint states/head camera pose
    - base pose (relative to world frame)

    This is an example collecting the data; not necessarily the way you should do it.
    """

    def __init__(
        self,
        robot,
        semantic_sensor=None,
        visualize_planner=False,
        voxel_size: float = 0.01,
    ):
        self.robot = robot  # Get the connection to the ROS environment via agent
        # Run detection here
        self.semantic_sensor = semantic_sensor
        self.started = False
        self.robot_model = HelloStretchKinematics(visualize=visualize_planner)
        self.voxel_map = SparseVoxelMap(resolution=voxel_size)

    def step(self, visualize_map=False):
        """Step the collector. Get a single observation of the world. Remove bad points, such as
        those from too far or too near the camera."""
        obs = self.robot.get_observation()

        rgb = obs.rgb
        depth = obs.depth
        xyz = obs.xyz
        camera_pose = obs.camera_pose
        base_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])

        # Semantic prediction
        obs = self.semantic_sensor.predict(obs)

        # TODO: remove debug code
        # For now you can use this to visualize a single frame
        # show_point_cloud(xyz, rgb / 255, orig=np.zeros(3))
        self.voxel_map.add(
            camera_pose,
            xyz,
            rgb,
            depth=depth,
            base_pose=base_pose,
            obs=obs,
            K=self.robot.head._ros_client.rgb_cam.K,
        )
        if visualize_map:
            # Now draw 2d
            self.voxel_map.get_2d_map(debug=True)

    def get_2d_map(self):
        """Get 2d obstacle map for low level motion planning and frontier-based exploration"""
        return self.voxel_map.get_2d_map()

    def show(self) -> Tuple[np.ndarray, np.ndarray]:
        """Display the aggregated point cloud."""
        return self.voxel_map.show()


def collect_data(
    rate,
    max_frames,
    visualize,
    manual_wait,
    pcd_filename,
    pkl_filename,
    voxel_size: float = 0.01,
    device_id: int = 0,
    visualize_map_at_start: bool = True,
    visualize_map: bool = True,
    blocking: bool = True,
    verbose: bool = True,
    **kwargs,
):
    """Collect data from a Stretch robot. Robot will move through a preset trajectory, stopping repeatedly."""

    print("- Loading configuration")
    config = load_config(visualize=False, **kwargs)

    print("- Connect to Stretch")
    robot = StretchClient()

    print("- Create and load vocabulary and perception model")
    semantic_sensor = OvmmPerception(config, device_id, verbose, module="detic")
    obj_name_to_id, rec_name_to_id = read_category_map_file(
        config.ENVIRONMENT.category_map_file
    )
    vocab = build_vocab_from_category_map(obj_name_to_id, rec_name_to_id)
    semantic_sensor.update_vocabulary_list(vocab, 0)
    semantic_sensor.set_vocabulary(0)

    collector = RosMapDataCollector(
        robot, semantic_sensor, visualize, voxel_size=voxel_size
    )

    # Tuck the arm away
    print("Sending arm to  home...")
    robot.switch_to_manipulation_mode()

    robot.head.look_front(blocking=False)
    robot.manip.goto_joint_positions(
        robot.manip._extract_joint_pos(STRETCH_NAVIGATION_Q)
    )
    print("... done.")

    rate = rospy.Rate(rate)

    # Move the robot
    robot.switch_to_navigation_mode()
    # Sequence information if we are executing the trajectory
    step = 0
    # Number of frames collected
    frames = 0

    trajectory = [
        (0, 0, 0),
        (0.4, 0, 0),
        (0.75, 0.15, np.pi / 4),
        (0.85, 0.3, np.pi / 4),
        (0.95, 0.5, np.pi / 2),
        (1.0, 0.55, np.pi),
        (0.6, 0.45, 5 * np.pi / 4),
        (0.0, 0.3, -np.pi / 2),
        (0, 0, 0),
        # (0.2, 0, 0),
        # (0.5, 0, 0),
        # (0.7, 0.2, np.pi / 4),
        # (0.7, 0.4, np.pi / 2),
        # (0.5, 0.4, np.pi),
        # (0.2, 0.2, -np.pi / 4),
        # (0, 0, -np.pi / 2),
        # (0, 0, 0),
    ]

    collector.step(visualize_map=visualize_map_at_start)  # Append latest observations

    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        ti = (rospy.Time.now() - t0).to_sec()
        print("t =", ti, trajectory[step])
        robot.nav.navigate_to(trajectory[step])
        print("... done navigating.")
        if manual_wait:
            input("... press enter ...")
        print("... capturing frame!")
        step += 1

        # Append latest observations
        collector.step()

        frames += 1
        if max_frames > 0 and frames >= max_frames or step >= len(trajectory):
            break

        rate.sleep()

    print("Done collecting data.")
    robot.nav.navigate_to((0, 0, 0))
    pc_xyz, pc_rgb = collector.show()

    if visualize_map:
        import matplotlib.pyplot as plt

        obstacles, explored = collector.get_2d_map()

        plt.subplot(1, 2, 1)
        plt.imshow(obstacles)
        plt.subplot(1, 2, 2)
        plt.imshow(explored)
        plt.show()

    # Create pointcloud
    if len(pcd_filename) > 0:
        pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
        open3d.io.write_point_cloud(pcd_filename, pcd)
    if len(pkl_filename) > 0:
        collector.voxel_map.write_to_pickle(pkl_filename)

    rospy.signal_shutdown("done")


DATA_MODES = ["ros", "pkl", "dir"]


@click.command()
@click.option(
    "--mode", type=click.Choice(DATA_MODES), default="ros"
)  # help="Choose data source. ROS requires connecting to a real stretch robot")
@click.option("--rate", default=5, type=int)
@click.option("--max-frames", default=20, type=int)
@click.option("--visualize", default=False, is_flag=True)
@click.option("--manual_wait", default=False, is_flag=True)
@click.option("--output-pcd-filename", default="output.ply", type=str)
@click.option("--output-pkl-filename", default="output.pkl", type=str)
@click.option("--from-ros", default=False, is_flag=True)
@click.option(
    "--input-path",
    type=click.Path(),
    default="output.pkl",
    help="Input path with default value 'output.npy'",
)
def main(
    mode,
    rate,
    max_frames,
    visualize,
    manual_wait,
    output_pcd_filename,
    output_pkl_filename,
    input_path: str = ".",
    voxel_size: float = 0.01,
    **kwargs,
):

    click.echo(f"Processing data in mode: {mode}")
    click.echo(f"Using input path: {input_path}")

    if mode == "ros":
        click.echo("Will connect to a Stretch robot and collect a short trajectory.")
        collect_data(
            rate,
            max_frames,
            visualize,
            manual_wait,
            output_pcd_filename,
            output_pkl_filename,
            voxel_size,
            **kwargs,
        )
    elif mode == "dir":
        # This is for backwards compatibility with the GOAT data
        click.echo(f"- Loading pickled observations from a directory at {input_path}.")
        input_path = Path(input_path)
        pkl_files = input_path.glob("*.pkl")
        sorted_pkl_files = sorted(pkl_files, key=lambda f: int(f.stem))
        voxel_map = SparseVoxelMap(resolution=voxel_size)
        camera_matrix = None
        hfov = 60.2
        for i, pkl_file in enumerate(sorted_pkl_files):
            print("-", i, pkl_file)
            obs = np.load(pkl_file, allow_pickle=True)
            if camera_matrix is None:
                camera = Camera.from_width_height_fov(
                    obs.rgb.shape[1], obs.rgb.shape[0], hfov
                )
                # camera_matrix = du.get_camera_matrix(obs.rgb.shape[1], obs.rgb.shape[0], hfov)
            xyt = np.array([obs.gps[0], obs.gps[1], obs.compass])
            if obs.xyz is None:
                # need to find camera matrix K
                assert obs.depth is not None, "need depth"
                xyz = camera.depth_to_xyz(obs.depth)
            else:
                xyz = obs.xyz
            # For backwards compatibility
            if obs.instance is None:
                # This copies instance map out of task data if it exists
                obs.instance = obs.task_observations["instance_map"]
            # Now try to create everything
            voxel_map.add(obs.camera_pose, xyz, obs.rgb, None, obs.depth, xyt, obs)
            if i > 10:
                break
        voxel_map.show_point_cloud()
    elif mode == "pkl":
        click.echo(
            f"- Loading pickled observations from a single file at {input_path}."
        )
        input_path = Path(input_path)
        voxel_map = SparseVoxelMap(resolution=voxel_size)
        voxel_map.read_from_pickle(input_path)
        voxel_map.show_point_cloud()
    else:
        raise NotImplementedError(f"- data mode {mode} not supported or recognized")


if __name__ == "__main__":
    """run the test script."""
    main()
