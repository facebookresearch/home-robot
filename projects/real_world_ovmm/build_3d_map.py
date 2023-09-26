# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import timeit
from pathlib import Path
from typing import Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import open3d
import rospy
import torch

import home_robot.utils.depth as du
from home_robot.agent.ovmm_agent import (
    OvmmPerception,
    build_vocab_from_category_map,
    read_category_map_file,
)
from home_robot.mapping import SparseVoxelMap, SparseVoxelMapNavigationSpace

# Import planning tools for exploration
from home_robot.motion.rrt_connect import RRTConnect
from home_robot.motion.shortcut import Shortcut
from home_robot.motion.stretch import HelloStretchKinematics
from home_robot.utils.geometry import xyt2sophus
from home_robot.utils.image import Camera
from home_robot.utils.point_cloud import numpy_to_pcd, pcd_to_numpy, show_point_cloud
from home_robot.utils.pose import convert_pose_habitat_to_opencv, to_pos_quat
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv
from home_robot_hw.remote import StretchClient
from home_robot_hw.ros.visualizer import Visualizer
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
        self.voxel_map = SparseVoxelMap(resolution=voxel_size, local_radius=0.1)

    def get_planning_space(self) -> SparseVoxelMapNavigationSpace:
        """return space for motion planning. Hard codes some parameters for Stretch"""
        return SparseVoxelMapNavigationSpace(
            self.voxel_map,
            self.robot_model,
            step_size=0.1,
            dilate_frontier_size=12,  # 0.6 meters back from every edge
            dilate_obstacle_size=2,
        )

    def step(self, visualize_map=False):
        """Step the collector. Get a single observation of the world. Remove bad points, such as
        those from too far or too near the camera."""
        obs = self.robot.get_observation()

        # Semantic prediction
        obs = self.semantic_sensor.predict(obs)

        # Add observation - helper function will unpack it
        self.voxel_map.add_obs(
            obs,
            K=torch.from_numpy(self.robot.head._ros_client.rgb_cam.K).float(),
        )
        if visualize_map:
            # Now draw 2d
            self.voxel_map.get_2d_map(debug=True)

    def get_2d_map(self):
        """Get 2d obstacle map for low level motion planning and frontier-based exploration"""
        return self.voxel_map.get_2d_map()

    def show(self, orig: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Display the aggregated point cloud."""
        return self.voxel_map.show(orig=orig)


def create_semantic_sensor(
    device_id: int = 0,
    verbose: bool = True,
    **kwargs,
):
    """Create segmentation sensor and load config. Returns config from file, as well as a OvmmPerception object that can be used to label scenes."""
    print("- Loading configuration")
    config = load_config(visualize=False, **kwargs)

    print("- Create and load vocabulary and perception model")
    semantic_sensor = OvmmPerception(config, device_id, verbose, module="detic")
    obj_name_to_id, rec_name_to_id = read_category_map_file(
        config.ENVIRONMENT.category_map_file
    )
    vocab = build_vocab_from_category_map(obj_name_to_id, rec_name_to_id)
    semantic_sensor.update_vocabulary_list(vocab, 0)
    semantic_sensor.set_vocabulary(0)
    return config, semantic_sensor


def run_fixed_trajectory(
    collector: RosMapDataCollector,
    robot: StretchClient,
    rate: int = 10,
    manual_wait: bool = False,
):
    """Go through a fixed robot trajectory"""
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
        (0.2, 0, 0),
        (0.5, 0, 0),
        (0.7, 0.2, np.pi / 4),
        (0.7, 0.4, np.pi / 2),
        (0.5, 0.4, np.pi),
        (0.2, 0.2, -np.pi / 4),
        (0, 0, -np.pi / 2),
        (0, 0, 0),
    ]

    # Sequence information if we are executing the trajectory
    step = 0
    # Number of frames collected
    frames = 0
    # Spin rate
    rate = rospy.Rate(rate)

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
        if step >= len(trajectory):
            break

        rate.sleep()


def run_exploration(
    collector: RosMapDataCollector,
    robot: StretchClient,
    rate: int = 10,
    manual_wait: bool = False,
    explore_iter: int = 20,
    try_to_plan_iter: int = 10,
    dry_run: bool = False,
    random_goals: bool = False,
    visualize: bool = False,
):
    """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

    Args:
        visualize(bool): true if we should do intermediate debug visualizations"""
    rate = rospy.Rate(rate)

    # Create planning space
    space = collector.get_planning_space()

    # Create a simple motion planner
    planner = Shortcut(RRTConnect(space, space.is_valid))

    print("Go to (0, 0, 0) to start with...")
    robot.nav.navigate_to([0, 0, 0])

    # Explore some number of times
    for i in range(explore_iter):
        print("\n" * 2)
        print("-" * 20, i, "-" * 20)
        start = robot.get_base_pose()
        start_is_valid = space.is_valid(start)
        print("Start is valid:", start_is_valid)
        # if start is not valid move backwards a bit
        if not start_is_valid:
            print("Start not valid. back up a bit.")
            robot.nav.navigate_to([-0.1, 0, 0], relative=True)
            continue
        print("       Start:", start)
        # sample a goal
        if random_goals:
            goal = next(space.sample_random_frontier()).cpu().numpy()
        else:
            # extract goal using fmm planner
            tries = 0
            failed = False
            for goal in space.sample_closest_frontier(start):
                if goal is None:
                    failed = True
                    break
                goal = goal.cpu().numpy()
                print("Sampled Goal:", goal)
                show_goal = np.zeros(3)
                show_goal[:2] = goal[:2]
                goal_is_valid = space.is_valid(goal)
                print("Start is valid:", start_is_valid)
                print(" Goal is valid:", goal_is_valid)
                if not goal_is_valid:
                    print(" -> resample goal.")
                    continue
                # plan to the sampled goal
                res = planner.plan(start, goal)
                print("Found plan:", res.success)
                if visualize:
                    obstacles, explored = collector.get_2d_map()
                    img = (10 * obstacles) + explored
                    space.draw_state_on_grid(img, start, weight=5)
                    space.draw_state_on_grid(img, goal, weight=5)
                    plt.imshow(img)
                    if res.success:
                        path = collector.voxel_map.plan_to_grid_coords(res)
                        x, y = get_x_and_y_from_path(path)
                        plt.plot(y, x)
                        plt.show()
                if res.success:
                    break
                else:
                    if visualize:
                        plt.show()
                    tries += 1
                    if tries >= try_to_plan_iter:
                        failed = True
                        break
                    continue
            else:
                print(" ------ no valid goals found!")
                failed = True
            if failed:
                print(
                    " ------ sampling and planning failed! Might be no where left to go."
                )
                continue

        if visualize:
            # After doing everything
            collector.show(orig=show_goal)

        # if it fails, skip; else, execute a trajectory to this position
        if res.success:
            print("Full plan:")
            for i, pt in enumerate(res.trajectory):
                print("-", i, pt.state)
            if not dry_run:
                robot.nav.execute_trajectory([pt.state for pt in res.trajectory])

        # Append latest observations
        collector.step()
        if manual_wait:
            input("... press enter ...")

    # Finally - plan back to (0,0,0)
    print("Go back to (0, 0, 0) to finish...")
    input("---")
    start = robot.get_base_pose()
    goal = np.array([0, 0, 0])
    res = planner.plan(start, goal)
    # if it fails, skip; else, execute a trajectory to this position
    if res.success:
        print("Full plan to home:")
        for i, pt in enumerate(res.trajectory):
            print("-", i, pt.state)
        if not dry_run:
            robot.nav.execute_trajectory([pt.state for pt in res.trajectory])
    else:
        print("WARNING: planning to home failed!")


def collect_data(
    rate,
    visualize,
    manual_wait,
    pcd_filename,
    pkl_filename,
    voxel_size: float = 0.01,
    device_id: int = 0,
    verbose: bool = True,
    visualize_map_at_start: bool = False,
    visualize_map: bool = False,
    blocking: bool = True,
    run_explore: bool = False,
    **kwargs,
):
    """Collect data from a Stretch robot. Robot will move through a preset trajectory, stopping repeatedly."""

    print("- Connect to Stretch")
    robot = StretchClient()

    config, semantic_sensor = create_semantic_sensor(device_id, verbose)

    print("- Start ROS data collector")
    collector = RosMapDataCollector(
        robot, semantic_sensor, visualize, voxel_size=voxel_size
    )

    # Tuck the arm away
    print("Sending arm to  home...")
    robot.switch_to_manipulation_mode()

    robot.move_to_nav_posture()
    robot.head.look_close(blocking=False)
    print("... done.")

    # Move the robot
    robot.switch_to_navigation_mode()
    collector.step(visualize_map=visualize_map_at_start)  # Append latest observations
    if run_explore:
        run_exploration(collector, robot, rate, manual_wait)
    else:
        run_fixed_trajectory(collector, robot, rate, manual_wait)

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
@click.option("--visualize", default=False, is_flag=True)
@click.option("--manual_wait", default=False, is_flag=True)
@click.option("--output-pcd-filename", default="output.ply", type=str)
@click.option("--output-pkl-filename", default="output.pkl", type=str)
@click.option("--run-explore", default=False, is_flag=True)
@click.option("--show-maps", default=False, is_flag=True)
@click.option("--show-paths", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option(
    "--input-path",
    type=click.Path(),
    default="output.pkl",
    help="Input path with default value 'output.npy'",
)
def main(
    mode,
    rate,
    visualize,
    manual_wait,
    output_pcd_filename,
    output_pkl_filename,
    run_explore: bool = False,
    input_path: str = ".",
    voxel_size: float = 0.01,
    device_id: int = 0,
    verbose: bool = True,
    show_maps: bool = False,
    show_paths: bool = False,
    random_goals: bool = True,
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        run_explore(bool): should sample frontier points and path to them; on robot will go there.
        show_maps(bool): show 2d maps
        show_paths(bool): display paths after planning
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """
    click.echo(f"Processing data in mode: {mode}")
    click.echo(f"Using input path: {input_path}")

    if run_explore and not (mode == "ros" or mode == "pkl"):
        raise RuntimeError("explore cannot be used without a robot to interact with")

    if mode == "ros":
        click.echo("Will connect to a Stretch robot and collect a short trajectory.")
        collect_data(
            rate,
            visualize,
            manual_wait,
            output_pcd_filename,
            output_pkl_filename,
            voxel_size,
            device_id,
            verbose,
            run_explore=run_explore,
            **kwargs,
        )
    elif mode == "dir":
        # This is for backwards compatibility with the GOAT data
        raise NotImplementedError("Output in directory mode not yet working.")
        input_path = Path(input_path)
        pkl_files = input_path.glob("*.pkl")
        sorted_pkl_files = sorted(pkl_files, key=lambda f: int(f.stem))
        voxel_map = SparseVoxelMap(resolution=voxel_size)
        camera = None
        hfov = 60.2
        for i, pkl_file in enumerate(sorted_pkl_files):
            print("-", i, pkl_file)
            obs = np.load(pkl_file, allow_pickle=True)
            if camera is None:
                camera = Camera.from_width_height_fov(
                    obs.rgb.shape[1], obs.rgb.shape[0], hfov
                )
            xyt = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
            base_pose_matrix = xyt2sophus(xyt).matrix()
            if obs.xyz is None:
                # need to find camera matrix K
                assert obs.depth is not None, "need depth"
                xyz = camera.depth_to_xyz(obs.depth)
                # show_point_cloud(xyz, obs.rgb / 255.0, orig=np.zeros(3))
                import trimesh
                import trimesh.transformations as tra

                import home_robot.utils.depth as du
                from home_robot.utils.point_cloud import show_point_cloud

                camera_matrix = du.get_camera_matrix(
                    obs.rgb.shape[1], obs.rgb.shape[0], hfov
                )
                agent_pos = obs.camera_pose[:3, 3]
                agent_height = agent_pos[2]
                import torch

                device = torch.device("cpu")
                point_cloud_t = du.get_point_cloud_from_z_t(
                    torch.Tensor(obs.depth).unsqueeze(0), camera_matrix, device, scale=1
                )
                # show_point_cloud(
                #    point_cloud_t[0].cpu().numpy(), obs.rgb / 255.0, orig=np.zeros(3)
                # )
                # xyz = point_cloud_t[0].cpu().numpy()
                # Now co
                import trimesh.transformations as tra

                angles = torch.Tensor(
                    [tra.euler_from_matrix(obs.camera_pose[:3, :3], "rzyx")]
                )
                tilt = angles[:, 1]
                point_cloud_base = du.transform_camera_view_t(
                    point_cloud_t,
                    agent_height,
                    torch.rad2deg(tilt).cpu().numpy(),
                    device,
                )
                xyz1 = point_cloud_t[0].cpu().numpy()
                xyz2 = point_cloud_base[0].cpu().numpy()
                show_point_cloud(xyz1, obs.rgb / 255.0, orig=np.zeros(3))
                show_point_cloud(xyz2, obs.rgb / 255.0, orig=np.zeros(3))
                breakpoint()

                camera_pose = convert_pose_habitat_to_opencv(obs.camera_pose)
                import trimesh.transformations as tra

                angles = torch.Tensor(
                    [tra.euler_from_matrix(obs.camera_pose[:3, :3], "rzyx")]
                )
                angles2 = tra.euler_from_matrix(obs.camera_pose[:3, :3], "rzyx")
                print("not corrected", angles)
                print("    corrected", angles2)
                R = tra.euler_matrix(angles2[0], 0, 0)  # angles2[2])
                cvt_xyz = trimesh.transform_points(xyz.reshape(-1, 3), R)
                show_point_cloud(cvt_xyz, obs.rgb / 255.0, orig=np.zeros(3))
                breakpoint()
            else:
                xyz = obs.xyz
            # For backwards compatibility
            if obs.instance is None:
                # This copies instance map out of task data if it exists
                obs.instance = obs.task_observations["instance_map"]
            # Put the camera pose into the world frame
            camera_pose_world_coords = obs.camera_pose @ base_pose_matrix
            # Now try to create everything
            voxel_map.add(
                camera_pose_world_coords, xyz, obs.rgb, None, obs.depth, xyt, obs
            )
            if i > 0:
                break
        voxel_map.show()
    elif mode == "pkl":
        click.echo(
            f"- Loading pickled observations from a single file at {input_path}."
        )
        input_path = Path(input_path)
        voxel_map = SparseVoxelMap(resolution=voxel_size)
        voxel_map.read_from_pickle(input_path)
        if show_maps:
            voxel_map.show(instances=True)
        voxel_map.get_2d_map(debug=show_maps)

        if run_explore:
            print(
                "Running exploration test on offline data. Will plan to various waypoints"
            )
            robot_model = HelloStretchKinematics()
            space = SparseVoxelMapNavigationSpace(voxel_map, robot_model, step_size=0.2)
            planner = Shortcut(RRTConnect(space, space.is_valid))

            rospy.init_node("build_3d_map")
            goal_visualizer = Visualizer("goto_controller/goal_abs")
            for i in range(10):
                print("-" * 10, i, "-" * 10)
                # NOTE: this is how you can sample a random free location
                # goal = space.sample_valid_location().cpu().numpy()
                # This lets you sample a free location only on the frontier
                if random_goals:
                    # Start at the center
                    start = np.zeros(3)
                    goal = next(space.sample_random_frontier()).cpu().numpy()
                else:
                    start = next(space.sample_valid_location()).cpu().numpy()
                    goal = next(space.sample_closest_frontier(start)).cpu().numpy()
                if goal is None:
                    print(" ------ sampling failed!")
                print("       Start:", start)
                print("Sampled Goal:", goal)
                print("Start is valid:", voxel_map.xyt_is_safe(start))
                print(" Goal is valid:", voxel_map.xyt_is_safe(goal))
                res = planner.plan(start, goal)
                print("Found plan:", res.success)

                if show_paths:
                    obstacles, explored = voxel_map.get_2d_map()
                    H, W = obstacles.shape
                    img = np.zeros((H, W, 3))
                    img[:, :, 0] = obstacles
                    img[:, :, 2] = explored
                    import torch

                    states = torch.zeros_like(obstacles).float()
                    space.draw_state_on_grid(states, start, weight=5)
                    space.draw_state_on_grid(states, goal, weight=5)
                    img[:, :, 1] = states.cpu().numpy()
                    plt.imshow(img)
                    if res.success:
                        path = voxel_map.plan_to_grid_coords(res)
                        x, y = get_x_and_y_from_path(path)
                        plt.plot(y, x)
                    plt.show()
                    show_orig = np.zeros(3)
                    show_orig[:2] = goal[:2]

                    # Send it to ROS
                    pose_goal = xyt2sophus(goal)
                    goal_visualizer(pose_goal.matrix())
                    voxel_map.show(orig=show_orig)
    else:
        raise NotImplementedError(f"- data mode {mode} not supported or recognized")


if __name__ == "__main__":
    """run the test script."""
    main()
