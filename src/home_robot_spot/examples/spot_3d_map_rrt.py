# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Dict, List, Optional

import numpy as np
import open3d
import matplotlib.pyplot as plt

from home_robot.agent.ovmm_agent import (
    OvmmPerception,
    build_vocab_from_category_map,
    read_category_map_file,
)
from home_robot.mapping.voxel import SparseVoxelMap  # Aggregate 3d information
from home_robot.mapping.voxel_map import (  # Sample positions in free space for our robot to move to
    SparseVoxelMapNavigationSpace,
)
from home_robot.motion.rrt_connect import RRTConnect
from home_robot.motion.shortcut import Shortcut
from home_robot.motion.spot import (  # Just saves the Spot robot footprint for kinematic planning
    SimpleSpotKinematics,
)
from home_robot.utils.config import get_config, load_config
from home_robot.utils.point_cloud import numpy_to_pcd
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_spot import SpotClient, VoxelMapSubscriber
from home_robot_spot.grasp_env import GraspController


def navigate_to_an_instance(spot, voxel_map, planner, instance_id, visualize=False):
    instances = voxel_map.get_instances()
    instance = instances[instance_id]

    # TODO this can be random
    view = instance.instance_views[0]

    cur_position = spot.current_relative_position
    goal_position = np.asarray(view.pose)
    print(type(goal_position))

    # This currently yields TypeError: unsupported operand type(s) for -: 'numpy.ndarray' and 'Tensor'
    # I guess we need to convert view.pose to tensor
    planner_response = planner.plan(cur_position, goal_position)

    if planner_response.success:
        path = voxel_map.plan_to_grid_coords(planner_response)
        x, y = get_x_and_y_from_path(path)
        print(f'Path: {path}')
    else:
        return False
    
    for node in planner_response.trajectory:
        spot.navigate_to(node.state, blocking=True)

    if visualize:
        cropped_image = view.cropped_image
        plt.imshow(cropped_image)
        plt.show()
        plt.imsave(f"instance_{instance_id}.png", cropped_image)

    return True


# def main(dock: Optional[int] = 549):
def main(dock: Optional[int] = None):
    # TODO add this to config
    spot_config = get_config("src/home_robot_spot/configs/default_config.yaml")[0]

    # TODO move these parameters to config
    parameters = {
        'step_size' : 2.0, # (originally .1, we can make it all the way to 2 maybe actually)
        'visualize': True,
        'exploration_steps': 5,

        # Voxel map
        'obs_min_height': .5, # Originally .1, floor appears noisy in the 3d map of fremont so we're being super conservative
        'obs_max_height': 1.8, # Originally 1.8, spot is shorter than stretch tho
        'obs_min_density': 25, # Originally 10, making it bigger because theres a bunch on noise
        'voxel_size' : 0.05,
        'local_radius': 0.5, # Can probably be bigger than original (.15)

        # Frontier 
        'min_size': 5, # Can probably be bigger than original (10) 
        'max_size': 20 # Can probably be bigger than original (10)

        # Other parameters tuned (footprint is a third of the real robot size)
    }

    # Create voxel map
    voxel_map = SparseVoxelMap(
        resolution=parameters['voxel_size'], 
        local_radius=parameters['local_radius'],
        obs_min_height=parameters['obs_min_height'],
        obs_max_height=parameters['obs_max_height'],
        obs_min_density=parameters['obs_min_density']
    )

    # Create kinematic model (very basic for now - just a footprint)
    robot_model = SimpleSpotKinematics()

    # Create navigation space example
    navigation_space = SparseVoxelMapNavigationSpace(
        voxel_map=voxel_map, robot=robot_model, step_size=parameters['step_size']
    )
    print(" - Created navigation space and environment")
    print(f"   {navigation_space=}")

    # Create segmentation sensor and load config. Returns config from file, as well as a OvmmPerception object that can be used to label scenes.
    print("- Loading configuration")
    config = load_config(visualize=False)

    print("- Create and load vocabulary and perception model")
    semantic_sensor = OvmmPerception(config, 0, True, module="detic")
    obj_name_to_id, rec_name_to_id = read_category_map_file(
        config.ENVIRONMENT.category_map_file
    )
    vocab = build_vocab_from_category_map(obj_name_to_id, rec_name_to_id)
    semantic_sensor.update_vocabulary_list(vocab, 0)
    semantic_sensor.set_vocabulary(0)

    planner = Shortcut(RRTConnect(navigation_space, navigation_space.is_valid))

    spot = SpotClient(config=spot_config, dock_id=dock, use_midas=False)
    try:
        # Turn on the robot using the client above
        spot.start()

        # Start thread to update voxel map
        voxel_map_subscriber = VoxelMapSubscriber(spot, voxel_map, semantic_sensor)
        voxel_map_subscriber.start()

        print("Waiting for observations...")
        # spot.navigate_to([0, 0, 0], blocking=True)
        # This slep allows the robot to receive some observations (we need to do something smarter)
        time.sleep(10)

        # Well do a 360 degree turn to get some observations (this helps debug the robot)
        # for _ in range(4):
        #     spot.move_base(0, .5)
        #     time.sleep(5)

        for step in range(parameters['exploration_steps']):

            # Get current position and goal
            start = spot.current_relative_position
            print("Start is valid:", voxel_map.xyt_is_safe(start))

            # TODO do something is start is not valid

            # Sample a goal in the frontier (TODO change to closest frontier)
            goal = navigation_space.sample_frontier(
                min_size=parameters['min_size'], max_size=parameters['max_size']
            )
            goal = goal.cpu().numpy()
            print(" Goal is valid:", voxel_map.xyt_is_safe(goal))

            #  Build plan
            res = planner.plan(start, goal)
            print("Res success:", res.success)
            
            if res.success:
                path = voxel_map.plan_to_grid_coords(res)
                x, y = get_x_and_y_from_path(path)
            else:
                continue

            # Print trajectory
            print("Trajectory:")
            print(res.trajectory)
            print('Path')
            print(path)


            # TODO this trajectory is really ineficient, we can interpolate smarter
            for node in res.trajectory:
                print(node.state)
                spot.navigate_to(node.state, blocking=True)

            if step % 5 == 0 and parameters['visualize']:
                print(
                    "Observations processed for the map so far: ",
                    voxel_map_subscriber.current_obs,
                )
                voxel_map.show(backend="open3d", instances=False)
                
                obstacles, explored = voxel_map.get_2d_map()
                img = (10 * obstacles) + explored
                start_unnormalized = spot.unnormalize_gps_compass(start)
                goal_unnormalized = spot.unnormalize_gps_compass(goal)

                navigation_space.draw_state_on_grid(img, start_unnormalized, weight=5)
                navigation_space.draw_state_on_grid(img, goal_unnormalized, weight=5)
                plt.imshow(img)
                plt.show()
                plt.imsave(f"exploration_step_{step}.png", img)

        print("Exploration complete!")


        print('Navigating to instance ')
        instances = voxel_map.get_instances()

        # Navigating to a random instance, add LLM here
        instance_id = np.random.randint(len(instances))
        print(f'Instance id: {instance_id}')
        success = navigate_to_an_instance(spot, voxel_map, planner, instance_id, visualize=parameters['visualize'])
        print(f'Success: {success}')



    except Exception as e:
        print("Exception caught:")
        print(e)
        raise e

    finally:
        print("Writing data...")
        pc_xyz, pc_rgb = voxel_map.show(
            backend="open3d", instances=False, orig=np.zeros(3)
        )
        pcd_filename = "spot_output.pcd"
        pkl_filename = "spot_output.pkl"

        # Create pointcloud
        if len(pcd_filename) > 0:
            pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
            open3d.io.write_point_cloud(pcd_filename, pcd)
            print(f"... wrote pcd to {pcd_filename}")
        if len(pkl_filename) > 0:
            voxel_map.write_to_pickle(pkl_filename)
            print(f"... wrote pkl to {pkl_filename}")

        # TODO dont repeat this code
        obstacles, explored = voxel_map.get_2d_map()
        img = (10 * obstacles) + explored
        start_unnormalized = spot.unnormalize_gps_compass(start)
        goal_unnormalized = spot.unnormalize_gps_compass(goal)
        navigation_space.draw_state_on_grid(img, start_unnormalized, weight=5)
        navigation_space.draw_state_on_grid(img, goal_unnormalized, weight=5)
        plt.imshow(img)
        plt.show()
        plt.imsave(f"exploration_step_final.png", img)

        # I am going to assume the robot is at its goal position here
        # gaze = GraspController(
        #     config=config,
        #     spot=spot,
        #     objects=[["penguin plush"]],
        #     confidence=0.1,
        #     show_img=True,
        #     top_grasp=False,
        #     hor_grasp=True,
        # )
        # spot.open_gripper()
        # time.sleep(1)
        # print("Resetting environment...")
        # success = gaze.gaze_and_grasp()
        # time.sleep(2)
        
        print("Safely stop the robot...")
        spot.stop()


if __name__ == "__main__":
    main()
