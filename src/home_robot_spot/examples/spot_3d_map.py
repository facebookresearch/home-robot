# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import numpy as np
import open3d

from home_robot.agent.ovmm_agent import (
    OvmmPerception,
    build_vocab_from_category_map,
    read_category_map_file,
)
from home_robot.mapping.voxel import SparseVoxelMap  # Aggregate 3d information
from home_robot.mapping.voxel_map import (  # Sample positions in free space for our robot to move to
    SparseVoxelMapNavigationSpace,
)
from home_robot.motion.spot import (  # Just saves the Spot robot footprint for kinematic planning
    SimpleSpotKinematics,
)
from home_robot.utils.config import get_config, load_config
from home_robot.utils.point_cloud import numpy_to_pcd
from home_robot_spot import SpotClient, VoxelMapSubscriber


# def main(dock: Optional[int] = 549):
def main(dock: Optional[int] = None):
    spot_config = get_config("src/home_robot_spot/configs/default_config.yaml")[0]

    # TODO move these parameters to config
    voxel_size = 0.05
    voxel_map = SparseVoxelMap(resolution=voxel_size, local_radius=0.1)

    # Create kinematic model (very basic for now - just a footprint)
    robot_model = SimpleSpotKinematics()

    # Create navigation space example
    navigation_space = SparseVoxelMapNavigationSpace(
        voxel_map=voxel_map, robot=robot_model, step_size=0.1
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

    spot = SpotClient(config=spot_config, dock_id=dock)
    try:
        # Turn on the robot using the client above
        spot.start()

        # Start thread to update voxel map
        voxel_map_subscriber = VoxelMapSubscriber(spot, voxel_map, semantic_sensor)
        voxel_map_subscriber.start()

        spot.move_base(0.0, 0.0)
        # breakpoint()

        linear = input("Input Linear: ")
        angular = input("Input Angular: ")
        action_index, visualization_frequency = 0, 10
        while linear != "" and angular != "":
            try:
                spot.move_base(float(linear), float(angular))
            except Exception:
                print("Error -- try again")

            if action_index % visualization_frequency == 0 and action_index > 0:
                print(
                    "Observations processed for the map so far: ",
                    voxel_map_subscriber.current_obs,
                )
                print("Actions taken so far: ", action_index)
                voxel_map.show(backend="open3d", instances=False)

            linear = input("Input Linear: ")
            angular = input("Input Angular: ")
            action_index += 1

            # viz_data = spot.make_3d_viz(viz_data)

    except Exception as e:
        print("Exception caught:")
        print(e)

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

        print("Safely stop the robot...")
        spot.stop()


if __name__ == "__main__":
    main()
