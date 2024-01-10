import time
import timeit
from threading import Event, Thread
from typing import Dict, List, Optional, Sequence, Tuple, Union

import bosdyn.client.frame_helpers as frame_helpers
import cv2
import numpy as np
import torch
import transforms3d as t3d
import trimesh.transformations as tra
from bosdyn.api import image_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    HAND_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
    get_vision_tform_body,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs
from spot_wrapper.basic_streaming_visualizer_numpy import obstacle_grid_points
from spot_wrapper.spot import Spot, build_image_request, image_response_to_cv2

from home_robot.core.interfaces import Action, Observations
from home_robot.motion import PlanResult
from home_robot.perception.midas import Midas
from home_robot.utils.bboxes_3d_plotly import plot_scene_with_bboxes
from home_robot.utils.config import Config, get_config
from home_robot.utils.geometry import (
    angle_difference,
    sophus2xyt,
    xyt2sophus,
    xyt_base_to_global,
)
from home_robot.utils.image import Camera as PinholeCamera
from home_robot.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates

try:
    config = get_config("projects/spot/configs/config.yaml")[0]
    spot = SpotClient(config=config)

    from home_robot.agent.ovmm_agent import (
        OvmmPerception,
        build_vocab_from_category_map,
        read_category_map_file,
    )
    from home_robot.mapping.voxel import SparseVoxelMap  # Aggregate 3d information
    from home_robot.utils.config import load_config

    # TODO move these parameters to config
    voxel_size = 0.05
    voxel_map = SparseVoxelMap(resolution=voxel_size, local_radius=0.1)

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

    # Turn on the robot using the client above
    spot.start()

    # Start thread to update voxel map
    voxel_map_subscriber = VoxelMapSubscriber(spot, voxel_map, semantic_sensor)
    voxel_map_subscriber.start()

    linear = input("Input Linear: ")
    angular = input("Input Angular: ")

    viz_data: Dict[str, List] = {
        "xyz": [],
        "colors": [],
        "depths": [],
        "Rs": [],
        "tvecs": [],
        "intrinsics": [],
    }

    action_index, visualization_frequency = 0, 7
    while linear != "" and angular != "":
        try:
            spot.move_base(float(linear), float(angular))
        except Exception:
            print("Error -- try again")

        # obs = spot.get_rgbd_obs()
        # obs = semantic_sensor.predict(obs)
        # voxel_map.add_obs(obs, xyz_frame="world")
        print("added, now display something")
        if action_index % visualization_frequency == 0 and action_index > 0:
            print(
                "Observations processed for the map so far: ",
                voxel_map_subscriber.current_obs,
            )
            print("Actions taken so far: ", action_index)

            voxel_map.show(backend="open3d", instances=False)

        # To navigate to an instance
        # instance_id = <set instance id>
        # instance_view_id = <set instance view id> (for now it can be random or the first one)
        # instances = voxel_map.get_instances()
        # instance = instances[instance_id]
        # view = instance.instance_views[instance_view_id]
        # gps, compass = view.gps (or pose?) this wint work is some rough pseudocode
        # position = np.array([gps[0], gps[1], compass[0]])
        # spot.navigate_to(position)

        linear = input("Input Linear: ")
        angular = input("Input Angular: ")
        # viz_data = spot.make_3d_viz(viz_data)
        action_index += 1

except Exception as e:
    print(e)
    spot.stop()
    raise e

    # finally:
    #     spot.stop()