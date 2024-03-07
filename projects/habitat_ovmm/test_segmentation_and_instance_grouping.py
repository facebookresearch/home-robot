root_path = "/home/xiaohan/accel-cortex/"

import pickle

import numpy as np

# with open(root_path + "debug_svm.pkl", "rb") as f:
#     svm = pickle.load(f)


# observations = svm.observations
# with open(root_path + "annotation.pkl", "rb") as f:
#     annotation = pickle.load(f)
with open(root_path + "stretch_output_2024-03-06_15-47-27.pkl", "rb") as f:
    obs_history = pickle.load(f)

# print(annotation["task"])
# key_frames = []
# key_obs = []
# for idx, obs in enumerate(observations):
#     perceived_ids = np.unique(obs.obs.task_observations["gt_instance_ids"])
#     for target_id in annotation["object_ids"]:
#         if (target_id + 1) in perceived_ids:
#             print("target observation found")
#             key_frames.append(obs)
#             key_obs.append(obs_history[idx])
# obs = key_frames[-1]
key_obs = obs_history["obs"]
obs = key_obs[-1]

import time
from pathlib import Path

import imageio
import yaml
from PIL import Image

from home_robot.agent.multitask import get_parameters
from home_robot.mapping.voxel import (
    SparseVoxelMap,
    SparseVoxelMapNavigationSpace,
    plan_to_frontier,
)
from home_robot.perception import create_semantic_sensor
from home_robot.perception.encoders import get_encoder

# image_array = np.array(obs.obs.rgb, dtype=np.uint8)
# print(image_array.shape)
# # image_array = image_array[..., ::-1]
# image = Image.fromarray(image_array)


parameters = yaml.safe_load(
    Path("/home/xiaohan/home-robot/src/home_robot_sim/configs/gpt4v.yaml").read_text()
)
config, semantic_sensor = create_semantic_sensor()
semantic_sensor

# parameters = get_parameters(cfg.agent_parameters)
encoder = get_encoder(parameters["encoder"], parameters["encoder_args"])

voxel_map = SparseVoxelMap(
    resolution=parameters["voxel_size"],
    local_radius=parameters["local_radius"],
    obs_min_height=parameters["obs_min_height"],
    obs_max_height=parameters["obs_max_height"],
    min_depth=parameters["min_depth"],
    max_depth=parameters["max_depth"],
    pad_obstacles=parameters["pad_obstacles"],
    add_local_radius_points=parameters.get("add_local_radius_points", True),
    remove_visited_from_obstacles=parameters.get(
        "remove_visited_from_obstacles", False
    ),
    obs_min_density=parameters["obs_min_density"],
    smooth_kernel_size=parameters["smooth_kernel_size"],
    encoder=encoder,
    use_median_filter=parameters.get("use_median_filter", False),
    median_filter_size=parameters.get("median_filter_size", 5),
    median_filter_max_error=parameters.get("median_filter_max_error", 0.01),
    use_derivative_filter=parameters.get("use_derivative_filter", False),
    derivative_filter_threshold=parameters.get("derivative_filter_threshold", 0.5),
    instance_memory_kwargs={
        "min_pixels_for_instance_view": parameters.get(
            "min_pixels_for_instance_view", 100
        )
    },
)

voxel_map.reset()
key_obs = [key_obs[4]]
for idx, obs in enumerate(key_obs):

    image_array = np.array(obs.rgb, dtype=np.uint8)
    # print(image_array.shape)
    # # image_array = image_array[..., ::-1]
    image = Image.fromarray(image_array)
    image.show()

    obs = semantic_sensor.predict(obs)
    voxel_map.add_obs(obs)
    voxel_map.show(
        instances=True,
        height=1000,
        boxes_plot_together=False,
        backend="open3d",
    )
