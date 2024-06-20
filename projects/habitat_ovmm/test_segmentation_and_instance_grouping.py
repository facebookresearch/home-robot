root_path = "/home/xiaohan/accel-cortex/"

import pickle

import numpy as np

from home_robot.core.interfaces import Observations

# with open(root_path + "debug_svm.pkl", "rb") as f:
#     svm = pickle.load(f)


# observations = svm.observations
# with open(root_path + "annotation.pkl", "rb") as f:
#     annotation = pickle.load(f)
with open("/home/xiaohan/Downloads/robot.pkl", "rb") as f:
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
key_obs = []
num_obs = len(obs_history["rgb"])

for obs_id in range(num_obs):
    key_obs.append(
        Observations(
            rgb=obs_history["rgb"][obs_id].numpy(),
            gps=obs_history["base_poses"][obs_id][:2].numpy(),
            compass=[obs_history["base_poses"][obs_id][2].numpy()],
            xyz=obs_history["xyz"][obs_id].numpy(),
            depth=obs_history["depth"][obs_id].numpy(),
            camera_pose=obs_history["camera_poses"][obs_id].numpy(),
            camera_K=obs_history["camera_K"][obs_id].numpy(),
        )
    )
if "obs" in obs_history:
    key_obs = obs_history["obs"]

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
    encoder=encoder,
    smooth_kernel_size=parameters.get("filters/smooth_kernel_size", -1),
    use_median_filter=parameters.get("filters/use_median_filter", False),
    median_filter_size=parameters.get("filters/median_filter_size", 5),
    median_filter_max_error=parameters.get("filters/median_filter_max_error", 0.01),
    use_derivative_filter=parameters.get("filters/use_derivative_filter", False),
    derivative_filter_threshold=parameters.get(
        "filters/derivative_filter_threshold", 0.5
    ),
    instance_memory_kwargs={
        "min_pixels_for_instance_view": parameters.get(
            "min_pixels_for_instance_view", 100
        ),
        "min_instance_thickness": parameters.get("min_instance_thickness", 0.01),
        "min_instance_vol": parameters.get("min_instance_vol", 1e-6),
        "max_instance_vol": parameters.get("max_instance_vol", 10.0),
        "min_instance_height": parameters.get("min_instance_height", 0.1),
        "max_instance_height": parameters.get("max_instance_height", 1.8),
        "open_vocab_cat_map_file": parameters.get("open_vocab_cat_map_file", None),
    },
)

voxel_map.reset()
# key_obs = key_obs[::4]
key_obs = [key_obs[40]]
for idx, obs in enumerate(key_obs):

    image_array = np.array(obs.rgb, dtype=np.uint8)
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
