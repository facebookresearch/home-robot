import math
from typing import Optional, Tuple

import gym
import numpy as np
from gym.core import ActType, ObsType
from rlbench_continual.utils.rlbench_dataset import RLBenchDataset


class RLBenchOfflineEnv(gym.Env):
    def __init__(
        self, dataset_dir, random_trajectory_start=True, single_step_trajectory=True
    ):
        self._random_trajectory_start = random_trajectory_start
        self._single_step_trajectory = single_step_trajectory

        self._loader = RLBenchDataset(
            dataset_dir,
            data_augmentation=False,
            first_keypoint_only=False,
            debug_closest_pt=False,
        )
        self._camera_keys = [
            "front_rgb",
            "front_depth",
            "front_xyz",
        ]  # Concat'd together -- TODO: make available in observation?
        self._current_trial = None
        self._current_keypoint_indices = None  # Indexes into the trial timesteps
        self._current_timestep = 0

        # TODO: lang using: self._current_trial['description'].asstr()[()]

        (
            self.observation_space,
            self.action_space,
        ) = self._get_stretch_obs_and_action_space()

    def _get_stretch_obs_and_action_space(self):
        channels = 7  # TODO: calculate from keys?
        state_size = 8  # TODO: should be 8...keeping for now so I can keep using my trained model
        action_size = 9  # 3 pos, 4 quat, 1 gripper, 1 completion fraction

        # Note, low and high for state_vector and action_space likely to be inaccurate...not currently used by USIP
        image_shape = self._loader.trials[0]["front_xyz"].shape[1:3]
        observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=1, shape=(*image_shape, channels), dtype=np.float32
                ),  # TODO: may not actually be bounded this way... includes xyz
                "state_vector": gym.spaces.Box(low=0, high=1, shape=(state_size,)),
            }
        )
        action_space = gym.spaces.Box(low=-math.pi, high=math.pi, shape=(action_size,))
        return observation_space, action_space

    def _get_combined_ee_state(self, trial, timestep, keypoint_indices):
        trial_time_index = keypoint_indices[timestep]
        ee_pos = trial["ee_xyz"][trial_time_index]
        ee_rot = trial["ee_rot"][trial_time_index]
        gripper_state = trial["gripper"][trial_time_index]

        adjusted_pose = np.concatenate(
            (ee_pos, ee_rot, np.array([gripper_state])), axis=-1
        )
        return adjusted_pose

    def _construct_image_observation_from_dict(self, obs_dict: dict) -> np.ndarray:
        image_observation = []
        for camera_key in self._camera_keys:
            timestep_data = np.array(obs_dict[camera_key])

            if "depth" in camera_key:
                timestep_data = np.expand_dims(timestep_data, axis=-1)

            if "rgb" in camera_key:
                timestep_data = (
                    timestep_data / 255.0
                )  # Since we're including xyz, just scale everything here (TODO?)

            image_observation.append(timestep_data)

        combined_image = np.concatenate(image_observation, axis=-1)
        return combined_image

    def _construct_observation(self, trial, timestep, keypoint_indices) -> ObsType:
        trial_time_index = keypoint_indices[timestep]
        ee_state = self._get_combined_ee_state(trial, timestep, keypoint_indices)
        extracted_timestep_data = {
            key: trial[key][trial_time_index] for key in trial.temporal_keys
        }
        combined_image = self._construct_image_observation_from_dict(
            extracted_timestep_data
        )

        return {"image": combined_image, "state_vector": ee_state}

    def _initialize_new_trial(self, options=None):
        trial_id = np.random.randint(len(self._loader.trials))
        self._current_trial = self._loader.trials[trial_id]

        # Include the first frame in the keypoints
        self._current_keypoint_indices = np.concatenate(
            (np.array([0]), np.array(self._current_trial["keypoints"]))
        )  # TODO: append last state too? More keypoints?
        self._current_timestep = (
            0
            if options is not None and options.get("ensure_first", False)
            else np.random.randint(len(self._current_keypoint_indices) - 1)
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> ObsType:
        self._initialize_new_trial(options)
        return self._construct_observation(
            self._current_trial, self._current_timestep, self._current_keypoint_indices
        )

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:

        max_timesteps = len(self._current_keypoint_indices)
        next_timestep = self._current_timestep + 1
        true_action = self._get_combined_ee_state(
            self._current_trial, next_timestep, self._current_keypoint_indices
        )

        # If we only have, effectively, 1 action (2 keypoints, but including the starting state), just say that one action is the final one
        # TODO: hmm...think more about the offsets here
        time_fraction = (
            (next_timestep - 1) / (max_timesteps - 2) if max_timesteps > 2 else 1
        )  # TODO: check
        true_action = np.concatenate((true_action, np.array([time_fraction])), axis=-1)

        done = self._single_step_trajectory or next_timestep == max_timesteps - 1
        reward = 0
        info = {"demo_action": true_action}
        obs = self._construct_observation(
            self._current_trial, next_timestep, self._current_keypoint_indices
        )

        self._current_timestep = next_timestep

        return obs, reward, done, info
