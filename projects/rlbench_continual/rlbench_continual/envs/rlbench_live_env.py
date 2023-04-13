import copy
from typing import Optional, Tuple

import numpy as np
import trimesh
from gym.core import ActType, ObsType
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench_continual.envs.rlbench_offline_env import RLBenchOfflineEnv
from rlbench_continual.utils.collect_rlbench_dataset import front as extract_front_view
from rlbench_continual.utils.collect_rlbench_dataset import (
    left_side as extract_left_side_view,
)
from rlbench_continual.utils.collect_rlbench_dataset import (
    overhead as extract_overhead_view,
)
from rlbench_continual.utils.collect_rlbench_dataset import (
    right_side as extract_right_side_view,
)
from rlbench_continual.utils.collect_rlbench_dataset import tasks as task_specs
from rlbench_continual.utils.collect_rlbench_dataset import wrist as extract_wrist_view


class RLBenchLiveEnv(RLBenchOfflineEnv):
    """
    A class that will run the policy in the simulation environment
    """

    # TODO: basing it on OfflineEnv to initialize our live env based on the tasks given to the offline env
    # reconsider if it diverges too far/doesn't make sense

    def __init__(self, dataset_dir, headless, views, language_embedding_model=None):
        super().__init__(
            dataset_dir,
            random_trajectory_start=False,
            trajectory_length=None,
            views=views,
            language_embedding_model=language_embedding_model,
        )

        obs_config = ObservationConfig(gripper_joint_positions=True)
        self._sim_env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(),
                gripper_action_mode=Discrete(),
            ),
            obs_config=obs_config,
            headless=headless,
        )
        self._current_task = None
        self._current_timestep = 0
        self._last_observation = None  # Used if we trigger an invalid action

    def __del__(self):
        self._sim_env.shutdown()

    def _convert_rlbench_observation_to_gym(self, rlbench_obs, language_command):
        state = np.concatenate(
            (rlbench_obs.gripper_pose, np.array([rlbench_obs.gripper_open])), axis=-1
        )  # TODO: gripper just bool right now --

        if self._language_embedding_model is not None:
            state = np.concatenate((state, language_command), axis=-1)

        # Pull out the view information for each supported camera (in particular, this generates the xyz)
        extracted_data = {}
        views = [
            ("front", extract_front_view),
            ("overhead", extract_overhead_view),
            ("left", extract_left_side_view),
            ("right", extract_right_side_view),
            ("wrist", extract_wrist_view),
        ]
        for view_name, view_extractor in views:
            extracted_data.update(
                dict(
                    zip(
                        [
                            f"{view_name}_name",
                            f"{view_name}_depth",
                            f"{view_name}_xyz",
                            f"{view_name}_rgb",
                        ],
                        view_extractor(rlbench_obs),
                    )
                )
            )

        image_obs = self._construct_image_observation_from_dict(extracted_data)

        gym_obs = {"image": image_obs, "state_vector": state}
        return gym_obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> ObsType:
        self._initialize_new_trial(options)

        # Load the task based on its stored name (TODO: de-dupe with collect_rlbench_dataset)
        task_name = self._current_trial["cmd"].asstr()[()]
        task_class, task_variant = task_specs[task_name]
        self._current_task = self._sim_env.get_task(task_class)
        if task_variant is not None:
            self._current_task.set_variation(task_variant)

        desc, rlbench_obs = self._current_task.reset()

        observation = self._convert_rlbench_observation_to_gym(
            rlbench_obs, self._current_language_command
        )
        self._last_observation = observation
        return observation

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # The action is (ee_pos, ee_rot, gripper state, time_frac), where time_frac can be used to indicate done...
        # The task expects the rotation to be already unitized, so do that
        ee_pos = action[:3]
        ee_rot = trimesh.unitize(action[3:7])
        gripper = action[7:8]
        time_frac = action[8:]

        task_action = np.concatenate((ee_pos, ee_rot, gripper), axis=-1)

        try:
            rlbench_obs, reward, done = self._current_task.step(task_action)

            """for _ in range(300):
                self._sim_env._pyrep.step_ui()
                import time
                time.sleep(0.1)"""

            obs = self._convert_rlbench_observation_to_gym(
                rlbench_obs, self._current_language_command
            )
            self._last_observation = copy.deepcopy(obs)
        except InvalidActionError:
            done = True
            reward = 0
            obs = self._last_observation

        self._current_timestep += 1

        trial_length = len(self._current_keypoint_indices)
        done_predicted = False  # time_frac > (trial_length - 0.9) / trial_length
        done_by_maxsteps_exceeded = self._current_timestep > trial_length * 3
        done = done or done_predicted or done_by_maxsteps_exceeded

        info = {}
        return obs, reward, done, info
