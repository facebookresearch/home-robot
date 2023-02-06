from typing import Tuple, Any, Dict, Optional
import numpy as np

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from home_robot_sim.env.habitat_abstract_env import HabitatEnv
import home_robot
from .constants import (
    HM3DtoCOCOIndoor,
    FloorplannertoMukulIndoor,
    mukul_33categories_padded,
    MIN_DEPTH_REPLACEMENT_VALUE,
    MAX_DEPTH_REPLACEMENT_VALUE,
)
from .visualizer import Visualizer


class HabitatObjectNavEnv(HabitatEnv):
    def __init__(self, habitat_env: habitat.core.env.Env, config):
        super().__init__(habitat_env)

        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self.visualizer = Visualizer(config)

        self.episodes_data_path = config.TASK_CONFIG.DATASET.DATA_PATH
        assert ("floorplanner" in self.episodes_data_path or "hm3d" in self.episodes_data_path)
        if "hm3d" in self.episodes_data_path:
            if config.AGENT.SEMANTIC_MAP.semantic_categories == "coco_indoor":
                self.semantic_category_mapping = HM3DtoCOCOIndoor()
            else:
                raise NotImplementedError
        elif "floorplanner" in self.episodes_data_path:
            if config.AGENT.SEMANTIC_MAP.semantic_categories == "mukul_indoor":
                self.semantic_category_mapping = FloorplannertoMukulIndoor()
            else:
                raise NotImplementedError

        if not self.ground_truth_semantics:
            from home_robot.agent.perception.detection.detic.detic_loader import get_detic
            if config.AGENT.SEMANTIC_MAP.semantic_categories == "mukul_indoor":
                self.segmentation = get_detic(
                    vocabulary="custom",
                    custom_vocabulary=",".join(mukul_33categories_padded),
                    sem_gpu_id=(-1 if config.NO_GPU else self.habitat_env.sim.gpu_device),
                )
            else:
                raise NotImplementedError

    def reset(self):
        habitat_obs = self.habitat_env.reset()
        self.semantic_category_mapping.reset_instance_id_to_category_id(self.habitat_env)
        self._last_obs = self._preprocess_obs(habitat_obs)
        self.visualizer.reset()

    def _preprocess_obs(self,
                        habitat_obs: habitat.core.simulator.Observations
                        ) -> home_robot.core_interfaces.Observations:
        depth = self._preprocess_depth(habitat_obs["depth"])

        semantic, semantic_vis = self._preprocess_semantic(
            rgb=habitat_obs["rgb"],
            depth=depth,
            semantic=habitat_obs["semantic"],
        )

        goal_id, goal_name = self._preprocess_goal(habitat_obs["objectgoal"])

        obs = home_robot.core_interfaces.Observations(
            rgb=habitat_obs["rgb"],
            depth=depth,
            semantic=semantic,
            compass=habitat_obs["compass"],
            gps=habitat_obs["gps"],
            task_observations={
                "goal_id": goal_id,
                "goal_name": goal_name,
                "semantic_frame": semantic_vis
            }
        )
        return obs

    def _preprocess_semantic(self,
                             rgb: np.ndarray,
                             depth: np.ndarray,
                             semantic: np.array,
                             ) -> Tuple[np.array, np.array]:
        if self.ground_truth_semantics:
            instance_id_to_category_id = self.semantic_category_mapping.instance_id_to_category_id
            semantic = instance_id_to_category_id[semantic[:, :, -1]]
            # TODO Ground-truth semantic visualization
            semantic_vis = rgb
        else:
            semantic, semantic_vis = self.segmentation.get_prediction(rgb, depth)
        return semantic.astype(int), semantic_vis

    def _preprocess_depth(self, depth: np.array) -> np.array:
        rescaled_depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        rescaled_depth[depth == 0.0] = MIN_DEPTH_REPLACEMENT_VALUE
        rescaled_depth[depth == 1.0] = MAX_DEPTH_REPLACEMENT_VALUE
        return rescaled_depth[:, :, -1]

    def _preprocess_goal(self, goal: np.array) -> Tuple[int, str]:
        return self.semantic_category_mapping.map_goal_id(goal[0])

    def _preprocess_action(self, action: home_robot.core_interfaces.Action) -> int:
        return HabitatSimActions[action.name]

    def _process_info(self, info: Dict[str, Any]) -> Any:
        self.visualizer.visualize(**info)
