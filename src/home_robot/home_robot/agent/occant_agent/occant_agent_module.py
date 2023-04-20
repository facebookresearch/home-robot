import time

import torch
import torch.nn as nn

from home_robot.mapping.geometric.geometric_map_module_anticipation import (
    GeometricMapModuleWithAnticipation,
)
from home_robot.mapping.occant_utils.configs.defaults import get_cfg
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.navigation_policy.exploration.frontier_exploration_policy import (
    FrontierExplorationPolicy,
)
from home_robot.navigation_policy.exploration.occant_policy import GlobalPolicy

# Do we need to visualize the frontier as we explore?
debug_frontier_map = False
USE_FRONTIER_POLICY = False


class OccAntAgentModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        device = torch.device("cuda:0")
        self.geometric_map_module = GeometricMapModuleWithAnticipation(
            frame_height=config.ENVIRONMENT.frame_height,
            frame_width=config.ENVIRONMENT.frame_width,
            camera_height=config.ENVIRONMENT.camera_height,
            hfov=config.ENVIRONMENT.hfov,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
            explored_radius=config.AGENT.SEMANTIC_MAP.explored_radius,
            been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
            map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
            occant_cfg_path=config.AGENT.SEMANTIC_MAP.occant_cfg_path,
            occant_ckpt_path=config.AGENT.SEMANTIC_MAP.occant_ckpt_path,
            min_depth=config.ENVIRONMENT.min_depth,
            max_depth=config.ENVIRONMENT.max_depth,
            device=device,
        )
        self.device = device
        self.occant_cfg = get_cfg(config.AGENT.SEMANTIC_MAP.occant_cfg_path)
        if USE_FRONTIER_POLICY:
            self.policy = FrontierExplorationPolicy(
                exploration_strategy=config.AGENT.exploration_strategy
            )
        else:
            self.policy = GlobalPolicy(self.occant_cfg.GLOBAL_POLICY)
            self.load_policy_weights(config.AGENT.SEMANTIC_MAP.occant_ckpt_path)
            self.policy.eval()
            self.policy.to(self.device)

    def load_policy_weights(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["global_state_dict"]
        state_dict = {k.replace("actor_critic.", ""): v for k, v in state_dict.items()}
        self.policy.load_state_dict(state_dict)
        print(
            "\n"
            + "=" * 10
            + " Successfully loaded OccAnt policy weights "
            + "=" * 10
            + "\n"
        )

    @property
    def goal_update_steps(self):
        return self.policy.goal_update_steps

    def forward(
        self,
        seq_obs,
        seq_pose_delta,
        seq_dones,
        seq_update_global,
        seq_camera_poses,
        init_local_map,
        init_local_map_seen,
        init_global_map,
        init_local_pose,
        init_global_pose,
        init_lmb,
        init_origins,
    ):
        """Update maps and poses with a sequence of observations, and predict
        high-level goals from map features.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation)
             of shape (batch_size, sequence_length, 3 + 1,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) done flags that
             indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            seq_camera_poses: sequence of (batch_size, 4, 4) camera poses
            init_local_map: initial local map before any updates of shape
             (batch_size, 4, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 4, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)

        Returns:
            seq_goal_map: sequence of binary maps encoding goal(s) of shape
             (batch_size, sequence_length, M, M)
            final_local_map: final local map after all updates of shape
             (batch_size, 4, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 4, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """
        # t0 = time.time()

        # Update map with observations and generate map features
        batch_size, sequence_length = seq_obs.shape[:2]
        (
            seq_map_features,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.geometric_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            seq_camera_poses,
            init_local_map,
            init_local_map_seen,
            init_global_map,
            init_local_pose,
            init_global_pose,
            init_lmb,
            init_origins,
        )

        # t1 = time.time()
        # print(f"[Semantic mapping] Total time: {t1 - t0:.2f}")

        # Predict high-level goals from map features
        # batched across sequence length x num environments
        if USE_FRONTIER_POLICY:
            map_features = seq_map_features.flatten(0, 1)
            goal_map = self.policy(map_features)
        else:
            assert sequence_length == 1
            global_policy_inputs = self._create_global_policy_inputs(
                final_local_map, seq_local_pose.flatten(0, 1)
            )
            _, global_action, _, _ = self.policy.act(
                global_policy_inputs, None, None, None
            )
            G = self.policy.G
            global_action_map_x = torch.fmod(
                global_action.squeeze(1), G
            ).float()  # (bs, )
            global_action_map_y = (global_action.squeeze(1) / G).float()  # (bs, )
            # Convert to MxM local map coordinates
            _, _, M, _ = final_local_map.shape
            global_action_map_x = (global_action_map_x * M / G).long()
            global_action_map_y = (global_action_map_y * M / G).long()
            goal_map = torch.zeros_like(final_local_map[:, :1])
            goal_map[:, :, global_action_map_y, global_action_map_x] = 1

        seq_goal_map = goal_map.view(batch_size, sequence_length, *goal_map.shape[-2:])

        # Compute the frontier map here
        frontier_map = goal_map
        seq_frontier_map = frontier_map.view(
            batch_size, sequence_length, *frontier_map.shape[-2:]
        )
        if debug_frontier_map:
            import matplotlib.pyplot as plt

            plt.subplot(121)
            plt.imshow(seq_frontier_map[0, 0].numpy())
            plt.subplot(122)
            plt.imshow(goal_map[0].numpy())
            plt.show()
            breakpoint()
        # t2 = time.time()
        # print(f"[Policy] Total time: {t2 - t1:.2f}")

        return (
            seq_goal_map,
            seq_frontier_map,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )

    def _create_global_policy_inputs(self, global_map, global_pose):
        """
        global_map  - (bs, N, V, V) - global obstacles, explored area, current and past position
        global_pose - (bs, 3) - (x, y, orientation) in real-world coordinates
        """
        obstacles = global_map[:, MC.OBSTACLE_MAP]
        explored = global_map[:, MC.EXPLORED_MAP]
        visited = global_map[:, MC.VISITED_MAP]
        current = global_map[:, MC.CURRENT_LOCATION]
        map_params = self.geometric_map_module.map_size_parameters
        global_loc = (global_pose[:, :2] * 100.0 / map_params.resolution).int()
        h_t = torch.stack(
            [obstacles, explored, visited, current], dim=1
        )  # (bs, 4, M, M)

        global_policy_inputs = {
            "pose_in_map_at_t": global_loc.to(self.device),
            "map_at_t": h_t.to(self.device),
        }

        return global_policy_inputs
