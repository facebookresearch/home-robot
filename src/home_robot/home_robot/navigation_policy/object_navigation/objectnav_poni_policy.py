# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from einops import asnumpy

from home_robot.mapping.semantic.constants import MapConstants as MC

from .poni.default import get_cfg
from .poni.model import PFModel
from .poni.visualization import visualize_area_pf, visualize_object_category_pf


class ObjectNavPONIPolicy(nn.Module):
    def __init__(
        self,
        pf_model_path: str,
        area_weight_coef: float = 0.5,
        pf_masking_opt: str = "explored",
        mask_nearest_locations: bool = True,
        device=None,
    ):
        super().__init__()
        self.area_weight_coef = area_weight_coef
        self.pf_masking_opt = pf_masking_opt
        self.mask_nearest_locations = mask_nearest_locations
        loaded_state = torch.load(pf_model_path, map_location="cpu")
        pf_model_cfg = get_cfg()
        pf_model_cfg.merge_from_other_cfg(loaded_state["cfg"])
        self.pf_model = PFModel(pf_model_cfg)
        # Remove dataparallel modules
        state_dict = {
            k.replace(".module", ""): v for k, v in loaded_state["state_dict"].items()
        }
        self.pf_model.load_state_dict(state_dict)
        self.eval()
        if device is not None:
            self.pf_model.to(device)

    @property
    def goal_update_steps(self):
        return 1

    def forward(self, map_features, object_category):
        goal_maps_f, found_goals = self.navigate_to_goal_if_found(
            map_features,
            object_category,
        )
        goal_maps_e = self.explore_otherwise(
            map_features,
            object_category,
        )
        mask = found_goals.unsqueeze(1).unsqueeze(2)
        goal_maps = goal_maps_f * mask + goal_maps_e * (1 - mask)
        return goal_maps, found_goals

    def navigate_to_goal_if_found(self, map_features, object_category):
        # map_features - (B, 10 + num_sem_categories, M, N)
        # object_category - (B, )
        goal_maps = torch.zeros_like(map_features[:, 0])
        found_goal = torch.zeros(map_features.shape[0]).to(map_features.device)
        for i in range(map_features.shape[0]):
            goal_i = int(object_category[i].item())
            goal_map_i = map_features[i, MC.NON_SEM_CHANNELS * 2 + goal_i]
            if torch.any(goal_map_i > 0.5):
                goal_maps[i] = goal_map_i
                found_goal[i] = 1.0
        return goal_maps, found_goal

    def explore_otherwise(self, map_features, object_category):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 10 + num_sem_categories, M, M)
            object_category: object goal category
            start_recep_category: start receptacle category
            end_recep_category: end receptacle category
            nav_to_recep: If both object_category and recep_category are specified, whether to navigate to receptacle
        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
            goal category of shape (batch_size,)
        """
        # Convert inputs to appropriate format for the PF model
        proc_inputs = self.do_proc(map_features)  # (B, N, H, W)

        with torch.no_grad():
            pfs, area_pfs = self.pf_model.infer(proc_inputs)

        # Take the mean with area_pfs
        init_pfs = pfs
        if area_pfs is not None:
            awc = self.area_weight_coef
            pfs = (1 - awc) * pfs + awc * area_pfs

        # Get action
        goal_cat_id = object_category.long()
        goal_maps = self.get_action(
            pfs,
            goal_cat_id,
            umap=1.0 - map_features[:, MC.EXPLORED_MAP],
            agent_locs=map_features[:, MC.CURRENT_LOCATION],
        )
        pred_maps = {
            "pfs": pfs,
            "raw_pfs": init_pfs,
            "area_pfs": area_pfs,
        }
        pred_maps = {
            k: asnumpy(v) if v is not None else v for k, v in pred_maps.items()
        }
        # Visualize the transformed PFs
        self._cached_visualizations = self.generate_pf_vis(
            proc_inputs,
            pred_maps,
            goal_cat_id,
            dset="gibson",
        )
        return goal_maps

    def do_proc(self, inputs):
        """
        Map consists of multiple channels containing the following:
        ----------- For local map -----------------
        1. Obstacle Map
        2. Explored Area
        3. Current Agent Location
        4. Past Agent Locations
        5. Been Close Map
        ----------- For global map -----------------
        6. Obstacle Map
        7. Explored Area
        8. Current Agent Location
        9. Past Agent Locations
        10. Been Close Map
        ----------- For semantic local map -----------------
        11,12,13,.. : Semantic Categories
        """
        # The input to PF model consists of Free map, Obstacle Map, Semantic Categories
        # The last semantic map channel is ignored since it belongs to unknown categories.
        obstacle_map = inputs[:, MC.OBSTACLE_MAP : MC.OBSTACLE_MAP + 1]
        explored_map = inputs[:, MC.EXPLORED_MAP : MC.EXPLORED_MAP + 1]
        semantic_map = inputs[:, 2 * MC.NON_SEM_CHANNELS : -1]
        free_map = ((obstacle_map < 0.5) & (explored_map >= 0.5)).float()
        outputs = torch.cat([free_map, obstacle_map, semantic_map], dim=1)
        return outputs

    def get_action(self, pfs, goal_cat_id, umap, agent_locs):
        """
        Computes distance from (agent -> location) + (location -> goal)
        based on PF predictions. It then selects goal as location with
        least distance.

        Args:
            pfs = (B, N + 2, H, W) potential fields
            goal_cat_id = (B, ) goal category
            umap = (B, H, W) unexplored map
            agent_locs = (B, H, W) agent map
        """
        B, _, _, W = pfs.shape[0], pfs.shape[1] - 2, pfs.shape[2], pfs.shape[3]
        goal_pfs = []
        for b in range(B):
            goal_pf = pfs[b, goal_cat_id[b].item() + 2, :]
            goal_pfs.append(goal_pf)
        goal_pfs = torch.stack(goal_pfs, dim=0)
        if self.pf_masking_opt == "unexplored":
            # Filter out explored locations
            goal_pfs = goal_pfs * umap
        # Filter out locations very close to the agent
        if self.mask_nearest_locations:
            goal_pfs[agent_locs == 1] = 0.0

        act_ixs = goal_pfs.view(B, -1).max(dim=1).indices
        # Convert action to (0, 1) values for x and y coors
        goal_maps = torch.zeros_like(pfs[:, 0])
        for b in range(B):
            act_ix = act_ixs[b].item()
            # Convert action to (0, 1) values for x and y coors
            act_x = act_ix % W
            act_y = act_ix // W
            goal_maps[b, act_y, act_x] = 1

        return goal_maps

    def generate_pf_vis(self, semantic_maps, pred_maps, goal_cat_ids, dset):
        vis_maps = []
        for i in range(semantic_maps.shape[0]):
            vis_maps_i = {}
            semmap = semantic_maps[i]
            pfs = pred_maps["pfs"][i]
            cat_id = goal_cat_ids[i].cpu().item()
            pfs_rgb = visualize_object_category_pf(semmap, pfs, cat_id, dset)
            vis_maps_i["pfs"] = pfs_rgb
            if "raw_pfs" in pred_maps and pred_maps["raw_pfs"] is not None:
                raw_pfs = pred_maps["raw_pfs"][i]
                raw_pfs_rgb = visualize_object_category_pf(
                    semmap,
                    raw_pfs,
                    cat_id,
                    dset,
                )
                vis_maps_i["raw_pfs"] = raw_pfs_rgb
            if "area_pfs" in pred_maps and pred_maps["area_pfs"] is not None:
                area_pfs = pred_maps["area_pfs"][i]
                area_pfs_rgb = visualize_area_pf(semmap, area_pfs, dset=dset)
                vis_maps_i["area_pfs"] = area_pfs_rgb
            vis_maps.append(vis_maps_i)
        return vis_maps
