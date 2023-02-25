import torch
import torch.nn as nn
import skimage.morphology

from home_robot.utils.morphology import binary_dilation


class FrontierExplorationPolicy(nn.Module):
    """
    Frontier exploration: select high-level exploration goals of the closest
    unexplored region.
    """

    def __init__(self):
        super().__init__()

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(10))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(1))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )

    @property
    def goal_update_steps(self):
        return 1

    def forward(self, map_features):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 8 + num_sem_categories, M, M)

        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
        """
        # Select unexplored area
        frontier_map = (map_features[:, [1], :, :] == 0).float()

        # Dilate explored area
        frontier_map = 1 - binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel
        )

        # Select the frontier
        frontier_map = (
            binary_dilation(frontier_map, self.select_border_kernel) - frontier_map
        )
        return frontier_map
