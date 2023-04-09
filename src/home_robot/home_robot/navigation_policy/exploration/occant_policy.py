# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

import home_robot.mapping.occant_utils.common as ocu


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GlobalPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.G = config.map_size

        self.actor = nn.Sequential(  # (8, G, G)
            nn.Conv2d(8, 8, 3, padding=1),  # (8, G, G)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 5, padding=2),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 2, 5, padding=2),  # (2, G, G)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 1, 5, padding=2),  # (1, G, G)
            Flatten(),  # (G*G, )
        )

        self.critic = nn.Sequential(  # (8, G, G)
            nn.Conv2d(8, 8, 3, padding=1),  # (8, G, G)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 5, padding=2),  # (4, G, G)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 2, 5, padding=2),  # (2, G, G)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 1, 5, padding=2),  # (1, G, G)
            Flatten(),
            nn.Linear(self.G * self.G, 1),
        )

        if config.use_data_parallel:
            self.actor = nn.DataParallel(
                self.actor,
                device_ids=config.gpu_ids,
                output_device=config.gpu_ids[0],
            )
            self.critic = nn.DataParallel(
                self.critic,
                device_ids=config.gpu_ids,
                output_device=config.gpu_ids[0],
            )

    @property
    def goal_update_steps(self):
        return 25

    def forward(self, inputs):
        raise NotImplementedError

    def _get_h12(self, inputs):
        x = inputs["pose_in_map_at_t"]
        h = inputs["map_at_t"]

        h_1 = ocu.crop_map(h, x[:, :2], self.G)
        h_2 = F.adaptive_max_pool2d(h, (self.G, self.G))

        h_12 = torch.cat([h_1, h_2], dim=1)

        return h_12

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        """
        Note: inputs['pose_in_map_at_t'] must obey the following conventions:
              origin at top-left, downward Y and rightward X in the map coordinate system.
        """
        h_12 = self._get_h12(inputs)
        action_logits = self.actor(h_12)
        dist = ocu.FixedCategorical(logits=action_logits)
        value = self.critic(h_12)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, prev_actions, masks):
        h_12 = self._get_h12(inputs)
        value = self.critic(h_12)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, prev_actions, masks, action):
        h_12 = self._get_h12(inputs)
        action_logits = self.actor(h_12)
        dist = ocu.FixedCategorical(logits=action_logits)
        value = self.critic(h_12)

        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
