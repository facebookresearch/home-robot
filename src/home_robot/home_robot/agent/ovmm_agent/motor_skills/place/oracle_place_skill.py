# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from habitat.tasks.rearrange.rearrange_sensors import (
    IsHoldingSensor,
    RelativeRestingPositionSensor,
)

from home_robot.agent.ovmm_agent.motor_skills import PickSkillPolicy
from home_robot.agent.ovmm_agent.motor_skills.skill import SkillPolicy


class OraclePlaceSkill(SkillPolicy):
    def __init__(
        self,
        config,
        observation_space,
        action_space,
        batch_size,
        env,
    ):
        super().__init__(
            config,
            action_space,
            batch_size,
            should_keep_hold_state=False,
        )
        self.env = env
        self.steps = 0
        self.thre_for_art_state = config.thre_for_art_state
        self.wait_time_for_obj_to_place = config.wait_time_for_obj_to_place

    def on_enter(self, batch_idxs, observations):
        super().on_enter(batch_idxs, observations)
        self.was_grasping = False
        self.just_entered = True
        self.steps = 0

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        # Is the agent not holding an object and is the end-effector at the
        # resting position?
        if not self.was_grasping:
            self.termination_message = 'Currently not grasping - cannot place'

        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1).type(torch.bool)

        is_done = (
            (~is_holding) & (self.steps >= self.wait_time_for_obj_to_place) & (self.was_grasping)
        )

        return is_done

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        action = torch.zeros(prev_actions.shape, device=masks.device)
        self.steps += 1

        if self.just_entered and self.env.sim.grasp_mgr.is_grasped:
            self.was_grasping = True

        self.just_entered = False

        if self.env.sim.grasp_mgr.is_grasped:

            # Is the receptacle within reach?
            ee_pos = np.array(self.env.sim.robot.ee_transform.translation)
            target_pos = self.target_pos
            ee_dist_to_target = np.linalg.norm(ee_pos - target_pos)
            if (ee_dist_to_target > self._config.placing_distance):
                self.termination_message = 'Not close enough to target position'
                return action, None

            # Is the target a receptacle or a surface? Cannot place on other object
            # if self.target not in self.env.scene_parser.receptacles and self.target not in self.env.scene_parser.surfaces:
            #     self.termination_message = 'Target is not a receptacle or surface'
            #     return action, None

            # # Is the receptacle open or not articulated?
            # receptacle_name = self.target
            # is_accessible = self.env.scene_parser.is_accessible(receptacle_name, self.thre_for_art_state)

            # if is_accessible:
            obj_id = self.env.sim.grasp_mgr._snapped_obj_id
            self.env.sim.grasp_mgr.desnap(True)
            self.env.sim.get_rigid_object_manager().get_object_by_id(
                obj_id
            ).translation = self.target_pos
            # else:
            #     self.termination_message = '[Privileged] Receptacle is closed, you need to open it first'
            #     return action, None

        return action, None
