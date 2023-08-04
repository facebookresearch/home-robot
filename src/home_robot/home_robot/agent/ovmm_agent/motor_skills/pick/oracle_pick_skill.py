# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
import torch
from habitat.tasks.rearrange.rearrange_sensors import IsHoldingSensor

from home_robot.agent.ovmm_agent.motor_skills.skill import SkillPolicy


class OraclePickSkill(SkillPolicy):
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

    def on_enter(self, batch_idxs, observations):
        super().on_enter(batch_idxs, observations)
        self.grasped = False
        self.steps = 0

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        # make self.grasped True a torch
        grasped = torch.tensor(self.grasped, dtype=torch.bool, device=masks.device)

        # combine it with is_holding
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        if is_holding and not grasped:
            self.termination_message = "Gripper already holding an object"
        is_holding = is_holding * grasped

        wait = (
            1.0
            if self.steps > self._config.wait_time_for_obj_to_grasp
            else 0.0
        )
        if (is_holding * wait).type(torch.bool):
            self.steps = 0
            return (is_holding * 1.0).type(torch.bool)
        else:
            return (is_holding * 0.0).type(torch.bool)

    def grasp(self, obj_idx):
        keep_T = mn.Matrix4.translation(mn.Vector3(0.1, 0.0, 0.0))
        self.env.sim.grasp_mgr.snap_to_obj(
            self.env.sim.scene_obj_ids[obj_idx],
            force=True,
            rel_pos=mn.Vector3(0.1, 0.0, 0.0),
            keep_T=keep_T,
        )

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
        # action[0, self.grip_action_index] = 1.0
        self.steps += 1

        # Teleport the object to the gripper
        if not self.env.sim.grasp_mgr.is_grasped:
            scene_obj_pos = self.env.sim.get_scene_pos()
            obj_idx = np.argmin(
                np.linalg.norm(scene_obj_pos - self.target_pos, ord=2, axis=-1)
            )

            # Is the object within reach?
            ee_pos = np.array(self.env.sim.robot.ee_transform.translation)
            target_pos = self.target_pos
            ee_dist_to_target = np.linalg.norm(ee_pos - target_pos)
            if (ee_dist_to_target > self._config.grasping_distance):
                # print('Object too far away, distance: ', ee_dist_to_target)
                self.termination_message = "Object too far away"
                return action, None

            # # Is the object something that can be grasped?
            # if self.target not in self.env.scene_parser.objects:
            #     # Probably trying to pick a "fridge" or "drawer"
            #     self.termination_message = "[Privileged] This is not a movable object"
            #     return action, None

            # # Is the object accessible?
            # surface = self.env.scene_parser.objects[self.target].receptacle
            # object_is_accesible = self.env.scene_parser.is_accessible(surface, self.thre_for_art_state)
            # if not object_is_accesible:
            #     self.termination_message = "[Privileged] Receptacle is closed, you need to open it first"
            #     return action, None

            self.grasp(obj_idx)
            self.grasped = True

        return action, None
