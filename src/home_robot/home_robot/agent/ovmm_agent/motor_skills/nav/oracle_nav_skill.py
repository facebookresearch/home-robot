# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import List, Optional, Tuple

import habitat_sim
import magnum as mn
import numpy as np
import torch
from habitat.tasks.utils import get_angle

from home_robot.agent.ovmm_agent.motor_skills.nn_skill import NnSkillPolicy
from home_robot.agent.ovmm_agent.motor_skills.skill import SkillPolicy
from home_robot.agent.ovmm_agent.motor_skills.utils import find_action_range


def place_agent_at_dist_from_pos(
    target_position: np.ndarray,
    rotation_perturbation_noise: float,
    distance_threshold: float,
    sim,
    num_spawn_attempts: int,
    physics_stability_steps: int,
    agent=None,
    navmesh_offset: Optional[List[Tuple[float, float]]] = None,
):
    """
    Places the robot at closest point if distance_threshold is -1.0 otherwise
    will place the robot at `distance_threshold` away.
    """
    if distance_threshold == -1.0:
        if navmesh_offset is not None:
            return place_robot_at_closest_point_with_navmesh(
                target_position, sim, navmesh_offset, agent=agent
            )
        else:
            return place_robot_at_closest_point(target_position, sim, agent=agent)
    else:
        raise NotImplementedError("distance_threshold != -1.0 not implemented.")


def place_robot_at_closest_point(
    target_position: np.ndarray,
    sim,
    agent=None,
):
    """
    Gets the agent's position and orientation at the closest point to the target position.
    :return: The robot's start position, rotation, and whether the placement was a failure (True for failure, False for success).
    """
    if agent is None:
        agent = sim.robot

    agent_pos = sim.safe_snap_point(target_position)
    desired_angle = get_angle_to_pos(np.array(target_position - agent_pos))

    return agent_pos, desired_angle, False


def place_robot_at_closest_point_with_navmesh(
    target_position: np.ndarray,
    sim,
    navmesh_offset: Optional[List[Tuple[float, float]]] = None,
    agent=None,
):
    """
    Gets the agent's position and orientation at the closest point to the target position.
    :return: The robot's start position, rotation, and whether the placement was a failure (True for failure, False for success).
    """
    if agent is None:
        agent = sim.robot

    agent_pos = sim.safe_snap_point(target_position)
    desired_angle = get_angle_to_pos(np.array(target_position - agent_pos))

    # Cache the initial location of the agent
    cache_pos = agent.base_pos
    # Make a copy of agent trans
    trans = mn.Matrix4(agent.sim_obj.transformation)

    # Set the base pos of the agent
    trans.translation = agent_pos
    # Project the nav pos
    nav_pos_3d = [np.array([xz[0], cache_pos[1], xz[1]]) for xz in navmesh_offset]  # type: ignore
    # Do transformation to get the location
    center_pos_list = [trans.transform_point(xyz) for xyz in nav_pos_3d]

    for center_pos in center_pos_list:
        # Update the transformation of the agent
        trans.translation = center_pos
        cur_pos = [trans.transform_point(xyz) for xyz in nav_pos_3d]
        # Project the height
        cur_pos = [np.array([xz[0], cache_pos[1], xz[2]]) for xz in cur_pos]

        is_collision = False
        for pos in cur_pos:
            if not sim.pathfinder.is_navigable(pos):
                is_collision = True
                break

        if not is_collision:
            return (
                np.array(center_pos),
                agent.base_rot,
                False,
            )

    return agent_pos, desired_angle, False


def get_angle_to_pos(rel_pos: np.ndarray) -> float:
    """
    :param rel_pos: Relative 3D positive from the robot to the target like: `target_pos - robot_pos`.
    :returns: Angle in radians.
    """

    forward = np.array([1.0, 0, 0])
    rel_pos = np.array(rel_pos)
    forward = forward[[0, 2]]
    rel_pos = rel_pos[[0, 2]]

    heading_angle = get_angle(forward, rel_pos)
    c = np.cross(forward, rel_pos) < 0
    if not c:
        heading_angle = -1.0 * heading_angle
    return heading_angle  # type: ignore


def compute_turn(rel, turn_vel, robot_forward):
    is_left = np.cross(robot_forward, rel) > 0
    if is_left:
        vel = [0, -turn_vel]
    else:
        vel = [0, turn_vel]
    return vel  # type: ignore


class SimpleVelocityControlEnv:
    """
    Simple velocity control environment for moving agent
    """

    def __init__(self, sim_freq=120.0):
        # the velocity control
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True
        self._sim_freq = sim_freq

    def act(self, trans, vel):
        linear_velocity = vel[0]
        angular_velocity = vel[1]
        # Map velocity actions
        self.vel_control.linear_velocity = mn.Vector3(
            [linear_velocity, 0.0, 0.0],
        )
        self.vel_control.angular_velocity = mn.Vector3(
            [0.0, angular_velocity, 0.0],
        )
        # Compute the rigid state
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()),
            trans.translation,
        )
        # Get the target rigit state based on the simulation frequency
        target_rigid_state = self.vel_control.integrate_transform(
            1 / self._sim_freq,
            rigid_state,
        )
        # Get the ending pos of the agent
        end_pos = target_rigid_state.translation
        # Offset the height
        end_pos[1] = trans.translation[1] + 1
        print('+1')
        # Construct the target trans
        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(),
            target_rigid_state.translation,
        )

        return target_trans  # type: ignore


class OracleNavSkill(SkillPolicy):
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
            should_keep_hold_state=True,
        )
        self.env = env
        self._has_reached_goal = torch.zeros(self._batch_size)
        # Defien the contorller
        self.base_vel_ctrl = habitat_sim.physics.VelocityControl()
        self.base_vel_ctrl.controlling_lin_vel = True
        self.base_vel_ctrl.lin_vel_is_local = True
        self.base_vel_ctrl.controlling_ang_vel = True
        self.base_vel_ctrl.ang_vel_is_local = True

        self.dist_thresh = config.dist_thresh
        self.turn_thresh = config.turn_thresh
        self.forward_velocity = config.forward_velocity
        self.turn_velocity = config.turn_velocity
        self.sim_freq = config.sim_freq

        self.navmesh_offset = config.navmesh_offset

        self.enable_backing_up = config.enable_backing_up

        self.action_range  = find_action_range(self.action_space, "base_velocity")
        self.linear_velocity_index = self.action_range[0]
        self.angular_velocity_index = self.action_range[1] - 1

    def _path_to_point(self, point):
        """
        Obtain path to reach the coordinate point. If agent_pos is not given
        the path starts at the agent base pos, otherwise it starts at the agent_pos
        value
        :param point: Vector3 indicating the target point
        """
        agent_pos = self.env.sim.robot.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self.env.sim.pathfinder.find_path(path)
        if not found_path:
            print('COULD NOT FIND PATH')
            return [agent_pos, point]
        self.path = path
        return path.points

    def is_collision(self, trans) -> bool:
        """
        The function checks if the agent collides with the object
        given the navmesh
        """
        nav_pos_3d = [
            np.array([xz[0], 0.0, xz[1]]) for xz in self.navmesh_offset
        ]  # type: ignore
        cur_pos = [trans.transform_point(xyz) for xyz in nav_pos_3d]
        cur_pos = [
            np.array([xz[0], self.env.sim.robot.base_pos[1], xz[2]]) for xz in cur_pos
        ]

        for pos in cur_pos:  # noqa: SIM110
            # Return true if the pathfinder says it is not navigable
            if not self.env.sim.pathfinder.is_navigable(pos):
                return True

        return False

    def fix_robot_leg(self):
        """
        Fix the robot leg's joint position
        """
        self.env.sim.robot.leg_joint_pos = self.env.sim.robot.params.leg_init_params

    def rotation_collision_check(
        self,
        next_pos,
    ):
        """
        This function checks if the robot needs to do backing-up action
        """
        # Make a copy of agent trans
        trans = mn.Matrix4(self.env.sim.robot.sim_obj.transformation)
        # Initialize the velocity controller
        vc = SimpleVelocityControlEnv(120.0)
        angle = float("inf")
        # Get the current location of the agent
        cur_pos = self.env.sim.robot.base_pos
        # Set the trans to be agent location
        trans.translation = self.env.sim.robot.base_pos
        return False

        while abs(angle) > self.turn_thresh:
            # Compute the robot facing orientation
            rel_pos = (next_pos - cur_pos)[[0, 2]]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(trans.transform_vector(forward))
            robot_forward = robot_forward[[0, 2]]
            angle = get_angle(robot_forward, rel_pos)
            vel = compute_turn(rel_pos, self.turn_velocity, robot_forward)
            trans = vc.act(trans, vel)
            cur_pos = trans.translation

            if self.is_collision(trans):
                return True

        return False

    def set_target(self, target, env):
        """Set the target (receptable, object) of the skill"""
        try:
            # For target is the x, y format
            pos = target.split(',')
            x, y = float(pos[0]), float(pos[1])
            target_pos = np.array([x, self.env.sim.robot.base_pos[1], y])
        except Exception:
            # For the target is the id of the receptacle/object
            target_pos = self.env.scene_parser.set_dynamic_target(target, self._config.name)
        self.target = target
        self.target_pos = target_pos

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # We do not feed any velocity command
        action = torch.zeros(prev_actions.shape, device=masks.device)

        # Place agent at target position that is collision free
        final_nav_targ, _, _ = place_agent_at_dist_from_pos(
            np.array(self.target_pos),
            0.0,
            -1.0,
            self.env.sim,
            100,
            1,
            # self.env.sim.robot,
            self.env.sim.robot,
            self.navmesh_offset,
        )

        # The location of the target objects
        obj_targ_pos = np.array(self.target_pos)

        # Get the base transformation for the robot
        base_T = self.env.sim.robot.base_transformation
        # Find the paths
        curr_path_points = self._path_to_point(final_nav_targ)
        self.curr_path_points = curr_path_points
        # Get the robot position
        robot_pos = np.array(self.env.sim.robot.base_pos)

        if curr_path_points is None:
            raise RuntimeError("Pathfinder returns empty list")
        else:
            # Compute distance and angle to target
            if len(curr_path_points) == 1:
                curr_path_points += curr_path_points

            cur_nav_targ = curr_path_points[1]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target
            rel_targ = cur_nav_targ - robot_pos

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
            # Get the angles
            angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)

            # Compute the distance
            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]],
            )
            at_goal = (
                dist_to_final_nav_targ < self.dist_thresh
                and angle_to_obj < self.turn_thresh
            )

            # Planning to see if the robot needs to do back-up
            need_move_backward = False
            if (
                dist_to_final_nav_targ >= self.dist_thresh
                and angle_to_target >= self.turn_thresh
                and not at_goal
            ):
                # check if there is a collision caused by rotation
                # if it does, we should block the rotation, and
                # only move backward
                need_move_backward = self.rotation_collision_check(
                    cur_nav_targ,
                )
            # print("need_move_backward", need_move_backward)
            # print('enable_backing_up', self.enable_backing_up)
            if need_move_backward and self.enable_backing_up:
                # Backward direction
                forward = np.array([-1.0, 0, 0])
                robot_forward = np.array(base_T.transform_vector(forward))
                # Compute relative target
                rel_targ = cur_nav_targ - robot_pos
                # Compute heading angle (2D calculation)
                robot_forward = robot_forward[[0, 2]]
                rel_targ = rel_targ[[0, 2]]
                rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
                # Get the angles
                angle_to_target = get_angle(robot_forward, rel_targ)
                angle_to_obj = get_angle(robot_forward, rel_pos)
                # Compute the distance
                dist_to_final_nav_targ = np.linalg.norm(
                    (final_nav_targ - robot_pos)[[0, 2]],
                )
                at_goal = (
                    dist_to_final_nav_targ < self.dist_thresh
                    and angle_to_obj < self.turn_thresh
                )

            if not at_goal:
                if dist_to_final_nav_targ < self.dist_thresh:
                    # Do not want to look at the object to reduce collision
                    # Look at the object
                    vel = compute_turn(
                        rel_pos,
                        self.turn_velocity,
                        robot_forward,
                    )
                elif angle_to_target < self.turn_thresh:
                    # Move towards the target
                    vel = [self.forward_velocity, 0]
                else:
                    # Look at the target waypoint.
                    vel = compute_turn(
                        rel_targ,
                        self.turn_velocity,
                        robot_forward,
                    )
                self._has_reached_goal[cur_batch_idx] = 0.0
            else:
                vel = [0, 0]
                self._has_reached_goal[cur_batch_idx] = 1.0

            if need_move_backward:
                vel[0] = -1 * vel[0]


        # Reset the robot's leg joints
        self.fix_robot_leg()

        action[cur_batch_idx, self.linear_velocity_index] = vel[0]
        action[cur_batch_idx, self.angular_velocity_index] = vel[1]
        # print("dist_to_final_nav_targ", dist_to_final_nav_targ, vel)
        # self.env.sim.robot.base_pos = np.array([self.env.sim.robot.base_pos[0], -2.0, self.env.sim.robot.base_pos[2]])
        # print(self.env.sim.robot.base_pos)

        return action, self._has_reached_goal[0]

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        return (self._has_reached_goal[batch_idx] > 0.0).to(masks.device)
