# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, List, Optional

import numpy as np
import rospy
from slap_manipulation.utils.slap_planner import CombinedSLAPPlanner

import home_robot
from home_robot.core.interfaces import (
    Action,
    ContinuousFullBodyAction,
    DiscreteNavigationAction,
    HybridAction,
    Observations,
)
from home_robot.motion.stretch import (
    STRETCH_ARM_EXTENSION,
    STRETCH_ARM_LIFT,
    STRETCH_HOME_Q,
    STRETCH_PREGRASP_Q,
    STRETCH_TO_GRASP,
)
from home_robot.utils.geometry import xyt2sophus, xyt_base_to_global, xyt_global_to_base
from home_robot.utils.pose import to_matrix
from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv
from home_robot_hw.ros.visualizer import ArrayVisualizer, Visualizer
from home_robot_hw.utils.grasping import GraspPlanner


class GeneralLanguageEnv(StretchPickandPlaceEnv):
    """Derivative environment for running free-form language experiments with
    SLAP being used for executing learnt skills"""

    def __init__(
        self,
        config,
        cat_map_file: str = None,
        visualize_planner: bool = False,
        ros_grasping: bool = True,
        test_grasping: bool = False,
        dry_run: bool = False,
        debug: bool = False,
        closed_loop: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            config,
            cat_map_file,
            visualize_planner=visualize_planner,
            ros_grasping=False,
            test_grasping=test_grasping,
            dry_run=dry_run,
            debug=debug,
            *args,
            **kwargs
        )
        if self.grasp_planner is None and ros_grasping:
            # create grasp_planner with preferred pregrasp_height
            self.grasp_planner = GraspPlanner(
                self.robot,
                self,
                visualize_planner=visualize_planner,
                pregrasp_height=1.4,
            )

        self.closed_loop = closed_loop
        self.current_goal_id = None
        self.current_goal_name = None
        self.skill_planner = CombinedSLAPPlanner(self.robot)
        self.debug_visualizer = Visualizer("orientation_goal", [0, 0.5, 0.5, 0.5])
        self.interaction_visualizer = Visualizer("interaction_point", [1, 0, 0, 0.5])
        self.action_visualizer = ArrayVisualizer("slap_actions", [0.5, 0.5, 0, 0.5])

    def _set_goal(self, info: Dict):
        """sets goals based on SLAP assumptions to either search for only one
        or two objects"""
        if len(info["object_list"]) > 1:
            goal_find = info["object_list"][0]
            goal_obj = info["object_list"][1]
            goal_place = None
        else:
            goal_place = info["object_list"][0]
            goal_find = None
            goal_obj = None
        self.set_goal(goal_find, goal_obj, goal_place, check_receptacles=False)

    def _switch_to_manip_mode(
        self, grasp_only: bool = False, pre_demo_pose: bool = False
    ):
        """Rotate the robot and put it in the right configuration for grasping"""

        # We switch to navigation mode in order to rotate by 90 degrees
        if not self.robot.in_navigation_mode() and not self.dry_run:
            self.robot.switch_to_navigation_mode()

        # Dummy out robot execution code for perception tests
        # Also do not rotate if you are just doing grasp testing
        if grasp_only:
            if pre_demo_pose:
                self.robot.move_to_pre_demo_posture()
            else:
                self.robot.move_to_manip_posture()
            return
        if not self.dry_run and not self.test_grasping:
            print("[ENV] Rotating robot")
            self.robot.nav.navigate_to([0, 0, np.pi / 2], relative=True, blocking=True)
            self.robot.move_to_manip_posture()
            rospy.sleep(5.0)

    def apply_action(
        self, action: Action, info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Handle all sorts of different actions we might be inputting into this class.
        We provide both a discrete and a continuous action handler."""
        # Process the action so we know what to do with it
        if not isinstance(action, HybridAction):
            action = HybridAction(action)
        # By default - no arm control
        joints_action = None
        gripper_action = 0
        # Handle discrete actions first
        if action.is_discrete():
            action = action.get()
            continuous_action = np.zeros(3)
            if action == DiscreteNavigationAction.MOVE_FORWARD:
                print("[GeneralLanguageEnv] Move forward")
                continuous_action[0] = self.forward_step
            elif action == DiscreteNavigationAction.TURN_RIGHT:
                print("[GeneralLanguageEnv] TURN RIGHT")
                continuous_action[2] = -self.rotate_step
            elif action == DiscreteNavigationAction.TURN_LEFT:
                print("[GeneralLanguageEnv] Turn left")
                continuous_action[2] = self.rotate_step
            elif action == DiscreteNavigationAction.STOP:
                # Do nothing if "stop"
                continuous_action = None
                return True
            elif action == DiscreteNavigationAction.EXTEND_ARM:
                """Extend the robot arm"""
                print("[GeneralLanguageEnv] Extending arm")
                joints_action = self.robot.model.create_action(
                    lift=STRETCH_ARM_LIFT, arm=STRETCH_ARM_EXTENSION
                ).joints
                continuous_action = None
            elif action == DiscreteNavigationAction.MANIPULATION_MODE:
                # set goal based on info dict here
                self._set_goal(info)
                # sleeping here so observation comes from view after turning head
                if not self.robot.in_manipulation_mode():
                    self._switch_to_manip_mode()
                    rospy.sleep(2.0)
                continuous_action = None
            elif action == DiscreteNavigationAction.NAVIGATION_MODE:
                # set goal based on info dict here
                self._set_goal(info)
                continuous_action = None
                if not self.robot.in_navigation_mode():
                    self._switch_to_nav_mode()
                    print("[GeneralLanguageEnv] Sending robot to 0,0,0")
                    self.robot.nav.navigate_to(
                        np.zeros(3), relative=False, blocking=True
                    )
                continuous_action = None
            elif action == DiscreteNavigationAction.PICK_OBJECT:
                print("[GeneralLanguageEnv] Discrete pick policy")
                continuous_action = None
                # Run in a while loop until we have succeeded
                while not rospy.is_shutdown():
                    if self.dry_run:
                        # Dummy out robot execution code for perception tests
                        break
                    ok = self.grasp_planner.try_grasping(
                        wait_for_input=self.debug,
                        visualize=self.test_grasping,
                        switch_mode=False,
                        z_standoff=0.6,
                    )
                    if ok:
                        break
            elif action == DiscreteNavigationAction.SNAP_OBJECT:
                # Close the gripper
                gripper_action = 1
            elif action == DiscreteNavigationAction.DESNAP_OBJECT:
                # Open the gripper
                gripper_action = -1
            else:
                print(
                    "[GeneralLanguageEnv] Action not implemented in pick-and-place environment:",
                    action,
                )
                continuous_action = None
        elif action.is_navigation():
            continuous_action = action.get()
        elif action.is_manipulation():
            if isinstance(action.action, ContinuousFullBodyAction):
                joints_action, continuous_action = action.get()
            else:
                pos, ori, gripper = action.get()
                continuous_action = None
                print("[GeneralLanguageEnv] Receiving a ContinuousEndEffectorAction")
                if "p2p-motion" in info.keys() and info["p2p-motion"]:
                    combined_action = np.concatenate((pos, ori, gripper), axis=-1)
                    ok = self.skill_planner.try_executing_skill(
                        combined_action,
                        False,
                        p2p_motion=True,
                        trimesh_format=True,
                    )
                else:
                    # visualize p_i
                    # convert p_i to global frame
                    import trimesh.transformations as tra

                    quat = np.array([0, 0, 0, 1])
                    p_i_global = np.copy(info["interaction_point"])
                    base_matrix = xyt2sophus(self.robot.nav.get_base_pose()).matrix()
                    p_i_global = tra.transform_points(
                        np.expand_dims(p_i_global, axis=0), base_matrix
                    )
                    p_i_matrix = to_matrix(p_i_global, quat)
                    self.interaction_visualizer.publish_2d(p_i_matrix)
                    p_i_global = p_i_matrix[:3, 3].reshape(3)
                    pose_matrix_array = []
                    for i in range(len(pos)):
                        pose_matrix = to_matrix(pos[i], ori[i], trimesh_format=True)
                        pose_matrix_array.append(pose_matrix)
                    # visualize actions so we can introspect
                    self.action_visualizer(pose_matrix_array, frame_id="base_link")
                    combined_action = np.concatenate((pos, ori, gripper), axis=-1)
                    ok = self.skill_planner.try_executing_skill(combined_action, False)

        # Move, if we are not doing anything with the arm
        if continuous_action is not None and not self.test_grasping:
            print(
                "[GeneralLanguageEnv] Execute navigation action:",
                continuous_action,
            )
            if not self.robot.in_navigation_mode():
                self.robot.switch_to_navigation_mode()
            if not self.dry_run:
                print("GOTO", continuous_action)
                if "SLAP" in info.keys():
                    quat = np.array([0, 0, 0, 1])
                    # visualize everything in rviz so we can introspect
                    p_i_local = np.copy(info["interaction_point"])
                    p_i_local_matrix = to_matrix(p_i_local, quat)
                    self.interaction_visualizer.publish_2d(
                        p_i_local_matrix, frame_id="base_link"
                    )

                    # convert p_i to global frame
                    import trimesh.transformations as tra

                    p_i_global = np.copy(info["interaction_point"])
                    base_matrix = xyt2sophus(self.robot.nav.get_base_pose()).matrix()
                    p_i_global = tra.transform_points(
                        np.expand_dims(p_i_global, axis=0), base_matrix
                    )
                    p_i_matrix = to_matrix(p_i_global, quat)
                    self.interaction_visualizer.publish_2d(p_i_matrix)
                    p_i_global = p_i_matrix[:3, 3].reshape(3)

                    # add global offsets and assign global orientation
                    desired_pose_in_global = (
                        p_i_global
                        + info["offset_distance"] * info["global_offset_vector"]
                    )
                    desired_pose_in_global[2] = info["global_orientation"]
                    # Visualize
                    goal_matrix = xyt2sophus(desired_pose_in_global).matrix()
                    self.debug_visualizer(goal_matrix)
                    self.robot.nav.navigate_to(
                        desired_pose_in_global, relative=False, blocking=True
                    )
                else:
                    self.robot.nav.navigate_to(
                        continuous_action, relative=True, blocking=True
                    )
        self._handle_joints_action(joints_action)
        self._handle_gripper_action(gripper_action)
        # Update the visualizer
        if self.visualizer is not None and info is not None and "viz" in info.keys():
            print("[GeneralLanguageEnv] visualizing")
            self.visualizer.visualize(**info["viz"])
        return False
