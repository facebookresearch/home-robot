import select
import sys
import time
import timeit

import numpy as np
import trimesh
from stretch_continual.envs.stretch_demo_base_env import StretchDemoBaseEnv

from home_robot.motion.stretch import HelloStretchIdx
from home_robot_hw.remote.api import StretchClient

# Returns the next observation from the demo, instead of the true observation from the robot. Useful for
# detecting if drift is the issue, or some fundamental mis-processing of the data
DEBUG = False


class StretchLiveEnv(StretchDemoBaseEnv):
    """
    Initializes from a demo, and then runs the predicted actions on the real robot
    """

    exec_tol = np.array(
        [
            1e-1,  # x
            1e-1,  # y
            0.01,  # theta
            0.001,  # lift
            0.01,  # arm
            0.01,  # gripper
            0.015,  # wrist roll
            0.05,  # wrist pitch 0.015 -- the wrist sometimes has trouble holding itself up...
            0.015,  # wrist yaw
            0.01,  # head pan
            0.01,  # head tilt
        ]
    )

    def __init__(
        self,
        demo_dir,
        camera_info_in_state=False,
        use_true_action=False,
        perturb_start_state=False,
        include_context=False,
    ):
        super().__init__(initialize_ros=True, include_context=include_context)

        self._demo_dir = demo_dir
        self._current_timestep = 0
        self._current_trajectory = None
        self._use_key_frames = True
        self._use_true_action = use_true_action  # "True" = real pose - current pose. Not="true" = predicted (as in from a NN - the input to step)
        self._camera_info_in_state = camera_info_in_state
        self._perturb_start_state = perturb_start_state
        self._cached_camera_data = None
        self._perturbation_limits = np.array(
            [
                0,  # x
                0,  # y
                0,  # theta
                0.5,  # lift
                0.1,  # arm
                0,  # gripper
                0,  # wrist roll
                0,  # wrist pitch
                0,  # wrist yaw
                0.1,  # head pan
                0,  # head tilt
            ]
        )

        self._client = None

        self._context_observation = None  # TODO: de-dupe with offline
        (
            self.observation_space,
            self.action_space,
        ) = self.get_stretch_obs_and_action_space(self._camera_info_in_state)

    @property
    def client(self):
        if self._client is None:
            self._client = StretchClient(
                urdf_path=self._urdf_path,
                init_node=self._initialize_ros,
                ik_type="pinocchio_optimize",
                grasp_frame=self.EE_LINK_NAME,
                ee_link_name=self.EE_LINK_NAME,
                manip_mode_controlled_joints=self.MANIP_MODE_CONTROLLED_JOINTS,
            )
            self._client.switch_to_manipulation_mode()
        return self._client

    @property
    def model(self):
        return self.client.robot_model

    def _get_robot_pose(self):
        return self.client.robot_joint_pos

    def _get_observation_for_timestep(self, context_observation):
        # TODO: de-dupe with offline
        if self._camera_info_in_state:
            (
                color_camera_info,
                depth_camera_info,
                camera_pose,
            ) = self.construct_camera_data_from_robot(self.client)

        else:
            color_camera_info, depth_camera_info, camera_pose = None, None, None

        rgb, depth = self.get_images_from_robot(self.client)
        pos = self._get_robot_pose()

        if DEBUG:
            rgb = self.get_numpy_image(self._current_trajectory["rgb"][f"0"])
            depth = self.get_numpy_image(self._current_trajectory["depth"][f"0"])
            pos = self._current_trajectory["q"][self._current_timestep]

        assert not np.all(
            pos == 0
        ), "Sometimes seeing a position with all 0s, which silently confuses everything. Asserting to figure out when it happens."

        obs = self.construct_observation(
            rgb,
            depth,
            pos,
            color_camera_info,
            depth_camera_info,
            camera_pose,
            camera_info_in_state=self._camera_info_in_state,
            model=self.model,
            context_observation=context_observation,
        )
        return obs

    def _goto(self, pose):
        # We store the full joint angles, but the client expects [Base translation, lift, arm, yaw, pitch, roll],
        # so translate

        reduced_pose = self.client.manip._extract_joint_pos(pose)
        gripper = pose[HelloStretchIdx.GRIPPER]
        reduced_pose[HelloStretchIdx.BASE_X] = 0  # Not supporting base motion

        self.client.manip.goto_joint_positions(reduced_pose, blocking=False)
        self.client.manip.move_gripper(gripper)
        self.client.head.set_pan_tilt(
            pose[HelloStretchIdx.HEAD_PAN],
            pose[HelloStretchIdx.HEAD_TILT],
        )

    def reset(self, ensure_first=True):
        self._current_trajectory = self.randomly_select_traj_from_dir(
            self._demo_dir, only_key_frames=self._use_key_frames, cache=True
        )
        self._current_timestep = 0
        pose = self._current_trajectory["q"][0]

        if self._perturb_start_state:
            perturbation = np.random.uniform(
                -self._perturbation_limits, self._perturbation_limits
            )
            pose += perturbation

        go_to_pose = True

        while go_to_pose:
            self._goto(pose)
            time.sleep(1)
            command = input("Robot in a good state to start? (r to retry)")
            go_to_pose = "r" in command.lower()

        # TODO: not caching camera at all...
        if self._include_context:
            self._context_observation = self._get_observation_for_timestep(
                context_observation=None,
            )

        initial_observation = self._get_observation_for_timestep(
            context_observation=self._context_observation,
        )

        return initial_observation

    def _give_user_stop_option(self):
        # TODO: look up what these variables mean
        # If in pycharm and this isn't working, see this: https://youtrack.jetbrains.com/issue/PY-42488
        # You can type into the input and hit enter *before* the prompt comes up, as well.
        print(
            "Hit x now to end the episode or b to end the action (will take effect before the next action): "
        )
        done = False
        continue_to_next = False
        i = True

        # Drain the buffer, so if the user enters several commands, they don't continue propagating into future actions
        while i:
            i, o, e = select.select(
                [sys.stdin], [], [], 0.1
            )  # 0.1 s to enter it -- the expectation is you'll spam it
            if i:
                command = sys.stdin.readline().strip()
                done = done or "x" in command.lower()
                continue_to_next = continue_to_next or "b" in command.lower()

        return done, continue_to_next

    def _interpolate_robot_to_in_joint_space(self, goal_joints):
        # Very naive (TODO) interpolation in joint space
        num_steps = 10
        original_joints = self._get_robot_pose()
        delta_joints = (goal_joints - original_joints) / num_steps

        # If our delta is "sufficiently large", step through the intermediate poses
        if np.any(np.abs(delta_joints) > self.exec_tol):
            for step_id in range(1, num_steps):
                next_joints = original_joints + step_id * delta_joints
                self._goto(next_joints)

        # Step to the desired end state regardless
        self._goto(goal_joints)
        current_pose = self._get_robot_pose()
        return current_pose

    def step(self, action):
        print("==========================STEP START================================")
        print(f"Original action (pos): {action}")
        done = False

        # Check before the action is done, so if the robot is in a bad state, we don't make it worse.
        done, _ = done or self._give_user_stop_option()

        # If the action is an end-effector pose, convert it to joint positions
        if action is not None and not done:
            pos = action[:3]
            rot = action[3:7]
            gripper = action[7:8]
            done = action[8] > (len(self._current_trajectory["q"]) - 0.9) / len(
                self._current_trajectory["q"]
            )  # TODO: threshold a bit arbitrary

            if done:
                print(f"Action predicted done, so finishing. Done: {action[8]}")

            original_head_pan = self._get_robot_pose()[HelloStretchIdx.HEAD_PAN]
            original_head_tilt = self._get_robot_pose()[HelloStretchIdx.HEAD_TILT]

            rot = trimesh.util.unitize(rot)
            action = self.gripper_ik(
                self.model, pos, rot, current_joints=self._get_robot_pose()
            )
            print(f"PB Action: {action}")

            if action is None:
                print("WARNING: IK failed. Ending episode early")
                done = True
            else:
                # print(f"Sanity check fk (EE): {self.gripper_fk(self.model, action)}")
                action[HelloStretchIdx.GRIPPER] = gripper
                action[HelloStretchIdx.HEAD_PAN] = original_head_pan
                action[HelloStretchIdx.HEAD_TILT] = original_head_tilt

        # Get the true action regardless of whether we're using it, for logging
        next_timestep = self._current_timestep + 1
        true_action = (
            self._current_trajectory["q"][self._current_timestep]
            if self._current_timestep < len(self._current_trajectory["q"])
            else None
        )

        if self._use_true_action:
            action = true_action
            self._current_timestep = next_timestep
        else:
            self._current_timestep += 1

        if not done:
            last_q = None
            next_q = None
            err = None
            start_time = timeit.default_timer()
            max_seconds = 60
            end_action = False

            # If each dimension is either close enough, or effectively stationary, move on
            while (
                not done
                and not end_action
                and timeit.default_timer() - start_time < max_seconds
                and (
                    err is None
                    or last_q is None
                    or np.any(
                        np.logical_and(
                            err > self.exec_tol,
                            np.abs(next_q - last_q) > self.exec_tol / 10,
                        )
                    )
                )
            ):
                last_q = next_q
                next_q = self._interpolate_robot_to_in_joint_space(action)
                err = np.abs(next_q - action)
                end_episode, end_action = self._give_user_stop_option()
                done = done or end_episode

        print(f"Commanding action (joints): {action} vs {true_action}")

        # Only record the demo action if we used it, otherwise let the reward play
        if self._use_true_action:
            info = {"demo_action": action}
        else:
            info = {}

        if self._use_true_action or DEBUG:
            done = done or self._current_timestep + 1 >= len(
                self._current_trajectory["q"]
            )
        else:
            done = (
                done
                or self._current_timestep + 1 > len(self._current_trajectory["q"]) * 2
            )

        observation = self._get_observation_for_timestep(
            context_observation=self._context_observation,
        )

        if done:
            reward = None
            while reward is None:
                try:
                    reward = float(input("Did the robot succeed? (r <= 1)"))
                    reward = (
                        reward if reward <= 1 else None
                    )  # Protection against typo'ing "9" or "2"
                except Exception as e:
                    print(f"Reward incorrectly entered. Failed with error: {e}")
        else:
            reward = 0
        print("==========================STEP COMPLETE================================")

        return observation, reward, done, info
