import numpy as np
from stretch_continual.envs.stretch_demo_base_env import StretchDemoBaseEnv

from home_robot.motion.stretch import HelloStretchIdx, HelloStretchKinematics


class StretchOfflineDemoEnv(StretchDemoBaseEnv):
    def __init__(
        self,
        demo_dir,
        camera_info_in_state=False,
        include_context=False,
        single_step_trajectory=True,
        eval_pos_only=False,
        use_key_frames=True,
    ):
        super().__init__(initialize_ros=False, include_context=include_context)
        self._demo_dir = demo_dir
        self._current_timestep = 0
        self._current_trajectory = None
        self._camera_info_in_state = camera_info_in_state
        self._use_key_frames = use_key_frames
        self._single_step_trajectory = single_step_trajectory
        self._random_trajectory_start = True
        self._context_observation = None
        self._eval_pos_only = eval_pos_only

        self._model = None

        (
            self.observation_space,
            self.action_space,
        ) = self.get_stretch_obs_and_action_space(self._camera_info_in_state)

    @property
    def model(self):
        # Note: should be consistent with StretchLiveEnv parameters. Not currently unified because StretchClient
        # creates its own instance of Kinematics, but I don't want the OfflineEnv to have a full StretchClient
        if self._model is None:
            self._model = HelloStretchKinematics(
                urdf_path=self._urdf_path,
                ik_type="pinocchio_optimize",
                grasp_frame=self.EE_LINK_NAME,
                ee_link_name=self.EE_LINK_NAME,
                manip_mode_controlled_joints=self.MANIP_MODE_CONTROLLED_JOINTS,
            )
        return self._model

    def _get_observation_for_timestep(self, trajectory, timestep, context_observation):
        if self._camera_info_in_state:
            (
                color_camera_info,
                depth_camera_info,
                camera_pose,
            ) = self._construct_camera_data_from_demo(trajectory, timestep=timestep)
        else:
            color_camera_info, depth_camera_info, camera_pose = None, None, None

        obs = self.get_numpy_image(trajectory["rgb"][f"{timestep}"])
        depth = self.get_numpy_image(trajectory["depth"][f"{timestep}"])

        obs = self.construct_observation(
            obs,
            depth,
            trajectory["q"][timestep],
            color_camera_info,
            depth_camera_info,
            camera_pose,
            camera_info_in_state=self._camera_info_in_state,
            model=self.model,
            context_observation=context_observation,
        )
        return obs

    def _construct_camera_data_from_demo(self, trajectory, timestep):
        color_camera_info = {
            k: np.array(v)[timestep] for k, v in trajectory["color_camera_info"].items()
        }
        depth_camera_info = {
            k: np.array(v)[timestep] for k, v in trajectory["depth_camera_info"].items()
        }
        camera_pose = np.array(trajectory["camera_pose"][timestep])
        return color_camera_info, depth_camera_info, camera_pose

    def reset(self, ensure_first=False):
        self._current_trajectory = self.randomly_select_traj_from_dir(
            self._demo_dir, only_key_frames=self._use_key_frames, cache=True
        )

        if self._random_trajectory_start and not ensure_first:
            self._current_timestep = np.random.randint(
                0, len(self._current_trajectory["q"]) - 1
            )
        else:
            self._current_timestep = 0

        if self._include_context:
            self._context_observation = self._get_observation_for_timestep(
                self._current_trajectory, timestep=0, context_observation=None
            )

        initial_observation = self._get_observation_for_timestep(
            self._current_trajectory,
            timestep=self._current_timestep,
            context_observation=self._context_observation,
        )
        return initial_observation

    def step(self, action):
        max_timesteps = len(self._current_trajectory["dq"])

        # Get the action (absolute position)
        next_timestep = self._current_timestep + 1
        next_demo_joints = self._current_trajectory["q"][next_timestep]

        self._current_timestep = next_timestep
        done = (
            self._single_step_trajectory or self._current_timestep + 1 >= max_timesteps
        )

        demo_pos, demo_rot = self.gripper_fk(self.model, next_demo_joints)
        gripper = next_demo_joints[HelloStretchIdx.GRIPPER]
        action_commanded = np.concatenate(
            (demo_pos, demo_rot, np.array([gripper])), axis=-1
        )

        time_fraction = (self._current_timestep - 1) / (max_timesteps - 2)
        action_commanded = np.concatenate(
            (action_commanded, np.array([time_fraction])), axis=-1
        )

        info = {"demo_action": action_commanded, "eval_pos_only": self._eval_pos_only}

        obs = self._get_observation_for_timestep(
            self._current_trajectory,
            timestep=self._current_timestep,
            context_observation=self._context_observation,
        )

        reward = 0

        return obs, reward, done, info
