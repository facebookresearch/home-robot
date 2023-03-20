import uuid
import numpy as np

from home_robot.hardware.stretch_ros import HelloStretchIdx
from home_robot.motion.robot import HelloStretch
from home_robot.envs.imitation.stretch_demo_base_env import StretchDemoBaseEnv


class StretchOfflineDemoEnv(StretchDemoBaseEnv):
    def __init__(self, demo_dir, camera_info_in_state=False, include_context=False,
                 single_step_trajectory=True, eval_pos_only=False, use_key_frames=True):
        super().__init__(initialize_ros=False, include_context=include_context)
        self._demo_dir = demo_dir
        self._current_timestep = 0
        self._current_trajectory = None
        self._camera_info_in_state = camera_info_in_state
        self._use_key_frames = use_key_frames
        self._model = None
        self._cached_camera_data = None
        self._single_step_trajectory = single_step_trajectory
        self._random_trajectory_start = True
        self._context_observation = None
        self._eval_pos_only = eval_pos_only

        self.observation_space, self.action_space = self.get_stretch_obs_and_action_space(self._camera_info_in_state)

    @property
    def model(self):
        if self._model is None:
            robot_name = f"robot_{uuid.uuid4()}"
            self._model = HelloStretch(name=robot_name, visualize=False, root="", urdf_path=self.urdf_path)
        return self._model

    def _get_observation_for_timestep(self, trajectory, timestep, cache_camera, use_camera_cache, context_observation):
        assert not (cache_camera and use_camera_cache), "Can't both recreate and use the camera cache"

        if self._camera_info_in_state:
            if use_camera_cache:
                color_camera_info, depth_camera_info, camera_pose = self._cached_camera_data
            else:
                color_camera_info, depth_camera_info, camera_pose = self.construct_camera_data_from_demo(trajectory, timestep=timestep)

                if cache_camera:
                    self._cached_camera_data = color_camera_info, depth_camera_info, camera_pose

        else:
            color_camera_info, depth_camera_info, camera_pose = None, None, None

        obs = self.get_numpy_image(trajectory['rgb'][f'{timestep}'])
        depth = self.get_numpy_image(trajectory['depth'][f'{timestep}'])

        obs = self.construct_observation(obs, depth, trajectory['q'][timestep],
                                         color_camera_info, depth_camera_info, camera_pose,
                                         camera_info_in_state=self._camera_info_in_state,
                                         current_time=timestep,
                                         max_time=len(trajectory['q']),
                                         model=self.model, context_observation=context_observation)
        return obs

    def reset(self, ensure_first=False):
        self._current_trajectory = self.randomly_select_traj_from_dir(self._demo_dir, only_key_frames=self._use_key_frames)

        if self._random_trajectory_start and not ensure_first:
            self._current_timestep = np.random.randint(0, len(self._current_trajectory['q']) - 1)
        else:
            self._current_timestep = 0

        # TODO: not caching camera here...
        if self._include_context:
            self._context_observation = self._get_observation_for_timestep(self._current_trajectory, timestep=0,
                                                                           cache_camera=False, use_camera_cache=False,
                                                                           context_observation=None)

        initial_observation = self._get_observation_for_timestep(self._current_trajectory, timestep=self._current_timestep,
                                                                 cache_camera=False, use_camera_cache=False,
                                                                 context_observation=self._context_observation)
        return initial_observation

    def step(self, action):
        max_timesteps = len(self._current_trajectory['dq'])

        # Get the action (absolute position)
        next_timestep = self._current_timestep + 1
        next_demo_joints = self._current_trajectory['q'][next_timestep]

        self._current_timestep = next_timestep
        done = self._single_step_trajectory or self._current_timestep + 1 >= max_timesteps

        demo_pos, demo_rot = self.gripper_fk(self.model, next_demo_joints)

        # Debugging - TODO spowers remove
        #recomputed_joints = self.gripper_ik(self.model, demo_pos, demo_rot)  # TODO: q0 is not consistent (e.g. if we stepped more than once)
        #print(f"Recomputed joints: {recomputed_joints} vs {next_demo_joints}")
        #recomp_pos, recomp_rot = self.gripper_fk(self.model, recomputed_joints)
        #print(f"Recomputed pos: {recomp_pos}, rot: {recomp_rot} vs original: {demo_pos}, {demo_rot}")

        gripper = next_demo_joints[HelloStretchIdx.GRIPPER]
        action_commanded = np.concatenate((demo_pos, demo_rot, np.array([gripper])), axis=-1)

        time_fraction = ((self._current_timestep - 1) / (max_timesteps - 2))  #/10  # TODO: ...check this
        action_commanded = np.concatenate((action_commanded, np.array([time_fraction])), axis=-1)

        info = {"demo_action": action_commanded,
                "eval_pos_only": self._eval_pos_only}

        obs = self._get_observation_for_timestep(self._current_trajectory, timestep=self._current_timestep,
                                                 cache_camera=False, use_camera_cache=False,
                                                 context_observation=self._context_observation)

        reward = 0

        return obs, reward, done, info
