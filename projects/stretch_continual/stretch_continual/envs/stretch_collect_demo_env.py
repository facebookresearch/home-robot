import datetime
import os

import geometry_msgs
import rospy
import tf2_ros
from stretch_continual.envs.stretch_demo_base_env import StretchDemoBaseEnv
from stretch_continual.envs.stretch_offline_demo_env import StretchOfflineDemoEnv

from home_robot_hw.remote.api import StretchClient
from home_robot_hw.ros.recorder import Recorder
from home_robot_hw.teleop.stretch_xbox_controller import StretchXboxController


class StretchCollectDemoEnv(StretchDemoBaseEnv):
    """
    This env uses the controller to actively collect and store demos.
    """

    def __init__(
        self,
        output_file_dir,
        task_name,
        camera_info_in_state=False,
        record_key_frames=False,
        record_all_frames=False,
        include_context=False,
    ):
        super().__init__(initialize_ros=True, include_context=include_context)
        self._episode_in_progress = False
        self._camera_info_in_state = camera_info_in_state
        self._task_name = task_name
        self._client = None

        self._controller_handler = StretchXboxController(
            self.client,
            start_button_callback=self._start_button_callback,
            back_button_callback=self._back_button_callback,
        )

        # Storing episode replay parameters
        os.makedirs(output_file_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%b_%d_%Y_%H.%M.%S.%f")
        output_filename = os.path.join(output_file_dir, f"demo_{timestamp}.h5")
        self._recorder = Recorder(output_filename) if record_all_frames else None

        key_frame_output_filename = os.path.join(
            output_file_dir, f"demo_{timestamp}_key_frames.h5"
        )
        self._key_frame_recorder = (
            Recorder(key_frame_output_filename) if record_key_frames else None
        )

        # Step-related parameters
        self._next_step_done = False  # Related to replay - when we say an episode is done, the next step should have done=True
        self._step_rate = rospy.Rate(10)  # hz
        self._current_timestep = 0

        (
            self.observation_space,
            self.action_space,
        ) = self.get_stretch_obs_and_action_space(self._camera_info_in_state)

        self._key_frame_poses = []
        self._pose_broadcaster = tf2_ros.TransformBroadcaster()

    def __del__(self):
        self.close()

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

    def close(self):
        print("Closing Collect Demo Env")
        if self._recorder is not None:
            self._recorder.close()

        if self._key_frame_recorder is not None:
            self._key_frame_recorder.close()

    def _publish_to_ros(self, id, pos, rot):
        pose_message = geometry_msgs.msg.TransformStamped()
        pose_message.header.stamp = rospy.Time.now()
        pose_message.header.frame_id = "base_link"

        pose_message.child_frame_id = f"key_frame_{id}"
        pose_message.transform.translation.x = pos[0]
        pose_message.transform.translation.y = pos[1]
        pose_message.transform.translation.z = pos[2]

        pose_message.transform.rotation.x = rot[0]
        pose_message.transform.rotation.y = rot[1]
        pose_message.transform.rotation.z = rot[2]
        pose_message.transform.rotation.w = rot[3]

        self._pose_broadcaster.sendTransform(pose_message)

    def _start_button_callback(self):
        # Episode is about to start
        if not self._episode_in_progress:
            if self._recorder is not None:
                self._recorder.start_recording(self._task_name)

            if self._key_frame_recorder is not None:
                self._key_frame_recorder.start_recording(self._task_name)

        else:
            # Episode is about to finish
            self._next_step_done = True

            # Stop publishing key frames
            self._key_frame_poses = []

        # Setting the toggle flag at the end as a quick-and-dirty "lock"
        self._episode_in_progress = not self._episode_in_progress
        print(f"Episode changed status. In progress? {self._episode_in_progress}")

    def _back_button_callback(self):
        # Call the start button to start recording before attempting to capture a keyframe.
        if self._key_frame_recorder is not None:
            if self._episode_in_progress:
                print("Saving key frame")
                rgb, depth, q, dq = self._key_frame_recorder.save_frame()

                pose = self.gripper_fk(self.model, q)
                self._key_frame_poses.append(pose)
            else:
                print(
                    "Attempted to save a keyframe before starting an episode. Please hit the start button first."
                )

    def _get_current_observation(self):
        rgb, depth = self.get_images_from_robot(self.client)
        (
            color_camera_info,
            depth_camera_info,
            camera_pose,
        ) = self.construct_camera_data_from_robot(self.client)
        observation = self.construct_observation(
            rgb,
            depth,
            self.client.robot_joint_pos,
            color_camera_info,
            camera_pose,
            camera_info_in_state=self._camera_info_in_state,
            model=self.model,
        )
        return observation

    def reset(self):
        self._current_timestep = 0
        observation = self._get_current_observation()
        return observation

    def step(self, _):
        # Don't hold the lock while we wait...
        while not self._episode_in_progress and not self._next_step_done:
            rospy.sleep(0.1)

        # We collect one more step after finishing, to collect the last action and observation
        done = self._next_step_done
        self._next_step_done = False
        observation = None

        if self._recorder is not None:
            # Note: the observation returned directly may not be identical to the one recorded
            self._recorder.save_frame()
            observation = self._get_current_observation()
            self._current_timestep += 1

        info = {"demo_action": None}
        reward = 0

        if done:
            if self._recorder is not None:
                self._recorder.finish_recording()

            if self._key_frame_recorder is not None:
                self._key_frame_recorder.finish_recording()

        # Publish our key frames, to visualize them in rviz
        for pose_id, pose in enumerate(self._key_frame_poses):
            self._publish_to_ros(pose_id, *pose)

        self._step_rate.sleep()

        return observation, reward, done, info


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()
    from stretch_continual.envs.stretch_live_env import StretchLiveEnv

    mode = "collect"  # ["collect", "replay", "offline"]

    base_dir = os.path.join(os.environ["DATASET_ROOT"], "demo_data/kitchen_data/")
    task_name = "bottle_to_sink"
    collection_id = "0"  # A setup identifier, for example
    demo_dir = os.path.join(base_dir, task_name, collection_id)

    if mode == "collect":
        # Stores an h5 containing key frames in the specified output directory
        env = StretchCollectDemoEnv(
            output_file_dir=demo_dir,
            camera_info_in_state=True,
            record_key_frames=True,
            task_name=task_name,
        )
    elif mode == "replay":
        # Will replay demos collected and stored in the specified dir, chosen randomly
        env = StretchLiveEnv(demo_dir=demo_dir, camera_info_in_state=True)
    elif mode == "offline":
        # Will select random trajectories; useful for debugging
        env = StretchOfflineDemoEnv(
            demo_dir=demo_dir, camera_info_in_state=True, use_key_frames=True
        )
    else:
        raise Exception(f"Unknown mode: {mode}")

    done = False
    env.reset()

    while True:
        try:
            obs, reward, done, info = env.step(None)

            if done:
                env.reset()
        except KeyboardInterrupt:
            env.close()
            raise
