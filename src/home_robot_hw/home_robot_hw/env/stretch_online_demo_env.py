import datetime
import os
import uuid

import geometry_msgs
import rospy
import tf2_ros
from home_robot.agent.motion.stretch import HelloStretch

from home_robot_hw.env.stretch_demo_base_env import StretchDemoBaseEnv
from home_robot_hw.ros.recorder import Recorder
from home_robot_hw.ros.stretch_ros import HelloStretchROSInterface
from home_robot_hw.teleop.stretch_xbox_controller import StretchXboxController


class StretchOnlineDemoEnv(StretchDemoBaseEnv):
    """
    This env uses the controller to actively collect and store demos, while also returning them (via step) in a form
    that can be trained on on-line.
    """

    def __init__(
        self,
        output_file_dir,
        camera_info_in_state=False,
        record_key_frames=False,
        record_all_frames=False,
    ):
        super().__init__(initialize_ros=True)
        self._episode_in_progress = False
        self._camera_info_in_state = camera_info_in_state

        # TODO: don't hardcode this path
        robot_name = f"robot_{uuid.uuid4()}"
        self._model = HelloStretch(
            name=robot_name, visualize=False, root="", urdf_path=self.urdf_path
        )
        self._robot = HelloStretchROSInterface(
            init_cameras=True, model=self._model, depth_buffer_size=None
        )  # TODO: would 0 work?
        self._controller_handler = StretchXboxController(
            self._model,
            start_button_callback=self._start_button_callback,
            back_button_callback=self._back_button_callback,
        )

        # Storing episode replay parameters
        os.makedirs(output_file_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%b_%d_%Y_%H.%M.%S.%f")
        output_filename = os.path.join(output_file_dir, f"demo_{timestamp}.h5")
        self._recorder = (
            Recorder(output_filename, model=self._model, robot=self._robot)
            if record_all_frames
            else None
        )

        key_frame_output_filename = os.path.join(
            output_file_dir, f"demo_{timestamp}_key_frames.h5"
        )
        self._key_frame_recorder = (
            Recorder(key_frame_output_filename, model=self._model, robot=self._robot)
            if record_key_frames
            else None
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

    def close(self):
        print("Closing Online Demo Env")
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
                self._recorder.start_recording()

            if self._key_frame_recorder is not None:
                self._key_frame_recorder.start_recording()

        else:
            # Episode is about to finish
            self._next_step_done = True

            # Stop publishing key frames
            self._key_frame_poses = []

        # Setting the toggle flag at the end as a quick-and-dirty "lock" for now (TODO)
        self._episode_in_progress = not self._episode_in_progress
        print(f"Episode changed status. In progress? {self._episode_in_progress}")

    def _back_button_callback(self):
        # Call the start button to start recording before attempting to capture a keyframe.
        if self._key_frame_recorder is not None:
            if self._episode_in_progress:
                print("Saving key frame")
                rgb, depth, q, dq = self._key_frame_recorder.save_frame()

                pose = self.gripper_fk(self._model, q)
                self._key_frame_poses.append(pose)
            else:
                print(
                    "Attempted to save a keyframe before starting an episode. Please hit the start button first."
                )

    def reset(self):
        rgb, depth = self.get_images_from_robot(self._robot)
        (
            color_camera_info,
            depth_camera_info,
            camera_pose,
        ) = self.construct_camera_data_from_robot(self._robot)
        self._current_timestep = 0
        observation = self.construct_observation(
            rgb,
            depth,
            self._robot.pos,
            color_camera_info,
            depth_camera_info,
            camera_pose,
            camera_info_in_state=self._camera_info_in_state,
            current_time=0,
            max_time=1,
            model=self._model,
        )  # TODO: we don't know the max time yet... (though in this case it's 0 regardless)
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
            rgb, depth, absolute_pose, delta_pose = self._recorder.save_frame()
            (
                color_camera_info,
                depth_camera_info,
                camera_pose,
            ) = self.construct_camera_data_from_robot(self._robot)

            # TODO: how to do time? We don't know the max (currently unused downstream anyway...) This is currently always just ratio of 1
            self._current_timestep += 1
            observation = self.construct_observation(
                rgb,
                depth,
                self._robot.pos,
                color_camera_info,
                depth_camera_info,
                camera_pose,
                camera_info_in_state=self._camera_info_in_state,
                current_time=self._current_timestep,
                max_time=self._current_timestep,
                model=self._model,
            )

        info = {
            "demo_action": None
        }  # If we start using the Online env in the training loop, this class will need to be updated to be consistent
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
    demo_dir = "/home/spowers/Datasets/Stretch/demo_data/kitchen_test/v3/bottle_to_sink/1"  # TODO get from CLI
    env = StretchOnlineDemoEnv(
        output_file_dir=demo_dir, camera_info_in_state=True, record_key_frames=True
    )
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
