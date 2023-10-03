import datetime
import math
import os
import tempfile
import time

import gym
import h5py
import numpy as np
from data_tools.image import img_from_bytes

from home_robot.motion.stretch import HelloStretchIdx

CAMERA_FAR_PLANE = 1  # m
CAMERA_SCALE_CONST = 10000
STRETCH_URDF_DIR = os.environ["STRETCH_URDF_DIR"]


class CacheContainer(object):
    """
    Spoof file-like object for when we want to cache the h5 information instead of re-loading it
    """

    def __init__(self, cached_data):
        self.cached_data = cached_data

    def __enter__(self):
        return self.cached_data

    def __exit__(self, *args):
        pass


class StretchDemoBaseEnv(gym.Env):
    # The URDFs provided with stretch_control do not support base_x_joint motion, so ignore that during IK for now
    MANIP_MODE_CONTROLLED_JOINTS = [
        "ignore",  # Originally base_x_joint
        "joint_lift",
        "joint_arm_l3",
        "joint_arm_l2",
        "joint_arm_l1",
        "joint_arm_l0",
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_wrist_roll",
    ]

    # This EE link does not require any offsets or conversions to use --
    # it is located directly in the middle of the gripper
    EE_LINK_NAME = "link_gripper_fingertip_center"

    def __init__(self, initialize_ros, include_context):
        super().__init__()
        self._urdf_path = STRETCH_URDF_DIR
        self._initialize_ros = initialize_ros
        self._include_context = include_context
        self._trajectory_cache = {"linked_files": {}}

    def _recursive_listdir(self, directory):
        # From: https://stackoverflow.com/questions/19309667/recursive-os-listdir
        return [os.path.join(dp, f) for dp, dn, fn in os.walk(directory) for f in fn]

    def _recursively_load_h5py(self, trajectory_data):
        extracted_data = {}
        for traj_key, traj_value in trajectory_data.items():
            if isinstance(traj_value, h5py.Group):
                extracted_data[traj_key] = self._recursively_load_h5py(traj_value)
            else:
                extracted_data[traj_key] = np.array(traj_value)

        return extracted_data

    def load_all_h5_from_dir(
        self, directory, only_key_frames, temp_aggregation_file, cache
    ):
        """
        We may have multiple h5 files, each of which may have multiple trajectories. This method loads all the h5 files
        into an aggregated h5 via ExternalLink. If we're caching, we'll load it into a local dictionary, and spoof
        that as a file-like object, so we can use these loading methods interchangeably.
        """
        h5_id = 0
        output_path = temp_aggregation_file.name
        max_tries = 5

        if not cache or len(self._trajectory_cache["linked_files"]) == 0:
            for try_id in range(max_tries):
                try:
                    with h5py.File(output_path, "w") as output_file:
                        linked_files_group = output_file.create_group("linked_files")

                        for filename in self._recursive_listdir(directory):
                            # If we expect it, it should be there. If we don't, it should not
                            file_allowed = only_key_frames == ("key_frames" in filename)
                            if (
                                os.path.splitext(filename)[-1].lower() == ".h5"
                                and "aggregated" not in filename
                                and file_allowed
                            ):
                                file_path = os.path.join(directory, filename)
                                linked_files_group[f"file_{h5_id}"] = h5py.ExternalLink(
                                    filename=file_path, path="."
                                )

                                if cache:
                                    with h5py.File(file_path, "r") as trajectory_file:
                                        self._trajectory_cache["linked_files"][
                                            f"file_{h5_id}"
                                        ] = self._recursively_load_h5py(
                                            trajectory_data=trajectory_file
                                        )

                                h5_id += 1

                    break
                except (BlockingIOError, OSError) as e:
                    if try_id == max_tries - 1:
                        print(
                            f"Failed to open or read {output_path} with error {e} after {max_tries} attempts."
                        )
                        raise e

                    print(
                        f"Failing to open or read {output_path} with error {e}. Retrying..."
                    )

        if cache:
            demo_data = CacheContainer(self._trajectory_cache)
        else:
            demo_data = h5py.File(output_path, "r")

        return demo_data

    def randomly_select_traj_from_dir(self, directory, only_key_frames, cache):
        max_tries = 25

        for try_id in range(max_tries):
            try:
                with tempfile.NamedTemporaryFile(
                    dir=directory
                ) as temp_aggregation_file:
                    with self.load_all_h5_from_dir(
                        directory,
                        only_key_frames,
                        temp_aggregation_file,
                        cache=cache,
                    ) as h5_file:
                        aggregated_h5 = h5_file["linked_files"]
                        traj_counts = [
                            (file_key, len(item.keys()))
                            if item is not None
                            else (None, 0)
                            for file_key, item in aggregated_h5.items()
                        ]
                        total_trajs = np.array(
                            [count[1] for count in traj_counts]
                        ).sum()
                        traj_id = np.random.randint(total_trajs)
                        curr_id = 0
                        traj = None

                        for file_id, count in traj_counts:
                            # If this file contains our selected trajectory (i.e. it's within the count for this file), grab it
                            if curr_id + count > traj_id:
                                traj = aggregated_h5[file_id][f"{traj_id - curr_id}"]
                                break

                            curr_id += count

                break
            except (BlockingIOError, OSError, KeyError) as e:
                time.sleep(0.1)

                if try_id == max_tries - 1:
                    print(f"Ran out of re-attempts to load the h5.")
                    raise e

        return traj

    def gripper_fk(self, model, pose):
        ee_pos, ee_rot = model.manip_fk(pose)
        return ee_pos, ee_rot

    def gripper_ik(self, model, pos, rot, current_joints):
        ros_pose, ik_success, ik_debug_info = model.manip_ik(
            (pos, rot), q0=current_joints
        )

        # Prevent full rotations of the wrist...
        raw_roll = ros_pose[HelloStretchIdx.WRIST_ROLL]
        positive_roll = raw_roll % (2 * np.pi)
        negative_roll = positive_roll - (2 * np.pi)
        current_roll = current_joints[HelloStretchIdx.WRIST_ROLL]

        if np.abs(current_roll - positive_roll) < np.abs(current_roll - negative_roll):
            ros_pose[HelloStretchIdx.WRIST_ROLL] = positive_roll
        else:
            ros_pose[HelloStretchIdx.WRIST_ROLL] = negative_roll

        return ros_pose

    def construct_observation(
        self,
        image,
        depth,
        pose,
        color_camera_info,
        depth_camera_info,  # Intentionally unused, to make it clearer how to pass this information if desired
        camera_pose,
        camera_info_in_state,
        model,
        context_observation=None,
    ):
        """
        Expects {color, depth}_camera_info to be a dict of D (5,), K (3x3), R (3x3), P (3x4)
        Expects camera_pose to be shape (1, 4, 4)
        """
        # The ee pose from fk doesn't include the offset to the gripper properly -- include it
        ee_pos, ee_rot = self.gripper_fk(model, pose)

        gripper_pose = pose[HelloStretchIdx.GRIPPER]
        adjusted_pose = np.concatenate(
            (ee_pos, ee_rot, np.array([gripper_pose, 0])), axis=-1  # TODO: remove the 0
        )

        # Just passing color camera info along for now, assuming depth is consistent
        if camera_info_in_state:
            color_camera_state = np.concatenate(
                (
                    color_camera_info["D"],
                    color_camera_info["K"].flatten(),
                    color_camera_info["R"].flatten(),
                    color_camera_info["P"].flatten(),
                )
            )
            camera_pose = camera_pose.flatten()
            adjusted_pose = np.concatenate(
                (adjusted_pose, color_camera_state, camera_pose)
            )

        depth = np.clip(depth, a_min=None, a_max=CAMERA_FAR_PLANE * CAMERA_SCALE_CONST)
        depth_scale = 2**8 / (
            CAMERA_SCALE_CONST * CAMERA_FAR_PLANE
        )  # Rescale to get the most we can out of our bits
        depth = depth * depth_scale

        # Convert from uint16 to uint8 (2**8/2**16 = 1/2**8)
        combined_image = np.concatenate(
            (image, np.expand_dims(depth, axis=-1)), axis=-1
        ).astype(np.uint8)

        if context_observation is not None:
            combined_image = np.concatenate(
                (context_observation["image"], combined_image), axis=-1
            )
            adjusted_pose = np.concatenate(
                (context_observation["state_vector"], adjusted_pose), axis=-1
            )

        return {"image": combined_image, "state_vector": adjusted_pose}

    def get_stretch_obs_and_action_space(self, camera_info_in_state):
        channels = 4
        state_size = 9  # TODO: should be 8...keeping for now so I can keep using my trained model
        state_size += 51 if camera_info_in_state else 0
        action_size = 9  # 3 pos, 4 quat, 1 gripper, 1 completion fraction

        if self._include_context:
            channels *= 2
            state_size *= 2

        # Note, low and high for state_vector and action_space likely to be inaccurate...not currently used by USIP
        observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(640, 480, channels), dtype=np.uint8
                ),
                "state_vector": gym.spaces.Box(low=-1, high=1, shape=(state_size,)),
            }
        )
        action_space = gym.spaces.Box(low=-math.pi, high=math.pi, shape=(action_size,))
        return observation_space, action_space

    def construct_camera_data_from_robot(self, robot):
        color_camera_info = {
            "D": robot.rgb_cam.D,
            "K": robot.rgb_cam.K,
            "R": robot.rgb_cam.R,
            "P": robot.rgb_cam.P,
        }
        depth_camera_info = {
            "D": robot.dpt_cam.D,
            "K": robot.dpt_cam.K,
            "R": robot.dpt_cam.R,
            "P": robot.dpt_cam.P,
        }
        camera_pose = robot.camera_pose
        return color_camera_info, depth_camera_info, camera_pose

    def get_images_from_robot(self, robot):
        rgb = robot.rgb_cam.get()
        depth = robot.dpt_cam.get()
        robot.dpt_cam.far_val = CAMERA_FAR_PLANE
        robot.dpt_cam.near_val = 0.0
        depth = robot.dpt_cam.fix_depth(depth)
        depth = (depth * CAMERA_SCALE_CONST).astype(
            np.uint16
        )  # CAMERA_SCALE_CONST consistent with recorder
        return rgb, depth

    def get_numpy_image(self, h5_image):
        image = img_from_bytes(np.array(h5_image))
        return image
