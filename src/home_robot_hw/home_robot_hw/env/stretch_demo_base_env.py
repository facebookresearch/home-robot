import datetime
import os
import tempfile
import time

import h5py
import numpy as np
from home_robot_hw.env.stretch_abstract_env import StretchEnv
import rospy
from home_robot.utils.data_tools.image import img_from_bytes
from home_robot.agent.motion.stretch import HelloStretch, HelloStretchIdx

CAMERA_FAR_PLANE = 1  # m
CAMERA_SCALE_CONST = 10000
STRETCH_URDF_DIR = os.environ["STRETCH_URDF_DIR"]  # TODO: pass in


class StretchDemoBaseEnv(StretchEnv):
    """base environment to work with demonstrations; both for recording and replaying"""
    NODE_INITIALIZED = False

    def __init__(self, initialize_ros):
        super().__init__()
        self.urdf_path = os.path.join(
            STRETCH_URDF_DIR, "planner_calibrated_simplified_fixed_base.urdf"
        )  # TODO: pass in urdf path

        if initialize_ros and not self.NODE_INITIALIZED:
            self.initialize_ros_node()

    @classmethod
    def initialize_ros_node(cls):
        name = rospy.get_name()
        if "unnamed" in name:
            timestamp = (
                datetime.datetime.now().timestamp()
            )  # TODO: fractions make dupes less likely, but also are technically not-ROS-y and might die
            rospy.get_master().getPid()
            rospy.init_node(f"demo_env_{timestamp}".replace(".", "_"))
        cls.NODE_INITIALIZED = True

    def _recursive_listdir(self, directory: str) -> list[str]:
        """returns a list of all files in a directory, recursively"""
        # From: https://stackoverflow.com/questions/19309667/recursive-os-listdir
        return [os.path.join(dp, f) for dp, dn, fn in os.walk(directory) for f in fn]

    # TODO: util functions should go where?
    def load_all_h5_from_dir(
        self, directory, only_key_frames, temp_aggregation_file
    ):  # TODO: where should this go
        """method to recursively load all H5s from given directory"""
        h5_id = 0
        output_path = temp_aggregation_file.name

        for _ in range(5):
            try:
                with h5py.File(output_path, "w") as output_file:
                    linked_files_group = output_file.create_group("linked_files")

                    for filename in self._recursive_listdir(directory):
                        file_allowed = only_key_frames == (
                            "key_frames" in filename
                        )  # If we expect it, it should be there. If we don't, it should not
                        if (
                            os.path.splitext(filename)[-1].lower() == ".h5"
                            and "aggregated" not in filename
                            and file_allowed
                        ):
                            file_path = os.path.join(directory, filename)
                            linked_files_group[f"file_{h5_id}"] = h5py.ExternalLink(
                                filename=file_path, path="."
                            )  # TODO: does this need to be a string. Doing it for consistency with DataWriter for now
                            h5_id += 1

                break
            except (BlockingIOError, OSError) as e:
                print(
                    f"Failing to open or read {output_path} with error {e}. Retrying..."
                )
                time.sleep(1)  # TODO: handle last try

        return h5py.File(output_path, "r")

    def randomly_select_traj_from_dir(self, directory, only_key_frames):
        """randomly selects a trajectory from a directory of H5s"""
        for _ in range(25):
            try:
                with tempfile.NamedTemporaryFile(
                    dir=directory
                ) as temp_aggregation_file:
                    with self.load_all_h5_from_dir(
                        directory, only_key_frames, temp_aggregation_file
                    ) as h5_file:
                        aggregated_h5 = h5_file[
                            "linked_files"
                        ]  # TODO: load from the cache that's being created or no?
                        traj_counts = [
                            (file_key, len(item.keys()))
                            if item is not None
                            else (None, 0)
                            for file_key, item in aggregated_h5.items()
                        ]
                        total_trajs = np.array(
                            [count[1] for count in traj_counts]
                        ).sum()
                        # print(f"Running with {total_trajs} total trajectories")
                        traj_id = np.random.randint(total_trajs)
                        # print(f"Running with traj id: {traj_id}")
                        curr_id = 0
                        traj = None

                        for file_id, count in traj_counts:
                            # If this file contains our selected trajectory (i.e. it's within the count for this file), grab it
                            if curr_id + count > traj_id:
                                # print(f"Loading entry: {traj_id - curr_id} from file {aggregated_h5[file_id].file}")
                                traj = aggregated_h5[file_id][f"{traj_id - curr_id}"]
                                break

                            curr_id += count

                break
            except (BlockingIOError, OSError, KeyError):
                time.sleep(1)  # TODO: handle last try

        return traj

    def convert_ros_pose_to_pinocchio(
        self, joint_angles
    ):  # TODO: the responsibility of robot.py, not this class
        # TODO: if I keep this, do it as lookups
        pin_compatible_joints = np.zeros(8)
        """
                "joint_lift",
                "joint_arm_l0",
                "joint_arm_l1",
                "joint_arm_l2",
                "joint_arm_l3",
                "joint_wrist_yaw",
                "joint_wrist_pitch",
                "joint_wrist_roll"
                """
        pin_compatible_joints[0] = joint_angles[HelloStretchIdx.LIFT]
        pin_compatible_joints[1] = pin_compatible_joints[2] = pin_compatible_joints[
            3
        ] = pin_compatible_joints[4] = (joint_angles[HelloStretchIdx.ARM] / 4)
        pin_compatible_joints[5] = joint_angles[HelloStretchIdx.WRIST_YAW]
        pin_compatible_joints[6] = joint_angles[HelloStretchIdx.WRIST_PITCH]
        pin_compatible_joints[7] = joint_angles[HelloStretchIdx.WRIST_ROLL]
        return pin_compatible_joints

    def convert_pinocchio_pose_to_ros(self, pin):
        joint_angles = np.zeros(11)
        joint_angles[HelloStretchIdx.LIFT] = pin[0]
        joint_angles[HelloStretchIdx.ARM] = pin[1] + pin[2] + pin[3] + pin[4]
        joint_angles[HelloStretchIdx.WRIST_YAW] = pin[5]
        joint_angles[HelloStretchIdx.WRIST_PITCH] = pin[6]
        joint_angles[HelloStretchIdx.WRIST_ROLL] = pin[7]
        return joint_angles

    def gripper_fk(self, model: HelloStretch, pose):
        pin_pose = self.convert_ros_pose_to_pinocchio(pose)
        ee_pos, ee_rot = model.fk(pin_pose)
        gripper_pos, gripper_rot = (
            ee_pos.copy(),
            ee_rot.copy(),
        )  # TODO: something is being changed in-place, this prevents that
        return gripper_pos, gripper_rot

    def gripper_ik(self, model, pos, rot, current_joints):
        shifted_pos, shifted_rot = pos, rot
        pose = model.pb_ik_solver_optimize((shifted_pos, shifted_rot))
        ros_pose = self.convert_pinocchio_pose_to_ros(
            pose.copy()
        )  # TODO: copying just to be safe...

        # Prevent full rotations of the wrist...TODO:??
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
        depth_camera_info,
        camera_pose,
        camera_info_in_state,
        current_time,
        max_time,
        model,
        pos_augmentation=None,
    ):  # TODO: better place
        """
        Expects {color, depth}_camera_info to be a dict of D (5,), K (3x3), R (3x3), P (3x4)
        Expects camera_pose to be shape (1, 4, 4) -- TODO check
        """
        time_fraction = 0  # current_time / max_time

        # The ee pose from fk doesn't include the offset to the gripper properly -- include it
        ee_pos, ee_rot = self.gripper_fk(model, pose)

        if pos_augmentation is not None:
            ee_pos += pos_augmentation

        gripper_pose = pose[HelloStretchIdx.GRIPPER]
        adjusted_pose = np.concatenate(
            (ee_pos, ee_rot, np.array([gripper_pose, time_fraction])), axis=-1
        )  # TODO: trying to remove the base pose+theta (TODO: don't hardcode)
        if camera_info_in_state:
            # TODO: depth and color current have the same intrinsics, so I'm just passing color along for now
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

        return {"image": combined_image, "state_vector": adjusted_pose}

    def get_stretch_obs_and_action_space(self, camera_info_in_state):
        channels = 4
        state_size = 9
        state_size += 51 if camera_info_in_state else 0
        action_size = 9  # 3 pos, 4 quat, 1 gripper, 1 done
        observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(640, 480, channels), dtype=np.uint8
                ),
                "state_vector": gym.spaces.Box(low=-1, high=1, shape=(state_size,)),
            }
        )  # TODO: low/high? Esp for x, y (7 > 2pi)
        action_space = gym.spaces.Box(
            low=-50, high=50, shape=(action_size,)
        )  # TODO: low and high - the high is based on brief observation - TODO: not currently used
        return observation_space, action_space

    def construct_camera_data_from_demo(self, trajectory, timestep):
        color_camera_info = {
            k: np.array(v)[timestep] for k, v in trajectory["color_camera_info"].items()
        }
        depth_camera_info = {
            k: np.array(v)[timestep] for k, v in trajectory["depth_camera_info"].items()
        }
        camera_pose = np.array(trajectory["camera_pose"][timestep])
        return color_camera_info, depth_camera_info, camera_pose

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
        camera_pose = robot.get_pose(robot.rgb_cam.get_frame(), timeout_s=10.0)
        return color_camera_info, depth_camera_info, camera_pose

    def get_images_from_robot(self, robot):
        rgb, depth = robot.get_images(
            filter_depth=False, compute_xyz=False
        )  # TODO: filter depth?
        depth = (depth * CAMERA_SCALE_CONST).astype(
            np.uint16
        )  # TODO: consistent with recorder
        return rgb, depth

    def get_numpy_image(self, h5_image):
        image = img_from_bytes(np.array(h5_image))
        return image
