# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import rospy
import tensorflow.compat.v1 as tf
import trimesh.transformations as tra

import home_robot
import home_robot_hw.ros
from home_robot_hw.ros.grasp_helper import GraspServer

tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.abspath(
    os.path.join(home_robot.__path__[0], "../../third_party/contact_graspnet")
)
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
import config_utils  # noqa: E402
from contact_grasp_estimator import GraspEstimator  # noqa: E402
from visualization_utils import show_image, visualize_grasps  # noqa: E402

from data import (  # noqa: E402
    depth2pc,
    load_available_input_data,
    regularize_pc_point_count,
)

GRIPPER_LENGTH = 0.1


def setup_gpu():
    """Configure GPUs for use by contact graspnet"""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(1024 * 3))],
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def inference(
    global_config,
    checkpoint_dir,
    K=None,
    local_regions=True,
    skip_border_objects=False,
    filter_grasps=True,
    segmap_id=None,
    z_range=[0.2, 1.8],
    forward_passes=1,
):
    """
    Predict 6-DoF grasp distribution for given model and input data

    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments.
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """

    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode="test")

    gripper2finger = np.eye(4)
    gripper2finger[2, 3] = GRIPPER_LENGTH
    in_base_frame = False  # produced grasps are in camera frame

    def get_grasps(pc_full, pc_colors, segmap, camera_pose):
        pc_segmap = segmap.reshape(-1)
        pc_segments = {}
        for i in np.unique(pc_segmap):
            if i == 0:
                continue
            else:
                pc_segments[i] = pc_full[pc_segmap == i]

        pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
            sess,
            pc_full,
            pc_segments=pc_segments,
            local_regions=local_regions,
            filter_grasps=filter_grasps,
            forward_passes=forward_passes,
        )
        # show_image(rgb, segmap)
        # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        # apply the correction here
        grasps = {}
        for k, v in pred_grasps_cam.items():
            fixed = np.zeros_like(v)
            # print(k, v.shape)
            fix = tra.euler_matrix(0, 0, -np.pi / 2)
            for i in range(v.shape[0]):
                fixed[i] = fix @ v[i]
                # print(i, v[i,:3,3], fixed[i,:3, 3])
                # pt = fix.T @ v[i, :, 3]
                # pt = fixed[i, :3, 3]
                # print(i, scores[k][i], pt)
                # fixed[i, :3, 3] = pt[:3]

            # "Grasps" should be pinch point poses
            grasps[k] = fixed @ gripper2finger

        return grasps, scores, in_base_frame

    _ = GraspServer(get_grasps)
    rospy.spin()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
        continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        default="checkpoints/scene_test_2048_bs3_hor_sigma_001",
        help="Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]",
    )
    parser.add_argument(
        "--K",
        default=None,
        help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"',
    )
    parser.add_argument(
        "--z_range",
        default=[0.2, 1.8],
        help="Z value threshold to crop the input point cloud",
    )
    parser.add_argument(
        "--local_regions",
        action="store_true",
        default=False,
        help="Crop 3D local regions around given segments.",
    )
    parser.add_argument(
        "--filter_grasps",
        action="store_true",
        default=False,
        help="Filter grasp contacts according to segmap.",
    )
    parser.add_argument(
        "--skip_border_objects",
        action="store_true",
        default=False,
        help="When extracting local_regions, ignore segments at depth map boundary.",
    )
    parser.add_argument(
        "--forward_passes",
        type=int,
        default=1,
        help="Run multiple parallel forward passes to mesh_utils more potential contact points.",
    )
    parser.add_argument(
        "--segmap_id",
        type=int,
        default=0,
        help="Only return grasps of the given object id",
    )
    parser.add_argument(
        "--arg_configs",
        nargs="*",
        type=str,
        default=[],
        help="overwrite config parameters",
    )
    FLAGS = parser.parse_args()
    rospy.init_node("contact_graspnet")
    setup_gpu()
    global_config = config_utils.load_config(
        FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs
    )

    print(str(global_config))
    print("pid: %s" % (str(os.getpid())))

    inference(
        global_config,
        FLAGS.ckpt_dir,
        z_range=eval(str(FLAGS.z_range)),
        K=FLAGS.K,
        local_regions=FLAGS.local_regions,
        filter_grasps=FLAGS.filter_grasps,
        segmap_id=FLAGS.segmap_id,
        forward_passes=FLAGS.forward_passes,
        skip_border_objects=FLAGS.skip_border_objects,
    )
