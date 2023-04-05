import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import rospy
import trimesh.transformations as tra

import home_robot
import home_robot_hw.ros
from home_robot_hw.ros.grasp_helper import GraspServer

GRASPNET_ROOT = os.path.abspath(
    os.path.join(home_robot.__path__[0], "../../third_party/pytorch_6dof-graspnet")
)
os.chdir(GRASPNET_ROOT)
sys.path.append(GRASPNET_ROOT)
from grasp_estimator import GraspEstimator  # noqa: E402
from utils import utils  # noqa: E402

# from utils.visualization_utils import visualize_grasps, show_image


def inference(args):
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

    # Initialize graspnet object
    grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False
    grasp_evaluator_args = utils.read_checkpoint_args(args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = False
    estimator = GraspEstimator(grasp_sampler_args, grasp_evaluator_args, args)

    # Create request function and initialze grasp server
    def get_grasps(pc_full, pc_colors, segmap):
        pc_segmap = segmap.reshape(-1)
        pc_segment = pc_full[pc_segmap == 1]
        grasps, scores = estimator.generate_and_refine_grasps(pc_segment)

        return grasps, scores

    server = GraspServer(get_grasps)

    # Spin
    rospy.spin()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
        continue

    del server


def make_parser():
    parser = argparse.ArgumentParser(
        description="6-DoF GraspNet Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--grasp_sampler_folder", type=str, default="checkpoints/gan_pretrained/"
    )
    parser.add_argument(
        "--grasp_evaluator_folder",
        type=str,
        default="checkpoints/evaluator_pretrained/",
    )
    parser.add_argument(
        "--refinement_method", choices={"gradient", "sampling"}, default="sampling"
    )
    parser.add_argument("--refine_steps", type=int, default=25)

    parser.add_argument("--npy_folder", type=str, default="demo/data/")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed",
    )
    parser.add_argument(
        "--choose_fn",
        choices={"all", "better_than_threshold", "better_than_threshold_in_sequence"},
        default="better_than_threshold",
        help="If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps",
    )

    parser.add_argument("--target_pc_size", type=int, default=1024)
    parser.add_argument("--num_grasp_samples", type=int, default=200)
    parser.add_argument(
        "--generate_dense_grasps",
        action="store_true",
        help="If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=30,
        help="Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory.",
    )
    parser.add_argument("--train_data", action="store_true")
    opts, _ = parser.parse_known_args()
    if opts.train_data:
        parser.add_argument(
            "--dataset_root_folder",
            required=True,
            type=str,
            help="path to root directory of the dataset.",
        )
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    rospy.init_node("6dof_graspnet")

    inference(args)
