import rospy
import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2

from home_robot.utils.point_cloud import numpy_to_pcd, pcd_to_numpy, show_point_cloud
from home_robot_hw.ros.grasp_helper import GraspServer
from home_robot.mapping.voxel import SparseVoxelMap
from scipy.spatial.transform import Rotation as R

#VERTICAL_GRIPPER_QUAT = [0.3794973, -0.15972253, -0.29782842, -0.86125998]
VERTICAL_GRIPPER_QUAT = [0.70988   , 0.70406461, 0.0141615 , 0.01276155]


def inference():
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

    def get_grasps(pc_full, pc_colors, segmap, camera_pose):
        pc_segmap = segmap.reshape(-1)
        pc_segment = pc_full[pc_segmap == 1]
        pc_color_segment = pc_colors[pc_segmap == 1]

        # Build voxel map (in abs frame)
        voxel_map = SparseVoxelMap(resolution=0.01, feature_dim=3)
        voxel_map.add(camera_pose, pc_segment, feats=pc_color_segment)

        # Flatten
        xyz, rgb = voxel_map.get_data()

        num_top = int(xyz.shape[0] * 0.1)
        idcs = np.argpartition(xyz[:, 2], -num_top)[-num_top:]
        xyz_top = xyz[idcs, :]

        rgb_colored = rgb.copy() / 255.
        rgb_colored[idcs, :] = np.array([0.0, 0.0, 1.0])[None, :]

        # Enumerate grasps
        scores_raw = [1.0]
        grasps_raw = np.zeros([1, 4, 4])
        grasps_raw[0, :3, :3] = R.from_quat(np.array(VERTICAL_GRIPPER_QUAT)).as_matrix()
        grasps_raw[0, :3, 3] = np.median(xyz_top, axis=0)

        #show_point_cloud(xyz, rgb_colored, orig=np.zeros(3), grasps=grasps_raw)

        # Postprocess grasps into dictionaries
        # (6dof graspnet only generates grasps for one object)
        grasps = {0: grasps_raw}
        scores = {0: scores_raw}
        return grasps, scores

    # Initialize server
    _server = GraspServer(get_grasps)

    # Spin
    rospy.spin()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
        continue


        
if __name__ == "__main__":
    rospy.init_node('simple_grasp_server')
    inference()
