import argparse
import glob
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R

from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.utils.point_cloud import numpy_to_pcd, pcd_to_numpy, show_point_cloud
from home_robot_hw.ros.grasp_helper import GraspServer

# VERTICAL_GRIPPER_QUAT = [0.3794973, -0.15972253, -0.29782842, -0.86125998] # camera frame
VERTICAL_GRIPPER_QUAT = [0.70988, 0.70406461, 0.0141615, 0.01276155]  # base frame

# Grasp generation params
TOP_PERCENTAGE = 0.1
VOXEL_RES = 0.01
GRASP_OUTER_RAD = 5
GRASP_INNER_RAD = 1
GRASP_THRESHOLD = 0.95
MIN_INNER_POINTS = 12


def _visualize_grasps(
    xyz: np.ndarray,
    rgb: np.ndarray,
    idcs: Optional[np.ndarray] = None,
    grasps: Optional[np.ndarray] = None,
) -> None:
    """Visualize grasps
    idcs: highlighed point indices
    grasps: list of poses (4x4 matrices)
    """
    rgb_colored = rgb.copy() / 255.0
    if idcs is not None:
        rgb_colored[idcs, :] = np.array([0.0, 0.0, 1.0])[None, :]
    show_point_cloud(xyz, rgb_colored, orig=np.zeros(3), grasps=grasps)


def _compute_grasp_scores(xy: Tuple, occ_map: np.ndarray) -> Tuple[float, float]:
    """Computes grasp scores given an occupancy map and a center point

    Computes two grasp scores, for grasps along the X and Y directions
    Scores are determined by two factors:
     - How populated the center region is
     - How empty the surrounding regions are
    """
    outer_area = 2 * (GRASP_OUTER_RAD - GRASP_INNER_RAD) ** 2

    xmax = occ_map.shape[0]
    ymax = occ_map.shape[1]

    xy0 = np.sum(
        occ_map[
            max(xy[0] - GRASP_INNER_RAD, 0) : min(xy[0] + GRASP_INNER_RAD + 1, xmax),
            max(xy[1] - GRASP_INNER_RAD, 0) : min(xy[1] + GRASP_INNER_RAD + 1, ymax),
        ]
    )

    x1 = np.sum(
        occ_map[
            max(xy[0] - GRASP_OUTER_RAD, 0) : min(xy[0] - GRASP_INNER_RAD + 1, xmax),
            max(xy[1] - GRASP_INNER_RAD, 0) : min(xy[1] + GRASP_INNER_RAD + 1, ymax),
        ]
    )
    x2 = np.sum(
        occ_map[
            max(xy[0] + GRASP_INNER_RAD, 0) : min(xy[0] + GRASP_OUTER_RAD + 1, xmax),
            max(xy[1] - GRASP_INNER_RAD, 0) : min(xy[1] + GRASP_INNER_RAD + 1, ymax),
        ]
    )

    y1 = np.sum(
        occ_map[
            max(xy[0] - GRASP_INNER_RAD, 0) : min(xy[0] + GRASP_INNER_RAD + 1, xmax),
            max(xy[1] - GRASP_OUTER_RAD, 0) : min(xy[1] - GRASP_INNER_RAD + 1, ymax),
        ]
    )
    y2 = np.sum(
        occ_map[
            max(xy[0] - GRASP_INNER_RAD, 0) : min(xy[0] + GRASP_INNER_RAD + 1, xmax),
            max(xy[1] + GRASP_INNER_RAD, 0) : min(xy[1] + GRASP_OUTER_RAD + 1, ymax),
        ]
    )

    x_score = max(1.0, xy0 / MIN_INNER_POINTS) - (x1 + x2) / outer_area
    y_score = max(1.0, xy0 / MIN_INNER_POINTS) - (y1 + y2) / outer_area

    return x_score, y_score


def _generate_grasp(xyz: np.ndarray, rz: float) -> np.ndarray:
    """Generate a vertical grasp pose given grasp location and z orientation"""
    grasp = np.zeros([4, 4])
    grasp[:3, :3] = (
        R.from_quat(np.array(VERTICAL_GRIPPER_QUAT)) * R.from_rotvec([0, 0, rz])
    ).as_matrix()
    grasp[:3, 3] = xyz

    return grasp


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

    in_base_frame = True  # produced grasps are in base frame

    def get_grasps(
        pc_full: np.ndarray,
        pc_colors: np.ndarray,
        segmap: np.ndarray,
        camera_pose: np.ndarray,
    ):
        """
        pc_full: full point cloud xyz (Nx3 matrix)
        pc_colors: full point cloud colors (Nx3 matrix)
        segmap: segmentation map (1 for object, 0 for background) (array of length N)
        camera_pose: 4x4 matrix
        """
        pc_segmap = segmap.reshape(-1)
        seg_idcs = np.logical_and(pc_segmap == 1, pc_full[:, 2] != 0.0)
        pc_segment = pc_full[seg_idcs]
        pc_color_segment = pc_colors[seg_idcs]

        # Build voxel map (in abs frame)
        voxel_map = SparseVoxelMap(resolution=VOXEL_RES, feature_dim=3)
        voxel_map.add(camera_pose, pc_segment, feats=pc_color_segment)

        # Extract highest points
        xyz, rgb = voxel_map.get_data()
        if xyz.shape[0] < 1:
            return {}, {}

        num_top = int(xyz.shape[0] * TOP_PERCENTAGE)
        top_idcs = np.argpartition(xyz[:, 2], -num_top)[-num_top:]
        xyz_top = xyz[top_idcs, :]

        # Flatten points into 2d occupancy map
        z_grasp = np.median(xyz_top[:, 2])

        orig = np.min(xyz_top[:, :2], axis=0)
        far_corner = np.max(xyz_top[:, :2], axis=0)
        occ_map = np.zeros(((far_corner - orig) / VOXEL_RES).astype(int) + 1)
        for point in xyz_top:
            i, j = ((point[:2] - orig) / VOXEL_RES).astype(int)
            occ_map[i, j] = 1

        # Compute x-direction and y-direction grasps on the occupancy map
        scores_raw = []
        grasps_raw = []
        for i in range(occ_map.shape[0]):
            for j in range(occ_map.shape[1]):
                if occ_map[i, j] == 0.0:
                    continue

                x_score, y_score = _compute_grasp_scores((i, j), occ_map)
                x, y = np.array((i, j)) * VOXEL_RES + orig

                if x_score >= GRASP_THRESHOLD:
                    grasp = _generate_grasp(np.array([x, y, z_grasp]), 0.0)
                    grasps_raw.append(grasp)
                    scores_raw.append(x_score)

                if y_score >= GRASP_THRESHOLD:
                    grasp = _generate_grasp(np.array([x, y, z_grasp]), np.pi / 2)
                    grasps_raw.append(grasp)
                    scores_raw.append(y_score)

        # _visualize_grasps(xyz, rgb, top_idcs, grasps_raw)

        # Postprocess grasps into dictionaries
        # (6dof graspnet only generates grasps for one object)
        grasps = {0: np.array(grasps_raw)}
        scores = {0: np.array(scores_raw)}
        return grasps, scores, in_base_frame

    # Initialize server
    _ = GraspServer(get_grasps)

    # Spin
    rospy.spin()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
        continue


if __name__ == "__main__":
    rospy.init_node("simple_grasp_server")
    inference()
