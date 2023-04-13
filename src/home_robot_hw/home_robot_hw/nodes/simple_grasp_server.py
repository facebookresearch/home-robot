import argparse
import glob
import os
import sys
import time
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R

from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.utils.point_cloud import numpy_to_pcd, pcd_to_numpy, show_point_cloud
from home_robot_hw.ros.grasp_helper import GraspServer

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


def _compute_grasp_scores(
    xy: Iterable[float], occ_map: np.ndarray
) -> Tuple[float, float]:
    """Computes grasp scores given an occupancy map and a center point

    Computes two grasp scores, for grasps along the X and Y directions
    Scores are determined by two factors:
     - How populated the center region is
     - How empty the surrounding regions are
    """
    outer_area = 2 * (GRASP_OUTER_RAD - GRASP_INNER_RAD) ** 2

    xmax = occ_map.shape[0]
    ymax = occ_map.shape[1]
    x_outer_lo = max(xy[0] - GRASP_OUTER_RAD, 0)
    x_inner_lo = max(xy[0] - GRASP_INNER_RAD, 0)
    x_inner_hi = min(xy[0] + GRASP_INNER_RAD + 1, xmax)
    x_outer_hi = min(xy[0] + GRASP_OUTER_RAD + 1, xmax)
    y_outer_lo = max(xy[1] - GRASP_OUTER_RAD, 0)
    y_inner_lo = max(xy[1] - GRASP_INNER_RAD, 0)
    y_inner_hi = min(xy[1] + GRASP_INNER_RAD + 1, ymax)
    y_outer_hi = min(xy[1] + GRASP_OUTER_RAD + 1, ymax)

    area_mid = np.sum(occ_map[x_inner_lo:x_inner_hi, y_inner_lo:y_inner_hi])
    area_x_lo = np.sum(occ_map[x_outer_lo:x_inner_lo, y_inner_lo:y_inner_hi])
    area_x_hi = np.sum(occ_map[x_inner_hi:x_outer_hi, y_inner_lo:y_inner_hi])
    area_y_lo = np.sum(occ_map[x_inner_lo:x_inner_hi, y_outer_lo:y_inner_lo])
    area_y_hi = np.sum(occ_map[x_inner_lo:x_inner_hi, y_inner_hi:y_outer_hi])

    mid_pop_score = max(1.0, area_mid / MIN_INNER_POINTS)
    x_grasp_score = mid_pop_score - (area_x_lo + area_x_hi) / outer_area
    y_grasp_score = mid_pop_score - (area_y_lo + area_y_hi) / outer_area

    return x_grasp_score, y_grasp_score


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
    Predict 6-DoF grasp distribution for given point cloud with a heuristic

    Heuristic is as follows:
        1. Voxelize the point cloud
        2. Extract the top 10% voxels with highest Z coordinates in world frame
        3. Project the said voxels into a 2D occupancy map
        4. Compute grasp scores for each voxel based on neighboring occupancies
        5. Generate top-down grasps if score > threshold
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
