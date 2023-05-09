from typing import Iterable, List, Optional, Tuple

import click
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R

from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.utils.point_cloud import show_point_cloud
from home_robot_hw.ros.grasp_helper import GraspServer

VERTICAL_GRIPPER_QUAT = [
    0.70988,
    0.70406461,
    0.0141615,
    0.01276155,
]  # base frame, gripper closing along x axis

# Grasp generation params
TOP_PERCENTAGE = 0.1
VOXEL_RES = 0.01
GRASP_OUTER_RAD = 5
GRASP_INNER_RAD = 1
GRASP_THRESHOLD = 0.9
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


def _filter_grasps(score_map: np.ndarray, grasp_direction: int) -> np.ndarray:
    """Filter an grid of grasps scores"""
    # Filter by neighboring grasps (multiply score of current score map with shifted versions of the map)
    mask1 = np.zeros_like(score_map)
    mask2 = np.zeros_like(score_map)

    if grasp_direction == 0:
        # Gripper closes along X: filter across Y
        mask1[:, :-1] = score_map[:, 1:]
        mask2[:, 1:] = score_map[:, :-1]
    elif grasp_direction == 1:
        # Gripper closes along Y: filter across X
        mask1[:-1, :] = score_map[1:, :]
        mask2[1:, :] = score_map[:-1, :]

    score_map = score_map * mask1 * mask2

    # Filter by thresholding
    thres_mask = score_map >= GRASP_THRESHOLD
    score_map = score_map * thres_mask

    return score_map


def _generate_grasp(xyz: np.ndarray, rz: float) -> np.ndarray:
    """Generate a vertical grasp pose given grasp location and z orientation"""
    grasp = np.zeros([4, 4])
    grasp[:3, :3] = (
        R.from_quat(np.array(VERTICAL_GRIPPER_QUAT)) * R.from_rotvec([0, 0, rz])
    ).as_matrix()
    grasp[:3, 3] = xyz

    return grasp


class VoxelGraspGenerator(object):
    """Create grasps based on simple voxel rules."""

    def __init__(self, in_base_frame=True, debug=False):
        self.in_base_frame = in_base_frame
        self.debug = debug

    def get_grasps(
        self,
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
        x_score_map = np.zeros_like(occ_map)
        y_score_map = np.zeros_like(occ_map)

        for i in range(occ_map.shape[0]):
            for j in range(occ_map.shape[1]):
                if occ_map[i, j] == 0.0:
                    continue

                x_score, y_score = _compute_grasp_scores((i, j), occ_map)
                x_score_map[i, j] = x_score
                y_score_map[i, j] = y_score

        # Filter grasps
        x_score_map_filtered = _filter_grasps(x_score_map, grasp_direction=0)
        y_score_map_filtered = _filter_grasps(y_score_map, grasp_direction=1)

        scores_raw = []
        grasps_raw = []

        # Extract grasps
        for i in range(occ_map.shape[0]):
            for j in range(occ_map.shape[1]):
                x, y = np.array((i, j)) * VOXEL_RES + orig
                grasp_pos = np.array([x, y, z_grasp])

                x_score = x_score_map_filtered[i, j]
                if x_score > 0.0:
                    grasp = _generate_grasp(grasp_pos, np.pi / 2)
                    grasps_raw.append(grasp)
                    scores_raw.append(x_score)

                y_score = y_score_map_filtered[i, j]
                if y_score > 0.0:
                    grasp = _generate_grasp(grasp_pos, 0.0)
                    grasps_raw.append(grasp)
                    scores_raw.append(y_score)

        # Debug and visualization
        if self.debug:
            print(f"# grasps = {len(grasps_raw)}")
            _visualize_grasps(xyz, rgb, top_idcs, grasps_raw)

        # Postprocess grasps into dictionaries
        # (this grasp generator only generates grasps for one object)
        grasps = {0: np.array(grasps_raw)}
        scores = {0: np.array(scores_raw)}

        return grasps, scores, self.in_base_frame
