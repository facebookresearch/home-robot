import csv
import os
from datetime import datetime

import cv2
import imageio
import numpy as np


def extract_map_from_vis(image):
    return image[50:530, 670:1150]


def clean_map(image):
    """
    Agent path: (244, 91, 66)
    Free space: (242, 242, 242)
    Unexplored space: (255, 255, 255)
    Goal: (30, 117, 178)
    Obstacle space: (153, 153, 153)
    """
    agent_path_mask = np.all(image == (244, 91, 66), axis=2)
    image[agent_path_mask, :] = (242, 242, 242)
    goal_mask = np.all(image == (30, 117, 178), axis=2)
    image[goal_mask, :] = (255, 255, 255)
    return image


def extract_clean_map(path):
    vis_image = imageio.imread(path)
    map_image = extract_map_from_vis(vis_image)
    map_image = clean_map(map_image)
    return map_image


def extract_binary_map(clean_image):
    """
    Free space: (242, 242, 242)
    Unexplored space: (255, 255, 255)
    Obstacle space: (153, 153, 153)
    """
    free_space_mask = np.all(clean_image == (242, 242, 242), axis=2)
    obstacle_space_mask = np.all(clean_image == (153, 153, 153), axis=2)
    explored_space_mask = free_space_mask | obstacle_space_mask
    clean_image_pixels = set()
    for p in clean_image.reshape(-1, 3):
        clean_image_pixels.add(tuple(p.tolist()))
    print(clean_image_pixels)
    print("# obstacle cells: ", np.count_nonzero(obstacle_space_mask))
    print("# explored cells: ", np.count_nonzero(explored_space_mask))
    return np.stack([obstacle_space_mask, explored_space_mask], axis=0).astype(
        np.float32
    )


def computeHomography(pairs):
    """Solves for the homography given any number of pairs of points. Visit
    http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf
    slide 9 for more details.

    Args:
        pairs (List[List[List]]): List of pairs of (x, y) points.

    Returns:
        np.ndarray: The computed homography.
    """
    A = []
    for x1, y1, x2, y2 in pairs:
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A)

    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(A)

    # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
    # of (A^T)A with the smalles eigenvalue. Reshape into 3x3 matrix.
    H = np.reshape(V[-1], (3, 3))

    # Normalization
    H = (1 / H.item(8)) * H
    return H


def dist(pair, H):
    """Returns the geometric distance between a pair of points given the
    homography H.

    Args:
        pair (List[List]): List of two (x, y) points.
        H (np.ndarray): The homography.

    Returns:
        float: The geometric distance.
    """
    # points in homogeneous coordinates
    p1 = np.array([pair[0], pair[1], 1])
    p2 = np.array([pair[2], pair[3], 1])

    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    return np.linalg.norm(np.transpose(p2) - p2_estimate)


def transform(x, y, H):
    p1 = np.array([x, y, 1])
    p2 = np.dot(H, np.transpose(p1))
    p2 = (1 / p2[2]) * p2
    x2, y2 = int(p2[0].item()), int(p2[1].item())
    return x2, y2


def RANSAC(point_map, threshold=0.6, num_iters=1000, match_distance=2, verbose=True):
    """Runs the RANSAC algorithm.

    Args:
        point_map (List[List[List]]): Map of (x, y) points from one image to the
            another image.
        threshold (float, optional): The minimum portion of points that should
            be inliers before the algorithm terminates. Defaults to THRESHOLD.
        verbose (bool, optional): True if additional information should be
            printed. Defaults to True.

    Returns:
        (np.ndarray, set(List[List])): The homography and set of inliers.
    """
    if verbose:
        print(f"Running RANSAC with {len(point_map)} points...")
    bestInliers = set()
    homography = None
    for i in range(num_iters):
        # randomly choose 4 points from the matrix to compute the homography
        pairs = [point_map[i] for i in np.random.choice(len(point_map), 4)]

        H = computeHomography(pairs)
        inliers = {
            (c[0], c[1], c[2], c[3]) for c in point_map if dist(c, H) < match_distance
        }

        if verbose:
            print(
                f"\x1b[2K\r└──> iteration {i + 1}/{num_iters} "
                + f"\t{len(inliers)} inlier"
                + ("s " if len(inliers) != 1 else " ")
                + f"\tbest: {len(bestInliers)}",
                end="",
            )

        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            homography = H
            if len(bestInliers) > (len(point_map) * threshold):
                break

    if verbose:
        print(f"\nNum matches: {len(point_map)}")
        print(f"Num inliers: {len(bestInliers)}")
        print(f"Min inliers: {len(point_map) * threshold}")

    return homography, bestInliers


EXPLORED_COLOR = (220, 183, 226)
GT_OBSTACLE_COLOR = (204, 204, 204)
CORRECT_OBSTACLE_COLOR = (51, 102, 0)
FALSE_OBSTACLE_COLOR = (102, 204, 0)
TRAJECTORY_COLOR = (0, 0, 0)


def generate_topdown_allocentric_map(
    global_map,
    pred_coverage_map,
    thresh_explored=0.6,
    thresh_obstacle=0.6,
    zoom=False,
):
    """
    Inputs:
        global_map        - (2, H, W) numpy array
        pred_coverage_map - (2, H, W) numpy array
        agent_positions   - (T, 3) numpy array --- (x, y, theta) map pose
    """
    H, W = global_map.shape[1:]
    colored_map = np.ones((H, W, 3), np.uint8) * 255
    global_obstacle_map = (global_map[0] == 1) & (global_map[1] == 1)

    # First show explored regions
    explored_map = pred_coverage_map[1] >= thresh_explored
    colored_map[explored_map, :] = np.array(EXPLORED_COLOR)

    # Show GT obstacles in explored regions
    gt_obstacles_in_explored_map = global_obstacle_map & explored_map
    colored_map[gt_obstacles_in_explored_map, :] = np.array(GT_OBSTACLE_COLOR)

    # Show correctly predicted obstacles in dark green
    pred_obstacles = (pred_coverage_map[0] >= thresh_obstacle) & explored_map
    correct_pred_obstacles = pred_obstacles & gt_obstacles_in_explored_map
    colored_map[correct_pred_obstacles, :] = np.array(CORRECT_OBSTACLE_COLOR)

    # Show in-correctly predicted obstacles in light green
    false_pred_obstacles = pred_obstacles & ~gt_obstacles_in_explored_map
    colored_map[false_pred_obstacles, :] = np.array(FALSE_OBSTACLE_COLOR)

    if zoom:
        # Add an initial padding to ensure a non-zero boundary.
        global_occ_map = np.pad(global_map[0], 5, mode="constant", constant_values=1.0)
        # Zoom into the map based on extents in global_map
        global_map_ysum = (1 - global_occ_map).sum(axis=0)  # (W, )
        global_map_xsum = (1 - global_occ_map).sum(axis=1)  # (H, )
        x_start = W
        x_end = 0
        y_start = H
        y_end = 0
        for i in range(W - 1):
            if global_map_ysum[i] == 0 and global_map_ysum[i + 1] > 0:
                x_start = min(x_start, i)
            if global_map_ysum[i] > 0 and global_map_ysum[i + 1] == 0:
                x_end = max(x_end, i)

        for i in range(H - 1):
            if global_map_xsum[i] == 0 and global_map_xsum[i + 1] > 0:
                y_start = min(y_start, i)
            if global_map_xsum[i] > 0 and global_map_xsum[i + 1] == 0:
                y_end = max(y_end, i)

        # Remove the initial padding
        x_start = max(x_start - 5, 0)
        y_start = max(y_start - 5, 0)
        x_end = max(x_end - 5, 0)
        y_end = max(y_end - 5, 0)

        # Some padding
        x_start = max(x_start - 5, 0)
        x_end = min(x_end + 5, W - 1)
        x_width = x_end - x_start + 1
        y_start = max(y_start - 5, 0)
        y_end = min(y_end + 5, H - 1)
        y_width = y_end - y_start + 1
        max_width = max(x_width, y_width)

        colored_map = colored_map[
            y_start : (y_start + max_width), x_start : (x_start + max_width)
        ]

    return colored_map
