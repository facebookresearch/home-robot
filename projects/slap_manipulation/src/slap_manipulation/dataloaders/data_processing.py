from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh.transformations as tra
from slap_manipulation.utils.data_visualizers import (
    show_point_cloud_with_keypt_and_closest_pt,
)

from home_robot.core.interfaces import Observations
from home_robot.utils.image import rotate_image
from home_robot.utils.point_cloud import numpy_to_pcd, show_point_cloud


def unrotate_image(images: List[np.ndarray]) -> List[np.ndarray]:
    new_images = [np.fliplr(np.flipud(np.rot90(x, 3))) for x in images]
    return new_images


def crop_around_voxel(
    xyz: np.ndarray,
    rgb: np.ndarray,
    feat: np.ndarray,
    crop_location: np.ndarray,
    crop_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a point cloud around given voxel"""
    mask = np.linalg.norm(xyz - crop_location, axis=1) < crop_size
    return xyz[mask, :], rgb[mask, :], feat[mask, :]


def get_local_action_prediction_problem(
    cfg,
    feat: np.ndarray,
    xyz: np.ndarray,
    p_i: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Returns a cropped version of the input point-cloud mean-centered around the predicted
    interaction point (p_i)
    """
    # crop from og pcd and mean-center it
    crop_xyz, crop_rgb = crop_around_voxel(feat, xyz, p_i, cfg.local_problem_size)
    crop_xyz = crop_xyz - p_i[None].repeat(crop_xyz.shape[0], axis=0)
    # show_point_cloud(crop_xyz, crop_rgb, orig=np.zeros(3))
    if crop_rgb.shape[0] > cfg.num_pts:
        # Downsample pt clouds
        downsample = np.arange(crop_rgb.shape[0])
        np.random.shuffle(downsample)
        if cfg.num_pts != -1:
            downsample = downsample[: cfg.num_pts]
        crop_rgb = crop_rgb[downsample]
        crop_xyz = crop_xyz[downsample]
    status = True
    if crop_xyz.shape[0] < 10:
        status = False
    return crop_rgb, crop_xyz, status


def drop_frames_from_input(xyzs, rgbs, imgs, probability_dropout=0.33):
    probability_keep = 1.0 - probability_dropout
    idx_dropout = np.random.choice(
        [False, True], size=len(rgbs), p=[probability_dropout, probability_keep]
    )
    rgbs = [rgbs[i] for i in idx_dropout]
    xyzs = [xyzs[i] for i in idx_dropout]
    imgs = [imgs[i] for i in idx_dropout]
    return xyzs, rgbs, imgs


def average_feats(feats, downsampled_index_trace):
    # downsampled_index_trace is a list of list of index
    # average feats over each list of index
    averaged_feats = []
    for idx in downsampled_index_trace:
        average_feats.append(np.mean(feats[idx, :], axis=0))
    averaged_feats = np.stack(averaged_feats, axis=0)
    return averaged_feats


def remove_duplicate_points(xyz, rgb, feats, depth, voxel_size=0.001):
    # heuristic based trimming
    mask = np.bitwise_and(depth < 1.5, depth > 0.3)
    rgb = rgb[mask]
    xyz = xyz[mask]
    debug_views = False
    if debug_views:
        print("xyz", xyz.shape)
        print("rgb", rgb.shape)
        show_point_cloud(xyz, rgb)

    xyz, feats = xyz.reshape(-1, 3), feats.reshape(-1, 3)
    # voxelize at a granular voxel-size rather than random downsample
    pcd = numpy_to_pcd(xyz, rgb)
    pcd_downsampled, downsampled_index_trace = pcd.voxel_down_sample_and_trace(
        voxel_size
    )
    rgb = np.asarray(pcd_downsampled.colors)
    xyz = np.asarray(pcd_downsampled.points)
    feats = average_feats(feats, downsampled_index_trace)

    debug_voxelization = False
    if debug_voxelization:
        # print(f"Number of points in this PCD: {len(pcd_downsampled2.points)}")
        show_point_cloud(xyz, rgb)

    return xyz, rgb, feats


def dr_crop_radius_around_interaction_point(
    xyz,
    rgb,
    feats,
    ref_ee_keyframe,
    crop_radius_chance=0.75,
    crop_radius_shift=0.05,
    crop_radius_range=[1.0, 2.0],
):
    """crop input point-cloud around the interaction point"""
    # crop out random points outside a certain distance from the gripper
    # this is to encourage it to learn only local features and skills
    cr_range = crop_radius_range[1] - crop_radius_range[0]
    cr_min = crop_radius_range[0]
    if np.random.random() < crop_radius_chance:
        # Now we do the cropping
        orig = ref_ee_keyframe[:3, 3][None].copy()
        crop_shift = ((np.random.random(3) * 2) - 1) * crop_radius_shift
        orig += crop_shift
        # Now here we apply some other stuff
        orig = np.repeat(orig, xyz.shape[0], axis=0)
        crop_dist = np.linalg.norm(xyz - orig, axis=-1)
        radius = (np.random.random() * cr_range) + cr_min
        crop_idx = crop_dist < radius
        rgb = rgb[crop_idx, :]
        feats = feats[crop_idx, :]
        xyz = xyz[crop_idx, :]
    return xyz, rgb, feats


def shuffle_and_downsample_point_cloud(xyz, rgb, feats, num_points=8000):
    """shuffle the xyz points to get a different input order, and downsample to num_points"""
    # Downsample pt clouds
    downsample = np.arange(rgb.shape[0])
    np.random.shuffle(downsample)
    if num_points != -1:
        downsample = downsample[:num_points]
    rgb = rgb[downsample]
    xyz = xyz[downsample]
    feats = feats[downsample]

    # mean center xyz
    center = np.mean(xyz, axis=0)
    # center = np.zeros(3)
    center[-1] = 0
    xyz = xyz - center[None].repeat(xyz.shape[0], axis=0)
    return xyz, rgb, feats, center


def adjust_keyframes_wrt_pcd_mean(ee_keyframes, interaction_ee_keyframe, pcd_center):
    """deduct pcd_center from all input keyframes"""
    N = ee_keyframes.shape[0]
    ee_keyframes = ee_keyframes - pcd_center.reshape(N, 3)
    interaction_ee_keyframe = interaction_ee_keyframe - pcd_center.reshape(1, 3)
    return ee_keyframes, interaction_ee_keyframe


def dr_rotation_translation(
    orig_xyz: np.ndarray,
    xyz: np.ndarray,
    ee_keyframe: np.ndarray,
    ref_ee_keyframe: np.ndarray,
    keyframes: List[np.ndarray],
    ori_dr_range: float = np.pi / 4,
    cart_dr_range: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """translate and rotate"""
    # Now that it is mean centered, apply data augmentation
    # Start with rotation
    rotation_matrix = tra.euler_matrix(
        0, 0, ori_dr_range * ((np.random.rand() * 2) - 1)
    )
    orig_xyz = tra.transform_points(orig_xyz, rotation_matrix)
    # note: above transforms points wrt translation and rotation provided
    # the second argument is a homogeneous matrix
    xyz = tra.transform_points(xyz, rotation_matrix)
    ee_keyframe = rotation_matrix @ ee_keyframe
    ref_ee_keyframe = rotation_matrix @ ref_ee_keyframe

    # Now add a random shift
    shift = ((np.random.rand(3) * 2) - 1) * cart_dr_range

    # Add shift to everything... yeah this could be written better
    ref_ee_keyframe[:3, 3] += shift
    xyz += shift[None].repeat(xyz.shape[0], axis=0)
    orig_xyz += shift[None].repeat(orig_xyz.shape[0], axis=0)
    ee_keyframe[:3, 3] += shift

    # Cropped trajectories
    # Loop over keyframes
    new_keyframes = []
    for keyframe in keyframes:
        keyframe = keyframe.copy()
        keyframe = rotation_matrix @ keyframe
        keyframe[:3, 3] += shift
        new_keyframes.append(keyframe)
    return (orig_xyz, xyz, ee_keyframe, ref_ee_keyframe, new_keyframes)


def voxelize_and_get_interaction_point(
    xyz, rgb, feats, interaction_ee_keyframe, voxel_size=0.01, debug=False
):
    """uniformly voxelizes the input point-cloud and returns the closest-point
    in the point-cloud to the task's interaction ee-keyframe"""
    # downsample another time to get sampled version
    input_pcd = numpy_to_pcd(xyz, rgb)
    downsampled_pcd, downsampled_index_trace = input_pcd.voxel_down_sample_and_trace(
        voxel_size
    )
    downsampled_xyz = np.asarray(downsampled_pcd.points)
    downsampled_rgb = np.asarray(downsampled_pcd.colors)
    downsampled_feats = average_feats(feats, downsampled_index_trace)

    # for the voxelized pcd
    if downsampled_xyz.shape[0] < 10:
        return (None, None, None, None, None, None)
    downsampled_pcd_tree = o3d.geometry.KDTreeFlann(downsampled_pcd)
    # Find closest points based on ref_ee_keyframe
    # This is used to supervise the location when we're detecting where the action
    # could have happened
    [_, target_idx_1, _] = downsampled_pcd_tree.search_knn_vector_3d(
        interaction_ee_keyframe[:3, 3], 1
    )
    target_idx_down_pcd = np.asarray(target_idx_1)[0]
    closest_pt_down_pcd = downsampled_xyz[target_idx_down_pcd]

    # this is for exact point
    # @Priyam I do not think the following is really needed
    input_pcd_tree = o3d.geometry.KDTreeFlann(input_pcd)
    [_, target_idx_2, _] = input_pcd_tree.search_knn_vector_3d(
        # ee_keyframe[:3, 3], 1
        interaction_ee_keyframe[:3, 3],
        1,
    )
    target_idx_og_pcd = np.asarray(target_idx_2)[0]
    closest_pt_og_pcd = xyz[target_idx_og_pcd]

    if debug:
        print("Closest point in downsampled pcd")
        show_point_cloud_with_keypt_and_closest_pt(
            downsampled_xyz,
            downsampled_rgb,
            interaction_ee_keyframe[:3, 3],
            interaction_ee_keyframe[:3, :3],
            downsampled_xyz[target_idx_down_pcd].reshape(3, 1),
        )
        print("Closest point in original pcd")
        show_point_cloud_with_keypt_and_closest_pt(
            xyz,
            rgb,
            interaction_ee_keyframe[:3, 3],
            interaction_ee_keyframe[:3, :3],
            xyz[target_idx_og_pcd].reshape(3, 1),
        )
    return (
        downsampled_xyz,
        downsampled_rgb,
        downsampled_feats,
        target_idx_down_pcd,
        closest_pt_down_pcd,
        target_idx_og_pcd,
        closest_pt_og_pcd,
    )


def get_local_problem(
    xyz,
    rgb,
    feat,
    interaction_pt,
    num_find_crop_tries=10,
    min_num_points=50,
    data_augmentation=False,
    multiplier=0.05,
    offset=-0.025,
    local_problem_size=0.1,
    num_pts=8000,
):
    """
    Crop given PCD around a perturbed interaction_point as input to action prediction problem
    """
    orig_crop_location = interaction_pt
    if data_augmentation:
        # Check to see if enough points are within the crop radius
        for _ in range(num_find_crop_tries):
            crop_location = orig_crop_location
            # Crop randomly within a few centimeters
            crop_location = orig_crop_location + (
                (np.random.random(3) * multiplier) + offset
            )
            # Make sure at least min_num_points are within this radius
            # get number of points in this area
            dists = np.linalg.norm(
                xyz - crop_location[None].repeat(xyz.shape[0], axis=0), axis=-1
            )
            # Make sure this is near some geometry
            if np.sum(dists < 0.1) > min_num_points:
                break
            else:
                crop_location = orig_crop_location
    else:
        crop_location = orig_crop_location

    # TODO: remove debug code
    # This should be at a totally reasonable location
    # show_point_cloud(xyz, rgb, orig=crop_location)

    # crop from og pcd and mean-center it
    crop_xyz, crop_rgb, crop_feat = crop_around_voxel(
        xyz, rgb, feat, crop_location, local_problem_size
    )
    crop_xyz = crop_xyz - crop_location[None].repeat(crop_xyz.shape[0], axis=0)
    # show_point_cloud(crop_xyz, crop_rgb, orig=np.zeros(3))
    if crop_rgb.shape[0] > num_pts:
        # Downsample pt clouds
        downsample = np.arange(crop_rgb.shape[0])
        np.random.shuffle(downsample)
        if num_pts != -1:
            downsample = downsample[:num_pts]
        crop_rgb = crop_rgb[downsample]
        crop_xyz = crop_xyz[downsample]
    status = True
    if crop_xyz.shape[0] < 10:
        status = False
    return crop_location, crop_xyz, crop_rgb, status


def get_local_commands(crop_location, ee_keyframe, ref_ee_keyframe, keyframes):
    """adjust keyframes in the pcd-center reference frame to be wrt crop-location instead"""
    # NOTE: copying the keyframes is EXTREMELY important
    crop_ee_keyframe = ee_keyframe.copy()
    crop_ee_keyframe[:3, 3] -= crop_location
    crop_ref_ee_keyframe = ref_ee_keyframe.copy()
    crop_ref_ee_keyframe[:3, 3] -= crop_location
    crop_keyframes = []
    for keyframe in keyframes:
        _keyframe = keyframe.copy()
        _keyframe[:3, 3] -= crop_location
        crop_keyframes.append(_keyframe)
    return crop_ref_ee_keyframe, crop_ee_keyframe, crop_keyframes


def format_commands(
    crop_ee_keyframe, keyframes, multi_step=False, ori_type="quaternion"
):
    """process keyframes and convert into the learnable format"""
    if multi_step:
        num_frames = len(keyframes)
        assert num_frames > 0
        positions = np.zeros((num_frames, 3))
        orientations = np.zeros((num_frames, 3, 3))
        # Set things up with the right shape
        if ori_type == "rpy":
            angles = np.zeros((num_frames, 3))
        elif ori_type == "quaternion":
            angles = np.zeros((num_frames, 4))
        else:
            raise RuntimeError("unsupported orientation type: " + str(ori_type))
        # Loop over the whole list of keyframes
        # Create a set of trajectory data so we can train all three waypoints at once
        for j, keyframe in enumerate(keyframes):
            orientations[j, :, :] = keyframe[:3, :3]
            positions[j, :] = keyframe[:3, 3]
            if ori_type == "rpy":
                angles[j, :] = tra.euler_from_matrix(keyframe[:3, :3])
            elif ori_type == "quaternion":
                angles[j, :] = tra.quaternion_from_matrix(keyframe[:3, :3])
    else:
        # Just one
        positions = crop_ee_keyframe[:3, 3]
        orientations = crop_ee_keyframe[:3, :3]
        if ori_type == "rpy":
            angles = tra.euler_from_matrix(crop_ee_keyframe[:3, :3])
        elif ori_type == "quaternion":
            angles = tra.quaternion_from_matrix(crop_ee_keyframe[:3, :3])
    return positions, orientations, angles


def compute_detic_mask(rgb_images, depth_images, segmentor):
    union_mask = []
    for i, img in enumerate(rgb_images):
        img, dimg = rotate_image([img, depth_images[i]])
        H, W, _ = img.shape

        # test DeticPerception
        # Create the observation
        obs = Observations(
            rgb=img.copy(),
            depth=dimg.copy(),
            xyz=None,
            gps=np.zeros(2),  # TODO Replace
            compass=np.zeros(1),  # TODO Replace
            task_observations={},
        )
        # Run the segmentation model here
        obs = segmentor.predict(obs, depth_threshold=0.5)
        breakpoint()
        plt.imshow(obs.task_observations["instance_map"])
        plt.show()
        mask = obs.task_observations["semantic_frame"] != 0
        # unrotate mask
        mask = unrotate_image([mask])
        union_mask.append(mask)
    union_mask = np.stack(union_mask, axis=0)
    return union_mask
