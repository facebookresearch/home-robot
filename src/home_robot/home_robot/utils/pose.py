# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import quaternion
import torch
import trimesh.transformations as tra

# Code adapted from the rotation continuity repo (https://github.com/papagina/RotationContinuity)


# T_poses num*3
# r_matrix batch*3*3
def compute_pose_from_rotation_matrix(T_pose, r_matrix):
    batch = r_matrix.shape[0]
    joint_num = T_pose.shape[0]
    r_matrices = (
        r_matrix.view(batch, 1, 3, 3)
        .expand(batch, joint_num, 3, 3)
        .contiguous()
        .view(batch * joint_num, 3, 3)
    )
    src_poses = (
        T_pose.view(1, joint_num, 3, 1)
        .expand(batch, joint_num, 3, 1)
        .contiguous()
        .view(batch * joint_num, 3, 1)
    )

    out_poses = torch.matmul(r_matrices, src_poses)  # (batch*joint_num)*3*1

    return out_poses.view(batch, joint_num, 3)


# batch*n
def normalize_vector(v, return_mag=False):
    v_mag = v.norm(dim=-1)
    v = v / (v_mag.view(-1, 1).repeat(1, 3))
    if return_mag is True:
        return v, v_mag
    else:
        return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat(
        (i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1
    )  # batch*3

    return out


# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def to_pos_quat(matrix):
    """utility to convert to (pos, quaternion) tuple in ROS quaternion format"""
    w, x, y, z = tra.quaternion_from_matrix(matrix)
    pos = matrix[:3, 3]
    return pos, np.array([x, y, z, w])


def to_matrix(pos, rot, trimesh_format=False) -> np.ndarray:
    """converts pos, quat to matrix format"""
    if trimesh_format:
        w, x, y, z = rot
    else:
        x, y, z, w = rot
    T = tra.quaternion_matrix([w, x, y, z])
    T[:3, 3] = pos
    return T


def get_l2_distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_pose(position, rotation):
    x = -position[2]
    y = -position[0]
    axis = quaternion.as_euler_angles(rotation)[0]
    if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
        o = quaternion.as_euler_angles(rotation)[1]
    else:
        o = 2 * np.pi - quaternion.as_euler_angles(rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o


def get_rel_pose_change(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2
    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1
    return dx, dy, do


def get_new_pose(pose, rel_pose_change):
    x, y, o = pose
    dx, dy, do = rel_pose_change
    global_dx = dx * np.sin(np.deg2rad(o)) + dy * np.cos(np.deg2rad(o))
    global_dy = dx * np.cos(np.deg2rad(o)) - dy * np.sin(np.deg2rad(o))
    x += global_dy
    y += global_dx
    o += np.rad2deg(do)
    if o > 180.0:
        o -= 360.0
    return x, y, o


def get_new_pose_batch(pose, rel_pose_change):
    const = 57.29577951308232
    pose[:, 1] += rel_pose_change[:, 0] * torch.sin(
        pose[:, 2] / const
    ) + rel_pose_change[:, 1] * torch.cos(pose[:, 2] / const)
    pose[:, 0] += rel_pose_change[:, 0] * torch.cos(
        pose[:, 2] / const
    ) - rel_pose_change[:, 1] * torch.sin(pose[:, 2] / const)
    pose[:, 2] += rel_pose_change[:, 2] * const
    pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
    pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0
    return pose


def threshold_poses(coords, shape):
    coords[0] = min(max(0, coords[0]), shape[0] - 1)
    coords[1] = min(max(0, coords[1]), shape[1] - 1)
    return coords


def normalize_angle(angle_in_degrees):
    angle_in_degrees = angle_in_degrees % 360.0
    if angle_in_degrees > 180:
        angle_in_degrees -= 360
    return angle_in_degrees


def normalize_radians(angle_in_radians):
    angle_in_radians = angle_in_radians % (2 * np.pi)
    if angle_in_radians > np.pi:
        angle_in_radians -= 2 * np.pi
    return angle_in_radians


def convert_pose_habitat_to_opencv(hab_pose: np.ndarray) -> np.ndarray:
    """Update axis convention of habitat pose to match the real-world axis convention"""
    hab_pose[[1, 2]] = hab_pose[[2, 1]]
    hab_pose[:, [1, 2]] = hab_pose[:, [2, 1]]

    hab_pose[0, 0] = -hab_pose[0, 0]
    hab_pose[1, 1] = -hab_pose[1, 1]
    hab_pose[0, 3] = -hab_pose[0, 3]

    return hab_pose
