#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def spatial_transform_map(p, x, invert=True, mode="bilinear"):
    """
    Inputs:
        p     - (bs, f, H, W) Tensor
        x     - (bs, 3) Tensor (x, y, theta) transforms to perform
    Outputs:
        p_trans - (bs, f, H, W) Tensor
    Conventions:
        Shift in X is rightward, and shift in Y is downward. Rotation is clockwise.

    Note: These denote transforms in an agent's position. Not the image directly.
    For example, if an agent is moving upward, then the map will be moving downward.
    To disable this behavior, set invert=False.
    """
    device = p.device
    H, W = p.shape[2:]

    trans_x = x[:, 0]
    trans_y = x[:, 1]
    # Convert translations to -1.0 to 1.0 range
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H / 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W / 2

    trans_x = trans_x / Wby2
    trans_y = trans_y / Hby2
    rot_t = x[:, 2]

    sin_t = torch.sin(rot_t)
    cos_t = torch.cos(rot_t)

    # This R convention means Y axis is downwards.
    A = torch.zeros(p.size(0), 3, 3).to(device)
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = -sin_t
    A[:, 1, 0] = sin_t
    A[:, 1, 1] = cos_t
    A[:, 0, 2] = trans_x
    A[:, 1, 2] = trans_y
    A[:, 2, 2] = 1

    # Since this is a source to target mapping, and F.affine_grid expects
    # target to source mapping, we have to invert this for normal behavior.
    Ainv = torch.inverse(A)

    # If target to source mapping is required, invert is enabled and we invert
    # it again.
    if invert:
        Ainv = torch.inverse(Ainv)

    Ainv = Ainv[:, :2]
    grid = F.affine_grid(Ainv, p.size())
    p_trans = F.grid_sample(p, grid, mode=mode)

    return p_trans


def crop_map(h, x, crop_size, mode="bilinear"):
    """
    Crops a tensor h centered around location x with size crop_size

    Inputs:
        h - (bs, F, H, W)
        x - (bs, 2) --- (x, y) locations
        crop_size - scalar integer

    Conventions for x:
        The origin is at the top-left, X is rightward, and Y is downward.
    """

    bs, _, H, W = h.size()
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2
    start = -(crop_size - 1) / 2 if crop_size % 2 == 1 else -(crop_size // 2)
    end = start + crop_size - 1
    x_grid = (
        torch.arange(start, end + 1, step=1)
        .unsqueeze(0)
        .expand(crop_size, -1)
        .contiguous()
        .float()
    )
    y_grid = (
        torch.arange(start, end + 1, step=1)
        .unsqueeze(1)
        .expand(-1, crop_size)
        .contiguous()
        .float()
    )
    center_grid = torch.stack([x_grid, y_grid], dim=2).to(
        h.device
    )  # (crop_size, crop_size, 2)

    x_pos = x[:, 0] - Wby2  # (bs, )
    y_pos = x[:, 1] - Hby2  # (bs, )

    crop_grid = center_grid.unsqueeze(0).expand(
        bs, -1, -1, -1
    )  # (bs, crop_size, crop_size, 2)
    crop_grid = crop_grid.contiguous()

    # Convert the grid to (-1, 1) range
    crop_grid[:, :, :, 0] = (
        crop_grid[:, :, :, 0] + x_pos.unsqueeze(1).unsqueeze(2)
    ) / Wby2
    crop_grid[:, :, :, 1] = (
        crop_grid[:, :, :, 1] + y_pos.unsqueeze(1).unsqueeze(2)
    ) / Hby2

    h_cropped = F.grid_sample(h, crop_grid, mode=mode)

    return h_cropped


def bottom_row_padding(p):
    V = p.shape[2]
    Vby2 = (V - 1) / 2 if V % 2 == 1 else V // 2
    left_h_pad = 0
    right_h_pad = int(V - 1)
    if V % 2 == 1:
        left_w_pad = int(Vby2)
        right_w_pad = int(Vby2)
    else:
        left_w_pad = int(Vby2) - 1
        right_w_pad = int(Vby2)

    # Pad so that the origin is at the center
    p_pad = F.pad(p, (left_w_pad, right_w_pad, left_h_pad, right_h_pad), "constant", 0)

    return p_pad


def bottom_row_cropping(p, map_size):
    bs = p.shape[0]
    V = map_size
    Vby2 = (V - 1) / 2 if V % 2 == 1 else V // 2
    device = p.device

    x_crop_center = torch.zeros(bs, 2).to(device)
    x_crop_center[:, 0] = V - 1
    x_crop_center[:, 1] = Vby2
    x_crop_size = V

    p_cropped = crop_map(p, x_crop_center, x_crop_size)

    return p_cropped


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def convert_world2map(world_coors, map_shape, map_scale):
    """
    World coordinate system:
        Agent starts at (0, 0) facing upward along X. Y is rightward.
    Map coordinate system:
        Agent starts at (W/2, H/2) with X rightward and Y downward.

    Inputs:
        world_coors: (bs, 2) --- (x, y) in world coordinates
        map_shape: tuple with (H, W)
        map_scale: scalar indicating the cell size in the map
    """
    H, W = map_shape
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2

    x_world = world_coors[:, 0]
    y_world = world_coors[:, 1]

    x_map = torch.clamp((Wby2 + y_world / map_scale), 0, W - 1).round()
    y_map = torch.clamp((Hby2 - x_world / map_scale), 0, H - 1).round()

    map_coors = torch.stack([x_map, y_map], dim=1)  # (bs, 2)

    return map_coors


def convert_map2world(map_coors, map_shape, map_scale):
    """
    World coordinate system:
        Agent starts at (0, 0) facing upward along X. Y is rightward.
    Map coordinate system:
        Agent starts at (W/2, H/2) with X rightward and Y downward.

    Inputs:
        map_coors: (bs, 2) --- (x, y) in map coordinates
        map_shape: tuple with (H, W)
        map_scale: scalar indicating the cell size in the map
    """
    H, W = map_shape
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2

    x_map = map_coors[:, 0]
    y_map = map_coors[:, 1]

    x_world = (Hby2 - y_map) * map_scale
    y_world = (x_map - Wby2) * map_scale

    world_coors = torch.stack([x_world, y_world], dim=1)  # (bs, 2)

    return world_coors


def subtract_pose(pose_a, pose_b):
    """
    Compute pose of pose_b in the egocentric coordinate frame of pose_a.
    Inputs:
        pose_a - (bs, 3) --- (x, y, theta)
        pose_b - (bs, 3) --- (x, y, theta)

    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction
    """

    x_a, y_a, theta_a = torch.unbind(pose_a, dim=1)
    x_b, y_b, theta_b = torch.unbind(pose_b, dim=1)

    r_ab = torch.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)  # (bs, )
    phi_ab = torch.atan2(y_b - y_a, x_b - x_a) - theta_a  # (bs, )
    theta_ab = theta_b - theta_a  # (bs, )
    theta_ab = torch.atan2(torch.sin(theta_ab), torch.cos(theta_ab))

    x_ab = torch.stack(
        [
            r_ab * torch.cos(phi_ab),
            r_ab * torch.sin(phi_ab),
            theta_ab,
        ],
        dim=1,
    )  # (bs, 3)

    return x_ab


def add_pose(pose_a, pose_ab):
    """
    Add pose_ab (in ego-coordinates of pose_a) to pose_a
    Inputs:
        pose_a - (bs, 3) --- (x, y, theta)
        pose_b - (bs, 3) --- (x, y, theta)

    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction
    """

    x_a, y_a, theta_a = torch.unbind(pose_a, dim=1)
    x_ab, y_ab, theta_ab = torch.unbind(pose_ab, dim=1)

    r_ab = torch.sqrt(x_ab**2 + y_ab**2)
    phi_ab = torch.atan2(y_ab, x_ab)

    x_b = x_a + r_ab * torch.cos(phi_ab + theta_a)
    y_b = y_a + r_ab * torch.sin(phi_ab + theta_a)
    theta_b = theta_a + theta_ab
    theta_b = torch.atan2(torch.sin(theta_b), torch.cos(theta_b))

    pose_b = torch.stack([x_b, y_b, theta_b], dim=1)  # (bs, 3)

    return pose_b


def flatten_two(x):
    return x.view(-1, *x.shape[2:])


def unflatten_two(x, sh1, sh2):
    return x.view(sh1, sh2, *x.shape[1:])


def transpose_image(img):
    """
    Inputs:
        img - (bs, H, W, C) torch Tensor
    """
    img_p = img.permute(0, 3, 1, 2)  # (bs, C, H, W)
    return img_p


def process_image(img, img_mean, img_std):
    """
    Convert HWC -> CHW, normalize image.
    Inputs:
        img - (bs, H, W, C) torch Tensor
        img_mean - list of per-channel means
        img_std - list of per-channel stds

    Outputs:
        img_p - (bs, C, H, W)
    """
    device = img.device

    img_p = rearrange(img.float(), "b h w c -> b c h w")
    img_p = img_p / 255.0  # (bs, C, H, W)

    if isinstance(img_mean, list):
        img_mean_t = rearrange(torch.Tensor(img_mean), "c -> () c () ()").to(device)
        img_std_t = rearrange(torch.Tensor(img_std), "c -> () c () ()").to(device)
    else:
        img_mean_t = img_mean.to(device)
        img_std_t = img_std.to(device)

    img_p = (img_p - img_mean_t) / img_std_t

    return img_p


def unprocess_image(img_p, img_mean, img_std):
    """
    Unnormalize image, Convert CHW -> HWC
    Inputs:
        img_p - (bs, C, H, W)
        img_mean - list of per-channel means
        img_std - list of per-channel stds

    Outputs:
        img - (bs, H, W, C) torch Tensor
    """
    device = img_p.device

    img_mean_t = rearrange(torch.Tensor(img_mean), "c -> () c () ()").to(device)
    img_std_t = rearrange(torch.Tensor(img_std), "c -> () c () ()").to(device)

    img = img_p * img_std_t + img_mean_t
    img = img * 255.0
    img = rearrange(img, "b c h w -> b h w c")

    return img


def padded_resize(x, size):
    """For an image tensor of size (bs, c, h, w), resize it such that the
    larger dimension (h or w) is scaled to `size` and the other dimension is
    zero-padded on both sides to get `size`.
    """
    h, w = x.shape[2:]
    top_pad = 0
    bot_pad = 0
    left_pad = 0
    right_pad = 0
    if h > w:
        left_pad = (h - w) // 2
        right_pad = (h - w) - left_pad
    elif w > h:
        top_pad = (w - h) // 2
        bot_pad = (w - h) - top_pad
    x = F.pad(x, (left_pad, right_pad, top_pad, bot_pad))
    x = F.interpolate(x, size, mode="bilinear", align_corners=False)
    return x


def grow_projected_map(proj_map, local_map, iterations=2):
    """
    proj_map - (H, W, 2) map
    local_map - (H, W, 2) map

    channel 0 - 1 if occupied, 0 otherwise
    channel 1 - 1 if explored, 0 otherwise
    """
    proj_map = np.copy(proj_map)
    HEIGHT, WIDTH = proj_map.shape[:2]

    explored_local_mask = local_map[..., 1] == 1
    free_local_mask = (local_map[..., 0] == 0) & explored_local_mask
    occ_local_mask = (local_map[..., 0] == 1) & explored_local_mask

    # Iteratively expand multiple times
    for i in range(iterations):
        # Generate regions which are predictable

        # ================ Processing free space ===========================
        # Pick only free areas that are visible
        explored_proj_map = (proj_map[..., 1] == 1).astype(np.uint8) * 255
        free_proj_map = ((proj_map[..., 0] == 0) & explored_proj_map).astype(
            np.uint8
        ) * 255
        occ_proj_map = ((proj_map[..., 0] == 1) & explored_proj_map).astype(
            np.uint8
        ) * 255

        # For the first iteration, aggressively close holes
        if i == 0:
            close_kernel = np.ones((3, 3), np.uint8)
            for itr in range(5):
                free_proj_map = cv2.morphologyEx(
                    free_proj_map, cv2.MORPH_CLOSE, close_kernel
                )
            free_proj_map = (free_proj_map > 0).astype(np.uint8) * 255

        dilate_kernel = np.ones((7, 7), np.uint8)

        # Expand only GT free area
        for itr in range(2):
            free_proj_map_edges = cv2.Canny(free_proj_map, 50, 100)
            free_proj_map_edges_dilated = cv2.dilate(
                free_proj_map_edges, dilate_kernel, iterations=3
            )
            free_mask = (
                (free_proj_map_edges_dilated > 0) | (free_proj_map > 0)
            ) & free_local_mask
            free_proj_map = free_mask.astype(np.uint8) * 255

        # Dilate to include some occupied area
        free_proj_map = cv2.dilate(free_proj_map, dilate_kernel, iterations=1)
        free_proj_map = (free_proj_map > 0).astype(np.uint8)

        # Close holes
        close_kernel = np.ones((3, 3), np.uint8)
        free_proj_map = cv2.morphologyEx(free_proj_map, cv2.MORPH_CLOSE, close_kernel)

        # ================ Processing occupied space ===========================
        # For the first iteration, aggressively close holes
        if i == 0:
            close_kernel = np.ones((3, 3), np.uint8)
            for itr in range(3):
                occ_proj_map = cv2.morphologyEx(
                    occ_proj_map, cv2.MORPH_CLOSE, close_kernel
                )
            occ_proj_map = (occ_proj_map > 0).astype(np.uint8) * 255

        dilate_kernel = np.ones((3, 3), np.uint8)

        # Expand only GT occupied area
        for itr in range(1):
            occ_proj_map_edges = cv2.Canny(occ_proj_map, 50, 100)
            occ_proj_map_edges_dilated = cv2.dilate(
                occ_proj_map_edges, dilate_kernel, iterations=3
            )
            occ_mask = (
                (occ_proj_map_edges_dilated > 0) | (occ_proj_map > 0)
            ) & occ_local_mask
            occ_proj_map = occ_mask.astype(np.uint8) * 255

        dilate_kernel = np.ones((9, 9), np.uint8)
        # Expand the free space around the GT occupied area
        for itr in range(2):
            occ_proj_map_dilated = cv2.dilate(occ_proj_map, dilate_kernel, iterations=3)
            free_mask_around_occ = (occ_proj_map_dilated > 0) & free_local_mask
            occ_proj_map = ((occ_proj_map > 0) | free_mask_around_occ).astype(
                np.uint8
            ) * 255

        # Close holes
        close_kernel = np.ones((3, 3), np.uint8)
        occ_proj_map = cv2.morphologyEx(occ_proj_map, cv2.MORPH_CLOSE, close_kernel)

        # Include originally present areas in proj_map
        predictable_regions_mask = (
            (explored_proj_map > 0) | (free_proj_map > 0) | (occ_proj_map > 0)
        )

        # Create new proj_map
        proj_map = np.zeros((HEIGHT, WIDTH, 2), np.float32)
        proj_map[predictable_regions_mask & occ_local_mask, 0] = 1
        proj_map[predictable_regions_mask, 1] = 1

    gt_map = proj_map

    return gt_map


def convert_gt2channel_to_gtrgb(gts):
    """
    Inputs:
        gts   - (H, W, 2) numpy array with values between 0.0 to 1.0
              - channel 0 is 1 if occupied space
              - channel 1 is 1 if explored space
    """
    H, W, _ = gts.shape

    exp_mask = (gts[..., 1] >= 0.5).astype(np.float32)
    occ_mask = (gts[..., 0] >= 0.5).astype(np.float32) * exp_mask
    free_mask = (gts[..., 0] < 0.5).astype(np.float32) * exp_mask
    unk_mask = 1 - exp_mask

    gt_imgs = np.stack(
        [
            0.0 * occ_mask + 0.0 * free_mask + 255.0 * unk_mask,
            0.0 * occ_mask + 255.0 * free_mask + 255.0 * unk_mask,
            255.0 * occ_mask + 0.0 * free_mask + 255.0 * unk_mask,
        ],
        axis=2,
    ).astype(
        np.uint8
    )  # (H, W, 3)

    return gt_imgs


def convert_gtrgb_to_gt2channel(gtrgb):
    """
    gt - (H, W, 3) RGB image with
         (0, 255, 0) for free space,
         (0, 0, 255) for occupied space,
         (0, 0, 0) for unknown space
    """
    gt2channel = np.zeros((*gtrgb.shape[:2], 2))
    free_space = np.all(gtrgb == np.array([0, 255, 0]), axis=-1)
    occ_space = np.all(gtrgb == np.array([0, 0, 255]), axis=-1)
    explored_space = free_space | occ_space

    gt2channel[occ_space, 0] = 1.0
    gt2channel[explored_space, 1] = 1.0

    return gt2channel


def dilate_tensor(x, size, iterations=1):
    """
    x - (bs, C, H, W)
    size - int / tuple of intes

    Assumes a kernel of ones with size 'size'.
    """
    if type(size) == int:
        padding = size // 2
    else:
        padding = tuple([v // 2 for v in size])
    for i in range(iterations):
        x = F.max_pool2d(x, size, stride=1, padding=padding)

    return x


def erode_tensor(x, size, iterations=1):
    """
    x - (bs, C, H, W)
    size - int / tuple of intes

    Assumes a kernel of ones with size 'size'.
    """
    if type(size) == int:
        padding = size // 2
    else:
        padding = tuple([v // 2 for v in size])
    for i in range(iterations):
        x = -F.max_pool2d(-x, size, stride=1, padding=padding)

    return x


def morphology_close(x, size, iterations=1):
    """
    x - (bs, C, H, W)
    size - int / tuple of intes

    Assumes a kernel of ones with size 'size'.
    """
    x = dilate_tensor(x, size, iterations)
    x = erode_tensor(x, size, iterations)
    return x


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = self.init_(nn.Linear(num_inputs, num_outputs))

    def init_(self, m):
        return init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
