# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh.transformations as tra


def numpy_to_pcd(xyz: np.ndarray, rgb: np.ndarray = None) -> o3d.geometry.PointCloud:
    """Create an open3d pointcloud from a single xyz/rgb pair"""
    xyz = xyz.reshape(-1, 3)
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def pcd_to_numpy(pcd: o3d.geometry.PointCloud) -> (np.ndarray, np.ndarray):
    """Convert an open3d point cloud into xyz, rgb numpy arrays and return them."""
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    return xyz, rgb


def show_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray = None,
    orig: np.ndarray = None,
    R: np.ndarray = None,
    save: str = None,
    grasps: list = None,
    size: float = 0.1,
):
    """Shows the point-cloud described by np.ndarrays xyz & rgb.
    Optional origin and rotation params are for showing origin coordinate.
    Optional grasps param for showing a list of 6D poses as coordinate frames.
    size controls scale of coordinate frame's size
    """
    pcd = numpy_to_pcd(xyz, rgb)
    show_pcd(pcd, orig=orig, R=R, save=save, grasps=grasps, size=size)


def show_pcd(
    pcd: o3d.geometry.PointCloud,
    orig: np.ndarray = None,
    R: np.ndarray = None,
    save: str = None,
    grasps: list = None,
    size: float = 0.1,
):
    """Shows the point-cloud described by open3d.geometry.PointCloud pcd
    Optional origin and rotation params are for showing origin coordinate.
    Optional grasps param for showing a list of 6D poses as coordinate frames.
    """
    geoms = create_visualization_geometries(
        pcd=pcd, orig=orig, R=R, grasps=grasps, size=size
    )
    o3d.visualization.draw_geometries(geoms)

    if save is not None:
        save_geometries_as_image(geoms, output_path=save)


def create_visualization_geometries(
    pcd: Optional[o3d.geometry.PointCloud] = None,
    xyz: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    orig: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    size: Optional[float] = 1.0,
    arrow_pos: Optional[np.ndarray] = None,
    arrow_size: Optional[float] = 1.0,
    arrow_R: Optional[np.ndarray] = None,
    arrow_color: Optional[np.ndarray] = None,
    sphere_pos: Optional[np.ndarray] = None,
    sphere_size: Optional[float] = 1.0,
    sphere_color: Optional[np.ndarray] = None,
    grasps: list = None,
):
    """
    Creates the open3d geometries for a point cloud (one of xyz or pcd must be specified), as well as, optionally, some
    helpful indicators for points of interest -- an origin (orig), an arrow (including direction), a sphere, and grasp
    indicators.
    """
    assert (pcd is not None) != (xyz is not None), "One of pcd or xyz must be specified"

    if xyz is not None:
        xyz = xyz.reshape(-1, 3)

    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
        if np.any(rgb > 1):
            print("WARNING: rgb values too high! Normalizing...")
            rgb = rgb / np.max(rgb)

    if pcd is None:
        pcd = numpy_to_pcd(xyz, rgb)

    geoms = [pcd]
    if orig is not None:
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
            origin=orig, size=size
        )
        if R is not None:
            coords = coords.rotate(R, orig)
        geoms.append(coords)

    if arrow_pos is not None:
        arrow = o3d.geometry.TriangleMesh.create_arrow()
        arrow = arrow.scale(
            arrow_size,
            center=np.zeros(
                3,
            ),
        )

        if arrow_color is not None:
            arrow = arrow.paint_uniform_color(arrow_color)

        if arrow_R is not None:
            arrow = arrow.rotate(arrow_R, center=(0, 0, 0))

        arrow = arrow.translate(arrow_pos)
        geoms.append(arrow)

    if sphere_pos is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)

        if sphere_color is not None:
            sphere = sphere.paint_uniform_color(sphere_color)

        sphere = sphere.translate(sphere_pos)
        geoms.append(sphere)

    if grasps is not None:
        for grasp in grasps:
            coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=grasp[:3, 3]
            )
            coords = coords.rotate(grasp[:3, :3])
            geoms.append(coords)

    return geoms


def save_geometries_as_image(
    geoms: list,
    camera_extrinsic: Optional[np.ndarray] = None,
    look_at_point: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    zoom: Optional[float] = None,
    point_size: Optional[float] = None,
    near_clipping: Optional[float] = None,
    far_clipping: Optional[float] = None,
    live_visualization: bool = False,
):
    """
    Helper function to allow manipulation of the camera to get a better image of the point cloud.
    The live_visualization flag can help debug issues, by also spawning an interactable window.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for geom in geoms:
        vis.add_geometry(geom)
        vis.update_geometry(geom)

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    if camera_extrinsic is not None:
        # The extrinsic seems to have a different convention - switch from our camera to open3d's version
        camera_extrinsic_o3d = camera_extrinsic.copy()
        camera_extrinsic_o3d[:3, :3] = np.matmul(
            camera_extrinsic_o3d[:3, :3], np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        )
        camera_extrinsic_o3d[:, 3] = np.matmul(
            camera_extrinsic_o3d[:, 3],
            np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        )

        camera_params.extrinsic = camera_extrinsic_o3d
        view_control.convert_from_pinhole_camera_parameters(camera_params)

    if look_at_point is not None:
        view_control.set_lookat(look_at_point)

    if zoom is not None:
        view_control.set_zoom(zoom)

    if near_clipping is not None:
        view_control.set_constant_z_near(near_clipping)

    if far_clipping is not None:
        view_control.set_constant_z_far(far_clipping)

    render_options = vis.get_render_option()

    if point_size is not None:
        render_options.point_size = point_size

    if live_visualization:
        vis.run()

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path, do_render=True)
    vis.destroy_window()


def fix_opengl_image(rgb, depth, camera_params=None):
    """opengl images are weird, we need to do a couple things here"""
    rgb = np.flip(rgb, axis=0)
    depth = np.flip(depth, axis=0)
    if camera_params is not None:
        depth = np.clip(depth, camera_params["near_val"], camera_params["far_val"])
    return rgb, depth


def depth_to_z(depth, near, far):
    """convert depth to metric depth - based on near/far values from camera"""
    depth = 2.0 * depth - 1.0
    z = 2.0 * near * far / (far + near - depth * (far - near))
    return z


def sim_depth_to_world_xyz(depth, width, height, view_matrix, proj_matrix):
    """get points from a camera image rendered in opengl. designed to work with pybullet
    from here: https://github.com/bulletphysics/bullet3/issues/1924#issuecomment-1091876325
    """
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth value
    y, x = np.mgrid[-1 : 1 : 2 / height, -1 : 1 : 2 / width]
    y *= -1.0
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    # pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3:4]
    points = points[:, :3]
    # show_point_cloud(points)
    # import pdb; pdb.set_trace()
    return points


# We apply this correction to xyz when computing it in sim
# R_CORRECTION = R1 @ R2
R_CORRECTION = tra.euler_matrix(0, 0, np.pi / 2)[:3, :3]


def opengl_depth_to_xyz(depth, camera):
    """get depth from numpy using simple pinhole camera model"""
    indices = np.indices((camera.height, camera.width), dtype=np.float32).transpose(
        1, 2, 0
    )
    z = depth
    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    # indices[..., 0] = np.flipud(indices[..., 0])
    x = (indices[:, :, 1] - camera.px) * (z / camera.fx)
    y = (indices[:, :, 0] - camera.py) * (z / camera.fy)  # * -1
    # Should now be height x width x 3, after this:
    xyz = np.stack([x, y, z], axis=-1) @ R_CORRECTION
    return xyz


def depth_to_xyz(depth, camera):
    """get depth from numpy using simple pinhole camera model"""
    indices = np.indices((camera.height, camera.width), dtype=np.float32).transpose(
        1, 2, 0
    )
    z = depth
    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    x = (indices[:, :, 1] - camera.px) * (z / camera.fx)
    y = (indices[:, :, 0] - camera.py) * (z / camera.fy)
    # Should now be height x width x 3, after this:
    xyz = np.stack([x, y, z], axis=-1)
    return xyz


def build_matrix_of_indices(height, width):
    """Builds a [height, width, 2] numpy array containing coordinates.
    @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)


def pose_from_camera_params(camera_params):
    # look_at = camera_params["look_at"]
    look_from = camera_params["look_from"]
    up_vector = camera_params["up_vector"]
    pose = np.eye(4)
    pose[:3, 3] = look_from
    pose[:3, 2] = up_vector
    return pose


##### Depth augmentations #####
# from: https://github.com/chrisdxie/uois/blob/master/src/data_augmentation.py#L68


def rotate_point_cloud(pcd):
    """
    Rotate point-cloud wrt z-axis by a random angle, anti-clockwise=+ve
    """
    rotation_matrix = tra.euler_matrix(0, 0, 2 * np.pi * np.random.random() - np.pi)
    pcd = tra.transform_points(pcd, rotation_matrix)
    return pcd


def add_multiplicative_noise(depth_img, gamma_shape=10000, gamma_scale=0.0001):
    """Add noise to depth image.
    This is adapted from the DexNet 2.0 code.
    Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
    @param depth_img: a [H x W] set of depth z values
    """
    # TODO(cpaxton): I am pretty sure we do not need to copy here, but verify this isn't
    # causing problems
    # depth_img = depth_img.copy()

    # Multiplicative noise: Gamma random variable
    # This will randomly shift around points locally
    multiplicative_noise = np.random.gamma(
        gamma_shape, gamma_scale, size=depth_img.shape
    )
    # Apply this noise to the depth image
    depth_img = multiplicative_noise * depth_img
    return depth_img


def add_additive_noise_to_xyz(
    xyz_img,
    gp_rescale_factor_range=[12, 20],
    gaussian_scale_range=[0.0, 0.003],
    valid_mask=None,
):
    """Add (approximate) Gaussian Process noise to ordered point cloud
    @param xyz_img: a [H x W x 3] ordered point cloud
    """
    xyz_img = xyz_img.copy()

    H, W, C = xyz_img.shape

    # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
    #                 which is rescaled with bicubic interpolation.
    gp_rescale_factor = np.random.randint(
        gp_rescale_factor_range[0], gp_rescale_factor_range[1]
    )
    gp_scale = np.random.uniform(gaussian_scale_range[0], gaussian_scale_range[1])

    small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
    additive_noise = np.random.normal(
        loc=0.0, scale=gp_scale, size=(small_H, small_W, C)
    )
    additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
    if valid_mask is not None:
        # use this to add to the image
        xyz_img[valid_mask, :] += additive_noise[valid_mask, :]
    else:
        xyz_img += additive_noise

    return xyz_img


def dropout_random_ellipses(
    depth_img, dropout_mean, gamma_shape=10000, gamma_scale=0.0001
):
    """Randomly drop a few ellipses in the image for robustness.
    This is adapted from the DexNet 2.0 code.
    Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
    @param depth_img: a [H x W] set of depth z values
    """
    depth_img = depth_img.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(dropout_mean)

    # Sample ellipse centers
    nonzero_pixel_indices = np.array(
        np.where(depth_img > 0)
    ).T  # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(
        nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout
    )
    dropout_centers = nonzero_pixel_indices[
        dropout_centers_indices, :
    ]  # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(gamma_shape, gamma_scale, size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(gamma_shape, gamma_scale, size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # dropout the ellipse
        # mask is always 2d even if input is not
        mask = np.zeros(depth_img.shape[:2])
        mask = cv2.ellipse(
            mask,
            tuple(center[::-1]),
            (x_radius, y_radius),
            angle=angle,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=-1,
        )
        depth_img[mask == 1] = 0

    return depth_img
