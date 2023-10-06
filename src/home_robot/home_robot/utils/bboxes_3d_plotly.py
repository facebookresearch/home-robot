# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from pytorch3d.viz.plotly_vis which has license:
# BSD License

# For PyTorch3D software

# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

#  * Neither the name Meta nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import warnings
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from pytorch3d.renderer import (
    HeterogeneousRayBundle,
    RayBundle,
    TexturesAtlas,
    TexturesVertex,
    ray_bundle_to_ray_points,
)
from pytorch3d.renderer.camera_utils import camera_to_eye_at_up
from pytorch3d.renderer.cameras import CamerasBase, get_world_to_view_transform
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene
from pytorch3d.vis.plotly_vis import (
    AxisArgs,
    Lighting,
    _add_camera_trace,
    _add_mesh_trace,
    _add_pointcloud_trace,
    _add_ray_bundle_trace,
    _is_ray_bundle,
    _scale_camera_to_bounds,
    _update_axes_bounds,
)
from torch import Tensor

from .bboxes_3d import BBoxes3D

Struct = Union[CamerasBase, Meshes, Pointclouds, RayBundle, HeterogeneousRayBundle]


def get_bbox_wireframe(
    bbox3d: torch.Tensor, add_cross_face_bars: Optional[bool] = False  # (N, 3, 2)
):  # pragma: no cover
    """
    Returns a wireframe of a 3D line-plot of a bounding box.
                    v4_____________________v5
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
              v0|    |                 |v1 |
                |    |                 |   |
                |    |                 |   |
                |    |                 |   |
                |    |_________________|___|
                |   / v7               |   /v6
                |  /                   |  /
                | /                    | /
                |/_____________________|/
                v3                     v2


                  + Y
                 /
                + - - + Z
                |
               X|
                +
    """
    assert bbox3d.shape[1] == 3 and bbox3d.shape[2] == 2
    minx, miny, minz = bbox3d[:, :, 0].unbind(1)
    maxx, maxy, maxz = bbox3d[:, :, 1].unbind(1)
    v0 = torch.stack([minx, miny, minz], axis=-1)
    v1 = torch.stack([minx, miny, maxz], axis=-1)
    v2 = torch.stack([maxx, miny, maxz], axis=-1)
    v3 = torch.stack([maxx, miny, minz], axis=-1)
    v4 = torch.stack([minx, maxy, minz], axis=-1)
    v5 = torch.stack([minx, maxy, maxz], axis=-1)
    v6 = torch.stack([maxx, maxy, maxz], axis=-1)
    v7 = torch.stack([maxx, maxy, minz], axis=-1)
    corner_points = [v0, v1, v2, v3, v0, v4, v5, v1, v5, v6, v2, v6, v7, v3, v7, v4]
    if add_cross_face_bars:
        corner_points = [
            v0,
            v5,
            v0,
            v7,
            v0,
            v2,
            v0,  # cross-face
            v1,
            v2,
            v3,
            v0,
            v4,
            v5,
            v1,
            v5,
            v6,
            v6,
            v1,
            v6,
            v3,
            v6,
            v4,
            v6,  # cross-face
            v2,
            v6,
            v7,
            v3,
            v7,
            v4,
        ]
    lines = torch.stack([x.float() for x in corner_points], axis=-2)
    return lines


def _add_bbox3d_trace(
    fig: go.Figure,
    box_wires: torch.Tensor,
    trace_name: str,
    subplot_idx: int,
    ncols: int,
    color: Optional[str] = None,
    wireframe_width: int = 1,
) -> None:  # pragma: no cover
    """
    Adds a trace rendering a Cameras object to the passed in figure, with
    a given name and in a specific subplot.

    Args:
        fig: plotly figure to add the trace within.
        bboxes_coords: Bounding boxes to render (N, 3, 2). It can be batched.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of subplots per row.
        color:
    """
    all_box_wires = box_wires.detach().cpu()
    if all_box_wires.ndim == 2:
        all_box_wires = all_box_wires.unsqueeze(0)

    nan_tensor = torch.Tensor([[float("NaN")] * 3])
    box_wires_padded = all_box_wires[0]
    for i, wire in enumerate(all_box_wires[1:]):
        # We combine camera points into a single tensor to plot them in a
        # single trace. The NaNs are inserted between sets of camera
        # points so that the lines drawn by Plotly are not drawn between
        # points that belong to different cameras.
        box_wires_padded = torch.cat((box_wires_padded, nan_tensor, wire))

    x, y, z = box_wires_padded.detach().cpu().numpy().T.astype(float)

    row, col = subplot_idx // ncols + 1, subplot_idx % ncols + 1
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            marker={
                "size": 1,
                "color": color,
            },
            line=dict(
                width=wireframe_width,
                color=color,
            ),
            name=trace_name,
        ),
        row=row,
        col=col,
    )

    # Access the current subplot's scene configuration
    plot_scene = "scene" + str(subplot_idx + 1)
    current_layout = fig["layout"][plot_scene]

    # flatten for bounds calculations
    flattened_wires = all_box_wires.flatten(0, 1)
    verts_center = flattened_wires.mean(0)
    max_expand = (flattened_wires.max(0)[0] - flattened_wires.min(0)[0]).max()
    _update_axes_bounds(verts_center, max_expand, current_layout)


def _add_separate_bbox3d_traces(
    fig: go.Figure,
    bboxes: BBoxes3D,
    trace_name: str,
    subplot_idx: int,
    ncols: int,
    bbox_names: List[str] = None,
    color: Optional[str] = None,
    wireframe_width: int = 1,
    add_cross_face_bars: bool = False,
    box_name_to_tracename_dict: Dict[int, str] = None,
    use_separate_traces: bool = True,
) -> None:  # pragma: no cover
    """
    Adds a trace rendering a Cameras object to the passed in figure, with
    a given name and in a specific subplot.

    Args:
        fig: plotly figure to add the trace within.
        bboxes: Bounding boxes to render (N, 3, 2). It can be batched.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of subplots per row.
        color:
    """
    # unrotate bounding boxes if they shouldn't be axis aligned.to(cameras.device)
    # we will need to update _add_bbox3d_trace
    bboxes = bboxes.detach().cpu()
    extrema_coords = bboxes.bounds_packed()
    features = bboxes.features_packed()
    n_boxes = len(extrema_coords)

    if not use_separate_traces:
        all_box_wires = get_bbox_wireframe(
            extrema_coords, add_cross_face_bars=add_cross_face_bars
        )
        _add_bbox3d_trace(
            fig=fig,
            box_wires=all_box_wires,
            trace_name=f"{trace_name}",
            subplot_idx=subplot_idx,
            ncols=ncols,
            color=color,
            wireframe_width=wireframe_width,
        )
        return

    if bbox_names is None:
        bbox_names = bboxes.names_packed()
        if bbox_names is None:
            pass
        elif box_name_to_tracename_dict is not None:
            bbox_names = [box_name_to_tracename_dict[int(name)] for name in bbox_names]
        else:
            bbox_names = [str(int(name)) for name in bbox_names]
    if bbox_names is None:
        bbox_names = [f"{int(i)}" for i in range(n_boxes)]

    bbox_color = [color] * n_boxes
    if features is not None:
        if features.shape[1] == 4:  # rgba
            template = "rgb(%d, %d, %d, %f)"
            rgb = (features[:, :3].clamp(0.0, 1.0) * 255).int()
            bbox_color = [
                template % (*rgb_, a_) for rgb_, a_ in zip(rgb, features[:, 3])
            ]

        if features.shape[1] == 3:
            template = "rgb(%d, %d, %d)"
            rgb = (features.clamp(0.0, 1.0) * 255).int()
            bbox_color = [template % (r, g, b) for r, g, b in rgb]

    all_box_wires = get_bbox_wireframe(
        extrema_coords, add_cross_face_bars=add_cross_face_bars
    )

    # row, col = subplot_idx // ncols + 1, subplot_idx % ncols + 1
    for (coords, name, color) in zip(all_box_wires, bbox_names, bbox_color):
        _add_bbox3d_trace(
            fig=fig,
            box_wires=coords,
            trace_name=f"{trace_name}.{name}",
            subplot_idx=subplot_idx,
            ncols=ncols,
            color=color,
            wireframe_width=wireframe_width,
        )


@torch.no_grad()
def plot_scene_with_bboxes(
    plots: Dict[str, Dict[str, Struct]],
    *,
    viewpoint_cameras: Optional[CamerasBase] = None,
    ncols: int = 1,
    camera_scale: float = 0.3,
    pointcloud_max_points: int = 20000,
    pointcloud_marker_size: int = 1,
    raybundle_max_rays: int = 20000,
    raybundle_max_points_per_ray: int = 1000,
    raybundle_ray_point_marker_size: int = 1,
    raybundle_ray_line_width: int = 1,
    boxes_wireframe_width: int = 1,
    boxes_add_cross_face_bars: bool = False,
    boxes_name_int_to_display_name_dict: Optional[Dict[int, str]] = None,
    boxes_plot_together: bool = False,
    height: int = None,
    width: int = None,
    use_orthographic: bool = False,
    **kwargs,
):  # pragma: no cover
    """
    Main function to visualize Cameras, Meshes, Pointclouds, and RayBundle.
    Plots input Cameras, Meshes, Pointclouds, and RayBundle data into named subplots,
    with named traces based on the dictionary keys. Cameras are
    rendered at the camera center location using a wireframe.

    Args:
        plots: A dict containing subplot and trace names,
            as well as the Meshes, Cameras and Pointclouds objects to be rendered.
            See below for examples of the format.
        viewpoint_cameras: an instance of a Cameras object providing a location
            to view the plotly plot from. If the batch size is equal
            to the number of subplots, it is a one to one mapping.
            If the batch size is 1, then that viewpoint will be used
            for all the subplots will be viewed from that point.
            Otherwise, the viewpoint_cameras will not be used.
        ncols: the number of subplots per row
        camera_scale: determines the size of the wireframe used to render cameras.
        pointcloud_max_points: the maximum number of points to plot from
            a pointcloud. If more are present, a random sample of size
            pointcloud_max_points is used.
        pointcloud_marker_size: the size of the points rendered by plotly
            when plotting a pointcloud.
        raybundle_max_rays: maximum number of rays of a RayBundle to visualize. Randomly
            subsamples without replacement in case the number of rays is bigger than max_rays.
        raybundle_max_points_per_ray: the maximum number of points per ray in RayBundle
            to visualize. If more are present, a random sample of size
            max_points_per_ray is used.
        raybundle_ray_point_marker_size: the size of the ray points of a plotted RayBundle
        raybundle_ray_line_width: the width of the plotted rays of a RayBundle
        **kwargs: Accepts lighting (a Lighting object) and any of the args xaxis,
            yaxis and zaxis which Plotly's scene accepts. Accepts axis_args,
            which is an AxisArgs object that is applied to all 3 axes.
            Example settings for axis_args and lighting are given at the
            top of this file.

    Example:

    ..code-block::python

        mesh = ...
        point_cloud = ...
        fig = plot_scene({
            "subplot_title": {
                "mesh_trace_title": mesh,
                "pointcloud_trace_title": point_cloud
            }
        })
        fig.show()

    The above example will render one subplot which has both a mesh and pointcloud.

    If the Meshes, Pointclouds, or Cameras objects are batched, then every object in that batch
    will be plotted in a single trace.

    ..code-block::python
        mesh = ... # batch size 2
        point_cloud = ... # batch size 2
        fig = plot_scene({
            "subplot_title": {
                "mesh_trace_title": mesh,
                "pointcloud_trace_title": point_cloud
            }
        })
        fig.show()

    The above example renders one subplot with 2 traces, each of which renders
    both objects from their respective batched data.

    Multiple subplots follow the same pattern:
    ..code-block::python
        mesh = ... # batch size 2
        point_cloud = ... # batch size 2
        fig = plot_scene({
            "subplot1_title": {
                "mesh_trace_title": mesh[0],
                "pointcloud_trace_title": point_cloud[0]
            },
            "subplot2_title": {
                "mesh_trace_title": mesh[1],
                "pointcloud_trace_title": point_cloud[1]
            }
        },
        ncols=2)  # specify the number of subplots per row
        fig.show()

    The above example will render two subplots, each containing a mesh
    and a pointcloud. The ncols argument will render two subplots in one row
    instead of having them vertically stacked because the default is one subplot
    per row.

    To view plotly plots from a PyTorch3D camera's point of view, we can use
    viewpoint_cameras:
    ..code-block::python
        mesh = ... # batch size 2
        R, T = look_at_view_transform(2.7, 0, [0, 180]) # 2 camera angles, front and back
        # Any instance of CamerasBase works, here we use FoVPerspectiveCameras
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        fig = plot_scene({
            "subplot1_title": {
                "mesh_trace_title": mesh[0]
            },
            "subplot2_title": {
                "mesh_trace_title": mesh[1]
            }
        },
        viewpoint_cameras=cameras)
        fig.show()

    The above example will render the first subplot seen from the camera on the +z axis,
    and the second subplot from the viewpoint of the camera on the -z axis.

    We can visualize these cameras as well:
    ..code-block::python
        mesh = ...
        R, T = look_at_view_transform(2.7, 0, [0, 180]) # 2 camera angles, front and back
        # Any instance of CamerasBase works, here we use FoVPerspectiveCameras
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        fig = plot_scene({
            "subplot1_title": {
                "mesh_trace_title": mesh,
                "cameras_trace_title": cameras,
            },
        })
        fig.show()

    The above example will render one subplot with the mesh object
    and two cameras.

    RayBundle visualization is also supproted:
    ..code-block::python
        cameras = PerspectiveCameras(...)
        ray_bundle = RayBundle(origins=..., lengths=..., directions=..., xys=...)
        fig = plot_scene({
            "subplot1_title": {
                "ray_bundle_trace_title": ray_bundle,
                "cameras_trace_title": cameras,
            },
        })
        fig.show()

    For an example of using kwargs, see below:
    ..code-block::python
        mesh = ...
        point_cloud = ...
        fig = plot_scene({
            "subplot_title": {
                "mesh_trace_title": mesh,
                "pointcloud_trace_title": point_cloud
            }
        },
        axis_args=AxisArgs(backgroundcolor="rgb(200,230,200)")) # kwarg axis_args
        fig.show()

    The above example will render each axis with the input background color.

    See the tutorials in pytorch3d/docs/tutorials for more examples
    (namely rendered_color_points.ipynb and rendered_textured_meshes.ipynb).
    """

    subplots = list(plots.keys())
    fig = _gen_fig_with_subplots(len(subplots), ncols, subplots)
    lighting = kwargs.get("lighting", Lighting())._asdict()
    axis_args_dict = kwargs.get("axis_args", AxisArgs())._asdict()

    # Set axis arguments to defaults defined at the top of this file
    x_settings = {**axis_args_dict}
    y_settings = {**axis_args_dict}
    z_settings = {**axis_args_dict}

    # Update the axes with any axis settings passed in as kwargs.
    x_settings.update(**kwargs.get("xaxis", {}))
    y_settings.update(**kwargs.get("yaxis", {}))
    z_settings.update(**kwargs.get("zaxis", {}))

    camera = {
        "up": {
            "x": 0.0,
            "y": 0.0,
            "z": 1.0,
        }  # set the up vector to match PyTorch3D world coordinates conventions
    }
    viewpoints_eye_at_up_world = None
    if viewpoint_cameras:
        n_viewpoint_cameras = len(viewpoint_cameras)
        if n_viewpoint_cameras == len(subplots) or n_viewpoint_cameras == 1:
            # Calculate the vectors eye, at, up in world space
            # to initialize the position of the camera in
            # the plotly figure
            viewpoints_eye_at_up_world = camera_to_eye_at_up(
                viewpoint_cameras.get_world_to_view_transform().cpu()
            )
        else:
            msg = "Invalid number {} of viewpoint cameras were provided. Either 1 \
            or {} cameras are required".format(
                len(viewpoint_cameras), len(subplots)
            )
            warnings.warn(msg)

    for subplot_idx in range(len(subplots)):
        subplot_name = subplots[subplot_idx]
        traces = plots[subplot_name]
        for trace_name, struct in traces.items():
            if isinstance(struct, Meshes):
                _add_mesh_trace(fig, struct, trace_name, subplot_idx, ncols, lighting)
            elif isinstance(struct, Pointclouds):
                _add_pointcloud_trace(
                    fig,
                    struct,
                    trace_name,
                    subplot_idx,
                    ncols,
                    pointcloud_max_points,
                    pointcloud_marker_size,
                )
            elif isinstance(struct, CamerasBase):
                _add_camera_trace(
                    fig, struct, trace_name, subplot_idx, ncols, camera_scale
                )
            elif isinstance(struct, BBoxes3D):
                _add_separate_bbox3d_traces(
                    fig,
                    struct,
                    trace_name,
                    subplot_idx,
                    ncols,
                    wireframe_width=boxes_wireframe_width,
                    add_cross_face_bars=boxes_add_cross_face_bars,
                    box_name_to_tracename_dict=boxes_name_int_to_display_name_dict,
                    use_separate_traces=not boxes_plot_together,
                )
            elif _is_ray_bundle(struct):
                _add_ray_bundle_trace(
                    fig,
                    struct,
                    trace_name,
                    subplot_idx,
                    ncols,
                    raybundle_max_rays,
                    raybundle_max_points_per_ray,
                    raybundle_ray_point_marker_size,
                    raybundle_ray_line_width,
                )
            else:
                raise ValueError(
                    "struct {} is not a Cameras, Meshes, BBoxes3D, Pointclouds,".format(
                        struct
                    )
                    + "RayBundle or HeterogeneousRayBundle object."
                )

        # Ensure update for every subplot.
        plot_scene = "scene" + str(subplot_idx + 1)
        current_layout = fig["layout"][plot_scene]
        xaxis = current_layout["xaxis"]
        yaxis = current_layout["yaxis"]
        zaxis = current_layout["zaxis"]

        # Update the axes with our above default and provided settings.
        xaxis.update(**x_settings)
        yaxis.update(**y_settings)
        zaxis.update(**z_settings)

        # update camera viewpoint if provided
        if viewpoints_eye_at_up_world is not None:
            # Use camera params for batch index or the first camera if only one provided.
            viewpoint_idx = min(n_viewpoint_cameras - 1, subplot_idx)

            eye, at, up = (i[viewpoint_idx] for i in viewpoints_eye_at_up_world)
            eye_x, eye_y, eye_z = eye.tolist()
            at_x, at_y, at_z = at.tolist()
            up_x, up_y, up_z = up.tolist()

            # scale camera eye to plotly [-1, 1] ranges
            x_range = xaxis["range"]
            y_range = yaxis["range"]
            z_range = zaxis["range"]

            eye_x = _scale_camera_to_bounds(eye_x, x_range, True)
            eye_y = _scale_camera_to_bounds(eye_y, y_range, True)
            eye_z = _scale_camera_to_bounds(eye_z, z_range, True)

            at_x = _scale_camera_to_bounds(at_x, x_range, True)
            at_y = _scale_camera_to_bounds(at_y, y_range, True)
            at_z = _scale_camera_to_bounds(at_z, z_range, True)

            up_x = _scale_camera_to_bounds(up_x, x_range, False)
            up_y = _scale_camera_to_bounds(up_y, y_range, False)
            up_z = _scale_camera_to_bounds(up_z, z_range, False)

            camera["eye"] = {"x": eye_x, "y": eye_y, "z": eye_z}
            camera["center"] = {"x": at_x, "y": at_y, "z": at_z}
            camera["up"] = {"x": up_x, "y": up_y, "z": up_z}
            camera["projection"] = {"type": "orthographic"}

        current_layout.update(
            {
                "xaxis": xaxis,
                "yaxis": yaxis,
                "zaxis": zaxis,
                # "aspectmode": "cube",
                "camera": camera,
            }
        )
    if width is not None or height is not None:
        fig.update_layout(width=width, height=height, aspectmode="data")
    if use_orthographic:
        # fig.update_scenes(aspectmode='data')
        fig.layout.scene.camera.projection.type = "orthographic"
    return fig


def _gen_fig_with_subplots(
    batch_size: int,
    ncols: int,
    subplot_titles: List[str],
    row_heights: Optional[List[int]] = None,
    column_widths: Optional[List[int]] = None,
):  # pragma: no cover
    """
    Takes in the number of objects to be plotted and generate a plotly figure
    with the appropriate number and orientation of titled subplots.
    Args:
        batch_size: the number of elements in the batch of objects to be visualized.
        ncols: number of subplots in the same row.
        subplot_titles: titles for the subplot(s). list of strings of length batch_size.

    Returns:
        Plotly figure with ncols subplots per row, and batch_size subplots.
    """
    fig_rows = batch_size // ncols
    if batch_size % ncols != 0:
        fig_rows += 1  # allow for non-uniform rows
    fig_cols = ncols
    fig_type = [{"type": "scene"}]
    specs = [fig_type * fig_cols] * fig_rows
    # subplot_titles must have one title per subplot
    fig = make_subplots(
        rows=fig_rows,
        cols=fig_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=[1.0] * fig_cols,
    )
    return fig


def create_triad_pointclouds(
    R: Tensor, T: Tensor, n_points: int = 1, scale: float = 0.1
):
    """
    Create a batch of 3D triads (coordinate systems) represented as point clouds.

    This function generates 3D point clouds for each instance in a batch. Each point cloud
    represents a triad, or a 3D coordinate system, that has been transformed by the given
    rotation matrices and translation vectors.

    Parameters:
    -----------
    R : torch.Tensor
        Batch of rotation matrices of shape (batch_size, 3, 3).
    T : torch.Tensor
        Batch of translation vectors of shape (batch_size, 3).
    n_points : int, optional
        Number of points along each axis in the triad. Default is 1.
    scale : float, optional
        Scaling factor for the size of the triads. Default is 0.1.

    Returns:
    --------
    Pointclouds : pytorch3d.structures.Pointclouds
        A batch of point clouds, each representing a transformed triad.
        The point clouds contain both the coordinates and the colors of the points.

    Example:
    --------
    >>> R = torch.eye(3).unsqueeze(0)
    >>> T = torch.tensor([[0.0, 0.0, 0.0]])
    >>> pointclouds = create_triad_pointclouds(R, T)
    """
    batch_size = R.shape[0]
    # Define the coordinates of the triad
    triad_coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],  # Origin
            [1.0, 0.0, 0.0, 1.0],  # X-axis
            [0.0, 1.0, 0.0, 1.0],  # Y-axis
            [0.0, 0.0, 1.0, 1.0],
        ]
    )  # Z-axis
    triad_coords = torch.cat(
        [triad_coords * (scale * 1.0 / n_points * i) for i in range(1, n_points + 1)],
        dim=0,
    )
    triad_coords[:, 3] = 1.0
    M = torch.zeros((batch_size, 4, 4))

    M[:, :3, :3] = R
    M[:, :3, 3] = T
    M[:, -1, -1] = 1.0

    triad_coords = triad_coords.unsqueeze(0).expand(
        batch_size, *triad_coords.shape[-2:]
    )
    triad_coords = torch.bmm(triad_coords, M.permute([0, 2, 1]))
    triad_coords = triad_coords[..., :3]
    # triad_coords += T.unsqueeze(1)

    # Create colors for each point (red, green, blue)
    colors = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # White
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
        ]
    )
    colors = (
        colors.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, n_points, 4, 3)
        .reshape(batch_size, n_points * 4, 3)
    )
    return Pointclouds(points=triad_coords[..., :3], features=colors)
