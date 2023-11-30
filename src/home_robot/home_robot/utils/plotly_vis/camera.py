# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from pytorch3d.vis.plotly_vis import (
    AxisArgs,
    _update_axes_bounds,
    get_camera_wireframe,
    plot_scene,
)
from torch import Tensor


def colormap_to_rgb_strings(
    data, colormap_name="viridis", include_alpha=False, min_val=None, max_val=None
):
    """
    Convert a range of numbers from a given dataset into a series of RGB or RGBA strings using a specified Matplotlib colormap.

    :param data: The dataset from which to derive color mappings.
    :param colormap_name: The name of the Matplotlib colormap to use.
    :param include_alpha: Boolean to decide if the alpha channel should be included in the RGB strings.
    :param min_val: Optional minimum value for colormap scaling.
    :param max_val: Optional maximum value for colormap scaling.
    :return: A list of color strings in the format 'rgb(R,G,B)' or 'rgba(R,G,B,A)'.
    """
    # Compute min and max from the data if not provided
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)

    # Normalize data within the provided or computed min and max range
    norm = plt.Normalize(min_val, max_val)
    colors = plt.cm.get_cmap(colormap_name)(norm(data))

    # Format color strings based on the include_alpha flag
    if include_alpha:
        return [
            "rgba({},{},{},{})".format(int(r * 255), int(g * 255), int(b * 255), a)
            for r, g, b, a in colors
        ]
    else:
        return [
            "rgb({},{},{})".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b in colors[:, :3]
        ]


def add_camera_poses(
    fig: Optional[go.Figure],
    poses: Tensor,
    linewidth: int = 3,
    color: Optional[str] = None,
    name: str = "cam",
    separate: bool = True,
    scale: float = 0.2,
    colormap_name: str = "plasma",
):
    """Add camera wireframe to a plotly figure.
    Args:
        fig (plotly.graph_objs.Figure): The figure to which the camera wireframe will be added.
        poses (list of numpy.ndarray): A list of camera poses, where each pose is a 4x4 matrix representing the transformation from the camera coordinate system to the world coordinate system.
        linewidth (int, optional): The width of the lines used to draw the camera wireframe. Defaults to 3.
        color (str or list of str, optional): The color of the lines used to draw the camera wireframe. If not specified, a different color will be chosen for each pose using a colormap.
        name (str, optional): The prefix for the name of the trace added to the figure. Defaults to "cam".
        separate (bool, optional): Whether to add a separate trace for each pose or to combine them into a single trace. Defaults to True.
        scale (float, optional): The scale factor for the camera wireframe. Defaults to 0.2.
        colormap_name (str, optional): The name of the colormap used to choose colors for the traces if `color` is not specified. Defaults to "plasma".
    Returns:
        None: The function modifies the input figure in place.
    """
    cam_points = get_camera_wireframe(scale)
    # Convert p3d (opengl) to opencv
    cam_points[:, 1] *= -1

    if color is None:
        colors = colormap_to_rgb_strings(
            list(range(len(poses))), colormap_name=colormap_name
        )
    else:
        colors = [color] * len(poses)
    for i, (pose, color) in enumerate(zip(poses, colors)):
        R = pose[:3, :3]
        t = pose[:3, -1]
        cam_points_world = cam_points @ R.T + t.unsqueeze(0)  # (cam_points @ R) # + t)
        x, y, z = [v.cpu().numpy().tolist() for v in cam_points_world.unbind(1)]
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                marker={
                    "size": 1,
                    "color": color,
                },
                line=dict(
                    width=linewidth,
                    color=color,
                ),
                name=f"{name}-{i}",
            )
        )


def _batch_points_for_plotly(cam_wires_trans):
    # if batch size is 1, unsqueeze to add dimension
    if len(cam_wires_trans.shape) < 3:
        cam_wires_trans = cam_wires_trans.unsqueeze(0)

    nan_tensor = torch.Tensor([[float("NaN")] * 3])
    all_cam_wires = cam_wires_trans[0]
    for wire in cam_wires_trans[1:]:
        # We combine camera points into a single tensor to plot them in a
        # single trace. The NaNs are inserted between sets of camera
        # points so that the lines drawn by Plotly are not drawn between
        # points that belong to different cameras.
        all_cam_wires = torch.cat((all_cam_wires, nan_tensor, wire))
    x, y, z = all_cam_wires.detach().cpu().numpy().T.astype(float)
    return x, y, z


def _add_camera_trace(
    fig: go.Figure,
    poses: Tensor,
    trace_name: str = "Cameras",
    subplot_idx: int = 0,
    ncols: int = 1,
    camera_scale: float = 0.2,
    colormap_name: str = "plasma",
    color: Optional[str] = None,
    separate_traces: bool = False,
    linewidth: int = 3,
) -> None:  # pragma: no cover
    """
    Adds a trace rendering a Cameras object to the passed in figure, with
    a given name and in a specific subplot.

    Args:
        fig: plotly figure to add the trace within.
        cameras: the Cameras object to render. It can be batched.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of subplots per row.
        camera_scale: the size of the wireframe used to render the Cameras object.
    """
    cam_wires = get_camera_wireframe(camera_scale).to(poses.device)

    cam_wires[:, 1] *= -1
    if color is None:
        colors = colormap_to_rgb_strings(
            list(range(len(poses))), colormap_name=colormap_name
        )
    else:
        colors = [color] * len(poses)
    cam_points_world_all = []
    for i, (pose, color) in enumerate(zip(poses, colors)):
        R = pose[:3, :3]
        t = pose[:3, -1]
        cam_points_world = cam_wires @ R.T + t.unsqueeze(0)  # (cam_points @ R) # + t)
        cam_points_world_all.append(cam_points_world)
    cam_wires_trans = torch.stack(cam_points_world_all, dim=0)
    if len(cam_wires_trans.shape) < 3:
        cam_wires_trans = cam_wires_trans.unsqueeze(0)

    row, col = subplot_idx // ncols + 1, subplot_idx % ncols + 1
    if separate_traces:
        for i, (cam_wires, color) in enumerate(zip(cam_wires_trans, colors)):
            x, y, z = cam_wires.detach().cpu().numpy().T.astype(float)

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    marker={
                        "size": 1,
                        "color": color,
                    },
                    line=dict(
                        width=linewidth,
                        color=color,
                    ),
                    name=f"{trace_name}-{i}",
                )
            )
    else:
        x, y, z = _batch_points_for_plotly(cam_wires_trans)
        fig.add_trace(
            go.Scatter3d(x=x, y=y, z=z, marker={"size": 1}, name=trace_name),
            row=row,
            col=col,
        )

    # Access the current subplot's scene configuration
    _plot_scene = "scene" + str(subplot_idx + 1)
    current_layout = fig["layout"][_plot_scene]

    # flatten for bounds calculations
    flattened_wires = cam_wires_trans.flatten(0, 1)
    verts_center = flattened_wires.mean(0)
    max_expand = (flattened_wires.max(0)[0] - flattened_wires.min(0)[0]).max()
    _update_axes_bounds(verts_center, max_expand, current_layout)
