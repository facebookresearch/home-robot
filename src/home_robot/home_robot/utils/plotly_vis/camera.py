# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from pytorch3d.vis.plotly_vis import AxisArgs, get_camera_wireframe, plot_scene


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
    fig,
    poses,
    linewidth=3,
    color=None,
    name="cam",
    separate=True,
    scale=0.2,
    colormap_name="plasma",
):

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
        # cam_points[:, 2] *= -1
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
