# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
from collections import Counter
from dataclasses import dataclass
from functools import cached_property

import dash
import dash_bootstrap_components as dbc
import plotly.colors as clrs
import plotly.graph_objects as go
import torch
from dash import Patch
from dash.exceptions import PreventUpdate

# from dash.dependencies import Input, Output, State
from dash_extensions.enrich import Input, Output, State, ctx, dcc, html
from loguru import logger
from pytorch3d.vis.plotly_vis import get_camera_wireframe
from torch_geometric.nn.pool.voxel_grid import voxel_grid

from home_robot.utils.bboxes_3d_plotly import get_bbox_wireframe
from home_robot.utils.point_cloud_torch import get_bounds

from .app import app, app_config, svm_watcher
from .core import DashComponent


@dataclass
class Realtime3dComponentConfig:
    figure: go.Figure
    update_on_start: bool = False
    update_frequency_ms: int = 3000
    update_camera_coord_frequence_ms: int = 0


class Realtime3dComponent(DashComponent):
    def __init__(self, name, config: Realtime3dComponentConfig):
        super().__init__(name)
        self.config = config
        camera = dict(eye=dict(x=0.0, y=0.0, z=app_config.camera_initial_distance))
        self.config.figure.update_layout(
            autosize=True,
            scene_camera=camera,
            # margin=dict(
            #     l=20,
            #     r=20,
            #     b=20,
            #     t=20,
            # ),
        )

    @cached_property
    def layout(self):
        children = [
            dcc.Graph(
                figure=self.config.figure,
                id=f"{self.name}-fig",
                style={"height": "90vh", "width": "100%"},
            ),
            dcc.Interval(
                id=f"{self.name}-interval",
                interval=int(self.config.update_frequency_ms),  # in milliseconds,
                disabled=not self.config.update_on_start,
            ),
            dcc.Store(id=f"{self.name}-obs-count", data=0),
            dcc.Store(id=f"{self.name}-fig-names", data=["Points"]),
            dcc.Store(id=f"{self.name}-camera-coords"),
        ]
        if self.config.update_camera_coord_frequence_ms > 0:
            children.extend(
                [
                    dcc.Interval(
                        id=f"{self.name}-camera-interval",
                        interval=int(
                            self.config.update_camera_coord_frequence_ms
                        ),  # in milliseconds,
                        disabled=False,
                    ),
                    dcc.Store(id=f"{self.name}-camera-count", data=-1),
                ]
            )
        return dbc.Container(children)

    def register_callbacks(self, app):
        app.callback(
            [
                Output(f"{self.name}-fig", "figure"),
                Output(f"{self.name}-obs-count", "data"),
                Output(f"{self.name}-fig-names", "data"),
            ],
            #   [Input('get-new-data-3d', 'n_clicks')],
            [
                Input(f"{self.name}-interval", "n_intervals"),
                Input(f"{self.name}-camera-coords", "data"),
            ],
            [
                State(f"{self.name}-fig-names", "data"),
                State(f"{self.name}-obs-count", "data"),
            ],
            blocking=False,
        )(self.add_new_points)

        if self.config.update_camera_coord_frequence_ms > 0:
            app.callback(
                [
                    Output(f"{self.name}-camera-coords", "data"),
                    Output(f"{self.name}-camera-count", "data"),
                ],
                # [Output(self.image_id, "src")],
                [Input(f"{self.name}-camera-interval", "n_intervals")],
                [
                    State(f"{self.name}-fig-names", "data"),
                    State(f"{self.name}-camera-count", "data"),
                ],
                #   [Input("gripper-feed-interval", "n_intervals")],
                blocking=False,
                prevent_initial_callback=False,
            )(update_camera_coords_store)
        return self

    def add_new_points(self, n_intervals, camera_coords, existing, next_obs):
        """
        For now,
        existing:
            data:
                0: pointcloud
                1: boxes
                2: path
                3: robot mesh
        """
        new_next_obs = len(svm_watcher.points)

        patched_figure = Patch()
        if (
            ctx.triggered_id == f"{self.name}-camera-coords"
            and get_plot_idx_by_name(existing, "Camera") is not None
        ):
            target_idx = get_plot_idx_by_name(existing, "Camera")
            update_or_create_camera_trace(
                patched_figure,
                target_idx,
                svm_watcher.cam_coords,
                wireframe_scale=0.2,
                linewidth=4,
                color="red",
                name="Camera",
            )
            target_idx = get_plot_idx_by_name(existing, "Target object")
            nan_tensor = torch.Tensor([[float("NaN")] * 3])

            target_box_idx = svm_watcher.target_instance_id
            target_box_bounds = (
                svm_watcher.box_bounds[new_next_obs - 1][target_box_idx]
                if target_box_idx is not None
                else nan_tensor.unsqueeze(-1).expand(1, 3, 2).squeeze(0)
            )
            update_or_create_box_trace(
                patched_figure,
                # existing,
                target_idx,
                box_bounds=target_box_bounds,
                box_name="Target object",
                linewidth=app_config.target_box_width,
                color="lime",
                mode="lines",
                linestyle="solid",
            )
            return patched_figure, next_obs, existing
        elif next_obs >= new_next_obs:
            logger.debug(
                f"[NOOP] Client has current obs (client: {next_obs}, server: {new_next_obs})"
            )
            raise PreventUpdate

        points = svm_watcher.points[next_obs:new_next_obs]
        rgb = svm_watcher.rgb[next_obs:new_next_obs]
        global_bounds = svm_watcher.bounds[new_next_obs - 1]
        box_bounds = svm_watcher.box_bounds[
            new_next_obs - 1
        ]  # For some reason doing this causes various index and other errors

        # Add new points
        points_idx = get_plot_idx_by_name(existing, "Points")
        if len(points) > 0:
            points = torch.cat(points, dim=0)
            x, y, z = [v.cpu().detach().numpy().tolist() for v in points.unbind(1)]
            points_trace = patched_figure["data"][points_idx]
            points_trace["x"].extend(x)
            points_trace["y"].extend(y)
            points_trace["z"].extend(z)

            rgb = torch.cat(rgb, dim=0).cpu().detach().numpy()
            rgb = [clrs.label_rgb(clrs.convert_to_RGB_255(c)) for c in rgb]
            points_trace["marker"]["color"].extend(rgb)
        trace_names = ["Points"]

        # Camera
        target_idx = get_plot_idx_by_name(existing, "Camera")
        if target_idx is None:
            update_or_create_camera_trace(
                patched_figure,
                target_idx,
                svm_watcher.cam_coords,
                wireframe_scale=0.2,
                linewidth=4,
                color="red",
                name="Camera",
            )
        trace_names += ["Camera"]

        # Add target box
        target_idx = get_plot_idx_by_name(existing, "Target object")
        nan_tensor = torch.Tensor([[float("NaN")] * 3])

        target_box_idx = svm_watcher.target_instance_id
        target_box_bounds = (
            svm_watcher.box_bounds[new_next_obs - 1][target_box_idx]
            if target_box_idx is not None
            else nan_tensor.unsqueeze(-1).expand(1, 3, 2).squeeze(0)
        )
        # logger.info(f"Target box idx: {target_box_idx}")
        update_or_create_box_trace(
            patched_figure,
            # existing,
            target_idx,
            box_bounds=target_box_bounds,
            box_name="Target object",
            linewidth=app_config.target_box_width,
            color="lime",
            mode="lines",
            linestyle="solid",
        )
        trace_names += ["Target object"]

        # # Add boxes
        # boxes_idx = get_plot_idx_by_name(existing["data"], "IB")
        # update_combined_box_trace(patched_figure['data'][boxes_idx], new_next_obs-1)
        # n_boxes = len(svm_watcher.box_names[new_next_obs-1])

        # Add boxes
        box_names = create_separate_box_traces(
            patched_figure, existing, len(trace_names), new_next_obs - 1
        )
        trace_names += box_names
        patched_figure["layout"]["annotations"][0][
            "text"
        ] = f"STATUS: {svm_watcher.robot_state}"
        # ] = f"STATUS: {box_names[target_box_idx]}"

        # Update bounds
        mins, maxs = global_bounds.unbind(-1)
        box_mins = svm_watcher.box_bounds[new_next_obs - 1].min(dim=0).values[..., 0]
        box_maxs = svm_watcher.box_bounds[new_next_obs - 1].max(dim=0).values[..., 1]
        mins, maxs = torch.min(mins, box_mins), torch.max(maxs, box_maxs)
        cam_coords_cat = torch.stack(
            [torch.tensor(svm_watcher.cam_coords[axis]) for axis in ["x", "y", "z"]],
            dim=0,
        )
        cam_mins = cam_coords_cat.min(dim=-1).values
        cam_maxs = cam_coords_cat.max(dim=-1).values
        mins, maxs = torch.min(mins, cam_mins), torch.max(maxs, cam_maxs)
        new_global_bounds = torch.stack([mins, maxs], dim=-1)
        maxlen = (maxs - mins).max().item()
        update_axis(
            maxlen, new_global_bounds[0], patched_figure["layout"]["scene"]["xaxis"]
        )
        update_axis(
            maxlen, new_global_bounds[1], patched_figure["layout"]["scene"]["yaxis"]
        )
        update_axis(
            maxlen, new_global_bounds[2], patched_figure["layout"]["scene"]["zaxis"]
        )
        patched_figure["layout"]["scene"]["aspectmode"] = "data"
        patched_figure["layout"]["uirevision"] = True

        logger.debug(
            f"[UPDATING CLIENT] obs {next_obs=} -> {new_next_obs=} (sending {len(points)} points & {len(box_names)} boxes)"
        )
        return [patched_figure, new_next_obs, trace_names]


def update_camera_coords_store(n_intervals, trace_names, count):
    new_count = svm_watcher.current_obs_number
    if (
        svm_watcher.rgb_jpeg is None
        or "Camera" not in trace_names  # is None
        or count >= new_count
    ):
        raise PreventUpdate
    return [
        svm_watcher.cam_coords,
        new_count,
    ]


def get_plot_idx_by_name(data, name: str) -> int:
    for i, trace in enumerate(data):
        if trace == name:
            return i
    return None


def update_axis(final_length, axis_range, axis_dict):
    axis_dict["range"] = [axis_range[0].item(), axis_range[1].item()]
    axis_dict["nticks"] = int(math.ceil(axis_range[1] - axis_range[0]))
    axis_dict["type"] = "scatter"


def update_combined_box_trace(patched_trace, obs_idx):
    box_names = svm_watcher.box_names[obs_idx]
    all_box_wires = get_bbox_wireframe(
        svm_watcher.box_bounds[obs_idx], add_cross_face_bars=False
    )
    if len(box_names) > 0:
        logger.info([svm_watcher._vocab[class_idx.item()] for class_idx in box_names])

    all_box_wires = all_box_wires.detach().cpu()
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

    box_x, box_y, box_z = box_wires_padded.detach().cpu().numpy().T.astype(float)
    patched_trace["x"] = box_x.tolist()
    patched_trace["y"] = box_y.tolist()
    patched_trace["z"] = box_z.tolist()


def update_or_create_box_trace(
    patched_figure,
    target_idx,
    box_bounds,
    box_name,
    linewidth=3,
    color=None,
    mode="lines+text",
    linestyle="solid",
    **scatter_kwargs,
):
    target_box_wires = get_bbox_wireframe(
        box_bounds.unsqueeze(0), add_cross_face_bars=False
    )[0]
    t_box_x, t_box_y, t_box_z = target_box_wires.detach().cpu().numpy().T.astype(float)

    if target_idx is None:
        text = [""] * (len(t_box_x))
        text[
            7  # Magic number -- we want to add labels _above_ a corner of the box. So we display text above a vertex that's on top
        ] = box_name
        patched_figure["data"].append(
            go.Scatter3d(
                x=t_box_x.tolist(),
                y=t_box_y.tolist(),
                z=t_box_z.tolist(),
                mode=mode,
                marker={
                    "size": 1,
                    "color": color,
                },
                text=text,
                textposition="top right",
                line=dict(width=linewidth, color=color, dash=linestyle),
                name=box_name,
            ),
        )
    else:
        patched_figure["data"][target_idx]["x"] = t_box_x.tolist()
        patched_figure["data"][target_idx]["y"] = t_box_y.tolist()
        patched_figure["data"][target_idx]["z"] = t_box_z.tolist()


def create_separate_box_traces(
    patched_figure, existing_figure, trace_start_idx, obs_idx
):
    counts = Counter()
    box_class_names = []
    for box_idx in range(len(svm_watcher.box_names[obs_idx])):
        target_idx = trace_start_idx + box_idx
        if len(existing_figure) <= target_idx:
            target_idx = None
        box_class = svm_watcher.box_names[obs_idx][box_idx].item()
        box_class_name = svm_watcher._vocab[box_class]
        counts[box_class] += 1
        box_name = f"{box_class_name}-{counts[box_class]}"
        box_class_names.append(box_name)
        update_or_create_box_trace(
            patched_figure=patched_figure,
            target_idx=target_idx,
            box_bounds=svm_watcher.box_bounds[obs_idx][box_idx],
            box_name=f"{box_class_name}-{counts[box_class]}",
        )
    return box_class_names


def update_or_create_camera_trace(
    patched_figure,
    trace_idx,
    cam_coords,
    wireframe_scale=0.5,
    linewidth=3,
    color="red",
    name="Camera",
):
    x, y, z = cam_coords["x"], cam_coords["y"], cam_coords["z"]
    if trace_idx is None:
        patched_figure["data"].append(
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
                name=name,
            )
        )
    else:
        patched_figure["data"][trace_idx]["x"] = x
        patched_figure["data"][trace_idx]["y"] = y
        patched_figure["data"][trace_idx]["z"] = z
