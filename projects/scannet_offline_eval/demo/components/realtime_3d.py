import logging

import dash
import dash_bootstrap_components as dbc
import plotly.colors as clrs
import torch
from dash import Patch, dcc, html
from dash.dependencies import Input, Output, State

from .app import app, svm_watcher

n_points_to_add = 1000
from loguru import logger
from torch_geometric.nn.pool.voxel_grid import voxel_grid

from home_robot.utils.bboxes_3d_plotly import get_bbox_wireframe
from home_robot.utils.point_cloud_torch import get_bounds


def make_layout(figure, update_frequency_ms):
    return dbc.Container(
        [
            dcc.Graph(
                figure=figure,
                id="realtime-3d-fig",
                style={"height": "90vh"},
            ),
            dcc.Interval(
                id="realtime-3d-interval",
                interval=int(update_frequency_ms),  # in milliseconds,
                disabled=True,
            ),
            dcc.Store(id="realtime-3d-obs-count"),
        ]
    )


def get_plot_idx_by_name(data, name: str) -> int:
    for i, trace in enumerate(data):
        if trace["name"] == name:
            return i
    return None


# @app.callback(Output('vis-3d', 'extendData'),
#             #   [Input('get-new-data-3d', 'n_clicks')],
#               [Input("viz3d-interval", "n_intervals")],
#               [State('vis-3d', 'figure')])
# def add_new_points(submit_n_clicks, existing):
#     """
#         Unfortunately extendData can only modify the points -- it doesn't allow updating of the marker colors
#         which we would need in order to show the colored pointcloud

#         So instead what we will need to do is sotre the figure data client-side in a Store object
#             and periodically update that store data.
#         Then trigger the client to update the figure on changes to that Store object.
#          [example here: https://community.plotly.com/t/clientside-callback-to-filter-data-and-update-graph/66861 ]

#         For now,
#         existing:
#             data:
#                 0: pointcloud
#                 1: boxes
#                 2: path
#                 3: robot mesh
#     """
#     print(existing['layout']['scene']['camera'])
#     points_idx = get_plot_idx_by_name(existing['data'], 'Points')
#     if points_idx is None:
#         raise ValueError("Unknown trace 'Points'")

#     if submit_n_clicks is None:
#         submit_n_clicks = 0
#     points_trace = existing['data'][points_idx]
#     start_points = min((n_points_to_add + 1) * submit_n_clicks, len(svm.voxel_map.voxel_pcd._points))
#     end_points = min(start_points + n_points_to_add, len(svm.voxel_map.voxel_pcd._points))

#     points = svm.voxel_map.voxel_pcd._points[start_points:end_points]
#     rgb = svm.voxel_map.voxel_pcd._rgb[start_points:end_points].cpu().detach().numpy()
#     rgb = [clrs.label_rgb(clrs.convert_to_RGB_255(c)) for c in rgb]

#     x, y, z = [v.cpu().detach().numpy().tolist() for v in points.unbind(1)]
#     return {'x': [x], 'y': [y], 'z': [z], 'marker.color': [rgb]}, [points_idx]


# svm = torch.load(
#     "/private/home/ssax/home-robot/projects/scannet_offline_eval/canned_scannet_scene.pth"
# )
# pointcloud = svm.voxel_map.show(backend="pytorch3d", pointcloud_max_points=100)

# @app.callback(
#     Output("realtime-3d-fig", "figure"),
#     #   [Input('get-new-data-3d', 'n_clicks')],
#     [Input("realtime-3d-interval", "n_intervals")],
#     [State("realtime-3d-fig", "figure")],
# )
# def add_new_points(submit_n_clicks, existing):
#     """
#     Unfortunately extendData can only modify the points -- it doesn't allow updating of the marker colors
#     which we would need in order to show the colored pointcloud

#     So instead what we will need to do is sotre the figure data client-side in a Store object
#         and periodically update that store data.
#     Then trigger the client to update the figure on changes to that Store object.
#      [example here: https://community.plotly.com/t/clientside-callback-to-filter-data-and-update-graph/66861 ]

#     For now,
#     existing:
#         data:
#             0: pointcloud
#             1: boxes
#             2: path
#             3: robot mesh
#     """
#     # print(existing['layout']['scene']['camera'])
#     points_idx = get_plot_idx_by_name(existing["data"], "Points")
#     if points_idx is None:
#         raise ValueError("Unknown trace 'Points'")

#     if submit_n_clicks is None:
#         submit_n_clicks = 0

#     patched_figure = Patch()

#     points_trace = existing["data"][points_idx]
#     start_points = min(
#         (n_points_to_add) * submit_n_clicks, len(svm.voxel_map.voxel_pcd._points)
#     )
#     end_points = min(
#         start_points + n_points_to_add, len(svm.voxel_map.voxel_pcd._points)
#     )

#     points = svm.voxel_map.voxel_pcd._points[start_points:end_points]
#     rgb = svm.voxel_map.voxel_pcd._rgb[start_points:end_points].cpu().detach().numpy()
#     rgb = [clrs.label_rgb(clrs.convert_to_RGB_255(c)) for c in rgb]

#     x, y, z = [v.cpu().detach().numpy().tolist() for v in points.unbind(1)]
#     # print(len(existing['data'][points_idx]['x']))
#     patched_figure["data"][points_idx]["x"].extend(x)
#     patched_figure["data"][points_idx]["y"].extend(y)
#     patched_figure["data"][points_idx]["z"].extend(z)
#     patched_figure["data"][points_idx]["marker"]["color"].extend(rgb)

#     # update the bounds of the axes for the current trace
#     mins = svm.voxel_map.voxel_pcd._points[:end_points].min(dim=0).values
#     maxs = svm.voxel_map.voxel_pcd._points[:end_points].max(dim=0).values
#     patched_figure["layout"]['scene']["xaxis"]['range'] = [mins[0].item(), maxs[0].item()]
#     patched_figure["layout"]['scene']["xaxis"]['type'] = 'scatter'

#     patched_figure["layout"]['scene']["yaxis"]['range'] = [mins[1].item(), maxs[1].item()]
#     patched_figure["layout"]['scene']["yaxis"]['type'] = 'scatter'

#     patched_figure["layout"]['scene']["zaxis"]['range'] = [mins[2].item(), maxs[2].item()]
#     patched_figure["layout"]['scene']["zaxis"]['type'] = 'scatter'
#     import pprint
#     pp = pprint.PrettyPrinter(width=80, compact=True)
#     pp.pprint(existing['layout']['scene'])
#     return patched_figure
#     # return {'x': [x], 'y': [y], 'z': [z], 'marker.color': [rgb]}, [points_idx]


# @app.callback(
#     Output("realtime-3d-interval", "disabled"),
#     #   [Input('get-new-data-3d', 'n_clicks')],
#     [Input("realtime-3d-interval", "n_intervals")],
#     [State("realtime-3d-interval", "disabled")],
# )
# def disable_streaming_on_mousepress():
#     pass


@app.callback(
    [Output("realtime-3d-fig", "figure"), Output("realtime-3d-obs-count", "data")],
    #   [Input('get-new-data-3d', 'n_clicks')],
    [Input("realtime-3d-interval", "n_intervals")],
    [State("realtime-3d-fig", "figure"), State("realtime-3d-obs-count", "data")],
)
def add_new_points(submit_n_clicks, existing, next_obs):
    """
    Unfortunately extendData can only modify the points -- it doesn't allow updating of the marker colors
    which we would need in order to show the colored pointcloud

    So instead what we will need to do is sotre the figure data client-side in a Store object
        and periodically update that store data.
    Then trigger the client to update the figure on changes to that Store object.
     [example here: https://community.plotly.com/t/clientside-callback-to-filter-data-and-update-graph/66861 ]

    For now,
    existing:
        data:
            0: pointcloud
            1: boxes
            2: path
            3: robot mesh
    """
    new_next_obs = len(svm_watcher.points)
    points_idx = get_plot_idx_by_name(existing["data"], "Points")
    if points_idx is None:
        raise ValueError("Unknown trace 'Points'")

    if submit_n_clicks is None:
        submit_n_clicks = 0

    if next_obs is None:
        next_obs = 0

    patched_figure = Patch()
    if next_obs >= new_next_obs:
        logger.info(f"No new obs beyond {next_obs} (only {new_next_obs} obs)")
        # No new observations
        return [patched_figure, next_obs]

    # Add new points
    points_trace = existing["data"][points_idx]

    points = svm_watcher.points[next_obs:new_next_obs]
    rgb = svm_watcher.rgb[next_obs:new_next_obs]
    bounds = svm_watcher.bounds[new_next_obs - 1]
    box_bounds = svm_watcher.bounds[new_next_obs - 1]

    points = torch.cat(points, dim=0)
    x, y, z = [v.cpu().detach().numpy().tolist() for v in points.unbind(1)]
    patched_figure["data"][points_idx]["x"].extend(x)
    patched_figure["data"][points_idx]["y"].extend(y)
    patched_figure["data"][points_idx]["z"].extend(z)

    rgb = torch.cat(rgb, dim=0).cpu().detach().numpy()
    rgb = [clrs.label_rgb(clrs.convert_to_RGB_255(c)) for c in rgb]

    patched_figure["data"][points_idx]["marker"]["color"].extend(rgb)

    logger.info(f"Adding {len(points)} points from {next_obs=} and {new_next_obs=}")
    # Update bounds
    mins, maxs = bounds.unbind(-1)
    patched_figure["layout"]["scene"]["xaxis"]["range"] = [
        mins[0].item(),
        maxs[0].item(),
    ]
    patched_figure["layout"]["scene"]["xaxis"]["type"] = "scatter"

    patched_figure["layout"]["scene"]["yaxis"]["range"] = [
        mins[1].item(),
        maxs[1].item(),
    ]
    patched_figure["layout"]["scene"]["yaxis"]["type"] = "scatter"

    patched_figure["layout"]["scene"]["zaxis"]["range"] = [
        mins[2].item(),
        maxs[2].item(),
    ]
    patched_figure["layout"]["scene"]["zaxis"]["type"] = "scatter"

    # Add boxes
    boxes_idx = get_plot_idx_by_name(existing["data"], "IB")
    all_box_wires = get_bbox_wireframe(
        svm_watcher.box_bounds[new_next_obs - 1], add_cross_face_bars=False
    )
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
    patched_figure["data"][boxes_idx]["x"] = box_x.tolist()
    patched_figure["data"][boxes_idx]["y"] = box_y.tolist()
    patched_figure["data"][boxes_idx]["z"] = box_z.tolist()
    logger.info(f"Now {len(all_box_wires)} boxes")

    # import pprint

    # pp = pprint.PrettyPrinter(width=80, compact=True)
    # pp.pprint(existing["data"][boxes_idx])

    return [patched_figure, new_next_obs]
