import dash
import dash_bootstrap_components as dbc
import plotly.colors as clrs
import torch
from dash import Patch, dcc, html
from dash.dependencies import Input, Output, State

from home_robot.mapping.voxel.voxel import SparseVoxelMap

from .app import app

svm = torch.load(
    "/private/home/ssax/home-robot/projects/scannet_offline_eval/canned_scannet_scene.pth"
)
pointcloud = svm.voxel_map.show(backend="pytorch3d", pointcloud_max_points=100)

n_points_to_add = 1000
POINTCLOUD_UPDATE_FREQ_MS = 1500


def make_layout():
    return dbc.Container(
        [
            dcc.Graph(
                figure=pointcloud,
                id="vis-3d",
                style={"height": "90vh"},
            ),
            dcc.Interval(
                id="viz3d-interval",
                interval=int(POINTCLOUD_UPDATE_FREQ_MS),  # in milliseconds,
                disabled=True,
            ),
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


@app.callback(
    Output("viz3d-interval", "disabled"),
    #   [Input('get-new-data-3d', 'n_clicks')],
    [Input("viz3d-interval", "n_intervals")],
    [State("viz3d-interval", "disabled")],
)
def disable_streaming_on_mousepress():
    pass


@app.callback(
    Output("vis-3d", "figure"),
    #   [Input('get-new-data-3d', 'n_clicks')],
    [Input("viz3d-interval", "n_intervals")],
    [State("vis-3d", "figure")],
)
def add_new_points(submit_n_clicks, existing):
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
    # print(existing['layout']['scene']['camera'])
    points_idx = get_plot_idx_by_name(existing["data"], "Points")
    if points_idx is None:
        raise ValueError("Unknown trace 'Points'")

    if submit_n_clicks is None:
        submit_n_clicks = 0

    patched_figure = Patch()

    points_trace = existing["data"][points_idx]
    start_points = min(
        (n_points_to_add + 1) * submit_n_clicks, len(svm.voxel_map.voxel_pcd._points)
    )
    end_points = min(
        start_points + n_points_to_add, len(svm.voxel_map.voxel_pcd._points)
    )

    points = svm.voxel_map.voxel_pcd._points[start_points:end_points]
    rgb = svm.voxel_map.voxel_pcd._rgb[start_points:end_points].cpu().detach().numpy()
    rgb = [clrs.label_rgb(clrs.convert_to_RGB_255(c)) for c in rgb]

    x, y, z = [v.cpu().detach().numpy().tolist() for v in points.unbind(1)]
    # print(len(existing['data'][points_idx]['x']))
    patched_figure["data"][points_idx]["x"].extend(x)
    patched_figure["data"][points_idx]["y"].extend(y)
    patched_figure["data"][points_idx]["z"].extend(z)
    patched_figure["data"][points_idx]["marker"]["color"].extend(rgb)
    return patched_figure
    # return {'x': [x], 'y': [y], 'z': [z], 'marker.color': [rgb]}, [points_idx]
