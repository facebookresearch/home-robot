# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import base64

import dash
import dash_bootstrap_components as dbc
import openai
from dash.exceptions import PreventUpdate
from dash_extensions import WebSocket
from dash_extensions.enrich import Input, Output, State, dcc, html
from loguru import logger

from .app import app, svm_watcher


def make_feed(height="30vh", update_frequency_ms=500):
    # return dbc.Col([
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(
                        html.H2(
                            ["Live Gripper Feed"],
                            className="text-secondary text-center",
                        ),
                    ),
                    # dbc.Row(
                    #     children=[
                    html.Img(
                        src=app.get_asset_url("images/stream_paused_4_3.jpg"),
                        # style={"width": '95%'},  # "float": "left",
                        id="gripper-feed-img",
                        className="gripper-img img-fluid",
                    ),
                    # WebSocket(id="gripper-feed-ws", url=f"ws://127.0.0.1:8901/gripper-feed-ws"),
                    dcc.Interval(
                        id="gripper-feed-interval",
                        interval=int(update_frequency_ms),  # in milliseconds,
                        disabled=True,
                    ),
                    #     ],
                    # )
                ],
                md=12,
                className=" gripper-feed",
            ),
        ]
    )


# @app.callback(
#     [Output("gripper-feed-img", "src")],
#     [Input("gripper-feed-interval", "n_intervals")],
#     blocking=True,
# )
# def update_gripper_feed(n_intervals):
#     if svm_watcher.rgb_jpeg is None:
#         raise PreventUpdate
#     return f"data:image/jpeg;base64, {base64.b64encode(svm_watcher.rgb_jpeg).decode()}"


@app.callback(
    [Output("gripper-feed-img", "src")],
    [Input("realtime-3d-interval", "n_intervals")],
    blocking=False,
)
def update_gripper_feed(n_intervals):
    if svm_watcher.rgb_jpeg is None:
        raise PreventUpdate
    return f"data:image/jpeg;base64, {base64.b64encode(svm_watcher.rgb_jpeg).decode()}"


# Instead of polling the server, we might prefer just establishing either
#   - a websocket connecton (Quart: https://www.dash-extensions.com/components/websocket)
#   - server-side events (Starlette: https://www.dash-extensions.com/components/event_source)
# For this we will need to start a Quart/Starlette server in another process, and that server
# will push updates to the client. We can tell the client what to do with the resulting data using
# the client-side callback, as below;
# Update div using websocket.
# app.clientside_callback(
#     "function(m){return m? m.data : '';}",
#     Output("gripper-feed-img", "src"),
#     [Input("gripper-feed-ws", "message")]
# )
