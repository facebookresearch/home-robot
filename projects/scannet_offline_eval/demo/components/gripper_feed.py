# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import base64
import math
from dataclasses import dataclass
from typing import Dict, Optional

import dash
import dash_bootstrap_components as dbc
import openai
import plotly.graph_objects as go
from dash import Patch
from dash.exceptions import PreventUpdate
from dash_extensions import WebSocket
from dash_extensions.enrich import DashProxy, Input, Output, State, dcc, html
from loguru import logger
from pytorch3d.vis.plotly_vis import get_camera_wireframe

from .app import SparseVoxelMapDirectoryWatcher, app, svm_watcher


@dataclass
class SourceConfig:
    # Instead of polling the server, we might prefer just establishing either
    #   - a websocket connecton (Quart: https://www.dash-extensions.com/components/websocket)
    #   - server-side events (Starlette: https://www.dash-extensions.com/components/event_source)
    # For this we will need to start a Quart/Starlette server in another process, and that server
    # will push updates to the client. We can tell the client what to do with the resulting data using
    # the client-side callback, as below;
    # Update div using websocket.
    image_id: str

    def make_html_element_and_setup_callbacks(self):
        raise NotImplementedError


@dataclass
class WebsocketSourceConfig(SourceConfig):
    """
    Connects to an existing webserver at websocket_url.
      - Creates a dash_extensions.Websocket element
      - Creates a callback to update Img based on websocket
    """

    app: DashProxy
    websocket_url: str
    trigger_id: str

    def make_html_element_and_setup_callbacks(self):
        self.app.clientside_callback(
            "function(m){return m? m.data : '';}",
            Output(self.image_id, "src"),
            [Input(self.trigger_id, "message")],
        )
        return WebSocket(id=self.trigger_id, url=self.websocket_url)


@dataclass
class ClientRequestSourceConfig(SourceConfig):
    """
    Connects to an existing webserver at
    """

    app: DashProxy
    trigger_id: str
    svm_watcher: SparseVoxelMapDirectoryWatcher
    trigger_exists: bool
    trigger_interval_kwargs: Optional[Dict] = None
    svm_watcher_attr: str = "obstacle"

    def make_html_element_and_setup_callbacks_with_cam_coords(self):
        @self.app.callback(
            [
                Output(self.image_id, "src"),
                Output("realtime-3d-camera-coords", "data"),
                Output(f"{self.image_id}-count", "data"),
            ],
            # [Output(self.image_id, "src")],
            [Input(self.trigger_id, "n_intervals")],
            [
                State("realtime-3d-fig-names", "data"),
                State(f"{self.image_id}-count", "data"),
            ],
            #   [Input("gripper-feed-interval", "n_intervals")],
            blocking=False,
            prevent_initial_callback=False,
        )
        def update_gripper_feed(n_intervals, trace_names, count):
            if count == None:
                count = -1
            new_count = svm_watcher.current_obs_number
            if (
                svm_watcher.rgb_jpeg is None
                or trace_names is None
                or count >= new_count
            ):
                raise PreventUpdate

            # logger.debug(f"Updating gripper feed image {svm_watcher.rgb_jpeg.shape}")
            return [
                f"data:image/jpeg;base64, {base64.b64encode(svm_watcher.rgb_jpeg).decode()}",
                svm_watcher.cam_coords,
                new_count,
            ]
            # return f"data:image/jpeg;base64, {base64.b64encode(svm_watcher.rgb_jpeg).decode()}"

        if not self.trigger_exists:
            if self.trigger_interval_kwargs is None:
                self.trigger_interval_kwargs = {}
            trigger_element = dcc.Interval(
                id=self.trigger_id,
                **self.trigger_interval_kwargs,
                # interval=int(update_frequency_ms),  # in milliseconds,
                # disabled=True,
            )
            return trigger_element

    def make_html_element_and_setup_callbacks(self):
        @self.app.callback(
            [
                Output(self.image_id, "src"),
                Output(f"{self.image_id}-count", "data"),
            ],
            # [Output(self.image_id, "src")],
            [Input(self.trigger_id, "n_intervals")],
            [
                State("realtime-3d-fig-names", "data"),
                State(f"{self.image_id}-count", "data"),
            ],
            blocking=False,
            prevent_initial_callback=False,
        )
        def update_gripper_feed(n_intervals, trace_names, count):
            if count == None:
                count = -1
            new_count = svm_watcher.current_obs_number
            if (
                getattr(svm_watcher, self.svm_watcher_attr) is None
                or trace_names is None
                or count >= new_count
            ):
                raise PreventUpdate

            return [
                f"data:image/jpeg;base64, {base64.b64encode(getattr(svm_watcher, self.svm_watcher_attr)).decode()}",
                new_count,
            ]

        if not self.trigger_exists:
            if self.trigger_interval_kwargs is None:
                self.trigger_interval_kwargs = {}
            trigger_element = dcc.Interval(
                id=self.trigger_id,
                **self.trigger_interval_kwargs,
                # interval=int(update_frequency_ms),  # in milliseconds,
                # disabled=True,
            )
            return trigger_element


@dataclass
class ServerSideEventSourceConfig(SourceConfig):
    app: DashProxy
    event_url: str
    trigger_id: str

    def make_html_element_and_setup_callbacks(self, app, image_id, trigger_id):
        raise NotImplementedError


def make_feed_with_cam_coord_callback(
    source_cfg, name="Live Gripper Feed", base_css_class="gripper"
):
    trigger_element = source_cfg.make_html_element_and_setup_callbacks_with_cam_coords()

    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(
                        html.H2(
                            [name],
                            className="text-secondary text-center",
                        ),
                    ),
                    html.Img(
                        src=app.get_asset_url("images/stream_paused_1_1_horiz.jpg"),
                        # style={"width": '95%'},  # "float": "left,
                        id=source_cfg.image_id,
                        className=f"{base_css_class}-img img-fluid",
                    ),
                    trigger_element,
                    dcc.Store(id=f"{source_cfg.image_id}-count"),
                ],
                md=12,
                className=f"{base_css_class}-feed",
            ),
        ]
    )


def make_feed(source_cfg, name="Live Gripper Feed", base_css_class="gripper"):
    trigger_element = source_cfg.make_html_element_and_setup_callbacks()

    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(
                        html.H2(
                            [name],
                            className="text-secondary text-center",
                        ),
                    ),
                    html.Img(
                        src=app.get_asset_url("images/stream_paused_1_1_horiz.jpg"),
                        # style={"width": '95%'},  # "float": "left,
                        id=source_cfg.image_id,
                        className=f"{base_css_class}-img img-fluid",
                    ),
                    trigger_element,
                    dcc.Store(id=f"{source_cfg.image_id}-count"),
                ],
                md=12,
                className=f"{base_css_class}-feed",
            ),
        ]
    )
