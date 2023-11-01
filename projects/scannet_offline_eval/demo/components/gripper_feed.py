# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import base64
import math
from dataclasses import dataclass
from functools import cached_property
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
from .core import DashComponent


@dataclass
class VideoStreamComponentConfig:
    image_id: str
    trigger_id: str
    trigger_exists: bool
    stream_obj: SparseVoxelMapDirectoryWatcher
    stream_attr: str
    title: str = "Video"
    base_css_class: str = "gripper"


class VideoStreamComponent(DashComponent):
    # Instead of polling the server, we might prefer just establishing either
    #   - a websocket connecton (Quart: https://www.dash-extensions.com/components/websocket)
    #   - server-side events (Starlette: https://www.dash-extensions.com/components/event_source)
    # For this we will need to start a Quart/Starlette server in another process, and that server
    # will push updates to the client. We can tell the client what to do with the resulting data using
    # the client-side callback, as below;
    # Update div using websocket.
    def __init__(self, name, config):
        super().__init__(name)
        self.config = config

    def register_callbacks(self, app):
        self.app = app
        return self

    @cached_property
    def layout(self):
        return dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            html.H2(
                                [self.config.title],
                                className="text-secondary text-center",
                            ),
                        ),
                        html.Img(
                            src=self.app.get_asset_url(
                                "images/stream_paused_1_1_horiz.jpg"
                            ),
                            # style={"width": '95%'},  # "float": "left,
                            id=self.config.image_id,
                            className=f"{self.config.base_css_class}-img img-fluid",
                        ),
                        self.trigger_component,
                        dcc.Store(id=f"{self.config.image_id}-count"),
                    ],
                    md=12,
                    className=f"{self.config.base_css_class}-feed",
                ),
            ]
        )

    @cached_property
    def trigger_component(self):
        raise NotImplementedError


@dataclass
class IntervalVideoStreamComponentConfig(VideoStreamComponentConfig):
    trigger_interval_kwargs: Optional[Dict] = None


@dataclass
class IntervalVideoStreamComponent(VideoStreamComponent):
    def __init__(self, name, config):
        super().__init__(name, config)

    def register_callbacks(self, app):
        super().register_callbacks(app)
        app.callback(
            [
                Output(self.config.image_id, "src"),
                Output(f"{self.config.image_id}-count", "data"),
            ],
            # [Output(self.image_id, "src")],
            [Input(self.config.trigger_id, "n_intervals")],
            [
                State(f"{self.config.image_id}-count", "data"),
            ],
            blocking=False,
            prevent_initial_callback=False,
        )(self.update_gripper_feed)
        return self

    @cached_property
    def trigger_component(self):
        if not self.config.trigger_exists:
            if self.config.trigger_interval_kwargs is None:
                self.config.trigger_interval_kwargs = {}
            trigger_element = dcc.Interval(
                id=self.config.trigger_id,
                **self.config.trigger_interval_kwargs,
            )
            return trigger_element

    def update_gripper_feed(self, n_intervals, count):
        if count == None:
            count = -1
        stream_obj, stream_attr = self.config.stream_obj, self.config.stream_attr
        new_count = stream_obj.current_obs_number
        if getattr(stream_obj, stream_attr) is None or count >= new_count:
            raise PreventUpdate

        return [
            f"data:image/jpeg;base64, {base64.b64encode(getattr(stream_obj, stream_attr)).decode()}",
            new_count,
        ]


####################
# Old setup
####################
@dataclass
class SourceConfig:

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
class ServerSideEventSourceConfig(SourceConfig):
    app: DashProxy
    event_url: str
    trigger_id: str

    def make_html_element_and_setup_callbacks(self, app, image_id, trigger_id):
        raise NotImplementedError


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
