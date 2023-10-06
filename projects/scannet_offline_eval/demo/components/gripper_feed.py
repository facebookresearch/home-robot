# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import base64
from dataclasses import dataclass
from typing import Dict, Optional

import dash
import dash_bootstrap_components as dbc
import openai
from dash.exceptions import PreventUpdate
from dash_extensions import WebSocket
from dash_extensions.enrich import DashProxy, Input, Output, State, dcc, html
from loguru import logger

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

    def make_html_element_and_setup_callbacks(self):
        @self.app.callback(
            [Output(self.image_id, "src")],
            [Input(self.trigger_id, "n_intervals")],
            #   [Input("gripper-feed-interval", "n_intervals")],
            blocking=False,
        )
        def update_gripper_feed(n_intervals):
            if svm_watcher.rgb_jpeg is None:
                raise PreventUpdate
            logger.debug(f"Updating gripper feed image {svm_watcher.rgb_jpeg.shape}")
            return f"data:image/jpeg;base64, {base64.b64encode(svm_watcher.rgb_jpeg).decode()}"

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


def make_feed(source_cfg):
    trigger_element = source_cfg.make_html_element_and_setup_callbacks()

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
                    html.Img(
                        src=app.get_asset_url("images/stream_paused_4_3.jpg"),
                        # style={"width": '95%'},  # "float": "left",
                        id=source_cfg.image_id,
                        className="gripper-img img-fluid",
                    ),
                    trigger_element,
                ],
                md=12,
                className=" gripper-feed",
            ),
        ]
    )
