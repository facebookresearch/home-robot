# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import cached_property

import dash
import dash_bootstrap_components as dbc
import openai
from dash import Patch, dcc, html
from dash.dependencies import Input, Output, State
from dash_extensions.enrich import DashProxy
from loguru import logger

from .app import app, app_config, svm_watcher
from .core import DashComponent


class DemoHeader(DashComponent):
    def __init__(self, name):
        super().__init__(name)

    def register_callbacks(self, app: DashProxy):
        app.callback(
            Output("stream-counter", "children"),
            [Input("realtime-3d-interval", "n_intervals")],
        )(display_count)
        if app_config.start_paused:
            app.callback(
                [
                    Output("realtime-3d-interval", "disabled"),
                    Output("get-new-data-3d", "children"),
                ],
                [Input("get-new-data-3d", "n_clicks")],
                [
                    State("realtime-3d-interval", "disabled"),
                    State("get-new-data-3d", "children"),
                ],
            )(toggle_interval)
        return self

    @cached_property
    def layout(self):
        stream_counter_children = []
        if app_config.start_paused:
            stream_counter_children.append(
                html.Button(
                    f"Begin streaming",
                    id="get-new-data-3d",
                    n_clicks=0,
                    className="button-primary",
                ),
            )
        stream_counter_children.append(html.P(id="stream-counter"))
        return dbc.Row(
            [
                dbc.Col(
                    children=stream_counter_children,
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Div(
                            children=[
                                html.H1(["Accel Cortex Demo: FAIR Conference"]),
                                # html.Img(
                                #     src=app.get_asset_url("images/VC1-cropped.svg"),
                                # ),
                            ],
                            className="text-primary text-center",
                        )
                    ],
                    md=8,
                ),
            ],
            className="header",
        )


def display_count(n):
    if n is None:
        n = 0
    return f"Interval has fired {n} times"


def toggle_interval(n, disabled, children):
    if n:
        is_now_disabled = not disabled
        children = ["Stop streaming data", "Begin streaming data"][int(is_now_disabled)]
        if is_now_disabled:
            svm_watcher.pause()
        else:
            logger.debug("Unpausing directory watcher...")
            svm_watcher.unpause()

        return [is_now_disabled, children]
    return [disabled, children]
