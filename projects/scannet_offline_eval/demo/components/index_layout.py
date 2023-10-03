# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import dash
import dash_bootstrap_components as dbc
import openai
from dash import Patch, dcc, html
from dash.dependencies import Input, Output, State
from loguru import logger

from .app import app, svm_watcher


def make_header_layout():
    # return dbc.Col([
    return dbc.Row(
        [
            dbc.Col(
                children=[
                    html.Button(
                        f"Begin streaming",
                        id="get-new-data-3d",
                        n_clicks=0,
                        className="button-primary",
                    ),
                    html.P(id="stream-counter"),
                ],
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


@app.callback(
    Output("stream-counter", "children"), [Input("realtime-3d-interval", "n_intervals")]
)
def display_count(n):
    if n is None:
        n = 0
    return f"Interval has fired {n} times"


@app.callback(
    [Output("realtime-3d-interval", "disabled"), Output("get-new-data-3d", "children")],
    [Input("get-new-data-3d", "n_clicks")],
    [State("realtime-3d-interval", "disabled"), State("get-new-data-3d", "children")],
)
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
