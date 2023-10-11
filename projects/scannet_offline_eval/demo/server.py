# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import time
from datetime import datetime
from textwrap import dedent

import dash
import dash_bootstrap_components as dbc
import openai
from components.app import app, app_config, svm_watcher
from components.chat import make_layout as make_chat_layout
from components.gripper_feed import (
    ClientRequestSourceConfig,
    WebsocketSourceConfig,
    make_feed,
)
from components.index_layout import make_header_layout
from components.realtime_3d import make_layout as make_realtime_3d_layout
from dash import Patch, dcc, html
from dash.dependencies import Input, Output, State
from loguru import logger

# os.environ[
#     "LOGURU_FORMAT"
# ] = "| <level>{level: <8}</level> |<cyan>{name:^45}</cyan>|<level>{function:^22}</level>| <cyan>{line:<3} |</cyan> - <level>{message}</level> <lg>@ [{time:YYYY-MM-DD HH:mm:ss.SSS}]</lg> "


figure = svm_watcher.svm.show(
    backend="pytorch3d",
    mock_plot=True,
    pointcloud_marker_size=4,  # int(app_config.pointcloud_voxel_size * 100)
)
figure.update_layout(
    autosize=True,
    margin=dict(
        l=20,
        r=20,
        b=20,
        t=20,
    ),
)

app.config.suppress_callback_exceptions = True
app.layout = dbc.Container(
    children=[
        make_header_layout(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        make_realtime_3d_layout(
                            figure,
                            app_config.pointcloud_update_freq_ms,
                        )
                    ],
                    width=8,
                ),
                dbc.Col(
                    children=[
                        make_feed(
                            # # Run publisher_server.py to set up server
                            # WebsocketSourceConfig(
                            #     app=app,
                            #     image_id="gripper-feed-img",
                            #     trigger_id="gripper-feed-ws",
                            #     websocket_url="ws://127.0.0.1:5000/gripper-feed-ws",
                            # ),
                            ClientRequestSourceConfig(
                                app=app,
                                image_id="gripper-feed-img",
                                trigger_id="gripper-feed-interval",  # realtime-3d
                                trigger_exists=False,
                                svm_watcher=svm_watcher,
                                trigger_interval_kwargs=dict(
                                    interval=int(app_config.video_feed_update_freq_ms),
                                    disabled=False,
                                ),
                            ),
                        ),
                        make_chat_layout(),
                    ],
                    width=3,
                ),
            ],
            className="main-body",
        ),
    ],
    fluid=True,
    className="h-100",
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=8901,
        help="The port to use (default: 8901)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger.warning("Starting server. Data consumer is currently paused:")
    svm_watcher.pause()
    svm_watcher.begin()
    # app.run(port=5000)

    app.run_server(debug=True, port=args.port)
