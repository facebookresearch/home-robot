# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import dash
import dash_bootstrap_components as dbc
import openai
from components.app import app, app_config, svm_watcher

# from components.chat_old import make_layout as make_chat_layout
from components.chat import Chatbox, ChatBoxConfig
from components.gripper_feed import (
    ClientRequestSourceConfig,
    WebsocketSourceConfig,
    make_feed,
    make_feed_with_cam_coord_callback,
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
    instances=False,
    mock_plot=True,
    pointcloud_marker_size=3,  # int(app_config.pointcloud_voxel_size * 100)
)
camera = dict(eye=dict(x=0.0, y=0.0, z=app_config.camera_initial_distance))
figure.update_layout(
    autosize=True,
    scene_camera=camera,
    margin=dict(
        l=20,
        r=20,
        b=20,
        t=20,
    ),
)

chatbox_config = ChatBoxConfig(
    chat_log_fpath=Path(os.path.dirname(app_config.directory_watch_path))
    / "demo_chat.json"
)
chatbox = Chatbox(chatbox_config, name="chatbox")

# app.config.suppress_callback_exceptions = True
app.layout = dbc.Container(
    children=[
        make_header_layout(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        make_feed_with_cam_coord_callback(
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
                            name="Egocentric Camera [RGBD]",
                            base_css_class="ego",
                            # name="Live 2D Map",
                        ),
                        make_feed(
                            ClientRequestSourceConfig(
                                app=app,
                                image_id="depth-feed-img",
                                trigger_id="gripper-feed-interval",  # realtime-3d
                                trigger_exists=True,
                                svm_watcher=svm_watcher,
                                svm_watcher_attr="depth_jpeg",
                                # trigger_interval_kwargs=dict(
                                #     interval=int(app_config.video_feed_update_freq_ms) * 2,
                                #     disabled=False,
                                # ),
                            ),
                            name="",
                            base_css_class="ego",
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        make_realtime_3d_layout(
                            figure,
                            app_config.pointcloud_update_freq_ms,
                        )
                    ],
                    width=7,
                ),
                dbc.Col(
                    children=[
                        make_feed(
                            ClientRequestSourceConfig(
                                app=app,
                                image_id="map-2d-img",
                                trigger_id="realtime-3d-interval",  # realtime-3d
                                trigger_exists=True,
                                svm_watcher=svm_watcher,
                                svm_watcher_attr="map_im",
                                # trigger_interval_kwargs=dict(
                                #     interval=int(app_config.video_feed_update_freq_ms) * 2,
                                #     disabled=False,
                                # ),
                            ),
                            name="Birds-Eye Obstacle Map",
                            base_css_class="map",
                        ),
                        # make_chat_layout(),
                        chatbox.register_callbacks(app).layout,
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
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="The path to the directory to watch (default: None)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    if args.path is not None:
        app_config.directory_watch_path = args.path

    logger.warning("Starting server. Data consumer is currently paused:")
    svm_watcher.pause()
    svm_watcher.begin()
    # app.run(port=5000)

    app.run_server(debug=True, port=args.port)
