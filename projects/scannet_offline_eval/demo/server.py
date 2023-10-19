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
from components.app import app, app_config, svm_watcher
from components.chat import Chatbox, ChatBoxConfig
from components.demo_header import DemoHeader  # make_header_layout
from components.gripper_feed import (  # ClientRequestSourceConfig,; WebsocketSourceConfig,; make_feed,; make_feed_with_cam_coord_callback,
    IntervalVideoStreamComponent,
    IntervalVideoStreamComponentConfig,
)
from components.realtime_3d import (  # make_layout as make_realtime_3d_layout
    Realtime3dComponent,
    Realtime3dComponentConfig,
)
from dash import Patch, dcc, html
from dash.dependencies import Input, Output, State
from loguru import logger

# os.environ[
#     "LOGURU_FORMAT"
# ] = "| <level>{level: <8}</level> |<cyan>{name:^45}</cyan>|<level>{function:^22}</level>| <cyan>{line:<3} |</cyan> - <level>{message}</level> <lg>@ [{time:YYYY-MM-DD HH:mm:ss.SSS}]</lg> "

##################
# Make components
##################
# Header
header = DemoHeader(name="demo-header")

# RGB
rgb_config = IntervalVideoStreamComponentConfig(
    title="Egocentric Camera [RGBD]",
    image_id="rgb-img",
    trigger_id="gripper-feed-interval",
    trigger_exists=False,
    stream_obj=svm_watcher,
    stream_attr="rgb_jpeg",
    base_css_class="ego",
    trigger_interval_kwargs=dict(
        interval=int(app_config.video_feed_update_freq_ms),
        disabled=False,
    ),
)
rgb_component = IntervalVideoStreamComponent(name="rgb-img", config=rgb_config)

# Depth
depth_config = IntervalVideoStreamComponentConfig(
    title="",
    image_id="depth-img",
    trigger_id="gripper-feed-interval",
    trigger_exists=True,
    stream_obj=svm_watcher,
    stream_attr="depth_jpeg",
    base_css_class="ego",
)
depth_component = IntervalVideoStreamComponent(name="depth-img", config=depth_config)

# Instance Map figure
r3d_config = Realtime3dComponentConfig(
    figure=svm_watcher.svm.show(
        backend="pytorch3d",
        instances=False,
        mock_plot=True,
        pointcloud_marker_size=2,  # int(app_config.pointcloud_voxel_size * 100)
    ),
    update_on_start=not app_config.start_paused,
    update_frequency_ms=app_config.pointcloud_update_freq_ms,
    update_camera_coord_frequence_ms=app_config.video_feed_update_freq_ms,
)
r3d_component = Realtime3dComponent(name="realtime-3d", config=r3d_config)

# Map
map_config = IntervalVideoStreamComponentConfig(
    title="Birds-Eye Obstacle Map",
    image_id="map-img",
    trigger_id="gripper-feed-interval",
    trigger_exists=True,
    stream_obj=svm_watcher,
    stream_attr="map_im",
    base_css_class="map",
)
map_component = IntervalVideoStreamComponent(name="map-img", config=map_config)

# Chatbox
chatbox_config = ChatBoxConfig(
    chat_log_fpath=Path(os.path.dirname(app_config.directory_watch_path))
    / "demo_chat.json"
)
chatbox = Chatbox(chatbox_config, name="chatbox")

######################


##############
# App layout
##############
# app.config.suppress_callback_exceptions = True
app.layout = dbc.Container(
    children=[
        header.register_callbacks(app).layout,
        dbc.Row(
            [
                dbc.Col(
                    [
                        rgb_component.register_callbacks(app).layout,
                        depth_component.register_callbacks(app).layout,
                    ],
                    width=2,
                ),
                dbc.Col(
                    [r3d_component.register_callbacks(app).layout],
                    width=7,
                ),
                dbc.Col(
                    children=[
                        map_component.register_callbacks(app).layout,
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
##############


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
    if app_config.start_paused:
        svm_watcher.pause()
    svm_watcher.begin()
    # app.run(port=5000)

    app.run_server(debug=True, port=args.port)
