import logging
import os
import time
from datetime import datetime
from textwrap import dedent

import dash
import dash_bootstrap_components as dbc
import openai
from components.app import app, svm_watcher
from components.chat import make_layout as make_chat_layout
from components.index_layout import make_header_layout
from components.realtime_3d import make_layout as make_realtime_3d_layout
from dash import Patch, dcc, html
from dash.dependencies import Input, Output, State
from loguru import logger

os.environ[
    "LOGURU_FORMAT"
] = "| <level>{level: <8}</level> |<cyan>{name:^45}</cyan>|<level>{function:^22}</level>| <cyan>{line:<3} |</cyan> - <level>{message}</level> <lg>@ [{time:YYYY-MM-DD HH:mm:ss.SSS}]</lg> "
# LOGURU_FORMAT=

# Authentication
POINTCLOUD_UPDATE_FREQ_MS = 2000

app.layout = dbc.Container(
    children=[
        make_header_layout(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        make_realtime_3d_layout(
                            svm_watcher.svm.show(backend="pytorch3d", mock_plot=True),
                            POINTCLOUD_UPDATE_FREQ_MS,
                        )
                    ],
                    width=9,
                ),
                dbc.Col(
                    children=[make_chat_layout()],
                    width=3,
                ),
            ]
        ),
    ],
    fluid=True,
    style={"height": "100vh"},
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

    logging.basicConfig(level=logging.DEBUG)  # , format=LOGURU_FORMAT)
    logger.warning("Starting server. Data consumer is currently paused:")
    svm_watcher.pause()
    svm_watcher.begin()
    app.run_server(debug=True, port=args.port)
