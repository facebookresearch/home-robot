import os
import time
from datetime import datetime
from textwrap import dedent

import dash
import dash_bootstrap_components as dbc
import openai
from components.app import app
from components.chat import make_layout as make_chat_layout
from components.index_layout import make_header_layout
from components.viz3d import make_layout as make_viz3d_layout
from dash import Patch, dcc, html
from dash.dependencies import Input, Output, State

# Authentication


app.layout = dbc.Container(
    children=[
        make_header_layout(),
        dbc.Row(
            [
                dbc.Col(
                    [make_viz3d_layout()],
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
    app.run_server(debug=False, port=8008)
