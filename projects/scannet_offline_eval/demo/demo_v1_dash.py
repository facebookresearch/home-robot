import os
import time
from datetime import datetime
from textwrap import dedent

import dash
import dash_bootstrap_components as dbc
import openai
from dash import dcc, html
from dash.dependencies import Input, Output, State
from demo_v1_dash_helpers import make_pointcloud
from PIL import Image

# Authentication
# openai.api_key = os.getenv("OPENAI_API_KEY")


###########################
# Chatbot
###########################
# Load images
def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("images/cute_robot.jpg"),
        style={"float": "right", "height": 60},
    )
    return dbc.Row([dbc.Col(title, md=8), dbc.Col(logo, md=4)])


def textbox(text, box="AI", name="Philippe"):
    text = text.replace(f"{name}:", "").replace("You:", "")
    style = {
        "max-width": "60%",
        "width": "max-content",
        "padding": "5px 10px",
        "border-radius": 25,
        "margin-bottom": 20,
    }

    if box == "user":
        style["margin-left"] = "auto"
        style["margin-right"] = 0

        return dbc.Card(text, style=style, body=True, color="primary", inverse=True)

    elif box == "AI":
        style["margin-left"] = 0
        style["margin-right"] = "auto"

        thumbnail = html.Img(
            src=app.get_asset_url("images/cheery_robot.jpg"),
            className="chat_thumbnail_im",
            style={
                # 'display': 'block',
                "border-radius": 50,
                "height": 50,
                # # "margin-right": 5,
                "margin-left": "auto",
                "margin-right": "auto",
                # # "float": "left",
                "text-align": "center",
            },
        )
        # user = html.P('Cortex', className='small font-weight-bold', style={"text-align": 'center'})
        time = html.P(
            f'{datetime.now().strftime("%H:%M:%S")}',
            className="chat_message_time text-muted small font-weight-light",
            style={"text-align": "center"},
        )
        full_thumbnail = html.Div([thumbnail, time])
        full_thumbnail = html.Div(
            [full_thumbnail], style={"float": "left", "margin-right": 5}
        )
        text = "EAI Cortex:\n" + text
        textbox = dbc.Card(text, style=style, body=True, color="light", inverse=False)

        return html.Div([full_thumbnail, textbox])

    else:
        raise ValueError("Incorrect option for `box`.")


description = """
Philippe is the principal architect at a condo-development firm in Paris. He lives with his girlfriend of five years in a 2-bedroom condo, with a small dog named Coco. Since the pandemic, his firm has seen a  significant drop in condo requests. As such, he’s been spending less time designing and more time on cooking,  his favorite hobby. He loves to cook international foods, venturing beyond French cuisine. But, he is eager  to get back to architecture and combine his hobby with his occupation. That’s why he’s looking to create a  new design for the kitchens in the company’s current inventory. Can you give him advice on how to do that?
"""


# Define Layout
conversation = html.Div(
    html.Div(id="display-conversation"),
    style={
        "overflow-y": "auto",
        "display": "flex",
        "height": "calc(90vh - 132px)",
        "flex-direction": "column-reverse",
    },
)

controls = dbc.InputGroup(
    children=[
        dbc.Input(
            id="user-input",
            placeholder="Write to the chatbot...",
            type="text",
            debounce=True,
        ),
    ]
)
######################

###########################
# Pointcloud
###########################
# pointcloud = make_pointcloud()

import torch

from home_robot.mapping.voxel.voxel import SparseVoxelMap

svm = torch.load(
    "/private/home/ssax/home-robot/projects/scannet_offline_eval/canned_scannet_scene.pth"
)
pointcloud = svm.voxel_map.show(backend="pytorch3d")

################


###########################
# App
###########################
# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                html.Div(
                    "Accel Cortex Demo: FAIR Conference",
                    className="text-primary app-header--title text-center fs-3",
                ),
                html.Hr(),
                # Header("Dash GPT-3 Chatbot", app),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        # dcc.Graph(figure={}, id='my-first-graph-final')
                        dcc.Graph(
                            figure=pointcloud,
                            id="my-first-graph-final",
                            style={"height": "90vh"},
                        )
                    ],
                    width=9,
                ),
                dbc.Col(
                    children=[
                        html.H3("Chat with Cortex", className="text-center"),
                        dcc.Store(id="store-conversation", data=""),
                        conversation,
                        dcc.Loading(
                            html.Div(id="loading-component"),
                            type="default",
                            className="gif-loading",
                            style={"margin-bottom": "132px", "margin-left": 0},
                        ),
                        controls,
                    ],
                    width=3,
                ),
            ]
        ),
    ],
    fluid=True,
    style={"height": "100vh"},
)
# app.layout = dbc.Container(
#     fluid=True,
#     children=[
#         Header("Dash GPT-3 Chatbot", app),
#         html.Hr(),
#         dcc.Store(id="store-conversation", data=""),
#         conversation,
#         dcc.Loading(html.Div(id="loading-component"), type="default", className='gif-loading', style={'margin-bottom': '132px', 'margin-left': 0}),
#         controls,
#     ],
# )


###########################
# Callbacks
###########################
@app.callback(
    Output("display-conversation", "children"), [Input("store-conversation", "data")]
)
def update_display(chat_history):
    return [
        textbox(x, box="user") if i % 2 == 0 else textbox(x, box="AI")
        for i, x in enumerate(chat_history.split("<split>")[:-1])
    ]


@app.callback(
    Output("user-input", "value"),
    [Input("user-input", "n_submit")],
)
def clear_input(n_submit):
    return ""


# @app.callback(
#     [Output("loading-component", "children")],
#     [State("store-conversation", "data")],
#     [State("user-input", "value"), State("store-conversation", "data")],
# )
# def
# Here we want to show the user message being submitted
# and the model message loading


@app.callback(
    [Output("store-conversation", "data"), Output("loading-component", "children")],
    [Input("user-input", "n_submit")],
    [State("user-input", "value"), State("store-conversation", "data")],
)
def run_chatbot(n_submit, user_input, chat_history):
    # n_submit, user_input, chat_history
    if n_submit is None:
        return "", None

    if user_input is None or user_input == "":
        return chat_history, None

    name = "Philippe"

    prompt = dedent(
        f"""
    {description}

    You: Hello {name}!
    {name}: Hello! Glad to be talking to you today.
    """
    )

    # First add the user input to the chat history
    chat_history += f"You: {user_input}<split>{name}:"

    model_input = prompt + chat_history.replace("<split>", "\n")

    # openai.api_key = os.environ['OPENAI_API_KEY']
    # "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"
    response = openai.Completion.create(
        engine="davinci",
        prompt=model_input,
        max_tokens=250,
        stop=["You:"],
        temperature=0.9,
    )
    model_output = response.choices[0].text.strip()

    chat_history += f"{model_output}<split>"

    return chat_history, None


if __name__ == "__main__":
    app.run_server(debug=True, port=8008)
