# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import base64
import logging
import os
import re
from datetime import datetime
from textwrap import dedent

import dash
import dash_bootstrap_components as dbc
import openai
from dash import Patch, dcc, html
from dash.exceptions import PreventUpdate

# from dash.dependencies import Input, Output, State
from dash_extensions.enrich import ALL, Input, Output, State, ctx, dcc, html

from home_robot.utils.demo_chat import DemoChat

from .app import app, app_config, svm_watcher

openai.api_key = os.getenv("OPENAI_API_KEY")
prompt_file = app.get_asset_url("../assets/prompts/demo.txt")
description = open("assets/prompts/demo.txt", "r").read()
import os

chat_log_dir = os.path.dirname(app_config.directory_watch_path)
if "viz_data" in chat_log_dir:
    chat_log_dir = os.path.dirname(chat_log_dir)
chat_log = DemoChat(log_file_path=f"{chat_log_dir}/demo_chat.json")
print(f"Chat log dir: {chat_log_dir}")
print(f"Chat log: {chat_log}")

# Conent of the demo_chat:
# [
#     {
#         "sender": "user",
#         "message": "bring me water."
#     },
#     {
#         "sender": "user",
#         "message": "y"
#     },
#     {
#         "sender": "system",
#         "message": "please type any task you want the robot to do: "
#     },
#     {
#         "sender": "system",
#         "message": "Plan: for task: y"
#     }
# ]

from loguru import logger


# Add modal that displays crop on click
# Look at us, using fancy patttern-matching callbacks to handle dynamic webpages
@app.callback(
    [
        Output("image-modal", "is_open"),
        Output("image-modal-im", "src"),
        Output("image-modal-title", "children"),
    ],
    Input({"type": "pattern-matched-image", "index": ALL}, "n_clicks"),
    [
        State({"type": "pattern-matched-image", "index": ALL}, "src"),
        State("image-modal", "is_open"),
    ],
)
def toggle_modal(n_clicks, src, is_open):
    n_clicks = ctx.triggered[0]["value"]
    if not n_clicks:
        raise PreventUpdate
    button_id = ctx.triggered_id
    logger.debug(f"Showing modal of {button_id}")
    # return [not is_open, src[0]]
    header = f"Crop of {button_id['index']}"
    if n_clicks:
        return [not is_open, src[0], header]
    return [is_open, src[0], header]


def make_layout(height="50vh"):
    # Define Layout
    return dbc.Row(
        [
            html.Div(
                html.H2("Chat with Cortex", className="text-center text-secondary"),
                className="chat-header",
            ),
            dcc.Store(id="store-conversation", data=""),
            html.Div(
                id="display-conversation",
                className="chat-conversation",
                style={
                    # "overflow-y": "auto",
                    # "display": "flex",
                    # "height": f"calc({height} - 140px)",
                    # "flex-direction": "column-reverse",
                },
            ),
            dcc.Loading(
                html.Div(id="loading-component"),
                type="default",
                className="gif-loading",
            ),
            dbc.InputGroup(
                children=[
                    dbc.Input(
                        id="user-input",
                        placeholder="Write to the chatbot...",
                        type="text",
                        debounce=True,
                        autoComplete="off",
                    ),
                ],
                className="chat-input",
            ),
            dbc.Modal(
                children=[
                    dbc.ModalHeader(
                        dbc.ModalTitle("Crop of Instance", id="image-modal-title")
                    ),
                    dbc.ModalBody(
                        children=[
                            html.Img(
                                src=None, className="chat-modal-im", id="image-modal-im"
                            ),
                        ],
                        className="center",
                    ),
                ],
                size="xl",
                is_open=False,
                id="image-modal",
            ),
        ],
        className="chat-window",
    )


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def replace_instance_with_image(text):
    # Find all instances of "instance id:x" in the text
    instances = re.findall(r"Instance id: (\d+)", text)
    text = re.sub(r"Instance id: \d+", "specified instance", text)

    image_src, instance = None, None
    for instance in instances:
        image_path = (
            f"{app_config.directory_watch_path}/instances/instance{instance}_view0.png"
        )

        # Encode the image to Base64
        base64_image = encode_image_to_base64(image_path)

        # Create an HTML image tag with the Base64 encoded image
        image_src = f"data:image/png;base64,{base64_image}"
    return image_src, instance


def textbox(text, box="user"):
    # Replace instances in text with their corresponding images
    image_src, instance = replace_instance_with_image(text)

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
        textbox = dbc.Card(
            children=[
                html.P(
                    "User",
                    className="chat-body chat-username",
                    style={"text-align": "right"},
                ),
                dbc.CardBody(text),
            ],
            style=style,
            color="primary",
            inverse=True,
        )
        return html.Div([textbox])
        # return dbc.Card(text, style=style, body=True, color="primary", inverse=True)

    elif box == "assistant":
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
        time = html.P(
            f'{datetime.now().strftime("%H:%M:%S")}',
            className="chat_message_time text-muted small font-weight-light",
            style={"text-align": "center"},
        )
        full_thumbnail = html.Div([thumbnail, time])
        full_thumbnail = html.Div(
            [full_thumbnail], style={"float": "left", "margin-right": 5}
        )
        children = [
            html.P("EAI-Cortex", className="chat-body chat-username"),
            dbc.CardBody(text),
        ]
        if image_src is not None:
            image_id = f"instance-{instance}"
            children.append(
                html.Img(
                    src=image_src,
                    className="chat_thumbnail_im",
                    id={"type": "pattern-matched-image", "index": image_id},
                    style={
                        # 'display': 'block',
                        # "border-radius": 50,
                        "height": 50,
                        # # "margin-right": 5,
                        "margin-left": "auto",
                        "margin-right": "auto",
                        # # "float": "left",
                        "text-align": "center",
                    },
                )
            )

        textbox = dbc.Card(
            children=children,
            style=style,
            color="light",
            inverse=False,
        )

        return html.Div([full_thumbnail, textbox])

    else:
        raise ValueError("Incorrect option for `box`.")


def msg_arr_to_str(msg_history_arr):
    return "<split>".join(
        [f"{msg['role']}<rolesep>{msg['content']}" for msg in msg_history_arr]
    )


def msg_str_to_arr(msg_history_str):
    if not msg_history_str:
        return []
    msg_history = msg_history_str.split("<split>")
    msg_history = [
        dict(role=msg.split("<rolesep>")[0], content=msg.split("<rolesep>")[1])
        for msg in msg_history
    ]
    for m in msg_history:
        logger.info(m)
    return msg_history


###########################
# Chat callback
###########################
@app.callback(
    [Output("display-conversation", "children")],
    [Input("store-conversation", "data")],
    # prevent_initial_call=True
)
def update_display(chat_history):
    # We shouldn't do this mod2 -- what if the user sends multiple messages?
    # Also recreating the chat history causes a brief flashing of objects
    # Instead we should just append a div
    msgs = msg_str_to_arr(chat_history)
    children = [
        textbox(msg["content"], msg["role"]) for msg in msgs if msg["role"] != "system"
    ]
    logger.info(len(children))
    return children
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
# Here we want to show the following:
#  1. user message being submitted immediately
#  2. display loading gif for the chatbot
#  3. when chatbot responds, show that message and remove the loading gif
@app.callback(
    [Output("store-conversation", "data"), Output("loading-component", "children")],
    [Input("user-input", "n_submit")],
    [State("user-input", "value"), State("store-conversation", "data")],
    prevent_initial_call=True,
)
def run_chatbot(n_submit, user_input, chat_history):
    # breakpoint()
    # n_submit, user_input, chat_history
    if n_submit is None:
        return "", None

    if user_input is None or user_input == "":
        return chat_history, None

    user_name = "User"
    system_name = "Guide"
    if not chat_history:
        chat_history = f"system<rolesep>{description}"
    # msg_history = msg_history_str.split("<split>") if msg_history_str else [f'system<rolesep>{description}']
    # if len(msg_history) == 0:
    #     msg_history = [f'role<rolesep>{description}'] #[{'role': 'system', 'content': description}]

    # First add the user input to the chat history
    msg_history = msg_str_to_arr(chat_history)
    msg_history.append({"role": "user", "content": user_input})
    # response = chat_log.input(user_input, role="user")
    # msg_history.append({"role": "assistant", "content": response})

    # TODO: REMOVE
    hard_coded_instance = 1
    if len(msg_history) <= 2:
        msg_history.extend(
            [
                {
                    "role": "assistant",
                    "content": f"Plan: goto('bottle-1') -- Instance id: 1",
                    "timestamp": f"{datetime.now():%Y-%m-%d-%H-%M-%S}",
                },
                {
                    "role": "assistant",
                    "content": "Execute this plan? [Y/N]",
                    "timestamp": f"{datetime.now():%Y-%m-%d-%H-%M-%S}",
                },
            ]
        )
    else:
        svm_watcher.target_instance_id = hard_coded_instance
        svm_watcher.unpause()
    ###########################

    # output(self, message: str, role: str = "system")
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=msg_history,
    #     max_tokens=250,
    #     stop=[f"{user_name}:"],
    #     temperature=0.2,
    # )
    # model_output = response.choices[0].message.content.strip()
    chat_history = msg_arr_to_str(msg_history)

    return chat_history, None
