# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from datetime import datetime
from textwrap import dedent

import dash
import dash_bootstrap_components as dbc
import openai
from dash import Patch, dcc, html
from dash.dependencies import Input, Output, State

from home_robot.utils.demo_chat import DemoChat

from .app import app, app_config

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


def make_layout(height="50vh"):
    # Define Layout
    # conversation =

    # controls =

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
        ],
        className="chat-window",
    )


def textbox(text, box="user"):
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
        textbox = dbc.Card(
            children=[
                html.P("EAI-Cortex", className="chat-body chat-username"),
                dbc.CardBody(text),
            ],
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
    Output("display-conversation", "children"),
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
#  2. display loading initially
#  3. when LLM responds, show that message and remove the loading
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
    response = chat_log.input(user_input, role="user")
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=msg_history,
    #     max_tokens=250,
    #     stop=[f"{user_name}:"],
    #     temperature=0.2,
    # )
    # model_output = response.choices[0].message.content.strip()
    msg_history.append({"role": "assistant", "content": response})
    chat_history = msg_arr_to_str(msg_history)

    return chat_history, None
