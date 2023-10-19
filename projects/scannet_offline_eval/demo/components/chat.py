# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import base64
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from textwrap import dedent

import dash
import dash_bootstrap_components as dbc
import openai
from dash import Patch, dcc, html
from dash.exceptions import PreventUpdate

# from dash.dependencies import Input, Output, State
from dash_extensions.enrich import ALL, DashProxy, Input, Output, State, ctx, dcc, html
from loguru import logger

from home_robot.utils.demo_chat import DemoChat

from .app import HARD_CODE_RESPONSES, app, app_config, svm_watcher
from .core import DashComponent


@dataclass
class ChatBoxConfig:
    chat_log_fpath: Path
    modal: bool = True


class Chatbox:
    def __init__(self, config: ChatBoxConfig, name="chatbox"):  #
        super().__init__()
        self.name = name
        self.chat_log = DemoChat(log_file_path=config.chat_log_fpath)

    def register_callbacks(self, app):
        app.callback(  # Modal
            [
                Output(f"{self.name}-modal", "is_open"),
                Output(f"{self.name}-modal-im", "src"),
                Output(f"{self.name}-modal-title", "children"),
                Output(f"{self.name}-modal-ctx", "data"),
            ],
            Input(
                {"type": f"{self.name}-pattern-matched-image", "index": ALL}, "n_clicks"
            ),
            [
                State(
                    {"type": f"{self.name}-pattern-matched-image", "index": ALL}, "src"
                ),
                State(f"{self.name}-modal", "is_open"),
                State(f"{self.name}-modal-ctx", "data"),
            ],
        )(toggle_modal)

        app.callback(  # User input
            Output(f"{self.name}-user-input", "value"),
            [Input(f"{self.name}-user-input", "n_submit")],
        )(clear_input)

        app.callback(  # User message
            [
                Output(f"{self.name}-store-conversation", "data", allow_duplicate=True),
                Output(
                    f"{self.name}-display-conversation",
                    "children",
                    allow_duplicate=True,
                ),
                Output(f"{self.name}-last-displayed-user-msg", "data"),
            ],
            [Input(f"{self.name}-user-input", "n_submit")],
            [
                State(f"{self.name}-user-input", "value"),
                State(f"{self.name}-store-conversation", "data"),
            ],
            prevent_initial_call=True,
        )(self.add_user_msg_callback)

        app.callback(  # User message
            [
                Output(f"{self.name}-store-conversation", "data", allow_duplicate=True),
                Output(
                    f"{self.name}-display-conversation",
                    "children",
                    allow_duplicate=True,
                ),
                Output(f"{self.name}-loading-component", "children"),
            ],
            [
                Input(f"{self.name}-last-displayed-user-msg", "data"),
            ],
            [State(f"{self.name}-store-conversation", "data")],
            prevent_initial_call=True,
        )(self.add_assistant_msg_callback)
        return self

    def add_assistant_msg_callback(self, user_input, msg_history):
        patched_children = Patch()
        patched_msg_hist = Patch()
        if True or not HARD_CODE_RESPONSES:
            response = self.chat_log.input(user_input, role="user")
            response_msg = {"role": "assistant", "content": response}
            self.patch_in_new_msg(
                response_msg, msg_history, patched_msg_hist, patched_children
            )
        else:
            # TODO: REMOVE
            hard_coded_instance = 1
            if (
                len(msg_history) <= 2
            ):  # No matter what the user types in, initially respond with this plan
                self.patch_in_new_msg(
                    {
                        "role": "assistant",
                        "content": f"Plan: goto('bottle-1') -- Instance id: 1",
                        "timestamp": f"{datetime.now():%Y-%m-%d-%H-%M-%S}",
                    },
                    msg_history,
                    patched_msg_hist,
                    patched_children,
                )
                self.patch_in_new_msg(
                    {
                        "role": "assistant",
                        "content": "Execute this plan? [Y/N]",
                        "timestamp": f"{datetime.now():%Y-%m-%d-%H-%M-%S}",
                    },
                    msg_history,
                    patched_msg_hist,
                    patched_children,
                )
            else:  # After that, assume the user confirmed and start the interface and
                svm_watcher.target_instance_id = hard_coded_instance
                svm_watcher.unpause()
            ###########################
        return patched_msg_hist, patched_children, None

    def add_user_msg_callback(self, n_submit, user_input, msg_history):
        if (n_submit is None) or (user_input is None) or (user_input == ""):
            raise PreventUpdate
        patched_children = Patch()
        patched_msg_hist = Patch()

        if len(msg_history) == 0:
            pass  # Add prompt information here
        # First add the user message
        user_msg = {"role": "user", "content": user_input}
        self.patch_in_new_msg(user_msg, msg_history, patched_msg_hist, patched_children)
        return patched_msg_hist, patched_children, user_input

    @cached_property
    def layout(self):
        # Define Layout
        return dbc.Row(
            [
                # Titel
                html.Div(
                    html.H2("Chat with Cortex", className="text-center text-secondary"),
                    className="chat-header",
                ),
                # Conversation display
                html.Div(
                    id=f"{self.name}-display-conversation",
                    className="chat-conversation",
                ),
                dcc.Loading(
                    html.Div(id=f"{self.name}-loading-component"),
                    type="default",
                    className="gif-loading",
                ),
                dcc.Store(
                    id=f"{self.name}-store-conversation", data=[]
                ),  # Complete message history
                dcc.Store(
                    id=f"{self.name}-last-displayed-user-msg", data=""
                ),  # Allows us to immediately display msg and then trigger VLM callback
                # User input box
                dbc.InputGroup(
                    children=[
                        dbc.Input(
                            id=f"{self.name}-user-input",
                            placeholder="Write to the chatbot...",
                            type="text",
                            debounce=True,
                            autoComplete="off",
                        ),
                    ],
                    className="chat-input",
                ),
                # Modal that shows crops
                dbc.Modal(
                    children=[
                        dbc.ModalHeader(
                            dbc.ModalTitle(
                                "Crop of Instance", id=f"{self.name}-modal-title"
                            )
                        ),
                        dbc.ModalBody(
                            children=[
                                html.Img(
                                    src=None,
                                    className="chat-modal-im",
                                    id=f"{self.name}-modal-im",
                                ),
                            ],
                            className="center",
                        ),
                    ],
                    size="xl",
                    is_open=False,
                    id=f"{self.name}-modal",
                ),
                dcc.Store(id=f"{self.name}-modal-ctx", data={}),
            ],
            className="chat-window",
        )

    def patch_in_new_msg(
        self, new_msg, msg_hist, patched_msg_hist, patched_convo_elems
    ):
        new_convo_elem = textbox(new_msg["content"], new_msg["role"], self.name)
        msg_hist.append(new_msg)
        if patched_msg_hist is not None:
            patched_msg_hist.append(new_msg)
        if patched_convo_elems is not None:
            patched_convo_elems.prepend(new_convo_elem)


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


# Add modal that displays crop on click
# Look at us, using fancy patttern-matching callbacks to handle dynamic webpages
# We could display all crops. If we don't need that, then this could be a client-side callback
def toggle_modal(n_clicks, src, is_open, last_ctx):
    n_clicks = ctx.triggered[0]["value"]
    trigger_id = ctx.triggered_id["index"]
    # curr_ctx = {'value': n_clicks, 'index': trigger_id}
    if not n_clicks or (trigger_id in last_ctx and last_ctx[trigger_id] == n_clicks):
        raise PreventUpdate
    # last_ctx[trigger_id] = n_clicks
    new_ctx = Patch()
    new_ctx[trigger_id] = n_clicks

    logger.debug(f"Showing modal of {trigger_id} {n_clicks}")
    # return [not is_open, src[0]]
    header = f"Crop of {trigger_id}"
    if n_clicks:
        return [not is_open, src[0], header, new_ctx]
    return [is_open, src[0], header, new_ctx]


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


def textbox(text, box="user", pattern_match_id=""):
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
                    id={
                        "type": f"{pattern_match_id}-pattern-matched-image",
                        "index": image_id,
                    },
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


def clear_input(n_submit):
    return ""


# chat_log_dir = os.path.dirname(app_config.directory_watch_path)
# if "viz_data" in chat_log_dir:
#     chat_log_dir = os.path.dirname(chat_log_dir)
# chat_log = DemoChat(log_file_path=f"{chat_log_dir}/demo_chat.json")
# print(f"Chat log dir: {chat_log_dir}")
# print(f"Chat log: {chat_log}")
