import os
from datetime import datetime

import openai
from astrowidgets import ImageWidget
from IPython.display import HTML, display
from ipywidgets import widgets
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma


def make_video_widget(arr):
    image = ImageWidget()
    image.load_array(arr)

    def update(arr):
        image.load_array(arr)

    return image, update


def make_pointcloud(n_points=100, **plot_scene_kwargs):
    import torch
    from pytorch3d.structures import Pointclouds
    from pytorch3d.vis.plotly_vis import AxisArgs, plot_scene

    from home_robot.utils.bboxes_3d_plotly import plot_scene_with_bboxes
    from home_robot.utils.data_tools.dict import update

    traces = {"Points": Pointclouds(points=torch.rand((1, n_points, 3)))}
    _default_plot_args = dict(
        xaxis={"backgroundcolor": "rgb(200, 200, 230)"},
        yaxis={"backgroundcolor": "rgb(230, 200, 200)"},
        zaxis={"backgroundcolor": "rgb(200, 230, 200)"},
        axis_args=AxisArgs(showgrid=True),
        pointcloud_marker_size=3,
        pointcloud_max_points=200_000,
        boxes_plot_together=True,
        boxes_wireframe_width=3,
    )
    fig = plot_scene_with_bboxes(
        plots={"Global scene": traces},
        **update(_default_plot_args, plot_scene_kwargs),
    )
    return fig


def make_chat_box_widget():
    # loader = DirectoryLoader("documents/", glob="*.txt")
    # txt_docs = loader.load_and_split()

    # embeddings = OpenAIEmbeddings()
    # txt_docsearch = Chroma.from_documents(txt_docs, embeddings)

    # file = open("prompts/demo.txt", "r")
    # prompt = file.read()
    # print(prompt)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, prompt=prompt)

    # qa = ConversationalRetrievalChain.from_llm(llm, retriever=txt_docsearch.as_retriever())

    file = open("prompts/demo.txt", "r")
    prompt = file.read()
    prompt_template = PromptTemplate(
        input_variables=["history", "input"], template=prompt
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    conversation_buf = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(ai_prefix="Guide", human_prefix="User"),
        prompt=prompt_template,
    )
    chat_history = []

    # def text_eventhandler(*args):
    def text_eventhandler(args):
        value = args.value
        if value == "":
            return

        # Show loading animation
        loading_bar.layout.display = "block"

        # Get question
        question = value

        # Reset text field
        args.value = ""

        # Formatting question for output
        q = (
            f'<div class="chat-message-right pb-4"><div>'
            + f'<img src="images/cheery_robot.jpg" class="rounded-circle mr-1" width="40" height="40">'
            + f'<div class="text-muted small text-nowrap mt-2">{datetime.now().strftime("%H:%M:%S")}</div></div>'
            + '<div class="flex-shrink-1 bg-light rounded py-2 px-3 ml-3">'
            + f'<div class="font-weight-bold mb-1">You</div>{question}</div>'
        )

        # Display formatted question
        output.append_display_data(HTML(q))

        try:
            # response = qa({"question": f"{question}", "chat_history": chat_history})
            # answer = response["answer"]
            answer = conversation_buf({"input": f"{question}", "history": chat_history})
            answer = answer["response"]
            chat_history.append((question, answer))
        except Exception as e:
            answer = "<b>Error:</b> " + str(e)

        # Formatting answer for output
        # Replacing all $ otherwise matjax would format them in a strange way
        answer_formatted = answer.replace("$", r"\$")
        a = (
            f'<div class="chat-message-left pb-4"><div>'
            + f'<img src="images/cute_robot.jpg" class="rounded-circle mr-1" width="40" height="40">'
            + f'<div class="text-muted small text-nowrap mt-2">{datetime.now().strftime("%H:%M:%S")}</div></div>'
            + '<div class="flex-shrink-1 bg-light rounded py-2 px-3 ml-3">'
            + f'<div class="font-weight-bold mb-1">EAI-Cortex</div>{answer_formatted}</div>'
        )

        # Turn off loading animation
        loading_bar.layout.display = "none"

        output.append_display_data(HTML(a))

    text_input_widget = widgets.Text(layout=widgets.Layout(height="auto", width="100%"))
    text_input_widget.on_submit(text_eventhandler)
    # # A more general way of handling events
    # in_text.continuous_update = False
    # in_text.observe(text_eventhandler, "value")

    output = widgets.Output()

    file = open("images/loading.gif", "rb")
    image = file.read()
    loading_bar = widgets.Image(
        value=image, format="gif", width="30", height="30", layout={"display": "None"}
    )

    conversation_history_widget = widgets.HBox(
        [loading_bar, output],
        layout=widgets.Layout(
            width="100%",
            max_height="500px",
            min_height="500px",
            display="inline-flex",
            flex_flow="column-reverse",
        ),
    )

    chat_title = widgets.Label(
        value="Cortex Chat",
        layout=widgets.Layout(
            width="100%",
            display="flex",
            justify_content="center",
        ),
    )

    chatbox = widgets.VBox(
        [chat_title, conversation_history_widget, text_input_widget],
        layout=widgets.Layout(
            width="100%",
            max_height="700px",
            display="flex",
            justify_content="space-between",
        ),
    )
    # user_input_widget = widgets.Box(
    #     children=[in_text],
    #     layout=widgets.Layout(width="100%", display="flex", flex_flow="row", object_position='right center'),
    # )
    return chatbox
