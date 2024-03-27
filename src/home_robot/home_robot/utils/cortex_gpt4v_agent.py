import base64
import io
import json
import logging
import os

import numpy as np
import requests
from PIL import Image
from torchvision import transforms

from home_robot.utils.cortex_base_agent import BaseAgent

logging.basicConfig(level=logging.INFO)


class CortexGPT4VAgent(BaseAgent):
    def initialize(self):
        self.prompt = self.cfg["prompt"]
        self.api_key = self.cfg["api_key"]
        self.max_tokens = self.cfg["max_tokens"]
        self.temperature = self.cfg["temperature"]
        self.to_pil = transforms.ToPILImage()
        self.resize = transforms.Resize((self.cfg["img_size"], self.cfg["img_size"]))

    def reset(self):
        self.errors = {}
        self.responses = {}
        self.current_round = 0
        self.goal = None

    def log_output(self, path):
        print("log gpt4v responses...")
        with open(os.path.join(path, "gpt4v_errs.json"), "w") as f:
            json.dump(self.errors, f, indent=4)
        with open(os.path.join(path, "responses.json"), "w") as f:
            json.dump(self.responses, f, indent=4)
        if self.goal:  # TODO: a few episodes' goals are None
            with open(os.path.join(path, "goal.txt"), "w") as f:
                f.write(self.goal)

    def _prepare_samples(self, obs, goal, debug_path=None):
        context_messages = [{"type": "text", "text": self.prompt}]
        # context_messages.append()
        self.goal = goal
        for img_id, object_image in enumerate(obs.object_images):
            # Convert to base64Image
            idx = object_image.crop_id
            # pil_image = self.resize(self.to_pil(object_image.image.permute(2, 0, 1)))
            pil_image = self.resize(
                Image.fromarray(np.array(object_image.image, dtype=np.uint8))
            )
            image_bytes = io.BytesIO()
            if debug_path:
                round_path = os.path.join(debug_path, str(self.current_round))
                os.makedirs(round_path, exist_ok=True)
                pil_image.save(os.path.join(round_path, str(img_id) + ".png"))
            pil_image.save(image_bytes, format="png")
            base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

            # Write context images
            text_pre = {"type": "text", "text": f"<img_{idx}>"}
            if idx == len(obs.object_images) - 1:
                text_post = {"type": "text", "text": f"</img_{idx}>"}
            else:
                text_post = {"type": "text", "text": f"</img_{idx}>, "}
            text_img = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
            context_messages.append(text_pre)
            context_messages.append(text_img)
            context_messages.append(text_post)

        scene_graph_text = "\n2. Scene descriptions: "
        for rel in obs.scene_graph:
            crop_id_a = next(
                (
                    obj_crop.crop_id
                    for obj_crop in obs.object_images
                    if obj_crop.instance_id == rel[0]
                ),
                None,
            )
            crop_id_b = next(
                (
                    obj_crop.crop_id
                    for obj_crop in obs.object_images
                    if obj_crop.instance_id == rel[1]
                ),
                None,
            )
            scene_graph_text += f"img_{crop_id_a} is {rel[2]} img_{crop_id_b}; "
        context_messages.append(
            {
                "type": "text",
                "text": scene_graph_text + "\n",
            }
        )

        context_messages.append({"type": "text", "text": f"3. Query: {self.goal}\n"})
        context_messages.append({"type": "text", "text": "4. Answer: "})
        chat_input = {
            "model": "gpt-4-vision-preview",
            "messages": [
                # {
                #     "role": "system",
                #     "content": [
                #         self.prompt
                #     ]
                # },
                {"role": "user", "content": context_messages}
            ],
            "max_tokens": self.max_tokens,
            # "temperature": self.temperature,
        }
        return chat_input

    def _request_gpt4v(self, chat_input):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=chat_input,
        )
        json_res = response.json()
        print(f">>>>>> the original output from gpt4v is: {json_res} >>>>>>>>>")
        if "choices" in json_res:
            res = json_res["choices"][0]["message"]["content"]
        elif "error" in json_res:
            self.errors[self.current_round] = json_res
            return "gpt4v API error"
        # the prompt come with "Answer: " prefix
        self.responses[self.current_round] = res
        # return " ".join(res.split(" ")[1:])
        return res

    def act_on_observations(
        self,
        episode_id,
        obs,
        goal=None,
        new_episode=False,
        task_complete=False,
        debug_path=None,
    ):
        if not obs:
            return None
        self.current_round += 1
        chat_input = self._prepare_samples(obs, goal, debug_path=debug_path)
        return self._request_gpt4v(chat_input)
