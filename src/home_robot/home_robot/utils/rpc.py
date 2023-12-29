# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import sys

import clip
import numpy as np
import torch
from loguru import logger

try:
    sys.path.append(os.path.expanduser(os.environ["ACCEL_CORTEX"]))
    import grpc
    import src.rpc
    import src.rpc.task_rpc_env_pb2
    from src.utils.observations import ObjectImage, Observations, ProtoConverter
    from task_rpc_env_pb2_grpc import AgentgRPCStub
except Exception as e:
    ## Temporary hack until we make accel-cortex pip installable
    print(
        "Make sure path to accel-cortex base folder is set in the ACCEL_CORTEX environment variable."
    )
    print("If you do not know what that means, this code is not for you!")
    raise (e)


def get_best_view(images, metric: str = "area"):
    """Get best view by some metric."""
    best_view = None
    # is_small = False
    if metric == "area":
        best_area = 0
        for view_instance in images:
            # view = Image.open(ovmm_data_dir+view_name)
            view = view_instance.cropped_image
            w, h = view.shape[0], view.shape[1]
            area = h * w
            if area > best_area:
                best_area = area
                best_view = view_instance
        # if best_area < min_num_pixel:
        #     is_small = True
    else:
        raise NotImplementedError(f"metric {metric} not supported")
    return best_view


def parse_pick_and_place_plan(world_representation, plan: str):
    """Simple parser to pull out high level actions from a plan of the form:

        goto(obj1);pickup(obj1);goto(obj2);placeon(obj1,obj2)

    Args:
        plan(str): contains a plan
    """
    pick_instance_id, place_instance_id = None, None
    if plan == "explore":
        return None, None

    for current_high_level_action in plan.split("; "):

        # addtional format checking of whether the current action is in the robot's skill set
        if not any(
            action in current_high_level_action
            for action in ["goto", "pickup", "placeon", "explore"]
        ):
            return None, None

        if "pickup" in current_high_level_action:
            img_id = current_high_level_action.split("(")[1].split(")")[0].split("_")[1]
            if img_id.isnumeric():
                pick_instance_id = int(
                    world_representation.object_images[int(img_id)].instance_id
                )
            else:
                pick_instance_id = None
        if "placeon" in current_high_level_action:
            img_id = (
                current_high_level_action.split("(")[1]
                .split(")")[0]
                .split(", ")[1]
                .split("_")[1]
                .split('"')[0]
            )
            if img_id.isnumeric():
                place_instance_id = int(
                    world_representation.object_images[int(img_id)].instance_id
                )
            else:
                place_instance_id = None
    return pick_instance_id, place_instance_id


def get_obj_centric_world_representation(
    instance_memory, max_context_length: int, sample_strategy: str, task: str = None
):
    """Get version that LLM can handle - convert images into torch if not already"""
    obs = Observations(object_images=[])
    candidate_objects = []

    for global_id, instance in enumerate(instance_memory):
        instance_crops = instance.instance_views
        crop = get_best_view(instance_crops)
        features = crop.embedding
        crop = crop.cropped_image
        if isinstance(crop, np.ndarray):
            crop = torch.from_numpy(crop)
        # loc = torch.mean(instance.point_cloud, axis=0)
        candidate_objects.append((crop, global_id, features))

    if len(candidate_objects) >= max_context_length:
        logger.warning(
            f"\nWarning: VLMs can only handle limited size of crops -- ignoring instances using strategy: {sample_strategy}..."
        )
        if sample_strategy == "major_vote":
            # Send all the crop images so the agent can implement divide and conquer
            pass
        elif sample_strategy == "random_subsample":
            pass
        elif sample_strategy == "first_seen":
            # Send the first images below the context length
            pass
        elif sample_strategy == "clip":
            # clip ranking
            torch.cuda.empty_cache()
            if task:
                print(f"clip sampling based on task: {task}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # device = "cpu"
                model, preprocess = clip.load("ViT-B/32", device=device)
                text = clip.tokenize([task]).to(device)
                text_features = model.encode_text(text).float()
                image_features = []
                for obj in candidate_objects:
                    image_features.append(obj[2])
                image_features = torch.stack(image_features).squeeze(1).to(device)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
                _, indices = similarity[0].topk(len(candidate_objects))
                sorted_objects = []
                for k in indices:
                    sorted_objects.append(candidate_objects[k.item()])
                candidate_objects = sorted_objects
        else:
            raise NotImplementedError

    for crop_id, obj in enumerate(
        candidate_objects[: min(max_context_length, len(candidate_objects))]
    ):
        obs.object_images.append(
            ObjectImage(crop_id=crop_id, image=obj[0].contiguous(), instance_id=obj[1])
        )
    return obs


def get_vlm_rpc_stub(vlm_server_addr: str, vlm_server_port: int):
    """Connect to a remote VLM server via RPC"""
    channel = grpc.insecure_channel(f"{vlm_server_addr}:{vlm_server_port}")
    stub = AgentgRPCStub(channel)
    return stub


def get_output_from_world_representation(stub, world_representation, goal: str):
    return stub.stream_act_on_observations(
        ProtoConverter.wrap_obs_iterator(
            episode_id=random.randint(1, 1000000),
            obs=world_representation,
            goal=goal,
        )
    )
