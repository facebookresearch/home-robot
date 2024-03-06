# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import sys

import numpy as np
import torch
from loguru import logger

try:
    sys.path.append(os.path.expanduser(os.environ["ACCEL_CORTEX"]))
    import grpc
    from task_rpc_env_pb2_grpc import AgentgRPCStub

    import src.rpc
    import src.rpc.task_rpc_env_pb2
    from src.utils.types.observations import Object, Observations, ProtoConverter
except Exception as e:
    ## Temporary hack until we make accel-cortex pip installable
    print(
        "Make sure path to accel-cortex base folder is set in the ACCEL_CORTEX environment variable."
    )
    print("If you do not know what that means, this code is not for you!")
    raise (e)


def parse_pick_and_place_plan(world_representation, plan: str):
    """Simple parser to pull out high level actions from a plan of the form:

        pick(obj1),place(obj2)

    Args:
        plan(str): contains a plan
    """

    if plan == "explore":
        return None, None
    # now it is hacky to get two instance ids
    # TODO: make it more general for all actions
    # get pick instance id
    current_high_level_action = plan.split("; ")[0]

    # addtional format checking of whether the current action is in the robot's skill set
    if not any(
        action in current_high_level_action
        for action in ["goto", "pickup", "placeon", "explore"]
    ):
        return None, None

    pick_instance_id = int(
        world_representation.object_images[
            int(
                current_high_level_action.split("(")[1]
                .split(")")[0]
                .split(", ")[0]
                .split("_")[1]
                .split('"')[0]
            )
        ].crop_id
    )
    place_instance_id = None
    # self.instance_ids['PICK'] = pick_instance_id
    if len(plan.split(": ")) > 2:
        # get place instance id
        current_high_level_action = plan.split("; ")[2]
        place_instance_id = int(
            world_representation.object_images[
                int(
                    current_high_level_action.split("(")[1]
                    .split(")")[0]
                    .split(", ")[0]
                    .split("_")[1]
                    .split('"')[0]
                )
            ].crop_id
        )
        # print("place_instance_id", place_instance_id)
        # self.instance_ids['PLACE'] = place_instance_id
    return pick_instance_id, place_instance_id


def get_obj_centric_world_representation(
    instance_memory, max_context_length: int, sample_strategy: str
):
    """Get version that LLM can handle - convert images into torch if not already"""

    if sample_strategy == "all":
        # Send all the crop images so the agent can implement divide and conquer
        pass
    elif sample_strategy == "random_subsample":
        pass
    elif sample_strategy == "first":
        # Send the first images below the context length
        pass
    else:
        pass

    obs = Observations(object_images=[])
    for global_id, instance in enumerate(instance_memory):
        if global_id >= max_context_length:
            logger.warning(
                "\nWarning: this version of minigpt4 can only handle limited size of crops -- ignoring instance..."
            )
        else:
            instance_crops = instance.instance_views
            crop = random.sample(instance_crops, 1)[0].cropped_image
            if isinstance(crop, np.ndarray):
                crop = torch.from_numpy(crop)
            obs.object_images.append(
                Object(
                    crop_id=global_id,
                    image=crop.contiguous(),
                )
            )

    # TODO: this code does not work as the global_ids have to be sequential and consecutive
    # # TODO: the model currenly can only handle 20 crops
    # if len(obs.object_images) > max_context_length:
    #     logger.warning(
    #         "\nWarning: this version of minigpt4 can only handle limited size of crops -- sampling a subset of crops from the instance memory..."
    #     )
    #     obs.object_images = random.sample(obs.object_images, max_context_length)

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
