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


def get_obj_centric_world_representation(instance_memory, max_context_length):
    """Get version that LLM can handle - convert images into torch if not already"""
    obs = Observations(object_images=[])
    for global_id, instance in enumerate(instance_memory):
        instance_crops = instance.instance_views
        crop = random.sample(instance_crops, 1)[0].cropped_image
        if isinstance(crop, np.ndarray):
            crop = torch.from_numpy(crop)
        obs.object_images.append(
            ObjectImage(
                crop_id=global_id,
                image=crop.contiguous(),
            )
        )
    # TODO: the model currenly can only handle 20 crops
    if len(obs.object_images) > max_context_length:
        logger.warning(
            "\nWarning: this version of minigpt4 can only handle limited size of crops -- sampling a subset of crops from the instance memory..."
        )
        obs.object_images = random.sample(obs.object_images, max_context_length)

    return obs


def get_vlm_rpc_stub(vlm_server_addr, vlm_server_port):
    """Connect to a remote VLM server via RPC"""
    channel = grpc.insecure_channel(f"{vlm_server_addr}:{vlm_server_port}")
    stub = AgentgRPCStub(channel)
    return stub


def get_output_from_world_representation(stub, world_representation, goal):
    return stub.stream_act_on_observations(
        ProtoConverter.wrap_obs_iterator(
            episode_id=random.randint(1, 1000000),
            obs=world_representation,
            goal=goal,
        )
    )
