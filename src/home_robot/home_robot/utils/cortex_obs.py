from dataclasses import dataclass
from typing import List, Text

import torch
from torch import Tensor

import home_robot.utils.task_rpc_env_pb2 as task_rpc_env_pb2


@dataclass
class Observations:
    low_level_output_messages: List[str] = None
    scene_images: List = None
    object_images: List = None
    scene_graph: List = None


@dataclass
class ObjectImage:
    image: Tensor = None
    position: List[float] = None
    crop_id: int = None
    object_class: Text = None
    instance_id: int = None


class ProtoConverter:
    @staticmethod
    def wrap_obs(obs: Observations) -> task_rpc_env_pb2.ProtoObservations:
        """Convert python observations data class into protobuf message"""
        return task_rpc_env_pb2.ProtoObservations(
            low_level_output_messages=obs.low_level_output_messages,
            scene_images=(
                [ProtoConverter.wrap_scene_image(x) for x in obs.scene_images]
                if obs.scene_images
                else None
            ),
            object_images=(
                [ProtoConverter.wrap_object_image(x) for x in obs.object_images]
                if obs.object_images
                else None
            ),
        )

    @staticmethod
    def wrap_scene_image(image: Tensor) -> task_rpc_env_pb2.ProtoSceneImage:
        """Convert scene images (Tensors) to protobuf message"""
        return task_rpc_env_pb2.ProtoSceneImage(
            shape=image.shape, values=image.reshape(-1)
        )

    @staticmethod
    def wrap_object_image(
        object_image: ObjectImage,
    ) -> task_rpc_env_pb2.ProtoObjectImage:
        """Convert ObjectImage python class into protobuf message"""
        return task_rpc_env_pb2.ProtoObjectImage(
            shape=object_image.image.shape,
            values=object_image.image.reshape(-1),
            position=object_image.position,
            crop_id=object_image.crop_id,
            object_class=object_image.object_class,
        )

    @staticmethod
    def unwrap_obs(obs: task_rpc_env_pb2.ProtoObservations) -> Observations:
        """Turn protomessage class into the python Observation class"""
        return Observations(
            low_level_output_messages=obs.low_level_output_messages,
            scene_images=[
                ProtoConverter.unwrap_scene_image(x) for x in obs.scene_images
            ],
            object_images=[
                ProtoConverter.unwrap_object_image(x) for x in obs.object_images
            ],
        )

    @staticmethod
    def unwrap_scene_image(image: task_rpc_env_pb2.ProtoSceneImage) -> Tensor:
        """Turn protomessage scene_image to python torch Tensor"""
        return torch.tensor(image.values).reshape(tuple(image.shape))

    @staticmethod
    def unwrap_object_image(
        object_image: task_rpc_env_pb2.ProtoObjectImage,
    ) -> ObjectImage:
        """Turn protobuf object_image to python class ObjectImage"""
        return ObjectImage(
            image=torch.tensor(object_image.values).reshape(tuple(object_image.shape)),
            position=object_image.position,
            crop_id=object_image.crop_id,
            object_class=object_image.object_class,
        )

    @staticmethod
    def wrap_obs_iterator(episode_id, obs: Observations, goal: str):
        yield task_rpc_env_pb2.ActOnObservationsArgs(episode_id=episode_id, goal=goal)

        for img in obs.object_images:
            yield task_rpc_env_pb2.ActOnObservationsArgs(
                obs=task_rpc_env_pb2.ProtoObservations(
                    object_images=[ProtoConverter.wrap_object_image(img)]
                )
            )
