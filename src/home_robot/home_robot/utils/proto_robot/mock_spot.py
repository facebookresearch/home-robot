import argparse
import io
import logging
import os
import pickle
import random
import shutil
import sys

import grpc
import robodata_pb2
import robodata_pb2_grpc
import torch
from PIL import Image
from torchvision import transforms

import home_robot.mapping.voxel.voxel as v


def tensor_to_bytes(x):
    buff = io.BytesIO()
    torch.save(x, buff)
    return buff.getvalue()


def tensor_to_robotensor(x):
    robo_tensor = robodata_pb2.RoboTensor()
    robo_tensor.dtype = str(x.dtype)
    tensorshape = x.shape
    dim_fields = [robo_tensor.d1, robo_tensor.d2, robo_tensor.d3]
    if len(tensorshape) <= 3:
        for i in range(len(tensorshape)):
            dim_fields[i] = tensorshape[i]
    robo_tensor.tensor_content = tensor_to_bytes(x)
    return robo_tensor


def create_msg():
    new_msg = robodata_pb2.RobotSummary()
    new_msg.message = "testing robot data"
    yield new_msg


def create_msg_from_file(data, idx):
    print("generating message for " + str(idx))
    new_msg = robodata_pb2.RobotSummary()

    new_msg.rgb_tensor.CopyFrom(tensor_to_robotensor(data["rgb"][idx]))
    new_msg.depth_tensor.CopyFrom(tensor_to_robotensor(data["depth"][idx]))

    new_msg.message = "testing robot data"
    new_msg.robot_obs.gps.lat = data["obs"][idx].gps[0]
    new_msg.robot_obs.gps.long = data["obs"][idx].gps[1]

    yield new_msg


def send_robot_data(stub, spot_filename):
    data = pickle.load(open(spot_filename, "rb"))
    for d in range(len(data) - 1):
        new_msg = create_msg_from_file(data, d)
        print(new_msg)
        responses = stub.ReceiveRobotData(new_msg)
        for response in responses:
            print("Received message %s " % (response.message))


def read_robot_data(stub):
    for i in range(4):
        response = stub.GetHistory(robodata_pb2.RobotSummary())
        print("Getting History...")
        for r in response:
            print(torch.load(io.BytesIO(r.rgb_tensor.tensor_content)))


def get_world_rep_msgs(task, world_rep):
    """Convert world_rep, a list of numpy arrays, into an Iterable of RobotSummary objects"""
    not_sent = True
    i = 0
    for obj in world_rep:
        robo_msg = robodata_pb2.RobotSummary()
        proto_obj = robo_msg.instance_image.add()
        proto_obj.CopyFrom(tensor_to_robotensor(obj))
        proto_obj.id = i
        i += 1
        if not_sent:
            robo_msg.message = task
            not_sent = False
        yield robo_msg


def test_conversation(stub, img_filename):
    test_img = Image.open(img_filename)

    convert_tensor = transforms.ToTensor()
    center_crop = transforms.CenterCrop((224, 224))
    test_img_tensor = center_crop(convert_tensor(test_img))

    new_conv = robodata_pb2.LLMInput()
    chat = new_conv.conversation.add()
    chat.role = "User"
    chat.content = "Hey! Whats up"
    img = new_conv.imgs.add()
    img.CopyFrom(tensor_to_robotensor(test_img_tensor))
    stub.Chat(new_conv)


def test_vlm(stub, voxel_file):
    voxel_map = v.SparseVoxelMap()
    voxel_map.read_from_pickle(voxel_file)
    print("generated voxel_map")

    def get_obj_centric_world_representation(instance_memory, max_context_length):
        crops = []
        for global_id, instance in enumerate(instance_memory):
            instance_crops = instance.instance_views
            crops.append((global_id, random.sample(instance_crops, 1)[0].cropped_image))
        # TODO: the model currenly can only handle 20 crops
        if len(crops) > max_context_length:
            print(
                "\nWarning: this version of minigpt4 can only handle limited size of crops -- sampling a subset of crops from the instance memory..."
            )
            crops = random.sample(crops, max_context_length)
        debug_path = "crops_for_planning/"
        shutil.rmtree(debug_path, ignore_errors=True)
        os.mkdir(debug_path)
        ret = []
        for id, crop in enumerate(crops):
            ret.append(crop[1].cpu().numpy())
        return ret

    world_rep = get_obj_centric_world_representation(voxel_map.get_instances(), 20)

    resp = stub.PlanHighLevelAction(get_world_rep_msgs("Where is the cup?", world_rep))
    print("response!")
    for r in resp:
        print(r)


def parse_spot_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--spot-filename",
        default="/private/home/ssax/home-robot/src/home_robot/home_robot/datasets/robot/data/spot_/09_18_fre_spot/spot_output.pkl",
        help="Path to a pickle file with robot data contained (RGB frames, camera intrinsics,etc)",
    )
    parser.add_argument(
        "--voxel-filename",
        default="/private/home/ssilwal/dev/home-robot/spot_output.pkl",
        help="Path to a pickle file with instances images and data that will be used to build a voxel map.",
    )
    parser.add_argument(
        "--img-filename", default="proto_robot/dog.webp", help="Pass in an image file."
    )
    return parser.parse_args()


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.

    # with grpc.insecure_channel("localhost:50051") as channel:
    maxMsgLength = 1024 * 1024 * 8
    channel = grpc.insecure_channel(
        "localhost:50051",
        options=[
            ("grpc.max_message_length", maxMsgLength),
            ("grpc.max_send_message_length", maxMsgLength),
            ("grpc.max_receive_message_length", maxMsgLength),
        ],
    )
    args = parse_spot_args()
    stub = robodata_pb2_grpc.RobotDataStub(channel)
    send_robot_data(stub, args.spot_filename)
    read_robot_data(stub)
    test_conversation(stub, args.img_filename)
    test_vlm(stub, args.voxel_filename)
    channel.close()


if __name__ == "__main__":
    logging.basicConfig()
    run()
