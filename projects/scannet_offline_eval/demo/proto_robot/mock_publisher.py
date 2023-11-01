import io
import logging
import pickle

import grpc
import robodata_pb2
import robodata_pb2_grpc
import torch
from PIL import Image
from torchvision import transforms

stretch_filename = "/private/home/ssax/home-robot/src/home_robot/home_robot/datasets/robot/data/stretch_/2023-09-06-demo01.pkl"
data = pickle.load(open(stretch_filename, "rb"))

img_filename = "proto_robot/dog.webp"
test_img = Image.open(img_filename)

convert_tensor = transforms.ToTensor()
center_crop = transforms.CenterCrop((224, 224))
test_img_tensor = center_crop(convert_tensor(test_img))


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


def send_robot_data(stub):
    for d in range(len(data) - 1):
        new_msg = create_msg_from_file(data, d)
        print(new_msg)
        responses = stub.ReceiveRobotData(new_msg)
        for response in responses:
            print("Received message %s at %s" % (response.message, response.robot_obs))


def test_conversation(stub):
    new_conv = robodata_pb2.LLMInput()
    chat = new_conv.conversation.add()
    chat.role = "User"
    chat.content = "Hey! Whats up"
    img = new_conv.imgs.add()
    img.CopyFrom(tensor_to_robotensor(test_img_tensor))
    stub.Chat(new_conv)


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.

    # with grpc.insecure_channel("localhost:50051") as channel:
    channel = grpc.insecure_channel("localhost:50051")
    stub = robodata_pb2_grpc.RobotDataStub(channel)
    send_robot_data(stub)
    test_conversation(stub)
    channel.close()


if __name__ == "__main__":
    logging.basicConfig()
    run()
