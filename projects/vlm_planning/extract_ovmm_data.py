import os
import pdb
import pickle

import numpy as np
import torch
from PIL import Image

dir_path = "/private/home/xiaohanzhang/data/ovmm_heuristic/"
data = {}
padding = 1.5
# total_num_images = 0
image_id = 0


def get_cropped_image(instance_memory, iv):
    image = instance_memory.images[0][iv.timestep]
    im_h = image.shape[1]
    im_w = image.shape[2]
    bbox = iv.bbox
    x = bbox[0, 1]
    y = bbox[0, 0]
    w = bbox[1, 1] - x
    h = bbox[1, 0] - y
    x = 0 if (x - (padding - 1) * w / 2) < 0 else int(x - (padding - 1) * w / 2)
    y = 0 if (y - (padding - 1) * h / 2) < 0 else int(y - (padding - 1) * h / 2)
    y2 = im_h if y + int(h * padding) >= im_h else y + int(h * padding)
    x2 = im_w if x + int(w * padding) >= im_w else x + int(w * padding)
    cropped_image = (
        image[
            :,
            y:y2,
            x:x2,
        ]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    return cropped_image


for i, epi in enumerate(os.listdir(dir_path)):

    os.makedirs(dir_path + epi + "/objects_cropped", exist_ok=True)
    os.makedirs(dir_path + epi + "/start_receps_cropped", exist_ok=True)
    os.makedirs(dir_path + epi + "/goal_receps_cropped", exist_ok=True)
    with open(dir_path + epi + "/instance_memory.pkl", "rb") as f:
        instance_memory = pickle.load(f)
    with open(dir_path + epi + "/info.pkl", "rb") as f:
        info = pickle.load(f)

    for instance_view in instance_memory.instances:
        for id in instance_view:
            if instance_view[id].category_id.cpu().numpy() == 1:
                for iv in instance_view[id].instance_views:
                    # image = iv.cropped_image
                    image = get_cropped_image(instance_memory, iv)
                    Image.fromarray(image).save(
                        dir_path + epi + "/objects_cropped/" + str(image_id) + ".png"
                    )
                    image_id += 1
            if instance_view[id].category_id.cpu().numpy() == 2:
                for iv in instance_view[id].instance_views:
                    # image = iv.cropped_image
                    image = get_cropped_image(instance_memory, iv)
                    Image.fromarray(image).save(
                        dir_path
                        + epi
                        + "/start_receps_cropped/"
                        + str(image_id)
                        + ".png"
                    )
                    image_id += 1
            if instance_view[id].category_id.cpu().numpy() == 3:
                for iv in instance_view[id].instance_views:
                    # image = iv.cropped_image
                    image = get_cropped_image(instance_memory, iv)
                    Image.fromarray(image).save(
                        dir_path
                        + epi
                        + "/goal_receps_cropped/"
                        + str(image_id)
                        + ".png"
                    )
                    image_id += 1
    print(epi)
    print(info["goal_name"] + "\n")

    with open(dir_path + epi + "/task.txt", "w") as f:
        f.write(info["goal_name"])
    # if i > 5:
# print(total_num_images)
