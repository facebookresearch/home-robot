import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


import home_robot
import home_robot_hw
from home_robot_hw.remote.api import StretchClient

client = StretchClient()
image = client.head.get_images()[0]

plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis("off")
plt.show()

import sys

sys.path.append("..")
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

sam_checkpoint = "./segment-anything/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

import timeit

t0 = timeit.default_timer()
masks = mask_generator.generate(image)
t1 = timeit.default_timer()
print(image.shape)
print(t1 - t0)

print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis("off")
plt.show()
