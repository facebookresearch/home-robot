# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import matplotlib.pyplot as plt
from fastsam import FastSAM, FastSAMPrompt

import home_robot
import home_robot_hw
from home_robot_hw.remote.api import StretchClient

client = StretchClient()
image = client.head.get_images()[0]

plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis("off")
plt.show()


model = FastSAM("./weights/FastSAM-x.pt")
DEVICE = "cpu"
everything_results = model(
    image,
    device=DEVICE,
    retina_masks=True,
    imgsz=1024,
    conf=0.4,
    iou=0.9,
)
prompt_process = FastSAMPrompt(image, everything_results, device=DEVICE)

# everything prompt
ann = prompt_process.everything_prompt()

# bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
# ann = prompt_process.box_prompt(bbox=[[200, 200, 300, 300]])

# text prompt
# ann = prompt_process.text_prompt(text='a photo of a dog')

# point prompt
# points default [[0,0]] [[x1,y1],[x2,y2]]
# point_label default [0] [1,0] 0:background, 1:foreground
# ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])

prompt_process.plot(
    annotations=ann,
    output_path="./output.jpg",
)
