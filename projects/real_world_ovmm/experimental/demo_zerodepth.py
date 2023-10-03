# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# To get this to work:
#  pip install torch-scatter einops lpip


import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch

from home_robot.utils.point_cloud import show_point_cloud
from home_robot_hw.remote.api import StretchClient

zerodepth_model = torch.hub.load(
    "TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True
)
intrinsics = torch.zeros((1, 3, 3))
rgb = torch.zeros((1, 3, 256, 256))
depth_pred = zerodepth_model(rgb, intrinsics)

client = StretchClient()

rgb, depth, xyz = client.head.get_images(compute_xyz=True)
plt.imshow(rgb)
plt.show()

orig_rgb = rgb / 255.0
rgb = torch.FloatTensor(orig_rgb[None]).permute(0, 3, 1, 2)
intrinsics = torch.FloatTensor(client.head.intrinsics()[None])
print("Predicting depth...")
t0 = timeit.default_timer()
pred_depth = zerodepth_model(rgb, intrinsics)[0, 0].detach().cpu().numpy()
t1 = timeit.default_timer()
print("...done. Took", t1 - t0, "seconds.")

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(depth)
plt.subplot(1, 2, 2)
plt.imshow(pred_depth)
plt.show()

pred_xyz = client.head.depth_to_xyz(pred_depth)
print("Original depth...")
show_point_cloud(xyz, orig_rgb, orig=np.zeros(3))
print("Predicted depth...")
show_point_cloud(pred_xyz, orig_rgb, orig=np.zeros(3))

breakpoint()
