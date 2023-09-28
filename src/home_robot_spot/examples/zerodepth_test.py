import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch

from home_robot.utils.config import get_config
from home_robot_spot import SpotClient
from home_robot.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates

# Create Zero depth model
zerodepth_model = torch.hub.load(
    "TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True
).to("cuda:1")
zerodepth_model = zerodepth_model.eval()
# Instanciate spot client
config = get_config("projects/spot/configs/config.yaml")[0]
spot = SpotClient(config=config)
spot.start()
# Get obs
obs = spot.get_rgbd_obs()
intrinsics = obs.camera_K[:3, :3]
rgb = obs.rgb
depth = obs.depth

# Show obs
plt.imshow(rgb)
plt.show()

# Predict depth and show it 
print("Predicting depth...")
with torch.no_grad():
    orig_rgb = rgb / 255.0
    rgb = torch.FloatTensor(orig_rgb[None]).permute(0, 3, 1, 2).to("cuda:1")
    intrinsics = torch.FloatTensor(intrinsics[None]).to("cuda:1")
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

breakpoint()

# Compare pointclouds
K = obs.camera_K
original_xyz = unproject_masked_depth_to_xyz_coordinates( 
    depth=depth.unsqueeze(0).unsqueeze(1),
    pose=obs.camera_pose.unsqueeze(0),
    inv_intrinsics=torch.linalg.inv(torch.tensor(K[:3, :3])).unsqueeze(0),
)

K = obs.camera_K
pred_xyz = unproject_masked_depth_to_xyz_coordinates( 
    depth=pred_depth.unsqueeze(0).unsqueeze(1),
    pose=obs.camera_pose.unsqueeze(0),
    inv_intrinsics=torch.linalg.inv(torch.tensor(K[:3, :3])).unsqueeze(0),
)

print("Original depth...")
show_point_cloud(original_xyz, orig_rgb, orig=np.zeros(3))


print("Predicted depth...")
show_point_cloud(pred_xyz, orig_rgb, orig=np.zeros(3))

breakpoint()
