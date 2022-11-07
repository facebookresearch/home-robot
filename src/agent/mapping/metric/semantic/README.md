# Metric Semantic Map

### Map Initialization
```
from semantic_map_state import SemanticMapState
from semantic_map_module import SemanticMapModule

# State holds global and local map and sensor pose
# See semantic_map_state.py for argument info
semantic_map = SemanticMapState(
    device=torch.device("cuda:0"),
    num_environments=1,
    num_sem_categories=16,
    map_resolution=5,
    map_size_cm=4800,
    global_downscaling=2,
)
semantic_map.init_map_and_pose()

# Module is responsible for updating the local and global maps and poses
# See semantic_map_module.py for argument info
semantic_map_module = SemanticMapModule(
    frame_height=480,
    frame_width=640,
    camera_height=0.88,
    hfov=79.0,
    num_sem_categories=16,
    map_size_cm=4800,
    map_resolution=5,
    vision_range=100,
    global_downscaling=2,
    du_scale=4,
    cat_pred_threshold=5.0,
    exp_pred_threshold=1.0,
    map_pred_threshold=1.0,
)
```

### Map Update
```
# See semantic_map_module.py for argument and return info
# obs: current frame containing (RGB, depth, segmentation) of shape
#  (num_environments, 3 + 1 + num_sem_categories, frame_height, frame_width)
# pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
#  of shape (num_environments, 3)
dones = torch.tensor([False] * num_environments)
update_global = torch.tensor([True] * num_environments)
(
    seq_map_features,
    semantic_map.local_map,
    semantic_map.global_map,
    seq_local_pose,
    seq_global_pose,
    seq_lmb,
    seq_origins,
) = semantic_map_module(
    obs.unsqueeze(1),
    pose_delta.unsqueeze(1),
    dones.unsqueeze(1),
    update_global.unsqueeze(1),
    semantic_map.local_map,
    semantic_map.global_map,
    semantic_map.local_pose,
    semantic_map.global_pose,
    semantic_map.lmb,
    semantic_map.origins,
)
semantic_map.local_pose = seq_local_pose[:, -1]
semantic_map.global_pose = seq_global_pose[:, -1]
semantic_map.lmb = seq_lmb[:, -1]
semantic_map.origins = seq_origins[:, -1]
```
