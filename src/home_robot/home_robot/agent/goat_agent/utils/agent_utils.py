import numpy as np
from tqdm import tqdm

from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory


def get_matches_against_memory(
    instance_memory: InstanceMemory,
    matching_fn,
    step,
    image_goal=None,
    language_goal=None,
    **kwargs
):
    all_matches, all_confidences = [], []
    instances = instance_memory.instance_views[0]
    all_views = []
    instance_view_counts = []
    steps_per_view = []
    for (inst_key, inst) in tqdm(
        instances.items(), desc="Matching goal image with instance views"
    ):
        inst_views = inst.instance_views
        for view_idx, inst_view in enumerate(inst_views):
            # if inst_view.cropped_image.shape[0] * inst_view.cropped_image.shape[1] < 2500 or (np.array(inst_view.cropped_image.shape[0:2]) < 15).any():
            #     continue
            img = instance_memory.images[0][inst_view.timestep].cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            all_views.append(img)
            steps_per_view.append(1000 * step + 10 * inst_key + view_idx)
        instance_view_counts.append(len(inst_views))

    if len(all_views) > 0:
        if image_goal is not None:
            _, _, all_matches, all_confidences = matching_fn(
                all_views,
                goal_image=image_goal,
                goal_image_keypoints=kwargs["goal_image_keypoints"],
                step=1000 * step + 10 * inst_key + view_idx,
            )
        elif language_goal is not None:
            all_matches, all_confidences = matching_fn(
                all_views,
                language_goal,
                step=1000 * step + 10 * inst_key + view_idx,
            )

    # unflatten based on number of views per instance
    all_matches = np.concatenate(all_matches, 0)
    all_confidences = np.concatenate(all_confidences, 0)
    all_matches = np.split(all_matches, np.cumsum(instance_view_counts)[:-1])
    all_confidences = np.split(all_confidences, np.cumsum(instance_view_counts)[:-1])
    return all_matches, all_confidences
