from pprint import pprint

import click
import numpy as np
import torch
import yaml

from home_robot.datasets.robopen_loader import (  # TODO change to robotloader in this proj
    RoboPenDataset,
)
from home_robot.policy.pt_query import QueryPointnet  # TODO this becomes IPM
from home_robot.policy.pt_regression_model import (  # TODO this becomes APM
    QueryRegressionModel,
)


class SLAP(object):
    def __init__(self, version):
        self.ipm = QueryPointnet(use_proprio=True, name="test-ipm-robopen")
        self.apm = []
        self.keypoints = 3
        for k in range(self.keypoints):
            self.apm.append(
                QueryRegressionModel(
                    name=f"test-apm-robopen-{k}",
                    multi_head=False,
                    num_heads=3,
                    orientation_type="quaternion",
                    use_cropped_pcd=True if version == "slap" else False,
                )
            )
        self.device = "cuda"
        self.version = version

    def load(self, ipm_path, apm_paths):
        self.ipm.load_state_dict(torch.load(ipm_path))
        for k, apm_path in enumerate(apm_paths):
            self.apm[k].load_state_dict(torch.load(apm_path))

    def model_to_device(self):
        self.ipm.to(self.device)
        for model in self.apm:
            model.to(self.device)

    def crop_around_voxel(self, xyz, rgb, p_i, radius=0.1):
        mask = np.linalg.norm(xyz - p_i, axis=1) < radius
        return xyz[mask], rgb[mask]

    def predict(self, batch):
        batch = self.to_device(batch)
        rgb = batch["rgb"][0]  # N x 3
        xyz = batch["xyz"][0]  # N x 3
        down_xyz = batch["xyz_downsampled"][0]
        down_rgb = batch["rgb_downsampled"][0]
        lang = batch["cmd"]  # list of 1
        target_idx = batch["closest_voxel_idx"][0]
        if self.ipm.use_proprio:
            proprio = batch["proprio"][0]
        else:
            proprio = None
        classification_probs, _, _ = self.ipm.forward(
            rgb,
            down_rgb,
            xyz,
            down_xyz,
            lang,
            proprio,
        )
        predicted_idx = torch.argmax(classification_probs, dim=-1)
        p_i = down_xyz[predicted_idx].detach().cpu().numpy()
        ipm_dist = np.linalg.norm(down_xyz[target_idx].detach().cpu().numpy() - p_i)
        k = batch["keypoint_idx"][0]
        if self.version == "slap" or self.version == "mixed-slap":
            crop_xyz, crop_rgb = self.crop_around_voxel(
                xyz.detach().cpu().numpy(),
                rgb.detach().cpu().numpy(),
                p_i,
            )
            positions, _, _ = self.apm[k].forward(
                torch.FloatTensor(crop_xyz).to(self.device),
                torch.FloatTensor(crop_rgb).to(self.device),
                proprio,
                lang,
            )
        else:
            positions, _, _ = self.apm[k].forward(
                xyz,
                rgb,
                proprio,
                lang,
            )
        target_pos = batch["ee_keyframe_pos"].detach().cpu().numpy()
        pred_ee_pos = p_i + positions.detach().cpu().numpy().reshape(1, 3)

        apm_dist = ((target_pos - pred_ee_pos) ** 2).sum()
        return ipm_dist, apm_dist

    def to_device(self, batch):
        new_batch = {}
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                new_batch[k] = v
            else:
                new_batch[k] = v.to(self.device)
        return new_batch

    def to_torch(self, batch):
        new_batch = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                new_batch[k] = torch.FloatTensor(v)
            else:
                new_batch[k] = v
        return new_batch


@click.command()
@click.option(
    "--version",
    default="slap",
    type=click.Choice(["slap", "ablated-slap", "mixed-slap"]),
)
def main(version):
    ROBOPEN_TASKS = {
        "place_in_drawer": "/home/priparashar/robopen_h5s/larp/rss_rebuttal/place_in_drawer",
        "open_top_drawer": "/home/priparashar/robopen_h5s/larp/rss_rebuttal/open_top_drawer",
        "close_drawer": "/home/priparashar/robopen_h5s/larp/rss_rebuttal/close_drawer",
        "place_in_basket": "/home/priparashar/robopen_h5s/larp/rss_rebuttal/place_in_basket",
    }
    IPM_PATH_SLAP = "./ipm_ablations/ipm-04-11-1-robopen/best_ipm-04-11-1-robopen.pth"
    IPM_PATH_ABLATED_SLAP = (
        "./ipm_ablations/ipm-04-11-2-robopen/best_ipm-04-11-2-robopen.pth"
    )
    APM_PATH_SLAP = [
        "./apm_ablations/action_predictor_robopen_key0_ablation1_2023-04-08_23-02/best_action_predictor_robopen_key0_ablation1.pth",
        "./apm_ablations/action_predictor_robopen_key1_ablation1_2023-04-08_23-14/best_action_predictor_robopen_key1_ablation1.pth",
        "./apm_ablations/action_predictor_robopen_key2_ablation1_2023-04-08_23-25/best_action_predictor_robopen_key2_ablation1.pth",
    ]
    APM_PATH_ABLATED_SLAP = [
        "./apm_ablations/action_predictor_robopen_key0_ablation3_2023-04-08_19-39/best_action_predictor_robopen_key0_ablation3.pth",
        "./apm_ablations/action_predictor_robopen_key1_ablation3_2023-04-08_19-43/best_action_predictor_robopen_key1_ablation3.pth",
        "./apm_ablations/action_predictor_robopen_key2_ablation3_2023-04-08_19-45/best_action_predictor_robopen_key2_ablation3.pth",
    ]
    split = "./assets/task_splits/test_heavy_split.yaml"
    with open(split, "r") as f:
        train_test_split = yaml.safe_load(f)

    ds = {}
    for task, path in ROBOPEN_TASKS.items():
        ds[task] = RoboPenDataset(
            path,
            trial_list=train_test_split["test"],
            num_pts=8000,
            data_augmentation=False,
            keypoint_range=[0, 1, 2],
            template="*.h5",
            verbose=True,
        )
    test_loader = {}
    for task, dataset in ds.items():
        test_loader[task] = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=8,
            shuffle=False,
        )

    # load SLAP model
    slap_model = SLAP(version)
    if version == "slap":
        ipm_path = IPM_PATH_SLAP
        apm_paths = APM_PATH_SLAP
    elif version == "ablated-slap":
        ipm_path = IPM_PATH_ABLATED_SLAP
        apm_paths = APM_PATH_ABLATED_SLAP
    else:
        ipm_path = IPM_PATH_ABLATED_SLAP
        apm_paths = APM_PATH_SLAP
    slap_model.load(ipm_path, apm_paths)
    slap_model.model_to_device()

    metrics = {}
    for task, loader in test_loader.items():
        print(f"Task: {task}")
        ipm_tot = 0.0
        apm_tot = 0.0
        samples = 0
        for batch in loader:
            ipm_dist, apm_dist = slap_model.predict(batch)
            ipm_tot += ipm_dist
            apm_tot += apm_dist
            samples += 1
        metrics[task] = {
            "ipm": ipm_tot / samples,
            "apm": apm_tot / samples,
        }
    pprint(metrics)


if __name__ == "__main__":
    main()
