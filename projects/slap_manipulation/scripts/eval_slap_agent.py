from pprint import pprint

import click
import numpy as np
import torch
import yaml
from slap_manipulation.agents.slap_agent import SLAPAgent


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
    slap_model = SLAPAgent(version)
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
