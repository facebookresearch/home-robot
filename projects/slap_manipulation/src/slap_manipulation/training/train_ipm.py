from time import time

import hydra
import numpy as np
import torch
import wandb
import yaml
from slap_manipulation.dataloaders.robot_loader import RobotDataset
from slap_manipulation.policy.interaction_prediction_module import (
    InteractionPredictionModule,
)


@hydra.main(
    version_base=None,
    config_path="./conf",
    config_name="interaction_predictor_training",
)
def main(cfg):
    if cfg.split:
        with open(cfg.split, "r") as f:
            train_test_split = yaml.safe_load(f)
        print(train_test_split)
        valid_list = train_test_split["val"]
        test_list = train_test_split["test"]
        train_list = train_test_split["train"]
    else:
        train_test_split = None
        train_list, valid_list, test_list = None, None, None
    # Set up data augmentation
    # This is a warning for you - if things are not going well
    if cfg.data_augmentation:
        print("-> Using data augmentation on training data.")
    else:
        print("-> NOT using data augmentation.")
    # Set up data loaders
    # Get the robopebn dataset
    # Dataset = RobotDataset
    train_dataset = RobotDataset(
        cfg.datadir,
        trial_list=train_list,
        data_augmentation=cfg.data_augmentation,
        ori_dr_range=np.pi / 8,
        num_pts=8000,
        random_idx=False,
        keypoint_range=[0, 1, 2],
        color_jitter=cfg.color_jitter,
        template=cfg.template,
        dr_factor=5,
    )
    valid_dataset = RobotDataset(
        cfg.datadir,
        num_pts=8000,
        data_augmentation=False,
        trial_list=valid_list,
        keypoint_range=[0, 1, 2],
        color_jitter=False,
        template=cfg.template,
        # template="**/*.h5",
    )
    test_dataset = RobotDataset(
        cfg.datadir,
        num_pts=8000,
        data_augmentation=False,
        trial_list=test_list,
        keypoint_range=[0, 1, 2],
        color_jitter=False,
        template=cfg.template,
        # template="**/*.h5",
    )

    # Create data loaders
    num_workers = 8 if not cfg.debug else 0
    B = 1
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=B,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )

    valid_data = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=B,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
    )

    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=B,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
    )
    # load the model
    model = InteractionPredictionModule(
        xent_loss=cfg.loss_fn == "xent",
        use_proprio=True,
        name=f"classify-{cfg.task_name}",
    )
    model.to(model.device)

    optimizer = model.get_optimizer()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    if cfg.resume:
        print(f"Resuming by loading the best model: {model.get_best_name()}")
        cfg.load = model.get_best_name()
    if cfg.load:
        # load the model now
        model.load_state_dict(torch.load(cfg.load))
        print("--> loaded last best <--")
    if cfg.validate:
        # Make sure we load something
        if not cfg.load:
            cfg.load = "best_%s.pth" % model.name
            print(
                f" --> No model name provided to validate. Using default...{cfg.load}"
            )

        if cfg.load:
            # load the model now
            model.load_state_dict(torch.load(cfg.load))

        with torch.no_grad():
            model.show_validation(train_data, viz=True, viz_mask=True)
    else:
        best_valid_loss = float("Inf")
        print("Starting training")
        if cfg.wandb:
            wandb.init(project="classification-v1", name=f"{model.name}")
            # wandb.config.data_voxel_1 = test_dataset._voxel_size
            # wandb.config.data_voxel_2 = test_dataset._voxel_size_2
            wandb.config.loss_fn = cfg.loss_fn
            wandb.config.loading_best = True

        model.start_time = time()
        for epoch in range(1, cfg.max_iter):
            res, avg_train_dist = model.do_epoch(train_data, optimizer, train=True)
            train_loss = res
            with torch.no_grad():
                res, avg_valid_dist = model.do_epoch(valid_data, optimizer, train=False)
            valid_loss = res
            print("avg train dist:", avg_train_dist)
            print("avg valid dist:", avg_valid_dist)
            if cfg.wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "avg_train_dist": avg_train_dist,
                        "avg_valid_dist": avg_valid_dist,
                    }
                )
            print(
                f"Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}"
            )
            # scheduler.step()
            best_valid_loss, updated = model.smart_save(
                epoch, valid_loss, best_valid_loss
            )
            if not updated:
                print(f"--> reloading best model from: {model.get_best_name()}")
                print(f"--> best loss was {best_valid_loss}")
                model.load_state_dict(torch.load(model.get_best_name()))
            if cfg.run_for > 0 and (time() - model.start_time) > cfg.run_for:
                print(f" --> Stopping training after {cfg.run_for} seconds")
                break

    print(f" --> Stopping training after {cfg.max_iter} iterations")


if __name__ == "__main__":
    main()
