# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Script to download LLM dataset from wandb"""
# %%
import wandb

run = wandb.init()
artifact = run.use_artifact(
    "larping/LangConditionedMobileSLAP/bring_x_from_y:v4", type="dataset"
)
artifact_dir = artifact.download(root="../../../datasets")
# %%
