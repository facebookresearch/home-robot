# %%
import wandb

run = wandb.init()
artifact = run.use_artifact(
    "larping/LangConditionedMobileSLAP/bring_x_from_y:v4", type="dataset"
)
artifact_dir = artifact.download(root="../../../datasets")
# %%
