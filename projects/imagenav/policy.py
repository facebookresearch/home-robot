from collections import OrderedDict

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict

import hydra
from omegaconf import OmegaConf

from habitat.config.default import Config as CN
from habitat_baselines.utils.common import batch_obs

from habitat_eaif.config import get_config
from habitat_eaif.rl.policy import EAIPolicy


# Turn numpy observations into torch tensors for consumption by policy
def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


class CortexPolicy:
    def __init__(
        self,
        config,
        checkpoint,
        observation_space,
        action_space,
        device,
        policy_class=EAIPolicy,
    ):
        print("Loading policy...")
        self.device = torch.device(device)

        self.policy = policy_class.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        # Move it to the device
        self.policy.to(self.device)

        # Load trained weights into the policy
        self.policy.load_state_dict(
           {k[len("actor_critic.") :]: v for k, v in checkpoint["state_dict"].items()}
        )

        self.prev_actions = None
        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.config = config
        self.num_actions = action_space.n
        self.reset_ran = False
        print("Policy loaded.")

    def reset(self):
        self.reset_ran = True
        self.test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments. Just one for real world.
            self.policy.net.num_recurrent_layers,
            self.config.RL.POLICY.hidden_size,
            device=self.device,
        )

        # We start an episode with 'done' being True (0 for 'not_done')
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, 1, device=self.device, dtype=torch.long)

    def act(self, observations):
        assert self.reset_ran, "You need to call .reset() on the policy first."
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            _, actions, _, self.test_recurrent_hidden_states = self.policy.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=True,
            )

        self.prev_actions.copy_(actions)
        self.not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        # GPU/CPU torch tensor -> numpy
        actions = actions.squeeze().cpu().numpy()

        return actions

@hydra.main(config_path="configs", config_name="config_imagenav_stretch")
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = CN(cfg)

    config = get_config()
    config.merge_from_other_cfg(cfg)

    h, w = (
        640,
        480,
    )

    obs_space = {
        "rgb": spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 3),
            dtype=np.uint8,
        ),
        "imagegoalrotation": spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 3),
            dtype=np.uint8,
        ),
    }
    obs_space = spaces.Dict(obs_space)

    action_space = spaces.Discrete(4)

    checkpoint = torch.load(
        config.checkpoint_path,
        map_location="cpu"
    )

    inav_policy = CortexPolicy(
        config,
        checkpoint,
        obs_space,
        action_space,
        device="cpu",
    )
    inav_policy.reset()
    observations = {
        "rgb": np.zeros([h, w, 3], dtype=np.uint8),
        "imagegoalrotation": np.zeros([h, w, 3], dtype=np.uint8),
    }
    actions = inav_policy.act(observations)
    print("actions:", actions)

if __name__ == "__main__":
    main()