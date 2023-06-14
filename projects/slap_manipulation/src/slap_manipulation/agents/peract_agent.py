import numpy as np
from slap_manipulation.agents.slap_agent import SLAPAgent
from slap_manipulation.policy.peract import PerceiverActorAgent, PerceiverIO


class PeractAgent(SLAPAgent):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def load_models(self):
        # initialize PerceiverIO Transformer
        self.perceiver_encoder = PerceiverIO(
            depth=6,
            iterations=1,
            voxel_size=self.cfg.PERACT.voxel_sizes[0],
            initial_dim=3 + 3 + 1 + 3,
            low_dim_size=3,
            layer=0,
            num_rotation_classes=72,
            num_grip_classes=2,
            num_collision_classes=2,
            num_latents=self.cfg.PERACT.num_latents,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
            activation="lrelu",
            input_dropout=0.1,
            attn_dropout=0.1,
            decoder_dropout=0.0,
            voxel_patch_size=5,
            voxel_patch_stride=5,
            final_dim=64,
        )
        # initialize PerceiverActor
        self.peract_agent = PerceiverActorAgent(
            coordinate_bounds=self.cfg.PERACT.scene_bounds,
            perceiver_encoder=self.perceiver_encoder,
            camera_names=self.cfg.PERACT.cameras,
            batch_size=1,
            voxel_size=self.cfg.PERACT.voxel_sizes[0],
            voxel_feature_size=3,
            num_rotation_classes=72,
            rotation_resolution=5,
            lr=0.0001,
            image_resolution=[128, 128],
            lambda_weight_l2=0.000001,
            transform_augmentation=False,
            optimizer_type="lamb",
            num_pts=8000,
        )
        self.peract_agent.build(training=True, device=self.device)
        self.peract_agent.load_weights(self.cfg.PERACT.model_path)
        print(f"---> loaded last best {self.cfg.PERACT.model_path} <---")

    def create_peract_input(self, input, t=0, num_actions=6):
        model_input = {}
        open_gripper_width = 0.16
        closed_gripper_width = -0.16
        closed_gripper_pour = 0.041
        if "pour" in input["lang"]:
            curr_gripper_width = closed_gripper_pour
            inferred_gripper_state = 1
        elif input["gripper"] < -0.0020:
            curr_gripper_width = closed_gripper_width
            inferred_gripper_state = 1
        else:
            curr_gripper_width = open_gripper_width
            inferred_gripper_state = 0
        time_index = float(2 * t / num_actions)
        gripper_states = np.array(
            [curr_gripper_width, inferred_gripper_state, time_index]
        )
        model_input["cmd"] = input["lang"]
        model_input["gripper_states"] = gripper_states
        model_input = self.to_torch(model_input)
        input.update(model_input)
        return input

    def predict(self, obs):
        info = {}
        action = None
        self._input = self.create_interaction_prediction_input_from_obs(
            obs, filter_depth=True, debug=False
        )
        for t in range(obs.task_observations["num-actions"]):
            self.model_input = self.create_peract_input(
                self._input, t=t, num_actions=obs.task_observations["num-actions"]
            )
            self.peract_agent.update_for_rollout(
                self.model_input, center=self.model_input["mean"].detach().cpu().numpy()
            )
