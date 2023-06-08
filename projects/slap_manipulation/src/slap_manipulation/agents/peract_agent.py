from slap_manipulation.agents.slap_agent import SLAPAgent
from slap_manipulation.policy.peract import PerceiverActorAgent


class PeractAgent(SLAPAgent):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def load_models(self):
        # initialize PerceiverIO Transformer
        self.perceiver_encoder = PerceiverIO(
            depth=6,
            iterations=1,
            voxel_size=VOXEL_SIZES[0],
            initial_dim=3 + 3 + 1 + 3,
            low_dim_size=3,
            layer=0,
            num_rotation_classes=72,
            num_grip_classes=2,
            num_collision_classes=2,
            num_latents=NUM_LATENTS,
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
            camera_names=CAMERAS,
            batch_size=BATCH_SIZE,
            voxel_size=ds.voxel_sizes[0],
            voxel_feature_size=3,
            num_rotation_classes=72,
            rotation_resolution=5,
            lr=0.0001,
            image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
            lambda_weight_l2=0.000001,
            transform_augmentation=False,
            optimizer_type="lamb",
            num_pts=8000,
        )
        self.peract_agent.build(training=True, device=device)
