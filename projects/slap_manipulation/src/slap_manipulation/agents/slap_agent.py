import numpy as np
import torch
from slap_manipulation.dataloaders.data_processing import (
    combine_multiple_views,
    compute_detic_features,
    drop_frames_from_input,
)
from slap_manipulation.dataloaders.robot_loader import RobotDataset

from home_robot.perception.detection.detic.detic_perception import DeticPerception

# from slap_manipulation.policy.action_prediction_module import \
#     ActionPredictionModule
# from slap_manipulation.policy.interaction_prediction_module import \
#     InteractionPredictionModule

REAL_WORLD_CATEGORIES = [
    "cup",
    "bottle",
    "drawer",
    "basket",
    "bowl",
]
VOXEL_SIZE_1 = 0.001
VOXEL_SIZE_2 = 0.01


class SLAPAgent(object):
    def __init__(self, version, drop_frames=False):
        # self.ipm = InteractionPredictionModule(
        #     use_proprio=True, name="test-ipm-robopen"
        # )
        # self.apm = []
        # self.keypoints = 3
        # for k in range(self.keypoints):
        #     self.apm.append(
        #         ActionPredictionModule(
        #             name=f"test-apm-robopen-{k}",
        #             multi_head=False,
        #             num_heads=3,
        #             orientation_type="quaternion",
        #             use_cropped_pcd=True if version == "slap" else False,
        #         )
        #     )
        self.ipm = None
        self.apm = None
        self.device = "cuda"
        self.version = version
        self.drop_frames = drop_frames
        self.segmentor = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=",".join(REAL_WORLD_CATEGORIES),
            sem_gpu_id=0,
        )

    # TODO move initializing of entire model to this method, rename "load_model"
    def load(self, ipm_path, apm_paths):
        if self.ipm and self.apm:
            self.ipm.load_state_dict(torch.load(ipm_path))
            for k, apm_path in enumerate(apm_paths):
                self.apm[k].load_state_dict(torch.load(apm_path))

    def model_to_device(self):
        if self.ipm and self.apm:
            self.ipm.to(self.device)
            for model in self.apm:
                model.to(self.device)

    def preprocessing(self, data):
        """
        agent-specific routine for preprocessing input data

        input:
            data is a dict containing:
                rgb_images
                depth_images
                computed_xyz
                interaction_ee_keyframe
                action_ee_keyframes (1 or many depending upon training; start with 1)
        """
        input_rgb_images = data["rgb_images"]
        input_xyz = data["computed_xyzs"]
        input_depth_images = data["depth_images"]
        if self.drop_frames:
            input_xyz, input_rgb_images, input_depth_images = drop_frames_from_input(
                input_xyz, input_rgb_images, input_depth_images
            )
        # get dense detic features
        features = compute_detic_features(
            input_rgb_images, input_depth_images, self.segmentor
        )
        features = np.expand_dims(features, axis=-1)
        rgb, xyz, features = combine_multiple_views(
            input_xyz, input_rgb_images, input_depth_images, features
        )

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


def test_processing():
    data_dir = "/home/priparashar/h5_test/pick_up_bottle/03-26/"
    template = "*.h5"
    robot = "stretch"
    loader = RobotDataset(
        data_dir,
        template=template,
        num_pts=8000,
        data_augmentation=True,
        crop_radius=True,
        ori_dr_range=np.pi / 8,
        cart_dr_range=0.0,
        first_frame_as_input=False,
        # first_keypoint_only=True,
        keypoint_range=[0, 1, 2],
        trial_list=[],
        orientation_type="quaternion",
        show_voxelized_input_and_reference=True,
        show_cropped=True,
        verbose=False,
        multi_step=False,
        visualize_interaction_estimates=True,
        visualize_cropped_keyframes=True,
        robot=robot,
    )
    agent = SLAPAgent("slap")
    for trial in loader.trials:
        print(f"Trial: {trial.name}")
        print(f"Task name: {trial.h5_filename}")
        num_keypt = trial.num_keypoints
        for i in range(num_keypt):
            print("Keypoint requested: ", i)
            data = loader.get_datum(trial, i, verbose=True)
            agent.preprocessing(data)
        # data = loader.get_datum(trial, 1, verbose=False)


if __name__ == "__main__":
    test_processing()
