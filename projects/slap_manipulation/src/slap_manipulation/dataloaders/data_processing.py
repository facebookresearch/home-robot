import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from home_robot.utils.image import rotate_image


class DataProcessor(object):
    def __init__(self) -> None:
        # setup segmentation pipeline
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.segmentor = SamAutomaticMaskGenerator(self.sam)
        # setup clip
        with torch.no_grad():
            self.clip_model = clip.load("ViT-B/32", device=device)[0]
        self.clip_image_size = (224, 224)
        self._clip_emb_dim = 512
        self.preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224))]
        )

    def get_dense_clip_features(self, data):
        # create per-point CLIP vectors before masking and downsampling
        # add all features to img_features vector based on idx
        # also keep count of number of overlapping features added to img_features (for averaging)

        raw_rgb = rotate_image([raw_rgb])[0]
        masks = self.segmentor.generate(raw_rgb)
        count_features = torch.tensor(np.zeros((raw_rgb.shape[0], raw_rgb.shape[1], 1)))
        image_features = torch.tensor(
            np.zeros((raw_rgb.shape[0], raw_rgb.shape[1], self._clip_emb_dim)),
            dtype=torch.float16,
        )

        plt.imshow(raw_rgb)
        self.show_anns(masks)
        plt.axis("off")
        plt.show()

        for ann in masks:
            masked_img = raw_rgb[
                ann["bbox"][1] : ann["bbox"][3], ann["bbox"][0] : ann["bbox"][2]
            ].copy()
            if masked_img.shape[0] == 0 or masked_img.shape[1] == 0:
                continue
            resized_masked_img = self.preprocess(masked_img).unsqueeze(0).cuda()
            masked_img_emb = self.clip_model.encode_image(resized_masked_img).cpu()
            image_features[ann["segmentation"]] = masked_img_emb
            count_features[ann["segmentation"]] += 1
        image_features = image_features / count_features

    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann["segmentation"]
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.35)))

    def process_training_data(self, data):
        """
        gets batch from dataloader and processes it to return:
            - voxelized point clouds at two resolutions
            - dense clip features associated with each point
            - clip task-name embedding
            - keyframe targets
            - index for interaction point

        """
        x_mask = xyz[:, 0] < 0.9
        rgb = rgb[x_mask]
        xyz = xyz[x_mask]
        xyz, rgb = xyz.reshape(-1, 3), rgb.reshape(-1, 3)

        # get EE keyframe
        current_ee_keyframe = self.get_gripper_pose(trial, int(current_keypoint_idx))
        interaction_ee_keyframe = self.get_gripper_pose(trial, int(interaction_pt))
        all_ee_keyframes = []
        if self.multi_step:
            target_gripper_state = np.zeros(len(keypoints))
            # Add all keypoints to this list
            for j, keypoint in enumerate(keypoints):
                all_ee_keyframes.append(self.get_gripper_pose(trial, int(keypoint)))
                target_gripper_state[j] = gripper_state[keypoint]
        else:
            # Pull out gripper state from the sim data
            target_gripper_state = gripper_state[current_keypoint_idx]

        # voxelize at a granular voxel-size then choose X points
        xyz, rgb = self.remove_duplicate_points(xyz, rgb)
        xyz, rgb = self.dr_crop_radius(xyz, rgb, interaction_ee_keyframe)
        orig_xyz, orig_rgb = xyz, rgb

        # Get the point clouds and shuffle them around a bit
        xyz, rgb, center = self.shuffle_and_downsample_point_cloud(xyz, rgb)

        # mean-center the keyframes wrt classifier-input pcd
        orig_xyz -= center[None].repeat(orig_xyz.shape[0], axis=0)
        current_ee_keyframe[:3, 3] -= center
        interaction_ee_keyframe[:3, 3] -= center
        for keyframe in all_ee_keyframes:
            keyframe[:3, 3] -= center

        (
            orig_xyz,
            xyz,
            current_ee_keyframe,
            interaction_ee_keyframe,
            all_ee_keyframes,
        ) = self.dr_rotation_translation(
            orig_xyz,
            xyz,
            current_ee_keyframe,
            interaction_ee_keyframe,
            all_ee_keyframes,
        )

        (
            xyz2,
            rgb2,
            target_idx_down_pcd,
            closest_pt_down_pcd,
            target_idx_og_pcd,
            closest_pt_og_pcd,
        ) = self.voxelize_and_get_interaction_point(xyz, rgb, interaction_ee_keyframe)
        if xyz2 is None:
            print("Couldn't find an interaction point")
            return {"data_ok_status": False}

        # Get the local version of the problem
        (crop_location, crop_xyz, crop_rgb, data_status) = self.get_local_problem(
            orig_xyz, orig_rgb, closest_pt_down_pcd
        )
        if verbose:
            print(f"Size of cropped xyz: {crop_xyz.shape}")
        # Get data for the regression training
        # This needs to happen before centering i guess
        (
            crop_ref_ee_keyframe,
            crop_ee_keyframe,
            crop_keyframes,
        ) = self.get_local_commands(
            crop_location,
            current_ee_keyframe,
            interaction_ee_keyframe,
            all_ee_keyframes,
        )

        positions, orientations, angles = self.get_commands(
            crop_ee_keyframe, crop_keyframes
        )
        self._assert_positions_match_ee_keyframes(crop_ee_keyframe, positions)

        if self._visualize_interaction_estimates:
            self.show_interaction_pt_and_keyframe(
                xyz2,
                rgb2,
                current_ee_keyframe,
                closest_pt_down_pcd,
                interaction_ee_keyframe,
            )
            print(
                "Showing current ee keyframe as the coordinate-frame and the interaction-point in PCD as yellow sphere"
            )
            show_point_cloud_with_keypt_and_closest_pt(
                xyz2,
                rgb2,
                current_ee_keyframe[:3, 3],
                current_ee_keyframe[:3, :3],
                closest_pt_down_pcd,
            )
            print(
                "Showing current ee keyframe as the coordinate-frame and the interaction-ee-position as yellow sphere"
            )
            show_point_cloud_with_keypt_and_closest_pt(
                xyz2,
                rgb2,
                current_ee_keyframe[:3, 3],
                current_ee_keyframe[:3, :3],
                interaction_ee_keyframe[:3, 3],
            )

        if self._visualize_cropped_keyframes:
            self.show_cropped_keyframes(
                crop_xyz, crop_rgb, crop_ee_keyframe, crop_ref_ee_keyframe
            )

        datum = {
            "trial_name": trial.name,
            "data_ok_status": data_status,
            # ----------
            "ee_keyframe_pos": torch.FloatTensor(current_ee_keyframe[:3, 3]),
            "ee_keyframe_ori": torch.FloatTensor(current_ee_keyframe[:3, :3]),
            "proprio": torch.FloatTensor(proprio),
            "target_gripper_state": torch.FloatTensor(target_gripper_state),
            "xyz": torch.FloatTensor(xyz),
            "rgb": torch.FloatTensor(rgb),
            "raw_rgb": torch.FloatTensor(raw_rgb),
            "cmd": cmd,
            "image_size": (self._image_height, self._image_width, self._image_channels),
            "keypoint_idx": keypoint_idx,
            # engineered features ----------------
            "closest_pos": torch.FloatTensor(closest_pt_og_pcd),
            "closest_pos_idx": torch.LongTensor([target_idx_og_pcd]),
            "closest_voxel": torch.FloatTensor(closest_pt_down_pcd),
            "closest_voxel_idx": torch.LongTensor([target_idx_down_pcd]),
            "xyz_downsampled": torch.FloatTensor(xyz2),
            "rgb_downsampled": torch.FloatTensor(rgb2),
            # used in pt_query.py; make sure this is being used with xyz_downsampled
            # TODO rename xyz_mask --> xyz_downsampled_mask to remove confusion
            "xyz_mask": torch.LongTensor(self.mask_voxels(xyz2, target_idx_down_pcd)),
            # Crop inputs -----------------
            "rgb_crop": torch.FloatTensor(crop_rgb),
            "xyz_crop": torch.FloatTensor(crop_xyz),
            "crop_ref_ee_keyframe_pos": torch.FloatTensor(crop_ref_ee_keyframe[:3, 3]),
            "crop_ref_ee_keyframe_ori": torch.FloatTensor(crop_ref_ee_keyframe[:3, :3]),
            "perturbed_crop_location": torch.FloatTensor(crop_location),
            # Crop goals ------------------
            # Goals for regression go here
            "ee_keyframe_pos_crop": torch.FloatTensor(positions),
            "ee_keyframe_ori_crop": torch.FloatTensor(orientations),
            "target_ee_angles": torch.FloatTensor(angles),
        }
        return datum

    def process_inference_data(self, data):
        pass
