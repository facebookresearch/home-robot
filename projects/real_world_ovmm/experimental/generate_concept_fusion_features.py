import os
from pathlib import Path
from typing import List

import h5py
import hydra
import matplotlib
import numpy as np
from omegaconf import DictConfig
import open_clip
import torch

from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from tqdm import trange

import seaborn as sns

custom_palette = sns.color_palette("viridis", 24)


def save_data(
        original_image: torch.Tensor,
        masks: List[dict],
        outfeat: torch.Tensor,
        idx: int,
        save_path: str,
    ):
    """
    Save original image, segmentation masks, and concept fusion features.

    Args:
        original_image (torch.Tensor): Original image.
        masks (list[dict]): List of segmentation masks.
        outfeat (torch.Tensor): Concept fusion features.
        idx (int): Index of image.
        save_path (str): Path to save data.
    """
    # create directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save original image
    original_image = Image.fromarray(original_image)
    file_path = os.path.join(save_path, "original_" + str(idx) + ".png")
    original_image.save(file_path)

    # save segmentation masks
    segmentation_image = torch.zeros(original_image.size[0], original_image.size[0], 3)
    for i, mask in enumerate(masks):
        segmentation_image += torch.from_numpy(mask["segmentation"]).unsqueeze(-1).repeat(1, 1, 3).float() * \
            torch.tensor(custom_palette[i%24]) * 255.0

    mask = Image.fromarray(segmentation_image.numpy().astype("uint8"))
    file_path = os.path.join(save_path, "mask_" + str(idx) + ".png")
    mask.save(file_path)

    # save concept fusion features
    file_path = os.path.join(save_path, "concept_fusion_features_" + str(idx) + ".pt")
    torch.save(outfeat.detach().cpu(), file_path)

class ConceptFusion:
    def __init__(
            self,
            args: DictConfig,
        ) -> None:
        """
        Initialize concept fusion model.

        Args:
            args (DictConfig): Hydra config.
        """
        self.args = args

        sam = sam_model_registry[args.model_type](checkpoint=Path(args.SAM_checkpoint_path))
        sam.to(device=args.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=args.points_per_side,
            pred_iou_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

        print(
            f"Initializing OpenCLIP model: {args.open_clip_model}"
            f" pre-trained on {args.open_clip_pretrained_dataset}..."
        )

        self.CLIP_model, _, self.CLIP_preprocess = open_clip.create_model_and_transforms(
            args.open_clip_model, args.open_clip_pretrained_dataset
        )
        self.CLIP_model.cuda()
        self.CLIP_model.eval()

        self.tokenizer = open_clip.get_tokenizer(args.open_clip_model)

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        self.feat_dim = None
    
    def generate_mask(self, img):
        masks = self.mask_generator.generate(img)
        
        # remove masks with zero area
        masks = list(filter(lambda x: x["bbox"][2] * x["bbox"][3] != 0, masks))
        
        return masks
    
    def generate_global_features(
        self,
        img: Image,
    ):
        # CLIP features global
        global_feat = None
        with torch.cuda.amp.autocast():
            _img = self.CLIP_preprocess(Image.fromarray(img)).unsqueeze(0)
            global_feat = self.CLIP_model.encode_image(_img.cuda())
            global_feat /= global_feat.norm(dim=-1, keepdim=True)

        global_feat = global_feat.half().cuda()
        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)

        if self.feat_dim is None:
            self.feat_dim = global_feat.shape[-1]

        return global_feat
    
    def generate_local_features(
        self,
        img: Image,
        masks: List[dict],
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate concept fusion features.

        Args:
            img (Image): Original image.
            masks (list[dict]): List of segmentation masks.
            global_feat (torch.Tensor): CLIP features global.

        Returns:
            torch.Tensor: Concept fusion features.
        """
        load_image_height, load_image_width = img.shape[0], img.shape[1]
        
        # CLIP features per ROI
        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []
        outfeat = torch.zeros(load_image_height, load_image_width, self.feat_dim, dtype=torch.half)

        for mask in masks:
            _x, _y, _w, _h = tuple(mask["bbox"])  # xywh bounding box

            # make sure _x, _y, _w, _h are ints
            _x, _y, _w, _h = int(_x), int(_y), int(_w), int(_h)

            nonzero_inds = torch.argwhere(torch.from_numpy(mask["segmentation"]))

            # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
            img_roi = img[_y : _y + _h, _x : _x + _w, :]

            img_roi = Image.fromarray(img_roi)
            img_roi = self.CLIP_preprocess(img_roi).unsqueeze(0).cuda()
            roifeat = self.CLIP_model.encode_image(img_roi)
            roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = self.cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)
        
        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)

        for maskidx in range(len(masks)):

            _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach().cpu().half()
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
                outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
            ).half()

        outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        outfeat = torch.nn.functional.interpolate(outfeat, [self.args.desired_height, self.args.desired_width], mode="nearest")
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0].half() # --> H, W, feat_dim

        return outfeat

    def text_query(
        self,
        query: str,
        map_features: torch.Tensor,
    ):
        text = self.tokenizer([query])
        textfeat = self.CLIP_model.encode_text(text.cuda())
        textfeat = torch.nn.functional.normalize(textfeat, dim=-1)
        textfeat = textfeat.unsqueeze(0)

        # make sure features are on cuda
        map_features = map_features.cuda()
        map_features = torch.nn.functional.normalize(map_features, dim=-1)

        similarity = self.cosine_similarity(textfeat, map_features)

        print("Max similarity: {}".format(similarity.max()))
        map_colors = np.zeros((similarity.shape[1], 3))

        if self.args.viz_type == "topk":
            # Viz topk points
            _, topk_ind = torch.topk(similarity, self.args.topk)
            map_colors[topk_ind.detach().cpu().numpy()] = np.array([1.0, 0.0, 0.0])

        elif self.args.viz_type == "thresh":
            # Viz thresholded "relative" attention scores
            similarity = (similarity + 1.0) / 2.0  # scale from [-1, 1] to [0, 1]
            # similarity = similarity.clamp(0., 1.)
            similarity_rel = (similarity - similarity.min()) / (
                similarity.max() - similarity.min() + 1e-12
            )
            similarity_rel[similarity_rel < self.args.similarity_thresh] = 0.0

            cmap = matplotlib.cm.get_cmap("jet")
            similarity_colormap = cmap(similarity_rel[0].detach().cpu().numpy())[:, :3]

            map_colors = 0.5 * map_colors + 0.5 * similarity_colormap

        return map_colors


@hydra.main(config_path="configs", config_name="concept_fusion")
def main(args: DictConfig):
    """
    Generate concept fusion features for a given episode file.
    
    Args:
        args (DictConfig): Hydra config.
    """

    torch.autograd.set_grad_enabled(False)

    dataset = h5py.File(args.episode_file, "r")["images"]

    # initialize concept fusion model
    concept_fusion = ConceptFusion(args)

    print("Extracting SAM masks...")
    for idx in trange(len(dataset)):
        img = dataset[idx]
        
        masks = concept_fusion.generate_mask(img)

        # CLIP features global
        global_feat = concept_fusion.generate_global_features(img)

        # CLIP features per ROI
        outfeat = concept_fusion.generate_local_features(img, masks, global_feat)

        # save data
        save_data(img, masks, outfeat, idx, args.save_path)
        


if __name__ == "__main__":
    main()