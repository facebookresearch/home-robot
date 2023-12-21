import glob

import clip
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

try:
    from trajectory_datasets.base import resize
except Exception:
    print("[WARNING] could not import mtm encoder!")

from .base_encoder import BaseImageTextEncoder

# THIS PATH IS HARD CODED, you will need to copy the configs here and point to your copies
_MY_PATH = "/private/home/vincentpierre/Documents/eaif/public/eai-foundations/vc_models/src/vc_models/conf/model"
configs = glob.glob(f"{_MY_PATH}/*.yaml")
VC1_CONFIG_PATHS = {}
for c in configs:
    name = c.replace(f"{_MY_PATH}/", "").replace(".yaml", "")
    VC1_CONFIG_PATHS[name] = c


def load_vc(model_name, step=None):
    assert (
        model_name in VC1_CONFIG_PATHS
    ), f"Model {model_name} not found. Model must be a file in minigpt4/configs/models/vc/*.yaml. Available models : {list(VC1_CONFIG_PATHS.keys())}"
    cfg = OmegaConf.load(VC1_CONFIG_PATHS[model_name])
    if step is not None:
        cfg.model.step = step
    return hydra.utils.call(cfg)


class ClipTokenizerWrapper:
    def __call__(self, texts):
        #### https://github.com/openai/CLIP/issues/212
        return clip.tokenize(texts, truncate=True)


class ModelWrapper:
    def __init__(self, mtm_model, encoder_only, device="cuda"):
        self._mtm_model = mtm_model
        self.device = device
        self._clip_model, _ = clip.load("ViT-L/14", device=self.device)
        self.encoder_only = encoder_only

    def encode_text(self, text_tokens):
        with torch.no_grad():
            return self._clip_model.encode_text(text_tokens).to(self.device)

    def encode_image(self, images):
        with torch.no_grad():
            if self.encoder_only:
                assert hasattr(
                    self._mtm_model._model, "contrastive_image_encoder"
                ), "Model set to use encoder only but model does not have proper contrastive loss"
                code = self._mtm_model(images)
                code = self._mtm_model._model.contrastive_image_encoder(code)
            else:
                _input = {
                    "images": images.unsqueeze(1),
                    "label": ["foo"] * images.shape[0],
                }  # the unsqueeze is for the sequence length of 1
                _input = self._mtm_model._tokenizer_manager.encode(_input)
                # _masks = {"label":torch.ones(1).to(self.device), "images":torch.zeros(_input["images"].shape[2]).to(self.device)}
                _masks = {
                    "label": torch.zeros(1).to(self.device),
                    "images": torch.ones(_input["images"].shape[2]).to(self.device),
                }
                decoded_trajs = self._mtm_model._model.forward(
                    trajectories=_input, masks=_masks
                )
                code = decoded_trajs["label"].squeeze(1)
            return code

    def eval(self):
        pass


# # Don't forget to activate or create a conda environment !
# conda create -n my_mtm_env python=3.9 -y
# conda activate my_mtm_env
# git clone https://github.com/facebookresearch/eai-foundations.git
# cd eai-foundations

# pip install -e ./vc_models
# pip install -e ./trajectory_datasets
# pip install git+https://github.com/openai/CLIP.git
# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116


# THIS IS THE ENCODER
class HomeRobotMTMEncoder(BaseImageTextEncoder):
    def __init__(self, version="oct1_laiononly_250k", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model, _, transform, _ = load_vc(version)
        self.transform = transform
        self._model_wrapper = ModelWrapper(model, True, device=device)

    def encode_image(self, image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy() * 255
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        transformed_image = self.transform(pil_image)
        transformed_image = transformed_image.unsqueeze(0)
        return self._model_wrapper.encode_image(transformed_image)

    def encode_text(self, text):
        text = clip.tokenize([text]).to(self.device)
        return self._model_wrapper.encode_text(text)


def test_models():
    import numpy as np

    image = np.random.random((32, 32, 3))
    text = "Is this really necessary ?"

    mtm_encoder = HomeRobotMTMEncoder(device="cuda")
    mtm_im = mtm_encoder.encode_image(image)
    print(mtm_im.shape)
    mtm_text = mtm_encoder.encode_text(text)
    print(mtm_text.shape)


if __name__ == "__main__":
    test_models()
