import torch

from .modules.models.lseg_net import LSegEncDecNet


def load_lseg_encoder_for_inference(checkpoint_path: str,
                                    device: torch.device,
                                    crop_size: int = 480,
                                    visualize=True) -> LSegEncDecNet:
    model = LSegEncDecNet(arch_option=0,
                          block_depth=0,
                          activation='lrelu',
                          crop_size=crop_size,
                          visualize=visualize)

    model_state_dict = model.state_dict()
    pretrained_state_dict = torch.load(checkpoint_path)
    pretrained_state_dict = {k.lstrip('net.'): v
                             for k, v in pretrained_state_dict['state_dict'].items()}
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict)

    model.eval()
    model = model.to(device)
    return model
