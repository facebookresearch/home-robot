import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================== Sub-parts of the U-Net model ============================


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class embedconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


# =================================== Component modules ===============================


class UNetEncoder(nn.Module):
    def __init__(
        self,
        n_channels,
        nsf=16,
        embedding_size=64,
        map_size=400,
    ):
        super().__init__()
        self.embed = embedconv(n_channels, embedding_size)
        self.nsf = nsf
        self.inc = inconv(embedding_size, nsf)
        self.down1 = down(nsf, nsf * 2)
        self.down2 = down(nsf * 2, nsf * 4)
        self.down3 = down(nsf * 4, nsf * 8)
        self.down4 = down(nsf * 8, nsf * 8)
        self.map_size = map_size
        self.n_channels = n_channels

    def forward(self, x):
        x = self.embed(x)
        x1 = self.inc(x)  # (bs, nsf, ..., ...)
        x2 = self.down1(x1)  # (bs, nsf*2, ... ,...)
        x3 = self.down2(x2)  # (bs, nsf*4, ..., ...)
        x4 = self.down3(x3)  # (bs, nsf*8, ..., ...)
        x5 = self.down4(x4)  # (bs, nsf*8, ..., ...)

        return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}

    def get_feature_map_shape(self):
        x = torch.randn(1, self.n_channels, self.map_size, self.map_size)
        x = self.inc(x)  # (bs, nsf, ..., ...)
        x = self.down1(x)  # (bs, nsf*2, ... ,...)
        x = self.down2(x)  # (bs, nsf*4, ..., ...)
        x = self.down3(x)  # (bs, nsf*8, ..., ...)
        x = self.down4(x)  # (bs, nsf*8, ..., ...)
        return x.shape


class UNetDecoder(nn.Module):
    def __init__(
        self,
        n_classes,
        nsf=16,
        bilinear=True,
    ):
        super().__init__()
        self.up1 = up(nsf * 16, nsf * 4, bilinear=bilinear)
        self.up2 = up(nsf * 8, nsf * 2, bilinear=bilinear)
        self.up3 = up(nsf * 4, nsf, bilinear=bilinear)
        self.up4 = up(nsf * 2, nsf, bilinear=bilinear)
        self.outc = outconv(nsf, n_classes)

    def forward(self, xin):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        x1 = xin["x1"]  # (bs, nsf, ..., ...)
        x2 = xin["x2"]  # (bs, nsf*2, ..., ...)
        x3 = xin["x3"]  # (bs, nsf*4, ..., ...)
        x4 = xin["x4"]  # (bs, nsf*8, ..., ...)
        x5 = xin["x5"]  # (bs, nsf*8, ..., ...)

        x = self.up1(x5, x4)  # (bs, nsf*4, ..., ...)
        x = self.up2(x, x3)  # (bs, nsf*2, ..., ...)
        x = self.up3(x, x2)  # (bs, nsf, ..., ...)
        x = self.up4(x, x1)  # (bs, nsf, ..., ...)
        x = self.outc(x)  # (bs, n_classes, ..., ...)

        return x


class ConfidenceDecoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = UNetEncoder(n_classes * 2, nsf=16)
        self.decoder = UNetDecoder(n_classes, nsf=16)

    def forward(self, xin, xpf):
        """
        xin - (bs, n_classes, H, W) semantic map
        xpf - (bs, n_classes, H, W) potential fields prediction
        """
        x_enc = self.encoder(torch.cat([xin, xpf], dim=1))
        x_dec = self.decoder(x_enc)
        return x_dec


class DirectionDecoder(nn.Module):
    def __init__(self, n_classes, n_dirs, nsf=16):
        super().__init__()
        self.n_classes = n_classes
        self.n_dirs = n_dirs
        self.conv1 = double_conv(8 * nsf, 16 * nsf)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = double_conv(16 * nsf, 32 * nsf)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32 * nsf, n_classes * n_dirs)

    def forward(self, xin):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        x5 = xin["x5"]  # (bs, nsf*8, ..., ...)

        x = self.conv1(x5)  # (bs, nsf*16, ..., ...)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)  # (bs, nsf*32, 1, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)  # (bs, N * D)
        x = x.view(x.shape[0], self.n_classes, self.n_dirs)

        return x


class PositionDecoder(nn.Module):
    def __init__(self, n_classes, nsf=16):
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = double_conv(8 * nsf, 16 * nsf)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = double_conv(16 * nsf, 32 * nsf)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32 * nsf, n_classes * 2)

    def forward(self, xin):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        x5 = xin["x5"]  # (bs, nsf*8, ..., ...)

        x = self.conv1(x5)  # (bs, nsf*16, ..., ...)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)  # (bs, nsf*32, 1, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)  # (bs, N * 2)
        x = x.view(x.shape[0], self.n_classes, 2)

        return x


class ActionDecoder(nn.Module):
    def __init__(self, n_classes, nsf=16, num_actions=4):
        super().__init__()
        self.n_classes = n_classes
        self.num_actions = num_actions
        self.conv1 = double_conv(8 * nsf, 16 * nsf)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = double_conv(16 * nsf, 32 * nsf)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32 * nsf, n_classes * num_actions)

    def forward(self, xin):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """
        x5 = xin["x5"]  # (bs, nsf*8, ..., ...)

        x = self.conv1(x5)  # (bs, nsf*16, ..., ...)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)  # (bs, nsf*32, 1, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)  # (bs, N * 2)
        x = x.view(x.shape[0], self.n_classes, self.num_actions)

        return x


def get_semantic_encoder_decoder(cfg):
    model_cfg = cfg.MODEL
    data_cfg = cfg.DATASET

    encoder, object_decoder, area_decoder = None, None, None
    output_type = model_cfg.output_type
    assert output_type in ["map", "dirs", "locs", "acts"]
    encoder = UNetEncoder(
        model_cfg.num_categories,
        model_cfg.nsf,
        model_cfg.embedding_size,
    )
    if output_type == "map":
        object_decoder = UNetDecoder(
            model_cfg.num_categories,
            model_cfg.nsf,
            bilinear=model_cfg.unet_bilinear_interp,
        )
    elif output_type == "dirs":
        object_decoder = DirectionDecoder(
            model_cfg.num_categories, model_cfg.ndirs, model_cfg.nsf
        )
    elif output_type == "locs":
        object_decoder = PositionDecoder(model_cfg.num_categories, model_cfg.nsf)
    elif output_type == "acts":
        object_decoder = ActionDecoder(
            model_cfg.num_categories, model_cfg.nsf, data_cfg.num_actions
        )
    if model_cfg.enable_area_head:
        area_decoder = UNetDecoder(
            1,
            model_cfg.nsf,
            bilinear=model_cfg.unet_bilinear_interp,
        )

    return encoder, object_decoder, area_decoder


def get_activation_fn(activation_type):
    assert activation_type in ["none", "sigmoid", "relu"]
    activation = nn.Identity()
    if activation_type == "sigmoid":
        activation = nn.Sigmoid()
    elif activation_type == "relu":
        activation = nn.ReLU()
    return activation
