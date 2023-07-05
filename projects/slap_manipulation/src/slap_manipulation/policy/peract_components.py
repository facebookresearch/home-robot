"""
Q-attention: Enabling Efficient Learning for Vision-based Robotic Manipulation

LICENCE AGREEMENT

WE (Imperial College of Science, Technology and Medicine, (“Imperial College London”))
ARE WILLING TO LICENSE THIS SOFTWARE TO YOU (a licensee “You”) ONLY ON THE
CONDITION THAT YOU ACCEPT ALL OF THE TERMS CONTAINED IN THE
FOLLOWING AGREEMENT. PLEASE READ THE AGREEMENT CAREFULLY BEFORE
DOWNLOADING THE SOFTWARE. BY EXERCISING THE OPTION TO DOWNLOAD
THE SOFTWARE YOU AGREE TO BE BOUND BY THE TERMS OF THE AGREEMENT.
SOFTWARE LICENCE AGREEMENT (EXCLUDING BSD COMPONENTS)

1.This Agreement pertains to a worldwide, non-exclusive, temporary, fully paid-up, royalty
free, non-transferable, non-sub- licensable licence (the “Licence”) to use the Q-attention
source code, including any modification, part or derivative (the “Software”).
Ownership and Licence. Your rights to use and download the Software onto your computer,
and all other copies that You are authorised to make, are specified in this Agreement.
However, we (or our licensors) retain all rights, including but not limited to all copyright and
other intellectual property rights anywhere in the world, in the Software not expressly
granted to You in this Agreement.

2. Permitted use of the Licence:

(a) You may download and install the Software onto one computer or server for use in
accordance with Clause 2(b) of this Agreement provided that You ensure that the Software is
not accessible by other users unless they have themselves accepted the terms of this licence
agreement.

(b) You may use the Software solely for non-commercial, internal or academic research
purposes and only in accordance with the terms of this Agreement. You may not use the
Software for commercial purposes, including but not limited to (1) integration of all or part of
the source code or the Software into a product for sale or licence by or on behalf of You to
third parties or (2) use of the Software or any derivative of it for research to develop software
products for sale or licence to a third party or (3) use of the Software or any derivative of it
for research to develop non-software products for sale or licence to a third party, or (4) use of
the Software to provide any service to an external organisation for which payment is
received.

Should You wish to use the Software for commercial purposes, You shall
email researchcontracts.engineering@imperial.ac.uk .

(c) Right to Copy. You may copy the Software for back-up and archival purposes, provided
that each copy is kept in your possession and provided You reproduce our copyright notice
(set out in Schedule 1) on each copy.

(d) Transfer and sub-licensing. You may not rent, lend, or lease the Software and You may
not transmit, transfer or sub-license this licence to use the Software or any of your rights or
obligations under this Agreement to another party.

(e) Identity of Licensee. The licence granted herein is personal to You. You shall not permit
any third party to access, modify or otherwise use the Software nor shall You access modify
or otherwise use the Software on behalf of any third party. If You wish to obtain a licence for
mutiple users or a site licence for the Software please contact us
at researchcontracts.engineering@imperial.ac.uk .

(f) Publications and presentations. You may make public, results or data obtained from,
dependent on or arising from research carried out using the Software, provided that any such
presentation or publication identifies the Software as the source of the results or the data,
including the Copyright Notice given in each element of the Software, and stating that the
Software has been made available for use by You under licence from Imperial College London
and You provide a copy of any such publication to Imperial College London.

3. Prohibited Uses. You may not, without written permission from us
at researchcontracts.engineering@imperial.ac.uk :

(a) Use, copy, modify, merge, or transfer copies of the Software or any documentation
provided by us which relates to the Software except as provided in this Agreement;

(b) Use any back-up or archival copies of the Software (or allow anyone else to use such
copies) for any purpose other than to replace the original copy in the event it is destroyed or
becomes defective; or

(c) Disassemble, decompile or "unlock", reverse translate, or in any manner decode the
Software for any reason.

4. Warranty Disclaimer

(a) Disclaimer. The Software has been developed for research purposes only. You
acknowledge that we are providing the Software to You under this licence agreement free of
charge and on condition that the disclaimer set out below shall apply. We do not represent or
warrant that the Software as to: (i) the quality, accuracy or reliability of the Software; (ii) the
suitability of the Software for any particular use or for use under any specific conditions; and
(iii) whether use of the Software will infringe third-party rights.
You acknowledge that You have reviewed and evaluated the Software to determine that it
meets your needs and that You assume all responsibility and liability for determining the
suitability of the Software as fit for your particular purposes and requirements. Subject to
Clause 4(b), we exclude and expressly disclaim all express and implied representations,
warranties, conditions and terms not stated herein (including the implied conditions or
warranties of satisfactory quality, merchantable quality, merchantability and fitness for
purpose).

(b) Savings. Some jurisdictions may imply warranties, conditions or terms or impose
obligations upon us which cannot, in whole or in part, be excluded, restricted or modified or
otherwise do not allow the exclusion of implied warranties, conditions or terms, in which
case the above warranty disclaimer and exclusion will only apply to You to the extent
permitted in the relevant jurisdiction and does not in any event exclude any implied
warranties, conditions or terms which may not under applicable law be excluded.

(c) Imperial College London disclaims all responsibility for the use which is made of the
Software and any liability for the outcomes arising from using the Software.

5. Limitation of Liability

(a) You acknowledge that we are providing the Software to You under this licence agreement
free of charge and on condition that the limitation of liability set out below shall apply.
Accordingly, subject to Clause 5(b), we exclude all liability whether in contract, tort,
negligence or otherwise, in respect of the Software and/or any related documentation
provided to You by us including, but not limited to, liability for loss or corruption of data,
loss of contracts, loss of income, loss of profits, loss of cover and any consequential or indirect
loss or damage of any kind arising out of or in connection with this licence agreement,
however caused. This exclusion shall apply even if we have been advised of the possibility of
such loss or damage.

(b) You agree to indemnify Imperial College London and hold it harmless from and against
any and all claims, damages and liabilities asserted by third parties (including claims for
negligence) which arise directly or indirectly from the use of the Software or any derivative
of it or the sale of any products based on the Software. You undertake to make no liability
claim against any employee, student, agent or appointee of Imperial College London, in
connection with this Licence or the Software.

(c) Nothing in this Agreement shall have the effect of excluding or limiting our statutory
liability.

(d) Some jurisdictions do not allow these limitations or exclusions either wholly or in part,
and, to that extent, they may not apply to you. Nothing in this licence agreement will affect
your statutory rights or other relevant statutory provisions which cannot be excluded,
restricted or modified, and its terms and conditions must be read and construed subject to any
such statutory rights and/or provisions.

6. Confidentiality. You agree not to disclose any confidential information provided to You by
us pursuant to this Agreement to any third party without our prior written consent. The
obligations in this Clause 6 shall survive the termination of this Agreement for any reason.

7. Termination.

(a) We may terminate this licence agreement and your right to use the Software at any time
with immediate effect upon written notice to You.

(b) This licence agreement and your right to use the Software automatically terminate if You:
(i) fail to comply with any provisions of this Agreement; or
(ii) destroy the copies of the Software in your possession, or voluntarily return the Software
to us.

(c) Upon termination You will destroy all copies of the Software.

(d) Otherwise, the restrictions on your rights to use the Software will expire 10 (ten) years
after first use of the Software under this licence agreement.

8. Miscellaneous Provisions.

(a) This Agreement will be governed by and construed in accordance with the substantive
laws of England and Wales whose courts shall have exclusive jurisdiction over all disputes
which may arise between us.

(b) This is the entire agreement between us relating to the Software, and supersedes any prior
purchase order, communications, advertising or representations concerning the Software.

(c) No change or modification of this Agreement will be valid unless it is in writing, and is
signed by us.

(d) The unenforceability or invalidity of any part of this Agreement will not affect the
enforceability or validity of the remaining parts.

BSD Elements of the Software

For BSD elements of the Software, the following terms shall apply:

Copyright as indicated in the header of the individual element of the Software.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to
endorse or promote products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import copy
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LRELU_SLOPE = 0.02


def act_layer(act):
    if act == "relu":
        return nn.ReLU()
    elif act == "lrelu":
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == "elu":
        return nn.ELU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "prelu":
        return nn.PReLU()
    else:
        raise ValueError("%s not recognized." % act)


def norm_layer2d(norm, channels):
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    elif norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, channels, affine=True)
    elif norm == "group":
        return nn.GroupNorm(4, channels, affine=True)
    else:
        raise ValueError("%s not recognized." % norm)


def norm_layer1d(norm, num_channels):
    if norm == "batch":
        return nn.BatchNorm1d(num_channels)
    elif norm == "instance":
        return nn.InstanceNorm1d(num_channels, affine=True)
    elif norm == "layer":
        return nn.LayerNorm(num_channels)
    else:
        raise ValueError("%s not recognized." % norm)


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        strides,
        norm=None,
        activation=None,
        padding_mode="replicate",
    ):
        super(Conv2DBlock, self).__init__()
        padding = (
            kernel_sizes // 2
            if isinstance(kernel_sizes, int)
            else (kernel_sizes[0] // 2, kernel_sizes[1] // 2)
        )
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_sizes,
            strides,
            padding=padding,
            padding_mode=padding_mode,
        )

        if activation is None:
            nn.init.xavier_uniform_(
                self.conv2d.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.conv2d.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.conv2d.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.conv2d.weight, nonlinearity="relu")
            nn.init.zeros_(self.conv2d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer2d(norm, out_channels)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv3DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes: Union[int, list] = 3,
        strides=1,
        norm=None,
        activation=None,
        padding_mode="replicate",
        padding=None,
    ):
        super(Conv3DBlock, self).__init__()
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_sizes,
            strides,
            padding=padding,
            padding_mode=padding_mode,
        )

        if activation is None:
            nn.init.xavier_uniform_(
                self.conv3d.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.conv3d.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.conv3d.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.conv3d.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.conv3d.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.conv3d.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.conv3d.weight, nonlinearity="relu")
            nn.init.zeros_(self.conv3d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            raise NotImplementedError("Norm not implemented.")
        if activation is not None:
            self.activation = act_layer(activation)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv3d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv2DUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        strides,
        norm=None,
        activation=None,
    ):
        super(Conv2DUpsampleBlock, self).__init__()
        layer = [
            Conv2DBlock(in_channels, out_channels, kernel_sizes, 1, norm, activation)
        ]
        if strides > 1:
            layer.append(
                nn.Upsample(scale_factor=strides, mode="bilinear", align_corners=False)
            )
        convt_block = Conv2DBlock(
            out_channels, out_channels, kernel_sizes, 1, norm, activation
        )
        layer.append(convt_block)
        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class Conv3DUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_sizes=3,
        norm=None,
        activation=None,
    ):
        super(Conv3DUpsampleBlock, self).__init__()
        layer = [
            Conv3DBlock(in_channels, out_channels, kernel_sizes, 1, norm, activation)
        ]
        if strides > 1:
            layer.append(
                nn.Upsample(scale_factor=strides, mode="trilinear", align_corners=False)
            )
        convt_block = Conv3DBlock(
            out_channels, out_channels, kernel_sizes, 1, norm, activation
        )
        layer.append(convt_block)
        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, norm=None, activation=None):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        if activation is None:
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.linear.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")
            nn.init.zeros_(self.linear.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer1d(norm, out_features)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class SiameseNet(nn.Module):
    def __init__(
        self,
        input_channels: List[int],
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        norm: str = None,
        activation: str = "relu",
    ):
        super(SiameseNet, self).__init__()
        self._input_channels = input_channels
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self.output_channels = filters[-1]  # * len(input_channels)

    def build(self):
        self._siamese_blocks = nn.ModuleList()
        for i, ch in enumerate(self._input_channels):
            blocks = []
            for i, (filt, ksize, stride) in enumerate(
                zip(self._filters, self._kernel_sizes, self._strides)
            ):
                conv_block = Conv2DBlock(
                    ch, filt, ksize, stride, self._norm, self._activation
                )
                blocks.append(conv_block)
            self._siamese_blocks.append(nn.Sequential(*blocks))
        self._fuse = Conv2DBlock(
            self._filters[-1] * len(self._siamese_blocks),
            self._filters[-1],
            1,
            1,
            self._norm,
            self._activation,
        )

    def forward(self, x):
        if len(x) != len(self._siamese_blocks):
            raise ValueError(
                "Expected a list of tensors of size %d." % len(self._siamese_blocks)
            )
        self.streams = [stream(y) for y, stream in zip(x, self._siamese_blocks)]
        y = self._fuse(torch.cat(self.streams, 1))
        return y


class CNNAndFcsNet(nn.Module):
    def __init__(
        self,
        siamese_net: SiameseNet,
        low_dim_state_len: int,
        input_resolution: List[int],
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        norm: str = None,
        fc_layers: List[int] = None,
        activation: str = "relu",
    ):
        super(CNNAndFcsNet, self).__init__()
        self._siamese_net = copy.deepcopy(siamese_net)
        self._input_channels = self._siamese_net.output_channels + low_dim_state_len
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._fc_layers = [] if fc_layers is None else fc_layers
        self._input_resolution = input_resolution

    def build(self):
        self._siamese_net.build()
        layers = []
        channels = self._input_channels
        for i, (filt, ksize, stride) in enumerate(
            list(zip(self._filters, self._kernel_sizes, self._strides))[:-1]
        ):
            layers.append(
                Conv2DBlock(channels, filt, ksize, stride, self._norm, self._activation)
            )
            channels = filt
        layers.append(
            Conv2DBlock(
                channels, self._filters[-1], self._kernel_sizes[-1], self._strides[-1]
            )
        )
        self._cnn = nn.Sequential(*layers)
        self._maxp = nn.AdaptiveMaxPool2d(1)

        channels = self._filters[-1]
        dense_layers = []
        for n in self._fc_layers[:-1]:
            dense_layers.append(DenseBlock(channels, n, activation=self._activation))
            channels = n
        dense_layers.append(DenseBlock(channels, self._fc_layers[-1]))
        self._fcs = nn.Sequential(*dense_layers)

    def forward(self, observations, low_dim_ins):
        x = self._siamese_net(observations)
        _, _, h, w = x.shape
        low_dim_latents = low_dim_ins.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
        combined = torch.cat([x, low_dim_latents], dim=1)
        x = self._cnn(combined)
        x = self._maxp(x).squeeze(-1).squeeze(-1)
        return self._fcs(x)


class Conv3DInceptionBlockUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor,
        norm=None,
        activation=None,
        residual=False,
    ):
        super(Conv3DInceptionBlockUpsampleBlock, self).__init__()
        layer = []

        convt_block = Conv3DInceptionBlock(in_channels, out_channels, norm, activation)
        layer.append(convt_block)

        if scale_factor > 1:
            layer.append(
                nn.Upsample(
                    scale_factor=scale_factor, mode="trilinear", align_corners=False
                )
            )

        convt_block = Conv3DInceptionBlock(out_channels, out_channels, norm, activation)
        layer.append(convt_block)

        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class Conv3DInceptionBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, norm=None, activation=None, residual=False
    ):
        super(Conv3DInceptionBlock, self).__init__()
        self._residual = residual
        cs = out_channels // 4
        assert out_channels % 4 == 0
        latent = 32
        self._1x1conv = Conv3DBlock(
            in_channels,
            cs * 2,
            kernel_sizes=1,
            strides=1,
            norm=norm,
            activation=activation,
        )

        self._1x1conv_a = Conv3DBlock(
            in_channels,
            latent,
            kernel_sizes=1,
            strides=1,
            norm=norm,
            activation=activation,
        )
        self._3x3conv = Conv3DBlock(
            latent, cs, kernel_sizes=3, strides=1, norm=norm, activation=activation
        )

        self._1x1conv_b = Conv3DBlock(
            in_channels,
            latent,
            kernel_sizes=1,
            strides=1,
            norm=norm,
            activation=activation,
        )
        self._5x5_via_3x3conv_a = Conv3DBlock(
            latent, latent, kernel_sizes=3, strides=1, norm=norm, activation=activation
        )
        self._5x5_via_3x3conv_b = Conv3DBlock(
            latent, cs, kernel_sizes=3, strides=1, norm=norm, activation=activation
        )
        self.out_channels = out_channels + (in_channels if residual else 0)

    def forward(self, x):
        yy = []
        if self._residual:
            yy = [x]
        return torch.cat(
            yy
            + [
                self._1x1conv(x),
                self._3x3conv(self._1x1conv_a(x)),
                self._5x5_via_3x3conv_b(self._5x5_via_3x3conv_a(self._1x1conv_b(x))),
            ],
            1,
        )


class SpatialSoftmax3D(torch.nn.Module):
    def __init__(self, depth, height, width, channel):
        super(SpatialSoftmax3D, self).__init__()
        self.depth = depth
        self.height = height
        self.width = width
        self.channel = channel
        self.temperature = 0.01
        pos_x, pos_y, pos_z = np.meshgrid(
            np.linspace(-1.0, 1.0, self.depth),
            np.linspace(-1.0, 1.0, self.height),
            np.linspace(-1.0, 1.0, self.width),
        )
        pos_x = torch.from_numpy(
            pos_x.reshape(self.depth * self.height * self.width)
        ).float()
        pos_y = torch.from_numpy(
            pos_y.reshape(self.depth * self.height * self.width)
        ).float()
        pos_z = torch.from_numpy(
            pos_z.reshape(self.depth * self.height * self.width)
        ).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)
        self.register_buffer("pos_z", pos_z)

    def forward(self, feature):
        feature = feature.view(
            -1, self.height * self.width * self.depth
        )  # (B, c*d*h*w)
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_z = torch.sum(self.pos_z * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y, expected_z], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 3)
        return feature_keypoints
