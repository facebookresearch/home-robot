# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from einops import rearrange, reduce, repeat
from perceiver_pytorch.perceiver_io import (
    FeedForward,
    PreNorm,
    cache_fn,
    default,
    exists,
)
from torch import einsum
from torch.nn import BatchNorm1d as BN
from torch.nn import Identity
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_cluster import fps
from torch_geometric.nn import PointNetConv, global_max_pool, radius
from torch_geometric.nn.conv import PointTransformerConv
from torch_geometric.nn.pool import knn
from torch_geometric.nn.unpool import knn_interpolate
from torch_scatter import scatter_max

# Simple policy code
# From:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_segmentation.py


# License: https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE
class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels])
        self.attn_nn = MLP([out_channels, 64, out_channels])
        # self.pos_nn = MLP([3, 64, out_channels], batch_norm=False)
        # self.attn_nn = MLP([out_channels, 64, out_channels], batch_norm=False)

        self.transformer = PointTransformerConv(
            in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


# License: https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE
class TransitionDown(torch.nn.Module):
    """
    Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality
    """

    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels])

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(
            pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch
        )

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(
            x[id_k_neighbor[1]],
            id_k_neighbor[0],
            dim_size=id_clusters.size(0),
            dim=0,
        )

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


# License: https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE
def MLP(channels, batch_norm=True):
    return Seq(
        *[
            Seq(
                Lin(channels[i - 1], channels[i]),
                BN(channels[i]) if batch_norm else Identity(),
                ReLU(),
            )
            for i in range(1, len(channels))
        ]
    )


# License: https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE
class TransitionUp(torch.nn.Module):
    """
    Reduce features dimensionnality and interpolate back to higher
    resolution and cardinality
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels], batch_norm=False)
        self.mlp = MLP([out_channels, out_channels], batch_norm=False)

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(
            x_sub, pos_sub, pos, k=3, batch_x=batch_sub, batch_y=batch
        )

        x = self.mlp(x) + x_interpolated

        return x


class PTLanguageFusionBlock(torch.nn.Module):
    """ling u-net fusion"""

    def __init__(self, inp_x, inp_lang, out):
        super(PTLanguageFusionBlock, self).__init__()
        # self.conv = Conv1d(inp_x + inp_lang, out, kernel_size=1, bias=False)
        self.linear = Lin(inp_x + inp_lang, out, bias=False)
        self.bn = BN(out)
        self.relu = ReLU()

        if inp_x != out:
            self.linear2 = Lin(inp_x, out, bias=False)
            self.bn2 = BN(out)
        else:
            self.linear2 = None
            self.bn2 = None

    def forward(self, x0, lang):
        B = x0.shape[0]
        lang = lang.view(1, -1)
        x = torch.cat([x0, lang.repeat(B, 1)], dim=1)
        # x = self.relu(x)
        x = self.linear(x)
        # x = self.bn(x)

        if self.linear2 is not None:
            y = self.linear2(x0)
            x0 = self.bn2(y)
        return self.relu(x0 + x)
        # return self.relu(x)


class SAModule(torch.nn.Module):
    def __init__(self, r, maxn, device, nn):
        super().__init__()
        self._query_radius = r
        self._max_neighbors = maxn
        self.conv = PointNetConv(nn, add_self_loops=False)
        self.device = device

    def some_function(self, batch):
        return batch

    def forward(self, feat, sampled_feat, pos, sampled_pos, batch):
        # get spheres associated with each centroid
        B = 1
        y2x_edge_index = radius(
            pos,
            sampled_pos,
            self._query_radius,
            # batch_x=B,
            # batch_y=B,
            max_num_neighbors=self._max_neighbors,
        )
        if y2x_edge_index.shape[1] < 2:
            breakpoint()

        # reverse edge_index because that is what PointConv expects
        idx = torch.LongTensor([1, 0])
        x2y_edge_index = y2x_edge_index[idx]

        # add another feature distinguishing voxel query points from real points
        # feat_real = torch.Tensor(np.zeros((feat.shape[0]))).to(self.device)
        # feat_query = torch.Tensor(np.ones((sampled_feat.shape[0]))).to(self.device)
        # feat = torch.column_stack((feat, feat_real))
        # sampled_feat = torch.column_stack((sampled_feat, feat_query))

        # NOTE: does PointConv need all points in the same object for message passing?
        # as seen in https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
        # combine the real points with query points
        # pos = torch.cat((pos, sampled_pos))
        # feat = torch.cat((feat, sampled_feat))
        # print(f"[DEBUG] shape of variable is {variable.shape}")
        # print(f"[DEBUG] shape of x2y_edge_index is {x2y_edge_index.shape}")
        # print(f"[DEBUG] shape of pos is {pos.shape}")
        # print(f"[DEBUG] shape of sampled_pos is {sampled_pos.shape}")
        # print(f"[DEBUG] shape of sampled_feat is {sampled_feat.shape}")
        # print(f"[DEBUG] shape of feat is {feat.shape}")

        x = self.conv(
            (feat, sampled_feat),
            (pos, sampled_pos),
            edge_index=x2y_edge_index,
        )
        sampled_batch = self.some_function(batch)
        return x, sampled_pos, sampled_batch


# License: https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE
class PtnetSAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


# License: https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE
class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PositionalEncoding(nn.Module):
    """
    From
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Modified for use in pt_query
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model % 2 != 0:
            raise RuntimeError("d_model should be even, you fool")

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# Dependencies 291-357 ported from: https://github.com/stepjam/ARM/blob/main/arm/network_utils.py
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


def norm_layer1d(norm, num_channels):
    if norm == "batch":
        return nn.BatchNorm1d(num_channels)
    elif norm == "instance":
        return nn.InstanceNorm1d(num_channels, affine=True)
    elif norm == "layer":
        return nn.LayerNorm(num_channels)
    else:
        raise ValueError("%s not recognized." % norm)


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


# Dependency taken from PerAct/Perceiver Code
# the one big difference which makes PerceiverIO work for 6DOF manip
# Perceiver IO implementation adpated for manipulation
# Source: https://github.com/lucidrains/perceiver-pytorch
# License: https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE
class Attention(nn.Module):  # is all you need. Living up to its name.
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)
