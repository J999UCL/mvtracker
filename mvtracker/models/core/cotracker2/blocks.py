# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
from itertools import repeat
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, padding_mode="zeros")
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2

        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)

        self.conv2 = nn.Conv2d(
            output_dim * 3 + output_dim // 4,
            output_dim * 2,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        def _bilinear_intepolate(x):
            return F.interpolate(
                x,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )

        a = _bilinear_intepolate(a)
        b = _bilinear_intepolate(b)
        c = _bilinear_intepolate(c)
        d = _bilinear_intepolate(d)

        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_mask=None):
        B, N1, _ = x.shape
        h = self.heads

        q = self.to_q(x).reshape(B, N1, h, -1).permute(0, 2, 1, 3)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        N2 = context.shape[1]
        k = k.reshape(B, N2, h, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, -1).permute(0, 2, 1, 3)

        sim = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            sim = sim.masked_fill(~attn_mask, float('-inf'))
        attn = sim.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, -1)
        return self.to_out(x)


class FlashAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)
        self.num_heads = num_heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_mask=None):
        B, N1, _ = x.shape
        h = self.num_heads

        q = self.to_q(x).reshape(B, N1, h, self.dim_head).transpose(1, 2)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        N2 = context.shape[1]
        k = k.reshape(B, N2, h, self.dim_head).transpose(1, 2)
        v = v.reshape(B, N2, h, self.dim_head).transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N1, -1)
        return self.to_out(x)


class AttnBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            attn_class: Callable[..., nn.Module] = Attention,
            **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_class(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            context_dim,
            num_heads,
            mlp_ratio=4.0,
            attn_class: Callable[..., nn.Module] = Attention,
            **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = attn_class(
            query_dim=hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, context, attn_mask=None):
        x = x + self.cross_attn(self.norm1(x), context=self.norm_context(context), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
            self,
            space_depth=6,
            time_depth=6,
            input_dim=320,
            hidden_size=384,
            num_heads=8,
            output_dim=130,
            mlp_ratio=4.0,
            add_space_attn=True,
            num_virtual_tracks=64,
            attn_class: Callable[..., nn.Module] = Attention,
            linear_layer_for_vis_conf=False,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        if self.linear_layer_for_vis_conf:
            self.flow_head = nn.Sequential(
                nn.Linear(hidden_size, output_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, output_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, output_dim - 2, bias=True)
            )
            self.vis_conf_head = torch.nn.Linear(hidden_size, 2, bias=True)
        else:
            self.flow_head = nn.Sequential(
                nn.Linear(hidden_size, output_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, output_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, output_dim, bias=True)
            )
        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(torch.randn(1, num_virtual_tracks, 1, hidden_size))
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=attn_class,
                )
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=attn_class,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size,
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=attn_class,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size,
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=attn_class,
                    )
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def xavier_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def trunc_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.001)

        # Apply xavier to all except flow_head
        self.apply(xavier_init)

        # Then override flow_head with trunc_normal
        self.flow_head.apply(trunc_init)
        if self.linear_layer_for_vis_conf:
            self.vis_conf_head.apply(trunc_init)

    def forward(self, input_tensor, point_mask=None):
        tokens = self.input_transform(input_tensor)
        B, _, T, _ = tokens.shape
        if point_mask is None:
            point_mask = torch.ones(
                B, tokens.shape[1], dtype=torch.bool, device=tokens.device
            )
        if point_mask.shape != (B, tokens.shape[1]):
            raise ValueError(
                f"point_mask must have shape {(B, tokens.shape[1])}, "
                f"got {tuple(point_mask.shape)}"
            )
        virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)
        _, N, _, _ = tokens.shape

        j = 0
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C
            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C
            if self.add_space_attn and (
                    i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0
            ):
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  # B N T C -> (B T) N C
                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks:]
                point_mask_bt = (
                    point_mask[:, None, :]
                    .expand(B, T, -1)
                    .reshape(B * T, -1)
                )
                point_key_mask = point_mask_bt[:, None, None, :]

                virtual_tokens = self.space_virtual2point_blocks[j](
                    virtual_tokens, point_tokens, attn_mask=point_key_mask
                )
                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens
                )
                point_tokens = point_tokens.masked_fill(
                    ~point_mask_bt[:, :, None], 0
                )
                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)  # (B T) N C -> B N T C
                j += 1
        tokens = tokens[:, : N - self.num_virtual_tracks]

        flow = self.flow_head(tokens)
        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)

        return flow
