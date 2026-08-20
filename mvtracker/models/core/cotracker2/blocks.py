# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import gc
from itertools import repeat
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


class _FusedQKVProjection(torch.autograd.Function):
    """One forward GEMM with the original two-linear backward decomposition."""

    @staticmethod
    def forward(ctx, x, q_weight, q_bias, kv_weight, kv_bias):
        autocast = x.is_cuda and torch.is_autocast_enabled("cuda")
        compute_dtype = torch.get_autocast_dtype("cuda") if autocast else x.dtype
        x_compute = x.to(compute_dtype)
        q_weight_compute = q_weight.to(compute_dtype)
        kv_weight_compute = kv_weight.to(compute_dtype)
        q_bias_compute = q_bias.to(compute_dtype)
        kv_bias_compute = kv_bias.to(compute_dtype)
        ctx.autocast = autocast
        ctx.compute_dtype = compute_dtype
        ctx.q_width = q_weight.shape[0]
        ctx.save_for_backward(x, q_weight, q_bias, kv_weight, kv_bias)
        return F.linear(
            x_compute,
            torch.cat((q_weight_compute, kv_weight_compute), dim=0),
            torch.cat((q_bias_compute, kv_bias_compute), dim=0),
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, q_weight, q_bias, kv_weight, kv_bias = ctx.saved_tensors
        q_grad, kv_grad = grad_output.split(
            (ctx.q_width, grad_output.shape[-1] - ctx.q_width), dim=-1
        )
        with torch.enable_grad(), torch.autocast(
            device_type="cuda",
            dtype=ctx.compute_dtype,
            enabled=ctx.autocast,
            cache_enabled=False,
        ):
            recompute_x = x.detach().requires_grad_(True)
            recompute_q_weight = q_weight.detach().requires_grad_(True)
            recompute_q_bias = q_bias.detach().requires_grad_(True)
            recompute_kv_weight = kv_weight.detach().requires_grad_(True)
            recompute_kv_bias = kv_bias.detach().requires_grad_(True)
            q = F.linear(recompute_x, recompute_q_weight, recompute_q_bias)
            kv = F.linear(recompute_x, recompute_kv_weight, recompute_kv_bias)
        return torch.autograd.grad(
            (q, kv),
            (
                recompute_x,
                recompute_q_weight,
                recompute_q_bias,
                recompute_kv_weight,
                recompute_kv_bias,
            ),
            (q_grad, kv_grad),
        )


class FusedFlashAttention(FlashAttention):
    """Flash attention with one QKV projection for self-attention."""

    def forward(self, x, context=None, attn_mask=None):
        B, N1, _ = x.shape
        h = self.num_heads

        if context is None:
            qkv = _FusedQKVProjection.apply(
                x,
                self.to_q.weight,
                self.to_q.bias,
                self.to_kv.weight,
                self.to_kv.bias,
            )
            q, k, v = qkv.split(
                (h * self.dim_head, h * self.dim_head, h * self.dim_head),
                dim=-1,
            )
            context_length = N1
        else:
            q = self.to_q(x)
            k, v = self.to_kv(context).chunk(2, dim=-1)
            context_length = context.shape[1]

        q = q.reshape(B, N1, h, self.dim_head).transpose(1, 2)
        k = k.reshape(B, context_length, h, self.dim_head).transpose(1, 2)
        v = v.reshape(B, context_length, h, self.dim_head).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N1, -1)
        return self.to_out(x)


UPDATEFORMER_TRACK_CAPACITIES = (512, 1024, 1280, 2048)


def updateformer_track_capacity(track_count):
    for capacity in UPDATEFORMER_TRACK_CAPACITIES:
        if track_count <= capacity:
            return capacity
    raise ValueError(
        f"fused UpdateFormer supports at most "
        f"{UPDATEFORMER_TRACK_CAPACITIES[-1]} tracks, got {track_count}"
    )


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
            checkpoint_updateformer=False,
            execution_backend="eager",
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.checkpoint_updateformer = checkpoint_updateformer
        self.execution_backend = execution_backend
        self._compiled_impl = None
        self._graphed_signature = None
        self._graphed_window_counts = None
        self._graphed_iterations = None
        self._graphed_callables = None
        self._graphed_cursor = 0
        if execution_backend not in {"eager", "fused", "graphed", "bucketed"}:
            raise ValueError(f"unknown UpdateFormer backend: {execution_backend}")
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
        if self.execution_backend == "graphed":
            return self._forward_graphed(input_tensor, point_mask)
        if self.execution_backend == "bucketed":
            return self._forward_bucketed(input_tensor, point_mask)
        if self.execution_backend == "fused":
            return self._forward_fused(input_tensor, point_mask)
        if (
            self.checkpoint_updateformer
            and self.training
            and torch.is_grad_enabled()
        ):
            output = checkpoint(
                self._forward_impl,
                input_tensor,
                point_mask,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            output = self._forward_impl(input_tensor, point_mask)
        return output

    def begin_graphed_sequence(self, window_counts, iterations, batch_size):
        if self.execution_backend != "graphed":
            return
        signature = (
            int(batch_size),
            tuple(int(value) for value in window_counts),
            int(iterations),
        )
        if self._graphed_signature != signature:
            self._graphed_signature = signature
            self._graphed_window_counts = signature[1]
            self._graphed_iterations = signature[2]
            self._graphed_callables = None
            gc.collect()
            torch.cuda.empty_cache()
        self._graphed_cursor = 0

    def end_graphed_sequence(self):
        if self.execution_backend != "graphed":
            return
        expected = len(self._graphed_window_counts) * self._graphed_iterations
        if self._graphed_cursor != expected:
            raise RuntimeError(
                f"graphed UpdateFormer consumed {self._graphed_cursor}/{expected} slots"
            )

    def _capture_graphed_sequence(self, input_tensor, point_mask):
        if self._graphed_signature is None:
            raise RuntimeError("begin_graphed_sequence must run before UpdateFormer")

        class Slot(nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core

            def forward(self, value, mask):
                return self.core._forward_impl(value, mask)

        batch_size = self._graphed_signature[0]
        slots = []
        sample_args = []
        for track_count in self._graphed_window_counts:
            for _ in range(self._graphed_iterations):
                slots.append(Slot(self))
                value = torch.zeros(
                    batch_size,
                    track_count,
                    input_tensor.shape[2],
                    input_tensor.shape[3],
                    dtype=input_tensor.dtype,
                    device=input_tensor.device,
                    requires_grad=input_tensor.requires_grad,
                )
                mask = torch.ones(
                    batch_size,
                    track_count,
                    dtype=torch.bool,
                    device=input_tensor.device,
                )
                sample_args.append((value, mask))
        autocast_enabled = torch.is_autocast_enabled("cuda")
        with torch.autocast(
            device_type="cuda",
            dtype=torch.get_autocast_dtype("cuda"),
            enabled=autocast_enabled,
            cache_enabled=False,
        ):
            callables = torch.cuda.make_graphed_callables(
                tuple(slots),
                tuple(sample_args),
                num_warmup_iters=3,
            )
        for parameter in self.parameters():
            parameter.grad = None
        object.__setattr__(self, "_graphed_callables", callables)

    def _forward_graphed(self, input_tensor, point_mask=None):
        if point_mask is None:
            point_mask = torch.ones(
                input_tensor.shape[:2],
                dtype=torch.bool,
                device=input_tensor.device,
            )
        if self._graphed_callables is None:
            self._capture_graphed_sequence(input_tensor, point_mask)
        if self._graphed_cursor >= len(self._graphed_callables):
            raise RuntimeError("graphed UpdateFormer sequence exhausted")
        expected_tracks = self._graphed_window_counts[
            self._graphed_cursor // self._graphed_iterations
        ]
        if input_tensor.shape[1] != expected_tracks:
            raise RuntimeError(
                f"graphed UpdateFormer slot expects {expected_tracks} tracks, "
                f"got {input_tensor.shape[1]}"
            )
        callable_ = self._graphed_callables[self._graphed_cursor]
        self._graphed_cursor += 1
        return callable_(input_tensor, point_mask)

    def _forward_fused(self, input_tensor, point_mask=None):
        batch_size, track_count, _, _ = input_tensor.shape
        if point_mask is None:
            point_mask = torch.ones(
                batch_size,
                track_count,
                dtype=torch.bool,
                device=input_tensor.device,
            )
        if self._compiled_impl is None:
            self._compiled_impl = torch.compile(
                self._forward_impl,
                fullgraph=True,
                dynamic=True,
                mode="max-autotune-no-cudagraphs",
            )
        output = self._compiled_impl(input_tensor, point_mask)
        return output

    def _forward_bucketed(self, input_tensor, point_mask=None):
        batch_size, track_count, _, _ = input_tensor.shape
        if point_mask is None:
            point_mask = torch.ones(
                batch_size,
                track_count,
                dtype=torch.bool,
                device=input_tensor.device,
            )
        capacity = updateformer_track_capacity(track_count)
        padding = capacity - track_count
        if padding:
            input_tensor = F.pad(input_tensor, (0, 0, 0, 0, 0, padding))
            point_mask = F.pad(point_mask, (0, padding), value=False)
        if self._compiled_impl is None:
            self._compiled_impl = torch.compile(
                self._forward_impl,
                fullgraph=True,
                dynamic=False,
                mode="max-autotune",
            )
        return self._compiled_impl(input_tensor, point_mask)[:, :track_count]

    def _forward_impl(self, input_tensor, point_mask=None):
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
        virtual_tokens = self.virual_tracks.expand(B, -1, T, -1)
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
