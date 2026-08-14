"""Fused indexed grouped feature correlation."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def _cuda_extension():
    bundled_cuda = Path(sys.prefix) / "cuda-toolkit"
    venv_bin = Path(sys.prefix) / "bin"
    if "CUDA_HOME" not in os.environ and bundled_cuda.is_dir():
        os.environ["CUDA_HOME"] = str(bundled_cuda)
    if (venv_bin / "ninja").is_file():
        os.environ["PATH"] = f"{venv_bin}:{os.environ['PATH']}"
    toolchain_bin = bundled_cuda / "bin"
    bundled_gcc = toolchain_bin / "x86_64-conda-linux-gnu-gcc"
    bundled_gxx = toolchain_bin / "x86_64-conda-linux-gnu-g++"
    capability = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{capability[0]}.{capability[1]}")
    from torch.utils import cpp_extension

    if cpp_extension.CUDA_HOME is None and bundled_cuda.is_dir():
        cpp_extension.CUDA_HOME = str(bundled_cuda)

    source_dir = Path(__file__).resolve().parent
    python_include = bundled_cuda / "include" / (
        f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    include_flag = f"-I{python_include}"
    previous_compilers = {key: os.environ.get(key) for key in ("CC", "CXX")}
    if bundled_gcc.is_file() and bundled_gxx.is_file():
        os.environ["CC"] = str(bundled_gcc)
        os.environ["CXX"] = str(bundled_gxx)
    try:
        return cpp_extension.load(
            name="mvtracker_indexed_correlation_cuda",
            sources=[
                str(source_dir / "indexed_correlation_cuda.cpp"),
                str(source_dir / "indexed_correlation_cuda.cu"),
            ],
            extra_cflags=["-O3", include_flag],
            extra_cuda_cflags=["-O3", include_flag],
            verbose=False,
        )
    finally:
        for key, value in previous_compilers.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _compiled_forward_expression(
    targets: torch.Tensor,
    source_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    batch_size, num_queries, channels = targets.shape
    num_points = source_features.shape[1]
    neighbors = neighbor_indices.shape[-1]
    channels_per_group = channels // groups
    batch_offsets = (
        torch.arange(
            batch_size,
            device=neighbor_indices.device,
            dtype=neighbor_indices.dtype,
        )
        * num_points
    )[:, None, None]
    flat_indices = (neighbor_indices + batch_offsets).reshape(-1)
    neighbor_features = source_features.reshape(
        batch_size * num_points, channels
    ).index_select(0, flat_indices)
    neighbor_features = neighbor_features.view(
        batch_size,
        num_queries,
        neighbors,
        groups,
        channels_per_group,
    )
    grouped_targets = targets.view(
        batch_size, num_queries, groups, channels_per_group
    )
    correlations = torch.einsum(
        "BMGc,BMKGc->BMKG", grouped_targets, neighbor_features
    )
    return correlations / (channels_per_group**0.5)


_compiled_forward = torch.compile(
    _compiled_forward_expression,
    backend="inductor",
    fullgraph=True,
    dynamic=True,
)


@triton.jit
def _correlation_target_backward_kernel(
    grad_output,
    source,
    indices,
    grad_targets,
    grad_output_stride_b,
    grad_output_stride_m,
    grad_output_stride_k,
    grad_output_stride_g,
    source_stride_b,
    source_stride_n,
    source_stride_c,
    index_stride_b,
    index_stride_m,
    index_stride_k,
    grad_target_stride_b,
    grad_target_stride_m,
    grad_target_stride_c,
    M,
    K: tl.constexpr,
    G: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    NORMALIZATION: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid % G
    pid //= G
    query = pid % M
    batch = pid // M

    neighbors = tl.arange(0, BLOCK_K)[:, None]
    channels_1d = tl.arange(0, BLOCK_C)
    channels = channels_1d[None, :]
    mask = (neighbors < K) & (channels < C_PER_GROUP)
    source_indices = tl.load(
        indices
        + batch * index_stride_b
        + query * index_stride_m
        + neighbors * index_stride_k,
        mask=neighbors < K,
        other=0,
    )
    grouped_channels = group * C_PER_GROUP + channels
    source_values = tl.load(
        source
        + batch * source_stride_b
        + source_indices * source_stride_n
        + grouped_channels * source_stride_c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    output_gradients = tl.load(
        grad_output
        + batch * grad_output_stride_b
        + query * grad_output_stride_m
        + neighbors * grad_output_stride_k
        + group * grad_output_stride_g,
        mask=neighbors < K,
        other=0.0,
    ).to(tl.float32)
    gradients = tl.sum(source_values * output_gradients, axis=0)
    gradients /= NORMALIZATION
    tl.store(
        grad_targets
        + batch * grad_target_stride_b
        + query * grad_target_stride_m
        + (group * C_PER_GROUP + channels_1d) * grad_target_stride_c,
        gradients,
        mask=channels_1d < C_PER_GROUP,
    )


def _eager_cpu_correlation(
    targets: torch.Tensor,
    source_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    batch_size, num_queries, channels = targets.shape
    num_points = source_features.shape[1]
    neighbors = neighbor_indices.shape[-1]
    batch_indices = torch.arange(batch_size, device=targets.device)[:, None, None]
    neighbor_features = source_features[batch_indices, neighbor_indices]
    grouped_targets = targets.view(batch_size, num_queries, groups, -1)
    grouped_neighbors = neighbor_features.view(
        batch_size, num_queries, neighbors, groups, -1
    )
    correlations = torch.einsum(
        "BMGc,BMKGc->BMKG", grouped_targets, grouped_neighbors
    )
    return correlations / ((channels / groups) ** 0.5)


class _IndexedGroupedCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, targets, source_features, neighbor_indices, groups):
        output = _compiled_forward(
            targets, source_features, neighbor_indices, groups
        )
        ctx.save_for_backward(targets, source_features, neighbor_indices)
        ctx.groups = groups
        return output

    @staticmethod
    def backward(ctx, grad_output):
        targets, source_features, neighbor_indices = ctx.saved_tensors
        batch_size, num_queries, channels = targets.shape
        neighbors = neighbor_indices.shape[-1]
        groups = ctx.groups
        channels_per_group = channels // groups
        grad_targets = torch.empty_like(targets)
        block_channels = triton.next_power_of_2(channels_per_group)

        _correlation_target_backward_kernel[(batch_size * num_queries * groups,)](
            grad_output,
            source_features,
            neighbor_indices,
            grad_targets,
            *grad_output.stride(),
            *source_features.stride(),
            *neighbor_indices.stride(),
            *grad_targets.stride(),
            M=num_queries,
            K=neighbors,
            G=groups,
            C_PER_GROUP=channels_per_group,
            NORMALIZATION=channels_per_group**0.5,
            BLOCK_K=triton.next_power_of_2(neighbors),
            BLOCK_C=block_channels,
        )
        grad_source = _cuda_extension().source_backward(
            targets,
            neighbor_indices,
            grad_output,
            source_features.shape[1],
        )
        return grad_targets, grad_source, None, None


def indexed_grouped_correlation(
    targets: torch.Tensor,
    source_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """Return grouped correlations without materializing neighbour features."""
    if targets.device.type == "cpu":
        return _eager_cpu_correlation(
            targets, source_features, neighbor_indices, groups
        )
    if targets.device.type != "cuda":
        raise ValueError(f"unsupported correlation device: {targets.device.type}")
    if source_features.device != targets.device or neighbor_indices.device != targets.device:
        raise ValueError("correlation inputs must be on the same device")
    if targets.dtype != source_features.dtype:
        raise ValueError("target and source feature dtypes must match")
    if targets.shape[-1] % groups:
        raise ValueError("feature channels must be divisible by groups")
    return _IndexedGroupedCorrelation.apply(
        targets, source_features, neighbor_indices, groups
    )
