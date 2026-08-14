"""Fused indexed grouped feature correlation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _correlation_forward_kernel(
    targets,
    source,
    indices,
    output,
    target_stride_b: tl.constexpr,
    target_stride_m: tl.constexpr,
    target_stride_c: tl.constexpr,
    source_stride_b: tl.constexpr,
    source_stride_n: tl.constexpr,
    source_stride_c: tl.constexpr,
    index_stride_b: tl.constexpr,
    index_stride_m: tl.constexpr,
    index_stride_k: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_m: tl.constexpr,
    output_stride_k: tl.constexpr,
    output_stride_g: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    NORMALIZATION: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid % G
    pid //= G
    neighbor = pid % K
    pid //= K
    query = pid % M
    batch = pid // M

    channels = tl.arange(0, BLOCK_C)
    channel_mask = channels < C_PER_GROUP
    source_index = tl.load(
        indices
        + batch * index_stride_b
        + query * index_stride_m
        + neighbor * index_stride_k
    )
    grouped_channels = group * C_PER_GROUP + channels
    target_values = tl.load(
        targets
        + batch * target_stride_b
        + query * target_stride_m
        + grouped_channels * target_stride_c,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    source_values = tl.load(
        source
        + batch * source_stride_b
        + source_index * source_stride_n
        + grouped_channels * source_stride_c,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    correlation = tl.sum(target_values * source_values, axis=0)
    correlation /= NORMALIZATION
    tl.store(
        output
        + batch * output_stride_b
        + query * output_stride_m
        + neighbor * output_stride_k
        + group * output_stride_g,
        correlation,
    )


@triton.jit
def _correlation_target_backward_kernel(
    grad_output,
    source,
    indices,
    grad_targets,
    grad_output_stride_b: tl.constexpr,
    grad_output_stride_m: tl.constexpr,
    grad_output_stride_k: tl.constexpr,
    grad_output_stride_g: tl.constexpr,
    source_stride_b: tl.constexpr,
    source_stride_n: tl.constexpr,
    source_stride_c: tl.constexpr,
    index_stride_b: tl.constexpr,
    index_stride_m: tl.constexpr,
    index_stride_k: tl.constexpr,
    grad_target_stride_b: tl.constexpr,
    grad_target_stride_m: tl.constexpr,
    grad_target_stride_c: tl.constexpr,
    M: tl.constexpr,
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
        batch_size, num_queries, channels = targets.shape
        neighbors = neighbor_indices.shape[-1]
        channels_per_group = channels // groups
        output = torch.empty(
            batch_size,
            num_queries,
            neighbors,
            groups,
            device=targets.device,
            dtype=targets.dtype,
        )
        _correlation_forward_kernel[(batch_size * num_queries * neighbors * groups,)](
            targets,
            source_features,
            neighbor_indices,
            output,
            *targets.stride(),
            *source_features.stride(),
            *neighbor_indices.stride(),
            *output.stride(),
            M=num_queries,
            K=neighbors,
            G=groups,
            C_PER_GROUP=channels_per_group,
            NORMALIZATION=channels_per_group**0.5,
            BLOCK_C=triton.next_power_of_2(channels_per_group),
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
        grad_source = torch.zeros_like(source_features)
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
        flat_grad_source = grad_source.view(-1, channels)
        grouped_targets = targets.view(
            batch_size, num_queries, groups, channels_per_group
        )
        batch_offsets = (
            torch.arange(batch_size, device=targets.device) * source_features.shape[1]
        )[:, None]
        for neighbor in range(neighbors):
            flat_indices = (
                neighbor_indices[:, :, neighbor].to(torch.int64) + batch_offsets
            ).reshape(-1)
            contributions = grouped_targets * grad_output[
                :, :, neighbor, :, None
            ]
            contributions = contributions.reshape(-1, channels)
            contributions = contributions / (channels_per_group**0.5)
            flat_grad_source.index_add_(0, flat_indices, contributions)
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
