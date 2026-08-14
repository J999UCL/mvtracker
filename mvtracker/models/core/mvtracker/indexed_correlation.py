"""Fused indexed grouped feature correlation."""

from __future__ import annotations

import torch


def _indexed_grouped_correlation(
    targets: torch.Tensor,
    source_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """Correlate targets with indexed source features without an eager gather."""
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


indexed_grouped_correlation = torch.compile(
    _indexed_grouped_correlation,
    backend="inductor",
    fullgraph=True,
    dynamic=True,
)
