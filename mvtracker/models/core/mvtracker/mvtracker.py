import logging
import os
from bisect import bisect_left
from collections import defaultdict
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch import nn as nn

from mvtracker.datasets.utils import transform_scene
from mvtracker.models.core.cotracker2.blocks import Attention, FlashAttention
from mvtracker.models.core.cotracker2.blocks import (
    EfficientUpdateFormer,
    FusedFlashAttention,
)
from mvtracker.models.core.embeddings import (
    get_3d_sincos_pos_embed_from_grid,
    get_1d_sincos_pos_embed_from_grid,
    get_3d_sincos_pos_embed_from_grid_cuda,
    get_3d_embedding,
)
from mvtracker.models.core.model_utils import smart_cat, init_pointcloud_from_rgbd, save_pointcloud_to_ply
from mvtracker.models.core.mvtracker.indexed_correlation import indexed_grouped_correlation
from mvtracker.models.core.spatracker.blocks import BasicEncoder
from mvtracker.utils.basic import time_now


# ---------- KNN backends ----------
def _knn_pointops(k: int, xyz_ref: torch.Tensor, xyz_query: torch.Tensor):
    """
    Efficient batched KNN using pointops library.

    This is slightly faster than torch.cdist + torch.topk and uses less memory:

    Example::

        Benchmarking KNN with different methods (HALF_PRECISION=True):
        torch.cdist+torch.topk   | Avg Time: 0.008380 s | Peak Memory: 1151.19 MB (min: 1151.19, max: 1151.19)
        pointops.knn_query       | Avg Time: 0.007477 s | Peak Memory:  47.22 MB (min:  47.22, max:  47.22)

        Benchmarking KNN with different methods (HALF_PRECISION=False):
        torch.cdist+torch.topk   | Avg Time: 0.014090 s | Peak Memory: 2249.88 MB (min: 2249.88, max: 2249.88)
        pointops.knn_query       | Avg Time: 0.007368 s | Peak Memory:  43.62 MB (min:  43.62, max:  43.62)

    Args:
        xyz_ref (Tensor): (B, N, 3)
        xyz_query (Tensor): (B, M, 3)

    Returns:
        Tuple[Tensor, Tensor]:
            - dist (Tensor): (B, M, k)
            - idx (Tensor): (B, M, k) int32 — indices into dimension N
    """
    # Fallback if tensors are not on CUDA
    if not xyz_ref.is_cuda:
        return _knn_torch(k, xyz_ref, xyz_query)

    from pointops import knn_query
    B, N, _ = xyz_ref.shape
    _, M, _ = xyz_query.shape
    orig_dtype = xyz_ref.dtype

    xyz_ref_flat = xyz_ref.contiguous().view(B * N, 3).to(torch.float32)
    xyz_query_flat = xyz_query.contiguous().view(B * M, 3).to(torch.float32)

    offset = torch.arange(1, B + 1, device=xyz_ref.device) * N
    new_offset = torch.arange(1, B + 1, device=xyz_query.device) * M
    idx, dists = knn_query(k, xyz_ref_flat, offset, xyz_query_flat, new_offset)

    # Remap global indices to local per-batch
    idx = idx.view(B, M, k)
    idx = idx - (torch.arange(B, device=idx.device).view(B, 1, 1) * N)
    dists = dists.view(B, M, k).to(orig_dtype)

    return dists, idx


def _knn_torch(k: int, xyz_ref: torch.Tensor, xyz_query: torch.Tensor):
    """Fallback KNN using torch.cdist + topk."""
    dists = torch.cdist(xyz_query, xyz_ref, p=2)  # (B, M, N)
    sorted_dists, indices = torch.topk(dists, k, dim=-1, largest=False, sorted=True)
    return sorted_dists, indices


def _knn_cuda_extension(
    k: int,
    xyz_ref: torch.Tensor,
    xyz_query: torch.Tensor,
    operation: str,
):
    if not xyz_ref.is_cuda:
        return _knn_torch(k, xyz_ref, xyz_query)
    from mvtracker.models.core.mvtracker import mvtracker_capturable_knn_cuda

    batch, reference_count, _ = xyz_ref.shape
    query_count = xyz_query.shape[1]
    reference = xyz_ref.contiguous().view(batch * reference_count, 3).float()
    query = xyz_query.contiguous().view(batch * query_count, 3).float()
    offsets = (
        torch.arange(1, batch + 1, device=xyz_ref.device, dtype=torch.int32)
        * reference_count
    )
    query_offsets = (
        torch.arange(1, batch + 1, device=xyz_ref.device, dtype=torch.int32)
        * query_count
    )
    indices = torch.empty(
        batch * query_count, k, dtype=torch.int32, device=xyz_ref.device
    )
    squared_distances = torch.empty(
        batch * query_count, k, dtype=torch.float32, device=xyz_ref.device
    )
    getattr(mvtracker_capturable_knn_cuda, operation)(
        k,
        reference,
        query,
        offsets,
        query_offsets,
        indices,
        squared_distances,
    )
    indices = indices.view(batch, query_count, k)
    indices = indices - (
        torch.arange(batch, device=indices.device, dtype=torch.int32)[:, None, None]
        * reference_count
    )
    return squared_distances.sqrt().view(batch, query_count, k).to(xyz_ref.dtype), indices


def _knn_capturable(k: int, xyz_ref: torch.Tensor, xyz_query: torch.Tensor):
    return _knn_cuda_extension(k, xyz_ref, xyz_query, "knn_query_out")


def _knn_tiled(k: int, xyz_ref: torch.Tensor, xyz_query: torch.Tensor):
    if k > 16:
        raise ValueError("tiled KNN supports at most 16 neighbors")
    return _knn_cuda_extension(k, xyz_ref, xyz_query, "tiled_knn_query_out")


knn = _knn_capturable


class MVTracker(nn.Module):
    def __init__(
            self,
            sliding_window_len=12,
            stride=4,
            normalize_scene_in_fwd_pass=False,
            fmaps_dim=128,
            add_space_attn=True,
            num_heads=6,
            hidden_size=384,
            space_depth=6,
            time_depth=6,
            num_virtual_tracks=64,
            use_flash_attention=True,
            corr_n_groups=1,
            corr_n_levels=4,
            corr_neighbors=16,
            corr_add_neighbor_offset=True,
            corr_add_neighbor_xyz=False,
            corr_filter_invalid_depth=False,
            knn_backend="serial",
            updateformer_backend="eager",
            checkpoint_updateformer=False,
    ):
        super().__init__()

        self.S = sliding_window_len
        self.stride = stride
        self.normalize_scene_in_fwd_pass = normalize_scene_in_fwd_pass
        self.latent_dim = fmaps_dim
        self.flow_embed_dim = 64
        self.b_latent_dim = self.latent_dim // 3
        self.corr_n_groups = corr_n_groups
        self.corr_n_levels = corr_n_levels
        self.corr_neighbors = corr_neighbors
        self.corr_pos_emb_size = 0
        self.corr_add_neighbor_offset = corr_add_neighbor_offset
        self.corr_add_neighbor_xyz = corr_add_neighbor_xyz
        self.corr_filter_invalid_depth = corr_filter_invalid_depth
        if knn_backend not in {"serial", "tiled"}:
            raise ValueError(f"unknown KNN backend: {knn_backend}")
        self.knn = _knn_tiled if knn_backend == "tiled" else _knn_capturable
        self.updateformer_backend = updateformer_backend
        if updateformer_backend not in {
            "eager",
            "qkv",
            "fused",
            "graphed",
            "graphed_bucketed",
            "compiled",
            "bucketed",
            "bucketed_reduce",
            "te_mlp",
        }:
            raise ValueError(f"unknown UpdateFormer backend: {updateformer_backend}")
        self.add_space_attn = add_space_attn
        self.updateformer_input_dim = (
            # The positional encoding of the 3D flow from t=i to t=0
                + (self.flow_embed_dim + 1) * 3

                # The correlation features (LRR) for the three planes (xy, yz, xz), concatenated
                + self.corr_neighbors * self.corr_n_levels
                * (self.corr_n_groups
                   + 3 * self.corr_add_neighbor_offset
                   + 3 * self.corr_add_neighbor_xyz
                   + self.corr_pos_emb_size)

                # The features of the tracked points, one for each of the three planes
                + self.latent_dim

                # The visibility mask
                + 1

                # The whether-the-point-is-tracked mask
                + 1
        )
        time_embed_dim = self.updateformer_input_dim
        if time_embed_dim % 2 != 0:
            time_embed_dim += 1
        time_grid = (
            torch.linspace(0, self.S - 1, self.S).reshape(1, self.S, 1) / self.S
        )
        time_embedding = torch.from_numpy(
            get_1d_sincos_pos_embed_from_grid(time_embed_dim, time_grid[0])
        )[None].float()[..., :self.updateformer_input_dim]
        self.register_buffer(
            "_updateformer_time_embedding",
            time_embedding,
            persistent=False,
        )

        # Feature encoder
        self.fnet = BasicEncoder(
            input_dim=3,
            output_dim=self.latent_dim,
            norm_fn="instance",
            dropout=0,
            stride=self.stride,
            Embed3D=False,
        )

        # Transformer for iterative updates
        self.updateformer_hidden_size = hidden_size
        self.updateformer = EfficientUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=self.updateformer_input_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_dim=3 + self.latent_dim,
            mlp_ratio=4.0,
            add_space_attn=add_space_attn,
            num_virtual_tracks=num_virtual_tracks,
            attn_class=(
                FusedFlashAttention
                if updateformer_backend in {"qkv", "fused"}
                else FlashAttention if use_flash_attention else Attention
            ),
            linear_layer_for_vis_conf=False,
            checkpoint_updateformer=checkpoint_updateformer,
            execution_backend=(
                "fused"
                if updateformer_backend == "compiled"
                else "bucketed"
                if updateformer_backend == "bucketed"
                else "bucketed_reduce"
                if updateformer_backend == "bucketed_reduce"
                else "graphed_bucketed"
                if updateformer_backend == "graphed_bucketed"
                else "te_mlp"
                if updateformer_backend == "te_mlp"
                else updateformer_backend
                if updateformer_backend in {"fused", "graphed"}
                else "eager"
            ),
        )

        # Feature update + visibility
        self.ffeats_norm = nn.GroupNorm(1, self.latent_dim)
        self.ffeats_updater = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.GELU())
        self.vis_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

        self.stats_pyramid = None
        self.stats_depth = None

    def fnet_fwd(self, rgbs_normalized, image_features=None):
        b, v, t, _, h, w = rgbs_normalized.shape
        rgbs_normalized = rgbs_normalized.reshape(-1, 3, h, w)
        with torch.profiler.record_function("mvtracker/cnn_feature_encoder"):
            return self.fnet(rgbs_normalized)

    def init_stats(self):
        self.stats_pyramid = defaultdict(list)
        self.stats_depth = []

    def consume_stats(self):
        # Per-pyramid-level summary of neighbor distances
        level_to_norms = defaultdict(list)
        for (level, _), norm_lists in self.stats_pyramid.items():
            level_to_norms[level].extend(norm_lists)
        level_summary = []
        for level, norm_lists in level_to_norms.items():
            norms = np.concatenate(norm_lists).astype(float)
            stats = pd.Series(norms).describe(percentiles=[.25, .5, .75])
            level_summary.append({
                "level": level,
                "count": int(stats["count"]),
                "mean": round(float(stats["mean"] * 100), 1),
                "std": round(float(stats["std"] * 100), 1),
                "min": round(float(stats["min"] * 100), 1),
                "25%": round(float(stats["25%"] * 100), 1),
                "50%": round(float(stats["50%"] * 100), 1),
                "75%": round(float(stats["75%"] * 100), 1),
                "max": round(float(stats["max"] * 100), 1),
            })
        df_level_summary = pd.DataFrame(level_summary).sort_values("level")
        logging.info(f"Neighbor distances across pyramid levels:\n{df_level_summary}")

        # Per-pyramid-level and per-iteration summary of neighbor distances
        summary = []
        for (level, it), norm_lists in self.stats_pyramid.items():
            norms = np.concatenate(norm_lists).astype(float)
            stats = pd.Series(norms).describe(percentiles=[.25, .5, .75])
            summary.append({
                "level": level,
                "iteration": it,
                "count": int(stats["count"]),
                "mean": round(float(stats["mean"] * 100), 1),
                "std": round(float(stats["std"] * 100), 1),
                "min": round(float(stats["min"] * 100), 1),
                "25%": round(float(stats["25%"] * 100), 1),
                "50%": round(float(stats["50%"] * 100), 1),
                "75%": round(float(stats["75%"] * 100), 1),
                "max": round(float(stats["max"] * 100), 1),
            })
        df_summary = pd.DataFrame(summary).sort_values(["level", "iteration"])
        logging.info(f"Neighbor distances across pyramid levels and iterations (in cm):\n{(df_summary)}")

        # Valid vs invalid depth stats
        depth_stats = pd.Series(self.stats_depth).describe(percentiles=[.25, .5, .75]).astype(float).round(1)
        logging.info(f"Depth stats (valid vs invalid):\n{depth_stats}")

        self.stats_pyramid = None
        self.stats_depth = None

    def forward_iteration(
            self,
            fmaps,
            depths,
            intrs,
            extrs,
            coords_init,
            vis_init,
            track_mask,
            track_padding_mask=None,
            intrs_inv=None,
            extrs_inv=None,
            pointcloud_grids=None,
            capture_safe=False,
            iters=4,
            feat_init=None,
            save_debug_logs=False,
            debug_logs_path="",
            debug_logs_prefix="",
            debug_logs_window_idx=None,
            save_rerun_logs: bool = False,
            rerun_fmap_coloring_fn: Optional[Callable] = None,
    ):
        B, V, S, D, H, W = fmaps.shape
        N = coords_init.shape[2]
        device = fmaps.device
        if coords_init.shape[1] < S:
            coords = torch.cat([coords_init, coords_init[:, -1].repeat(1, S - coords_init.shape[1], 1, 1)], dim=1)
            vis_init = torch.cat([vis_init, vis_init[:, -1].repeat(1, S - vis_init.shape[1], 1, 1)], dim=1)
        else:
            coords = coords_init.clone()
        if track_mask.shape[1] < S:
            track_mask = torch.cat([
                track_mask,
                torch.zeros_like(track_mask[:, 0]).repeat(1, S - track_mask.shape[1], 1, 1),
            ], dim=1)
        assert D == self.latent_dim
        assert fmaps.shape == (B, V, S, D, H, W)
        assert depths.shape == (B, V, S, 1, H, W)
        assert intrs.shape == (B, V, S, 3, 3)
        assert extrs.shape == (B, V, S, 3, 4)
        assert coords.shape == (B, S, N, 3)
        assert vis_init.shape == (B, S, N, 1)
        assert track_mask.shape == (B, S, N, 1)
        assert feat_init is None or feat_init.shape == (B, S, N, self.latent_dim)
        if track_padding_mask is None:
            track_padding_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        assert track_padding_mask.shape == (B, N)

        if save_debug_logs:
            requested = track_mask.any(1).squeeze(-1)
            assert (requested | track_padding_mask).all(), (
                "All real points should be requested to be tracked at least once"
            )

        fcorr_fns = {}
        for lvl in range(self.corr_n_levels):
            with torch.profiler.record_function("mvtracker/build_pointcloud_pyramid"):
                pc = init_pointcloud_from_rgbd(
                    fmaps=fmaps,
                    depths=depths,
                    intrs=intrs,
                    extrs=extrs,
                    stride=self.stride,
                    level=lvl,
                    return_validity_mask=self.corr_filter_invalid_depth or save_rerun_logs,
                    intrs_inv=intrs_inv,
                    extrs_inv=extrs_inv,
                    pixel_xy_homo=(
                        pointcloud_grids[lvl]
                        if pointcloud_grids is not None else None
                    ),
                )
            if self.corr_filter_invalid_depth or save_rerun_logs:
                pc_xyz, pc_fvec, pc_valid = pc
            else:
                pc_xyz, pc_fvec = pc
                pc_valid = None
            fcorr_fns[lvl] = PointcloudCorrBlock(
                k=self.corr_neighbors,
                groups=self.corr_n_groups,
                xyz=pc_xyz,
                fvec=pc_fvec,
                filter_invalid=self.corr_filter_invalid_depth,
                valid=pc_valid,
                corr_add_neighbor_offset=self.corr_add_neighbor_offset,
                corr_add_neighbor_xyz=self.corr_add_neighbor_xyz,
                rerun_fmap_coloring_fn=rerun_fmap_coloring_fn,
                knn_fn=self.knn,
            )

        # Positional/time embeddings (keep shapes identical to before)
        embed_dim = self.updateformer_input_dim
        if embed_dim % 6 != 0:
            embed_dim += 6 - (embed_dim % 6)
        pos_embed_fn = (
            get_3d_sincos_pos_embed_from_grid_cuda
            if capture_safe else get_3d_sincos_pos_embed_from_grid
        )
        pos_embed = pos_embed_fn(embed_dim, coords[:, 0:1]).float()[:, 0].permute(0, 2, 1)
        if embed_dim > self.updateformer_input_dim:
            pos_embed = pos_embed[:, :self.updateformer_input_dim, :]
        pos_embed = rearrange(pos_embed, "b e n -> (b n) e").unsqueeze(1)

        times_embed = self._updateformer_time_embedding.expand(B, -1, -1)
        times_embed = (
            times_embed[:, None]
            .expand(B, N, S, self.updateformer_input_dim)
            .reshape(B * N, S, self.updateformer_input_dim)
        )

        coord_predictions = []

        ffeats = feat_init.clone()
        track_mask_and_vis = torch.cat([track_mask, vis_init], dim=3).permute(0, 2, 1, 3).reshape(B * N, S, 2)
        for it in range(iters):
            coords = coords.detach()

            # Sample correlation features around each point
            fcorrs = []
            for lvl in range(self.corr_n_levels):
                fcorr_fn = fcorr_fns[lvl]
                with torch.profiler.record_function("mvtracker/knn_and_correlation"):
                    fcorrs_level = (
                        fcorr_fn
                        .corr_sample(
                            targets=ffeats.reshape(B * S, N, self.latent_dim),
                            coords_world_xyz=coords.reshape(B * S, N, 3),
                            save_debug_logs=False,
                            debug_logs_path=debug_logs_path,
                            debug_logs_prefix=debug_logs_prefix + f"__iter_{it}__pyramid_level_{lvl}",
                            save_rerun_logs=save_rerun_logs,
                        )
                        .reshape(B, S, N, -1)
                    )
                fcorrs.append(fcorrs_level)
                if self.stats_pyramid is not None:
                    self.stats_pyramid[(lvl, it)] += [
                        np.linalg.norm(fcorrs_level.reshape(-1, 4)[:, 1:].detach().cpu().numpy(), axis=-1)
                    ]
            fcorrs = torch.cat(fcorrs, dim=-1)
            LRR = fcorrs.shape[3]
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)

            # Flow embedding
            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 3)
            flows_ = get_3d_embedding(flows_, self.flow_embed_dim, cat_coords=True)

            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            transformer_input = torch.cat([flows_, fcorrs_, ffeats_, track_mask_and_vis], dim=2)
            assert transformer_input.shape[-1] == pos_embed.shape[-1]
            x = transformer_input + pos_embed + times_embed
            x = rearrange(x, "(b n) t d -> b n t d", b=B)

            with torch.profiler.record_function("mvtracker/update_transformer"):
                delta = self.updateformer(x, point_mask=~track_padding_mask)
            delta = rearrange(delta, " b n t d -> (b n) t d")

            d_coord = delta[:, :, :3].reshape(B, N, S, 3).permute(0, 2, 1, 3)

            d_feats = delta[:, :, 3:self.latent_dim + 3]
            d_feats = self.ffeats_norm(d_feats.view(-1, self.latent_dim))
            d_feats = self.ffeats_updater(d_feats).view(B, N, S, self.latent_dim).permute(0, 2, 1, 3)

            coords = coords + d_coord
            ffeats = ffeats + d_feats

            if save_debug_logs and torch.isnan(coords).any():
                logging.error("Got NaN values in coords, perhaps the training exploded")
                import ipdb
                ipdb.set_trace()

            coord_predictions.append(coords.clone())

        vis_e = self.vis_predictor(ffeats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)

        return coord_predictions, vis_e, feat_init

    def forward(
            self,
            rgbs,
            depths,
            query_points,
            intrs,
            extrs,
            track_padding_mask=None,
            iters=4,
            image_features=None,
            is_train=False,
            save_debug_logs=False,
            debug_logs_path="",
            save_rerun_logs: bool = False,
            save_rerun_logs_output_rrd_path: Optional[str] = None,
            execution_schedule=None,
            camera_inverses=None,
            pointcloud_grids=None,
            capture_safe=False,
            **kwargs,
    ):
        device = extrs.device
        if save_debug_logs:
            if kwargs:
                logging.info(f"Unused kwargs: {kwargs.keys()}")

        batch_size, num_views, num_frames, _, height, width = rgbs.shape
        _, num_points, _ = query_points.shape
        logging.info(f"FWD pass: {num_views=} {num_frames=} {num_points=} "
                     f"{height=} {width=} {iters=} {num_points=} {rgbs.dtype=}")

        # I made a video tutorial here if it is easier to follow: https://www.youtube.com/watch?v=dQw4w9WgXcQ

        assert rgbs.shape == (batch_size, num_views, num_frames, 3, height, width)
        assert depths.shape == (batch_size, num_views, num_frames, 1, height, width)
        assert query_points.shape == (batch_size, num_points, 4)
        assert intrs.shape == (batch_size, num_views, num_frames, 3, 3)
        assert extrs.shape == (batch_size, num_views, num_frames, 3, 4)
        if track_padding_mask is None:
            track_padding_mask = torch.zeros(
                batch_size, num_points, dtype=torch.bool, device=query_points.device
            )
        else:
            track_padding_mask = track_padding_mask.to(
                device=query_points.device, dtype=torch.bool
            )
        if track_padding_mask.shape != (batch_size, num_points):
            raise ValueError(
                f"track_padding_mask must have shape {(batch_size, num_points)}, "
                f"got {tuple(track_padding_mask.shape)}"
            )
        if execution_schedule is None and track_padding_mask.all(dim=1).any():
            raise ValueError("every scene must contain at least one real trajectory")

        if execution_schedule is None:
            query_frames_for_schedule = query_points[:, :, 0].long().masked_fill(
                track_padding_mask, num_frames
            )
            schedule_starts = (
                query_frames_for_schedule.amin(dim=1).detach().cpu().tolist()
            )
        else:
            schedule_starts = list(execution_schedule["schedule_starts"])
        if batch_size > 1 and len(set(schedule_starts)) > 1:
            if self.normalize_scene_in_fwd_pass:
                raise ValueError("batched VGGT scene normalization is not supported")
            if save_rerun_logs:
                raise ValueError("batched Rerun logging requires aligned query schedules")
            grouped_results = {}
            for schedule_start in sorted(set(schedule_starts)):
                scene_indices = [
                    index for index, value in enumerate(schedule_starts)
                    if value == schedule_start
                ]
                index = torch.tensor(scene_indices, device=query_points.device)
                result = self.forward(
                    rgbs=rgbs.index_select(0, index),
                    depths=depths.index_select(0, index),
                    query_points=query_points.index_select(0, index),
                    intrs=intrs.index_select(0, index),
                    extrs=extrs.index_select(0, index),
                    track_padding_mask=track_padding_mask.index_select(0, index),
                    iters=iters,
                    image_features=None,
                    is_train=is_train,
                    save_debug_logs=save_debug_logs,
                    debug_logs_path=debug_logs_path,
                    execution_schedule={
                        name: [values[index] for index in scene_indices]
                        for name, values in execution_schedule.items()
                    } if execution_schedule is not None else None,
                    camera_inverses=(
                        camera_inverses[0].index_select(0, index),
                        camera_inverses[1].index_select(0, index),
                    ) if camera_inverses is not None else None,
                    pointcloud_grids=pointcloud_grids,
                    capture_safe=capture_safe,
                )
                for local_index, scene_index in enumerate(scene_indices):
                    grouped_results[scene_index] = {
                        "traj_e": result["traj_e"][local_index],
                        "vis_e": result["vis_e"][local_index],
                        "feat_init": result["feat_init"][local_index],
                        "train_data": (
                            result["train_data"]["scenes"][local_index]
                            if is_train else None
                        ),
                    }
            results = {
                name: torch.stack([
                    grouped_results[index][name] for index in range(batch_size)
                ])
                for name in ("traj_e", "vis_e", "feat_init")
            }
            if is_train:
                results["train_data"] = {
                    "scenes": [
                        grouped_results[index]["train_data"]
                        for index in range(batch_size)
                    ]
                }
            return results

        if save_debug_logs:
            os.makedirs(debug_logs_path, exist_ok=True)

        if save_rerun_logs:
            assert save_rerun_logs_output_rrd_path is not None
            import rerun as rr
            rr.init("3dpt", recording_id="v0.16")
            rr.set_time_seconds("frame", 0)

        if self.stats_depth is not None:
            self.stats_depth += [(depths == 0).float().mean().item() * 100]

        # Scene normalization (optional): Rigid transformation to center first camera and rescale the scene like VGGT
        if save_debug_logs:
            qp_range_before = np.stack([
                query_points[0, :, 1:].min(0).values.cpu().numpy().round(2),
                query_points[0, :, 1:].max(0).values.cpu().numpy().round(2),
            ])
        if self.normalize_scene_in_fwd_pass:
            assert batch_size == 1, "VGGT normalization assumes batch size 1"
            max_depth = 24
            _d = depths.clone()
            _d[_d < max_depth] = max_depth
            T_scale, T_rot, T_translation = compute_vggt_scene_normalization_transform(
                _d[0], extrs[0].to(_d.device), intrs[0].to(_d.device)
            )
            T_scale_inv = 1 / T_scale
            T_rot_inv = T_rot.transpose(0, 1)
            T_translation_inv = -T_translation @ T_rot_inv

            query_points, extrs = query_points[0], extrs[0]  # Remove batch dimension
            extrs, query_points, _, _ = transform_scene(T, extrs, query_points, None, None)
            query_points, extrs = query_points[None], extrs[None]  # Add batch dimension
        if save_debug_logs:
            qp_range_after = np.stack([
                query_points[0, :, 1:].min(0).values.cpu().numpy().round(2),
                query_points[0, :, 1:].max(0).values.cpu().numpy().round(2),
            ])
            logging.info(f"Query points range before normalization:\n{qp_range_before}")
            logging.info(f"Query points range after normalization: \n{qp_range_after}")

        self.is_train = is_train

        # Unpack the query points
        query_points_t = query_points[:, :, 0].long()
        query_points_xyz_worldspace = query_points[:, :, 1:]

        # Interpolate the rgbs and depthmaps to the stride of the SpaTracker
        strided_height = height // self.stride
        strided_width = width // self.stride

        ind_array = torch.arange(num_frames, device=query_points.device)[None, :, None]
        track_mask = (
            (ind_array >= query_points_t[:, None, :])
            & ~track_padding_mask[:, None, :]
        ).unsqueeze(-1)

        # Prepare the initial coordinates and visibility
        coords_init = query_points_xyz_worldspace.unsqueeze(1).repeat(1, self.S, 1, 1)
        vis_init = query_points.new_ones((batch_size, self.S, num_points, 1)) * 10

        # Sort each scene independently. Padded tracks stay at the end.
        query_points_t_for_sort = query_points_t.masked_fill(
            track_padding_mask, num_frames + 1
        )
        sort_inds = torch.argsort(query_points_t_for_sort, dim=1)
        inv_sort_inds = torch.argsort(sort_inds, dim=1)

        def gather_tracks(value, track_dim):
            shape = [1] * value.ndim
            shape[0] = batch_size
            shape[track_dim] = num_points
            index = sort_inds.view(shape)
            expand_shape = list(value.shape)
            expand_shape[track_dim] = num_points
            return torch.gather(value, track_dim, index.expand(expand_shape))

        if save_debug_logs:
            restored = torch.gather(
                torch.gather(query_points_t, 1, sort_inds), 1, inv_sort_inds
            )
            assert torch.equal(query_points_t, restored)

        query_points_t_ = gather_tracks(query_points_t, 1)
        query_points_xyz_worldspace_ = gather_tracks(query_points_xyz_worldspace, 1)
        coords_init_ = gather_tracks(coords_init, 2).clone()
        vis_init_ = gather_tracks(vis_init, 2).clone()
        track_mask_ = gather_tracks(track_mask, 2).clone()
        track_padding_mask_ = gather_tracks(track_padding_mask, 1)
        if execution_schedule is None:
            real_track_counts = (
                (~track_padding_mask_).sum(dim=1).detach().cpu().tolist()
            )
            query_times = [
                query_points_t_[batch_index, :count].detach().cpu().tolist()
                for batch_index, count in enumerate(real_track_counts)
            ]
        else:
            real_track_counts = list(execution_schedule["real_track_counts"])
            query_times = [list(values) for values in execution_schedule["query_times"]]

        # Delete the unsorted variables (for safety)
        del coords_init, vis_init, query_points_t, query_points, query_points_xyz_worldspace, track_mask

        # Placeholders for the results (for the sorted points)
        traj_e_ = coords_init_.new_zeros((batch_size, num_frames, num_points, 3))
        vis_e_ = coords_init_.new_zeros((batch_size, num_frames, num_points))

        w_idx_start = query_times[0][0]
        if any(times[0] != w_idx_start for times in query_times):
            raise RuntimeError("internal schedule grouping failed")
        p_idx_starts = [0] * batch_size
        scene_records = [
            {
                "coord_predictions": [],
                "vis_predictions": [],
                "p_idx_end_list": [],
                "window_starts": [],
                "sort_inds": sort_inds[index],
                "real_track_count": real_track_counts[index],
            }
            for index in range(batch_size)
        ]
        fmaps_seq, depths_seq, feat_init, rerun_fmap_coloring_fn = None, None, None, None
        graph_window_counts = []
        graph_window_start = w_idx_start
        while graph_window_start < num_frames - self.S // 2:
            graph_window_counts.append(max(
                bisect_left(times, graph_window_start + self.S)
                for times in query_times
            ))
            graph_window_start += self.S // 2
        self.updateformer.begin_graphed_sequence(
            graph_window_counts,
            iters,
            batch_size,
        )
        graph_window_index = 0
        while w_idx_start < num_frames - self.S // 2:
            p_idx_ends = [
                bisect_left(times, w_idx_start + self.S)
                for times in query_times
            ]
            p_idx_end = max(p_idx_ends)
            if p_idx_end == 0:
                raise RuntimeError("window contains no query trajectories")

            intrs_seq = intrs[:, :, w_idx_start:w_idx_start + self.S]
            extrs_seq = extrs[:, :, w_idx_start:w_idx_start + self.S]
            intrs_inv_seq = (
                camera_inverses[0][
                    :, :, w_idx_start:w_idx_start + self.S
                ] if camera_inverses is not None else None
            )
            extrs_inv_seq = (
                camera_inverses[1][
                    :, :, w_idx_start:w_idx_start + self.S
                ] if camera_inverses is not None else None
            )

            # Compute fmaps and interpolated depth on a rolling basis
            # to reduce peak GPU memory consumption, but don't recompute
            # for the overlapping part of a window
            if fmaps_seq is None:
                assert depths_seq is None
                new_seq_t0 = w_idx_start
            else:
                fmaps_seq = fmaps_seq[:, :, self.S // 2:]
                depths_seq = depths_seq[:, :, self.S // 2:]
                new_seq_t0 = w_idx_start + self.S // 2
            new_seq_t1 = w_idx_start + self.S

            with torch.profiler.record_function("mvtracker/downsample_depth"):
                _depths_seq_new = nn.functional.interpolate(
                    input=depths[:, :, new_seq_t0:new_seq_t1].to(device).reshape(-1, 1, height, width),
                    scale_factor=1.0 / self.stride,
                    mode="nearest",
                ).reshape(batch_size, num_views, -1, 1, strided_height, strided_width)
            depths_seq = smart_cat(depths_seq, _depths_seq_new, dim=2)

            _fmaps_seq_new = self.fnet_fwd(
                (2 * (rgbs[:, :, new_seq_t0: new_seq_t1].to(device) / 255.0) - 1.0),
                image_features,
            )
            assert _fmaps_seq_new.shape[-2:] == (strided_height, strided_width)
            _fmaps_seq_new = _fmaps_seq_new.reshape(
                batch_size,
                num_views,
                -1,
                self.latent_dim,
                strided_height,
                strided_width,
            )
            fmaps_seq = smart_cat(fmaps_seq, _fmaps_seq_new, dim=2)
            if feat_init is None:
                feat_init = _fmaps_seq_new.new_zeros(
                    batch_size, self.S, num_points, self.latent_dim
                )

            if save_rerun_logs and rerun_fmap_coloring_fn is None:
                valid_depths_mask = depths_seq.detach().cpu().squeeze(3) > 0
                fvec_flat = fmaps_seq.detach().cpu().permute(0, 1, 2, 4, 5, 3)[valid_depths_mask].numpy()
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=3)
                reducer.fit(fvec_flat)
                fvec_reduced = reducer.transform(fvec_flat)
                reducer_min = fvec_reduced.min(axis=0)
                reducer_max = fvec_reduced.max(axis=0)

                def fvec_to_rgb(fvec):
                    input_shape = fvec.shape
                    assert input_shape[-1] == self.latent_dim
                    fvec_reduced = reducer.transform(fvec.reshape(-1, self.latent_dim))
                    fvec_reduced = np.clip(fvec_reduced, reducer_min[None, :], reducer_max[None, :])
                    fvec_reduced_rescaled = (fvec_reduced - reducer_min) / (reducer_max - reducer_min)
                    fvec_reduced_rgb = (fvec_reduced_rescaled * 255).astype(int)
                    fvec_reduced_rgb = fvec_reduced_rgb.reshape(input_shape[:-1] + (3,))
                    return fvec_reduced_rgb

                rerun_fmap_coloring_fn = fvec_to_rgb

            S_local = fmaps_seq.shape[2]
            if S_local < self.S:
                diff = self.S - S_local
                fmaps_seq = torch.cat([fmaps_seq, fmaps_seq[:, :, -1:].repeat(1, 1, diff, 1, 1, 1)], 2)
                depths_seq = torch.cat([depths_seq, depths_seq[:, :, -1:].repeat(1, 1, diff, 1, 1, 1)], 2)
                intrs_seq = torch.cat([intrs_seq, intrs_seq[:, :, -1:].repeat(1, 1, diff, 1, 1)], 2)
                extrs_seq = torch.cat([extrs_seq, extrs_seq[:, :, -1:].repeat(1, 1, diff, 1, 1)], 2)
                if intrs_inv_seq is not None:
                    intrs_inv_seq = torch.cat([
                        intrs_inv_seq,
                        intrs_inv_seq[:, :, -1:].repeat(1, 1, diff, 1, 1),
                    ], 2)
                    extrs_inv_seq = torch.cat([
                        extrs_inv_seq,
                        extrs_inv_seq[:, :, -1:].repeat(1, 1, diff, 1, 1),
                    ], 2)

            # Compute query features independently per scene and query frame.
            if any(end > start for start, end in zip(p_idx_starts, p_idx_ends)):
                with torch.profiler.record_function("mvtracker/query_feature_pointcloud"):
                    rgbd_xyz, rgbd_fvec = init_pointcloud_from_rgbd(
                        fmaps=_fmaps_seq_new,
                        depths=_depths_seq_new,
                        intrs=intrs[:, :, new_seq_t0:new_seq_t1],
                        extrs=extrs[:, :, new_seq_t0:new_seq_t1],
                        stride=self.stride,
                        intrs_inv=(
                            camera_inverses[0][:, :, new_seq_t0:new_seq_t1]
                            if camera_inverses is not None else None
                        ),
                        extrs_inv=(
                            camera_inverses[1][:, :, new_seq_t0:new_seq_t1]
                            if camera_inverses is not None else None
                        ),
                        pixel_xy_homo=(
                            pointcloud_grids[0]
                            if pointcloud_grids is not None else None
                        ),
                    )

                new_num_frames = _fmaps_seq_new.shape[2]
                rgbd_xyz = rgbd_xyz.reshape(batch_size, new_num_frames, num_views, strided_height * strided_width, 3)
                rgbd_fvec = rgbd_fvec.reshape(batch_size, new_num_frames, num_views, strided_height * strided_width,
                                              self.latent_dim)

                for batch_idx in range(batch_size):
                    for t in range(new_seq_t0, new_seq_t1):
                        query_start = max(
                            p_idx_starts[batch_idx],
                            bisect_left(query_times[batch_idx], t),
                        )
                        query_end = min(
                            p_idx_ends[batch_idx],
                            bisect_left(query_times[batch_idx], t + 1),
                        )
                        if query_start == query_end:
                            continue
                        query_points_world = query_points_xyz_worldspace_[
                            batch_idx, query_start:query_end
                        ]
                        rgbd_xyz_current = rgbd_xyz[
                            batch_idx, t - new_seq_t0
                        ].reshape(-1, 3)
                        rgbd_fvec_current = rgbd_fvec[
                            batch_idx, t - new_seq_t0
                        ].reshape(-1, self.latent_dim)
                        with torch.profiler.record_function("mvtracker/query_feature_knn"):
                            _, neighbor_indices = self.knn(
                                1,
                                rgbd_xyz_current[None],
                                query_points_world[None],
                            )
                        neighbor_fvec = rgbd_fvec_current[neighbor_indices[0, :, 0]]
                        feat_init[
                            batch_idx, :, query_start:query_end
                        ] = neighbor_fvec[None].expand(self.S, -1, -1)

            # Update the initial coordinates and visibility for non-first windows
            for batch_idx, previous_count in enumerate(p_idx_starts):
                if previous_count == 0:
                    continue
                last_coords = coords[-1][
                    batch_idx:batch_idx + 1, self.S // 2:, :previous_count
                ].clone()
                coords_init_[
                    batch_idx:batch_idx + 1, :self.S // 2, :previous_count
                ] = last_coords
                coords_init_[
                    batch_idx:batch_idx + 1, self.S // 2:, :previous_count
                ] = last_coords[:, -1:].expand(-1, self.S // 2, -1, -1)

                last_vis = vis[
                    batch_idx:batch_idx + 1, self.S // 2:, :previous_count
                ][..., None]
                vis_init_[
                    batch_idx:batch_idx + 1, :self.S // 2, :previous_count
                ] = last_vis
                vis_init_[
                    batch_idx:batch_idx + 1, self.S // 2:, :previous_count
                ] = last_vis[:, -1:].expand(-1, self.S // 2, -1, -1)

            track_mask_current = track_mask_[:, w_idx_start: w_idx_start + self.S, :p_idx_end]
            if S_local < self.S:
                track_mask_current = torch.cat([
                    track_mask_current,
                    track_mask_current[:, -1:].repeat(1, self.S - S_local, 1, 1),
                ], 1)

            active_counts = (
                execution_schedule["active_count_tensors"][graph_window_index]
                if execution_schedule is not None
                else torch.tensor(p_idx_ends, device=device)
            )[:, None]
            active_padding_mask = (
                torch.arange(p_idx_end, device=device)[None] >= active_counts
            ) | track_padding_mask_[:, :p_idx_end]

            coords, vis, _ = self.forward_iteration(
                fmaps=fmaps_seq,
                depths=depths_seq,
                intrs=intrs_seq,
                extrs=extrs_seq,
                coords_init=coords_init_[:, :, :p_idx_end],
                feat_init=feat_init[:, :, :p_idx_end],
                vis_init=vis_init_[:, :, :p_idx_end],
                track_mask=track_mask_current,
                track_padding_mask=active_padding_mask,
                intrs_inv=intrs_inv_seq,
                extrs_inv=extrs_inv_seq,
                pointcloud_grids=pointcloud_grids,
                capture_safe=capture_safe,
                iters=iters,
                save_debug_logs=save_debug_logs,
                debug_logs_path=debug_logs_path,
                debug_logs_prefix=f"__widx-{w_idx_start}_pidx-{p_idx_end}",
                debug_logs_window_idx=w_idx_start,
                save_rerun_logs=save_rerun_logs,
                rerun_fmap_coloring_fn=rerun_fmap_coloring_fn,
            )

            if is_train:
                for batch_idx, count in enumerate(p_idx_ends):
                    record = scene_records[batch_idx]
                    record["coord_predictions"].append([
                        coord[batch_idx:batch_idx + 1, :S_local, :count]
                        if not self.normalize_scene_in_fwd_pass
                        else transform_scene(
                            T_scale_inv, T_rot_inv, T_translation_inv,
                            None, None, None,
                            coord[batch_idx, :S_local, :count], None,
                        )[2][None]
                        for coord in coords
                    ])
                    record["vis_predictions"].append(
                        vis[batch_idx:batch_idx + 1, :S_local, :count]
                    )
                    record["p_idx_end_list"].append(count)
                    record["window_starts"].append(w_idx_start)

            for batch_idx, count in enumerate(p_idx_ends):
                traj_e_[
                    batch_idx, w_idx_start:w_idx_start + S_local, :count
                ] = coords[-1][batch_idx, :S_local, :count]
                vis_e_[
                    batch_idx, w_idx_start:w_idx_start + S_local, :count
                ] = torch.sigmoid(vis[batch_idx, :S_local, :count])
                track_mask_[
                    batch_idx, :w_idx_start + self.S, :count
                ] = 0.0
            w_idx_start = w_idx_start + self.S // 2
            p_idx_starts = p_idx_ends
            graph_window_index += 1

        self.updateformer.end_graphed_sequence()

        if save_debug_logs:
            import gpustat
            torch.cuda.empty_cache()
            logging.info(f"Forward pass GPU usage: {gpustat.new_query()}")

        if save_rerun_logs:
            import rerun as rr
            rr.save(save_rerun_logs_output_rrd_path)
            logging.info(f"Saved Rerun recording to: {os.path.abspath(save_rerun_logs_output_rrd_path)}.")

        traj_e = torch.gather(
            traj_e_, 2,
            inv_sort_inds[:, None, :, None].expand(-1, num_frames, -1, 3),
        )
        vis_e = torch.gather(
            vis_e_, 2,
            inv_sort_inds[:, None, :].expand(-1, num_frames, -1),
        )

        # Un-normalize the scene
        if self.normalize_scene_in_fwd_pass:
            traj_e = transform_scene(T_scale_inv, T_rot_inv, T_translation_inv,
                                     None, None, None, traj_e[0], None)[2][None]

        results = {
            "traj_e": traj_e,
            "feat_init": feat_init,
            "vis_e": vis_e,
        }
        if self.is_train:
            results["train_data"] = {"scenes": scene_records}
        return results


def compute_vggt_scene_normalization_transform(depths, extrs, intrs):
    V, T, _, H, W = depths.shape
    device = depths.device

    extrs_square = torch.eye(4, device=device)[None, None].repeat(V, T, 1, 1)
    extrs_square[:, :, :3, :] = extrs
    extrs_inv = torch.inverse(extrs_square.float()).type(extrs.dtype)

    intrs_inv = torch.inverse(intrs.float()).type(intrs.dtype)

    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    homog = torch.stack([x, y, torch.ones_like(x)], dim=-1).float().reshape(-1, 3)
    homog = homog[None].expand(V, -1, -1).type(depths.dtype)

    cam_points = torch.einsum("vij,vnj->vni", intrs_inv[:, 0], homog) * depths[:, 0].reshape(V, -1, 1)
    cam_points_h = torch.cat([cam_points, torch.ones_like(cam_points[..., :1])], dim=-1)
    world_points_h = torch.einsum("vij,vnj->vni", extrs_inv[:, 0], cam_points_h)

    world_points_in_first = torch.einsum("ij,vnj->vni", extrs[0, 0], world_points_h)

    mask = (depths[:, 0] > 0).reshape(V, -1)
    valid_points = world_points_in_first[mask]
    avg_dist = valid_points.norm(dim=1).mean()
    scale = 1.0 / avg_dist

    rot = extrs[0, 0, :3, :3]
    translation = extrs[0, 0, :3, 3] * scale
    return scale, rot, translation


class PointcloudCorrBlock:
    def __init__(
            self,
            k: int,
            groups,
            xyz: torch.Tensor,
            fvec: torch.Tensor,
            corr_add_neighbor_offset: bool,
            corr_add_neighbor_xyz: bool,
            filter_invalid: bool = False,
            valid: Optional[torch.Tensor] = None,
            rerun_fmap_coloring_fn: Optional[Callable] = None,
            knn_fn: Callable = _knn_capturable,
    ):
        self.B, self.N, self.C = fvec.shape
        assert xyz.shape == (self.B, self.N, 3)
        assert fvec.shape == (self.B, self.N, self.C)
        assert k <= self.N, "k should be less than or equal to N"
        assert groups <= self.C, "number of correlation groups should not be larger than the number of channels"
        assert self.C % groups == 0, "number of channels must be divisible by the number of groups (for convenience)"
        assert not filter_invalid or valid is not None

        self.k = k
        self.groups = groups
        self.xyz = xyz
        self.fvec = fvec
        self.corr_add_neighbor_offset = corr_add_neighbor_offset
        self.corr_add_neighbor_xyz = corr_add_neighbor_xyz
        self.filter_invalid = filter_invalid
        self.valid = valid
        self.rerun_fmap_coloring_fn = rerun_fmap_coloring_fn
        self.knn = knn_fn

    def corr_sample(
            self,
            targets: torch.Tensor,
            coords_world_xyz: torch.Tensor,
            save_debug_logs=False,
            debug_logs_path=".",
            debug_logs_prefix="corr",
            save_rerun_logs=False,
    ):
        # Check inputs
        _, M, _ = targets.shape
        assert targets.shape == (self.B, M, self.C)
        assert coords_world_xyz.shape == (self.B, M, 3)

        # Find neighbors for each of the N target points
        if not self.filter_invalid:
            neighbor_dists, neighbor_indices = self.knn(
                self.k, self.xyz, coords_world_xyz
            )
        else:
            neighbor_dists = []
            neighbor_indices = []
            for xyz_i, valid_i, coords_world_xyz_i in zip(self.xyz, self.valid, coords_world_xyz):
                xyz_i = xyz_i[valid_i]
                neighbor_dists_i, neighbor_indices_i = self.knn(
                    self.k, xyz_i[None], coords_world_xyz_i[None]
                )
                neighbor_dists.append(neighbor_dists_i)
                neighbor_indices.append(neighbor_indices_i)
            neighbor_dists = torch.cat(neighbor_dists)
            neighbor_indices = torch.cat(neighbor_indices)
        batch_idx = torch.arange(self.B, device=self.xyz.device)[:, None, None]
        neighbor_xyz = self.xyz[batch_idx, neighbor_indices]

        # Compute the local correlations
        corrs = indexed_grouped_correlation(
            targets, self.fvec, neighbor_indices, self.groups
        )

        output = corrs

        # Append the distance/direction features to the correlation
        neighbor_offset_in_world_xyz = neighbor_xyz - coords_world_xyz[..., None, :]
        if self.corr_add_neighbor_offset:
            output = torch.cat([corrs, neighbor_offset_in_world_xyz], -1)

        # Append the neighbor xyz to the correlation
        if self.corr_add_neighbor_xyz:
            output = torch.cat([output, neighbor_xyz], -1)

        if save_debug_logs or save_rerun_logs:
            neighbor_fvec = self.fvec[batch_idx, neighbor_indices]

        if save_debug_logs:

            from sklearn.decomposition import PCA
            fvec_flat = self.fvec.reshape(-1, self.C).detach().cpu().numpy()
            reducer = PCA(n_components=3)
            reducer.fit(fvec_flat)

            fvec_reduced = reducer.transform(fvec_flat)
            reducer_min = fvec_reduced.min(axis=0)
            reducer_max = fvec_reduced.max(axis=0)

            def fvec_to_rgb(fvec):
                fvec_reduced = reducer.transform(fvec)
                fvec_reduced_rescaled = (fvec_reduced - reducer_min) / (reducer_max - reducer_min)
                fvec_reduced_rgb = (fvec_reduced_rescaled * 255).astype(int)
                return fvec_reduced_rgb

            for b in [0, self.B - 1]:
                # Save all points
                xyz = self.xyz[b].detach().cpu().numpy()
                xyz_colors = fvec_to_rgb(self.fvec[b].detach().cpu().numpy())
                save_pointcloud_to_ply(os.path.join(debug_logs_path, f"{time_now()}{debug_logs_prefix}_all_b{b}.ply"),
                                       xyz, xyz_colors)

                for n in range(3):
                    neighbors = neighbor_xyz[b, n].detach().cpu().numpy()
                    neighbors_colors = fvec_to_rgb(neighbor_fvec[b, n].detach().cpu().numpy())
                    save_pointcloud_to_ply(
                        os.path.join(debug_logs_path, f"{time_now()}{debug_logs_prefix}_neighbors_b{b}_n{n}.ply"),
                        neighbors, neighbors_colors)

                for n in range(3):
                    neighbors = neighbor_xyz[b, n].detach().cpu().numpy()
                    neighbors_colors = fvec_to_rgb(neighbor_fvec[b, n].detach().cpu().numpy())
                    query_point = coords_world_xyz[b, n].detach().cpu().numpy()
                    query_point_color = fvec_to_rgb(targets[b, n].detach().cpu().numpy().reshape(1, -1))
                    combined_points = np.vstack([query_point, neighbors])
                    combined_colors = np.vstack([query_point_color, neighbors_colors])
                    query_point_index = 0
                    neighbor_indices = np.arange(1, len(neighbors) + 1)
                    edges = np.array([[query_point_index, i] for i in neighbor_indices])
                    save_pointcloud_to_ply(os.path.join(debug_logs_path,
                                                        f"{time_now()}{debug_logs_prefix}_query_b{b}_n{n}_with_edges.ply"),
                                           combined_points, combined_colors, edges=edges)

        # Visualize the results with rerun.io
        if save_rerun_logs:
            import rerun as rr
            import re

            assert self.C > 1
            rerun_fps = 30
            log_feature_maps = True
            log_knn_neighbors = False
            knn_line_coloring = "static"
            knn_neighbors_to_log = 6

            logging.info(f"rerun for {debug_logs_prefix} started")

            ## Mask out target scene area
            # xyz = self.xyz.detach().cpu().numpy()
            # bbox = np.array([[-4, 4], [-3, 3.7], [1.2, 5.2]]) # Softball bbox
            # mask = (
            #         (xyz[..., 0] > bbox[0, 0])
            #         & (xyz[..., 0] < bbox[0, 1])
            #         & (xyz[..., 1] > bbox[1, 0])
            #         & (xyz[..., 1] < bbox[1, 1])
            #         & (xyz[..., 2] > bbox[2, 0])
            #         & (xyz[..., 2] < bbox[2, 1])
            # )
            xyz = self.xyz.detach().cpu().numpy()
            mask = np.ones_like(xyz[..., 0]).astype(bool)
            if self.valid is not None:
                mask = self.valid.detach().cpu().numpy()

            # PCA-based feature coloring
            if self.rerun_fmap_coloring_fn is None:
                fvec_flat = self.fvec.detach().cpu().numpy()[mask]
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=3)
                reducer.fit(fvec_flat)
                fvec_reduced = reducer.transform(fvec_flat)
                reducer_min = fvec_reduced.min(axis=0)
                reducer_max = fvec_reduced.max(axis=0)

                def fvec_to_rgb(fvec):
                    input_shape = fvec.shape
                    assert input_shape[-1] == self.C
                    fvec_reduced = reducer.transform(fvec.reshape(-1, self.C))
                    fvec_reduced = np.clip(fvec_reduced, reducer_min[None, :], reducer_max[None, :])
                    fvec_reduced_rescaled = (fvec_reduced - reducer_min) / (reducer_max - reducer_min)
                    fvec_reduced_rgb = (fvec_reduced_rescaled * 255).astype(int)
                    fvec_reduced_rgb = fvec_reduced_rgb.reshape(input_shape[:-1] + (3,))
                    return fvec_reduced_rgb

                self.rerun_fmap_coloring_fn = fvec_to_rgb

            fvec_colors = self.rerun_fmap_coloring_fn(self.fvec.detach().cpu().numpy())
            targets_colors = self.rerun_fmap_coloring_fn(targets.detach().cpu().numpy())
            neighbor_fvec_colors = self.rerun_fmap_coloring_fn(neighbor_fvec.detach().cpu().numpy())

            import re
            pattern = r'__widx-(\d+)_pidx-(\d+)-(\d+)__iter_(\d+)__pyramid_level_(\d+)'
            match = re.search(pattern, debug_logs_prefix)
            assert match
            t_start = int(match.group(1))
            pidx_start = int(match.group(2))
            pidx_end = int(match.group(3))
            iteration = int(match.group(4))
            pyramid_level = int(match.group(5))

            # # Log fmaps as images for the pipeline figure
            # import os
            # from PIL import Image
            # png_outdir = os.path.join(debug_logs_path, "feature_maps_pngs_2")
            # os.makedirs(png_outdir, exist_ok=True)
            # if pyramid_level == 0 and iteration == 0:
            #     for b in range(self.B):
            #         t = t_start + b
            #         for v in range(8):
            #             fvec_rgb_uint8 = fvec_colors[b].reshape(8, 96, 128, 3)[v].astype(np.uint8)
            #             fname = f"fmap__view{v:02d}__frame{t:05d}.png"
            #             fpath = os.path.join(png_outdir, fname)
            #             Image.fromarray(fvec_rgb_uint8).save(fpath)

            # Log feature map points
            # if log_feature_maps and pyramid_level in [0, 1, 2, 3] and iteration == 0:
            if log_feature_maps and pyramid_level in [0] and iteration == 0:
                if t_start > 0:
                    bs = range(self.B)
                else:
                    bs = range(self.B // 2, self.B)
                for b in bs:
                    rr.set_time_seconds("frame", (t_start + b) / rerun_fps)
                    rr.log(f"fmaps/pyramid-{pyramid_level}", rr.Points3D(
                        xyz[b][mask[b]],
                        colors=fvec_colors[b][mask[b]],
                        radii=0.042,
                        # radii=-2.53,
                    ))

            # Log neighbors
            if log_knn_neighbors and pyramid_level in [0, 1, 2, 3] and iteration in [0]:
                for b in range(self.B):
                    rr.set_time_seconds("frame", (t_start + b) / rerun_fps)
                    for n in range(min(neighbor_xyz.shape[1], knn_neighbors_to_log)):  # Iterate over queries
                        prefix = f"knn/track-{n:03d}/iter-{iteration}/pyramid-{pyramid_level}"
                        rr.log(f"{prefix}/queries", rr.Points3D(
                            coords_world_xyz[b, n].cpu().numpy(),
                            colors=targets_colors[b, n],
                            radii=0.072,
                            # radii=-9.0,
                        ))

                        rr.log(f"{prefix}/neighbors", rr.Points3D(
                            neighbor_xyz[b, n].cpu().numpy(),
                            colors=neighbor_fvec_colors[b, n],
                            radii=0.054,
                            # radii=-5.0,
                        ))

                        if knn_line_coloring == "correlation":
                            # Compute correlation strength for line coloring
                            corr_strength = corrs[b, n,].squeeze(-1).cpu().numpy()
                            corr_strength_normalized = (corr_strength / corr_strength.max()) * 1.0 + 0.0
                            line_colors = (corr_strength_normalized[:, None] * np.array([9, 208, 239])).astype(int)
                            line_colors = np.hstack([line_colors, np.full((line_colors.shape[0], 1), 204)])  # RGBA 80%

                        elif knn_line_coloring == "static":
                            # Make the lines sun flower yellow (241, 196, 15)
                            line_colors = np.array([241, 196, 15])[None].repeat(self.k, 0).astype(int)

                        # Draw edges between query and its neighbors
                        strips = np.stack([
                            coords_world_xyz[b, n].cpu().numpy()[None].repeat(neighbor_xyz.shape[2], axis=0),
                            neighbor_xyz[b, n].cpu().numpy(),
                        ], axis=-2)
                        rr.log(f"{prefix}/arrows", rr.Arrows3D(
                            origins=strips[:, 0],
                            vectors=strips[:, 1] - strips[:, 0],
                            colors=line_colors,
                            radii=0.016,
                            # radii=-1.2,
                        ))
            logging.info(f"rerun for {debug_logs_prefix} done")
        return output
