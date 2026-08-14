# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from mvtracker.models.core.model_utils import reduce_masked_mean

EPS = 1e-6

sigma = 3
x_grid = torch.arange(-7, 8, 1)
y_grid = torch.arange(-7, 8, 1)
x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing="ij")
gridxy = torch.stack([x_grid, y_grid], dim=-1).float()
gs_kernel = torch.exp(-torch.sum(gridxy ** 2, dim=-1) / (2 * sigma ** 2))


def _reduce_per_scene(values, mask):
    """Reduce a [B, ...] masked tensor to one value per scene.

    Keep the call to ``reduce_masked_mean`` in its original two-argument form:
    this is also used by lightweight CPU test harnesses that provide the
    project's reduction primitive.
    """
    return torch.stack([
        reduce_masked_mean(values[scene_index], mask[scene_index])
        for scene_index in range(values.shape[0])
    ])


def balanced_ce_loss(pred, gt, valid=None):
    total_balanced_loss = None
    for j in range(len(gt)):
        B, S, N = gt[j].shape
        # pred and gt are the same shape
        for (a, b) in zip(pred[j].size(), gt[j].size()):
            assert a == b  # some shape mismatch!
        # if valid is not None:
        for (a, b) in zip(pred[j].size(), valid[j].size()):
            assert a == b  # some shape mismatch!

        pos = (gt[j] > 0.95).float()
        neg = (gt[j] < 0.05).float()

        label = pos * 2.0 - 1.0
        a = -label * pred[j]
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

        # Reduce each scene independently before averaging the batch.  This
        # keeps a scene with more valid tracks from dominating a mixed batch.
        pos_loss = _reduce_per_scene(loss, pos * valid[j])
        neg_loss = _reduce_per_scene(loss, neg * valid[j])

        balanced_loss = pos_loss + neg_loss
        total_balanced_loss = (
            balanced_loss
            if total_balanced_loss is None
            else total_balanced_loss + balanced_loss
        )
    if total_balanced_loss is None:
        raise ValueError("balanced_ce_loss requires at least one prediction window")
    return total_balanced_loss.mean()


def sequence_loss_3d(flow_preds, flow_gt, vis, valids, gamma=0.8, dmin=0.1, dmax=65, Dz=128):
    """Loss function defined over sequence of flow predictions with z component post-processing"""
    total_flow_loss = None
    J = len(flow_gt)
    for j in range(J):
        B, S, N, D = flow_gt[j].shape
        assert D == 3
        B, S1, N = vis[j].shape
        B, S2, N = valids[j].shape
        assert S == S1
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i]
            flow_gt_j = flow_gt[j].clone()
            flow_pred[..., 2] = (flow_pred[..., 2] - dmin) / (dmax - dmin) * Dz
            flow_gt_j[..., 2] = (flow_gt_j[..., 2] - dmin) / (dmax - dmin) * Dz
            i_loss = (flow_pred - flow_gt_j).abs()  # B, S, N, 3
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            # Reduce each scene independently before averaging the batch.  Keep
            # the reduction primitive's original two-argument API so this
            # function remains usable by lightweight CPU harnesses.
            per_scene_loss = torch.stack([
                reduce_masked_mean(i_loss[scene_index], valids[j][scene_index])
                for scene_index in range(i_loss.shape[0])
            ])
            flow_loss += i_weight * per_scene_loss
        flow_loss = flow_loss / n_predictions
        total_flow_loss = (
            flow_loss
            if total_flow_loss is None
            else total_flow_loss + flow_loss
        )
    if total_flow_loss is None:
        raise ValueError("sequence_loss_3d requires at least one ground-truth window")
    # Window averaging is unchanged; average scenes only after all windows have
    # been accumulated so each scene contributes equally.
    return (total_flow_loss / float(J)).mean()
