"""CPU contracts for deterministic homogeneous TAPVid request batches."""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_sampler_api():
    path = ROOT / "mvtracker" / "datasets" / "utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and getattr(node, "name", "") in {"SampleRequest", "HomogeneousViewBatchSampler"}
    ]
    namespace = {
        "dataclass": dataclass,
        "np": np,
        "Sequence": Sequence,
        "torch": torch,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace["SampleRequest"], namespace["HomogeneousViewBatchSampler"]


def _load_collate_api():
    path = ROOT / "mvtracker" / "datasets" / "utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and getattr(node, "name", "") in {"Datapoint", "collate_fn"}
    ]
    namespace = {
        "Any": Any,
        "List": List,
        "Optional": Optional,
        "dataclass": dataclass,
        "np": np,
        "torch": torch,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace["Datapoint"], namespace["collate_fn"]


SampleRequest, HomogeneousViewBatchSampler = _load_sampler_api()
Datapoint, collate_fn = _load_collate_api()


class _Dataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def _batches(batch_size, *, rank=0, world_size=1, epoch=0):
    sampler = HomogeneousViewBatchSampler(
        _Dataset(36),
        batch_size,
        rank=rank,
        world_size=world_size,
        seed=23,
    )
    sampler.set_epoch(epoch)
    return list(sampler)


def test_batches_are_homogeneous_for_arbitrary_physical_sizes():
    for batch_size in (1, 2, 3):
        batches = _batches(batch_size)
        assert len(batches) == 36 // batch_size
        assert all(len({request.view_count for request in batch}) == 1 for batch in batches)
        assert all(isinstance(request, SampleRequest) for batch in batches for request in batch)


def test_rank_batches_share_schedule_and_partition_indices():
    rank_zero = _batches(3, rank=0, world_size=2)
    rank_one = _batches(3, rank=1, world_size=2)
    assert len(rank_zero) == len(rank_one)
    for batch_zero, batch_one in zip(rank_zero, rank_one):
        assert {request.view_count for request in batch_zero} == {request.view_count for request in batch_one}
        indices_zero = {request.virtual_index for request in batch_zero}
        indices_one = {request.virtual_index for request in batch_one}
        assert indices_zero.isdisjoint(indices_one)
    all_indices = {
        request.virtual_index
        for batch in rank_zero + rank_one
        for request in batch
    }
    assert all_indices == set(range(36))


def test_epoch_changes_schedule_but_is_reproducible():
    first = _batches(2, epoch=0)
    repeat = _batches(2, epoch=0)
    next_epoch = _batches(2, epoch=1)
    assert first == repeat
    assert [batch[0].view_count for batch in first] != [batch[0].view_count for batch in next_epoch]


def test_collate_pads_only_tracks_and_returns_scene_scales():
    def scene(track_count, scale):
        return Datapoint(
            video=torch.zeros(2, 3, 3, 4, 4),
            videodepth=torch.zeros(2, 3, 1, 4, 4),
            segmentation=torch.zeros(2, 3, 1, 4, 4),
            trajectory=torch.zeros(2, track_count, 2),
            trajectory_3d=torch.zeros(2, track_count, 3),
            visibility=torch.ones(2, track_count, dtype=torch.bool),
            valid=torch.ones(2, track_count),
            query_points_3d=torch.zeros(track_count, 4),
            intrs=torch.eye(3).repeat(2, 3, 1, 1),
            extrs=torch.zeros(2, 3, 3, 4),
            track_upscaling_factor=scale,
        )

    batch, gotit = collate_fn([(scene(2, 1.5), True), (scene(5, 2.5), True)])
    assert gotit == [True, True]
    assert batch.video.shape[1] == 2
    assert batch.trajectory.shape == (2, 2, 5, 2)
    assert batch.track_padding_mask.tolist() == [[False, False, True, True, True], [False] * 5]
    assert batch.track_upscaling_factor.tolist() == [1.5, 2.5]
