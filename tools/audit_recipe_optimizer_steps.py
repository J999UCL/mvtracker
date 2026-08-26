"""Replay selected mixed-recipe samples and audit their optimizer gradients.

The audit is deliberately a diagnostic: it loads one checkpoint, replays the
eight logical samples in each requested recipe step, performs forward/backward
individually, and never calls ``optimizer.step``.  Optimizer steps are
1-indexed by the training logs while ``steps.jsonl`` is 0-indexed; therefore
the defaults 625 and 688 replay recipe steps 624 and 687 respectively.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch
import numpy as np


DEFAULT_OPTIMIZER_STEPS = (625, 688)
DEFAULT_SKETCH_SIZE = 2048
DEFAULT_SKETCH_SEED = 0
SOURCE_NAMES = ("diegesis", "syn4d", "mvkubric")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float | None:
    denominator = first.float().norm() * second.float().norm()
    if float(denominator) == 0.0:
        return None
    return float((torch.dot(first.float(), second.float()) / denominator).clamp(-1, 1))


def _gradient_relations(records: list[dict[str, Any]]) -> dict[str, Any]:
    sketches = [torch.as_tensor(record["gradient_sketch"], dtype=torch.float32) for record in records]
    pairwise = []
    for first in range(len(sketches)):
        for second in range(first + 1, len(sketches)):
            one, two = sketches[first], sketches[second]
            pairwise.append({
                "first_logical_index": int(records[first].get("logical_index", first)),
                "second_logical_index": int(records[second].get("logical_index", second)),
                "cosine": _cosine(one, two),
                "dot": float(torch.dot(one, two)),
                "influence": float(torch.dot(one, two)),
            })
    leave_one_out = []
    for index, sketch in enumerate(sketches):
        others = torch.stack([value for position, value in enumerate(sketches) if position != index]).mean(0)
        leave_one_out.append({
            "logical_index": int(records[index].get("logical_index", index)),
            "cosine_to_others_mean": _cosine(sketch, others),
            "dot_to_others_mean": float(torch.dot(sketch, others)),
            "influence": float(torch.dot(sketch, others)),
            "norm": float(sketch.norm()),
            "others_norm": float(others.norm()),
        })
    return {"pairwise": pairwise, "leave_one_out": leave_one_out}


def _concentration(batch, output: Mapping[str, Any], cfg) -> dict[str, Any]:
    """Decompose the exact weighted ``sequence_loss_3d`` by track."""
    trace = output.get("training_trace")
    if trace is None:
        raise RuntimeError("trajectory concentration requires captured training trace")
    query_times = sorted(
        int(value) for value in batch.query_points_3d[0, :, 0].detach().cpu().tolist()
    )
    sort_indices = torch.as_tensor(
        sorted(
            range(len(query_times)),
            key=lambda index: int(batch.query_points_3d[0, index, 0]),
        ),
        device=batch.trajectory_3d.device,
        dtype=torch.long,
    )
    sequence_length = int(cfg.model.sliding_window_len)
    frame_count = int(batch.trajectory_3d.shape[1])
    track_count = int(batch.trajectory_3d.shape[2])
    per_track = torch.zeros(track_count, device=batch.trajectory_3d.device, dtype=torch.float32)
    trace_index = 0
    window_count = 0
    for window_start in range(query_times[0], frame_count - sequence_length // 2, sequence_length // 2):
        p_idx_end = bisect_left(query_times, window_start + sequence_length)
        if p_idx_end == 0:
            raise RuntimeError("trajectory trace window contains no query tracks")
        target = batch.trajectory_3d.index_select(2, sort_indices)
        target = target[:, window_start:window_start + sequence_length, :p_idx_end].float()
        valid = batch.valid.index_select(2, sort_indices)
        valid = valid[:, window_start:window_start + sequence_length, :p_idx_end].float()
        valid_count = valid.sum().clamp_min(1.0)
        for prediction_index in range(int(cfg.trainer.train_iters)):
            if trace_index >= len(trace["coordinates"]):
                raise RuntimeError("captured trajectory trace ended before all model windows")
            prediction = trace["coordinates"][trace_index].to(batch.trajectory_3d.device).float()
            trace_index += 1
            prediction = prediction[..., :p_idx_end, :]
            prediction_z = (prediction[..., 2] - 0.1) / (65.0 - 0.1) * 128.0
            target_z = (target[..., 2] - 0.1) / (65.0 - 0.1) * 128.0
            error = torch.stack((prediction[..., 0] - target[..., 0], prediction[..., 1] - target[..., 1], prediction_z - target_z), dim=-1).abs().mean(dim=-1)
            weight = float(cfg.trainer.gamma) ** (int(cfg.trainer.train_iters) - prediction_index - 1)
            per_track[:p_idx_end] += (error * valid).sum(dim=(0, 1)) * weight / valid_count
        window_count += 1
    if trace_index != len(trace["coordinates"]):
        raise RuntimeError("captured trajectory trace contains unexpected extra windows")
    scale = float(batch.track_upscaling_factor[0].item())
    values = per_track * scale / max(window_count, 1) / int(cfg.trainer.train_iters)
    valid_sorted = batch.valid.index_select(2, sort_indices)
    real = values[valid_sorted[0].sum(dim=0) > 0]
    total = float(real.sum())
    result = {"track_count": int(real.numel()), "total_trajectory_loss": total}
    if total == 0.0:
        return {**result, "worst_1_percent_share": 0.0, "worst_5_percent_share": 0.0, "worst_10_percent_share": 0.0}
    descending = torch.sort(real, descending=True).values
    for percentage in (1, 5, 10):
        count = max(1, math.ceil(len(descending) * percentage / 100.0))
        result[f"worst_{percentage}_percent_share"] = float(descending[:count].sum() / total)
    return result


def _scene_loss_report(output: Mapping[str, Any]) -> dict[str, float]:
    scene_losses = output["scene_losses"]
    flow = float(scene_losses["flow"][0].float().item())
    visibility_raw = float(scene_losses["visibility_raw"][0].float().item())
    visibility = float(scene_losses["visibility"][0].float().item())
    return {
        "trajectory": flow,
        "visibility_raw": visibility_raw,
        "visibility": visibility,
        "total": flow + visibility,
    }


def _load_recipe_step(reader, recipe_step: int) -> tuple[Any, ...]:
    from mvtracker.datasets.training_recipe import RecipeRecord

    if recipe_step < 0:
        raise ValueError("recipe step must be non-negative")
    for payload in reader.steps(start_step=recipe_step):
        if int(payload["step"]) != recipe_step:
            raise ValueError(f"recipe step numbering diverges at {recipe_step}")
        records = tuple(RecipeRecord.from_dict(item) for item in payload["logical_samples"])
        if len(records) != int(reader.manifest["logical_samples_per_step"]):
            raise ValueError(f"recipe step {recipe_step} does not contain the expected logical samples")
        return records
    raise FileNotFoundError(f"recipe has no step {recipe_step}")


def _validate_plan(record, plan) -> None:
    observed = (
        int(plan.seed), str(plan.sequence), tuple(int(value) for value in plan.frame_indices),
        tuple(int(value) for value in plan.views), int(plan.track_count),
        tuple(int(value) for value in plan.selected_global_track_indices), str(plan.depth_source),
    )
    expected = (
        int(record.seed), str(record.scene), tuple(record.frames), tuple(record.views),
        int(record.track_count), tuple(record.tracks), str(record.depth_source),
    )
    if observed != expected:
        raise RuntimeError(f"recipe replay diverged for {record.source} logical sample {record.logical_index}: {observed} != {expected}")


def _counterfactual_batch(batch, plan, dataset, radius: float = 30.0):
    """Remove the source tracks rejected by the current 30m radius rule.

    DIEGESIS applies that rule before scene-transform augmentation.  Re-read
    only its indexed track array so the counterfactual does not accidentally
    classify a translated/ scaled training sample by its augmented radius.
    """
    manifest = dataset._manifest(plan.sequence)
    source_root = Path(dataset.raw_root) / manifest["source_sequence"]
    tracks = np.asarray(
        dataset._mmap(source_root / "tracks_xyz.npy"), dtype=np.float32
    )
    selected = np.asarray(plan.selected_global_track_indices, dtype=np.int64)
    tracks = tracks[np.asarray(plan.frame_indices, dtype=np.int64)][:, selected]
    anchor = np.asarray(manifest.get("world_anchor", (0.0, 0.0, 0.0)), dtype=np.float32)
    radii = np.linalg.norm(tracks - anchor, axis=-1).max(axis=0)
    radii = torch.as_tensor(radii)
    keep = radii <= float(radius)
    if bool(keep.all()):
        return None, int((~keep).sum().item())
    return dataclasses.replace(
        batch,
        trajectory=batch.trajectory[:, :, keep],
        trajectory_3d=batch.trajectory_3d[:, keep],
        visibility=batch.visibility[:, :, keep],
        valid=batch.valid[:, :, keep],
        query_points=batch.query_points[:, keep] if batch.query_points is not None else None,
        query_points_3d=batch.query_points_3d[:, keep],
        track_padding_mask=None,
    ), int((~keep).sum().item())


def _build_config(args, recipe_manifest):
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.merge(
        OmegaConf.load(repo_root / "configs/train.yaml"),
        OmegaConf.load(repo_root / "configs/experiment/diegesis_syn4d_mvkubric_recipe_da3_ddp_1000.yaml"),
    )
    cfg.datasets.root = str(args.mvkubric_root)
    cfg.datasets.train.sources.diegesis.root = str(args.diegesis_root)
    cfg.datasets.train.sources.syn4d.root = str(args.syn4d_root)
    cfg.datasets.train.sources.mvkubric.root = str(args.mvkubric_root)
    cfg.datasets.train.kubric_metadata_index_root = str(args.mvkubric_index_root)
    cfg.datasets.train.mvkubric_storage = "native"
    # The recipe predates the radius filter currently used by DIEGESIS.
    cfg.datasets.diegesis_max_track_radius = float("inf")
    # The recipe was generated with the 65m Syn4D radius.
    cfg.datasets.syn4d_max_track_radius = 65.0
    cfg.datasets.train.sources.diegesis.include_scene_ids = list(recipe_manifest["scene_lists"].get("diegesis", ()))
    cfg.datasets.train.sources.syn4d.include_scene_ids = list(recipe_manifest["scene_lists"].get("syn4d", ()))
    cfg.datasets.train.sources.mvkubric.include_scene_ids = list(recipe_manifest["scene_lists"].get("mvkubric", ()))
    return cfg


def _build_datasets(cfg, args, recipe_manifest):
    from mvtracker.datasets import KubricMultiViewDataset, Syn4DMultiViewDataset, TapVid3DMultiViewDataset
    from mvtracker.datasets.kubric_gpu_dataset import GpuDecodedKubricMultiViewDataset

    fabric = SimpleNamespace(world_size=int(recipe_manifest["world_size"]), global_rank=0)
    datasets = {}
    source_cfg = cfg.datasets.train.sources
    for source in SOURCE_NAMES:
        if source not in recipe_manifest.get("scene_lists", {}):
            continue
        if source == "diegesis":
            kwargs = TapVid3DMultiViewDataset.from_name(
                source_cfg[source].name, str(args.diegesis_root), cfg, fabric,
                just_return_kwargs=True, include_scene_ids=list(recipe_manifest["scene_lists"][source]),
            )
            datasets[source] = TapVid3DMultiViewDataset(**kwargs)
        elif source == "syn4d":
            kwargs = Syn4DMultiViewDataset.from_name(
                source_cfg[source].name, str(args.syn4d_root), cfg, fabric,
                just_return_kwargs=True, include_scene_ids=list(recipe_manifest["scene_lists"][source]),
            )
            datasets[source] = Syn4DMultiViewDataset(**kwargs)
        else:
            kwargs = KubricMultiViewDataset.from_name(
                source_cfg[source].name, str(args.mvkubric_root), cfg, fabric,
                just_return_kwargs=True, include_scene_ids=list(recipe_manifest["scene_lists"][source]),
            )
            kwargs["metadata_index_root"] = str(args.mvkubric_index_root)
            datasets[source] = GpuDecodedKubricMultiViewDataset(**kwargs)
        print(f"AUDIT event=dataset_ready source={source} scenes={len(recipe_manifest['scene_lists'][source])}", flush=True)
    return datasets


def _load_model(cfg, checkpoint: Path, device: torch.device):
    import hydra

    model = hydra.utils.instantiate(cfg.model).to(device)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_payload.get("model") if isinstance(checkpoint_payload, Mapping) else None
    if state is None:
        raise ValueError(f"{checkpoint}: expected a training checkpoint with a model state")
    model.load_state_dict(state, strict=True)
    model.train()
    print(f"AUDIT event=checkpoint_loaded path={checkpoint}", flush=True)
    return model


def _materialize(dataset, record, plan, args, runtime_store):
    if args.depth_mode == "gt":
        plan = dataclasses.replace(plan, depth_source="gt")
        sample, gotit = dataset.materialize_sample(plan)
    else:
        if runtime_store is None:
            raise ValueError("--depth-mode runtime requires --runtime-depth-root")
        runtime_depth = runtime_store.load(int(record.step), int(record.logical_index))
        sample, gotit = dataset.materialize_sample(plan, runtime_depth=runtime_depth)
    if not gotit or sample is None:
        raise RuntimeError(f"materialization failed for {record.source} logical sample {record.logical_index}")
    return sample


def _decode(sample, decoder):
    prepared = SimpleNamespace(samples=(sample,))
    batch, _ = decoder.decode_async(prepared)
    return batch


def _run_forward(model, batch, cfg, recipe_step, diagnostics):
    from mvtracker.cli.train import forward_batch_multi_view

    model.zero_grad(set_to_none=True)
    diagnostics.begin()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = forward_batch_multi_view(
            batch=batch, model=model, cfg=cfg, step=recipe_step,
            train_iters=int(cfg.trainer.train_iters), gamma=float(cfg.trainer.gamma),
            run_expensive_diagnostics=False, capture_training_trace=True,
        )
        loss = output["flow"]["loss"] + output["visibility"]["loss"]
    loss.backward()
    gradient = diagnostics.finish(unscale_factor=1.0)
    if gradient is None:
        raise RuntimeError("no gradients were observed during audit backward")
    return output, gradient


def run(args) -> dict[str, Any]:
    if args.device != "cuda" and not args.device.startswith("cuda:"):
        raise ValueError("the optimizer-step audit requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BF16 recipe replay")
    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "model_000500.pth"
    recipe_path = Path(args.recipe) if args.recipe else run_dir / "recipe"
    output_path = Path(args.output) if args.output else run_dir / "recipe_optimizer_step_audit.json"
    from mvtracker.datasets.estimated_depth import RuntimeRecipeDepthStore
    from mvtracker.datasets.training_recipe import RecipeReader

    reader = RecipeReader(recipe_path)
    cfg = _build_config(args, reader.manifest)
    device = torch.device(args.device)
    model = _load_model(cfg, checkpoint, device)
    datasets = _build_datasets(cfg, args, reader.manifest)
    from mvtracker.datasets.mixed_source_schedule import ScheduledSampleRequest
    from mvtracker.datasets.mixed_physical_loader import PhysicalBatchDecoder
    from mvtracker.cli.train import _MicrobatchGradientDiagnostics
    runtime_store = RuntimeRecipeDepthStore(args.runtime_depth_root) if args.depth_mode == "runtime" else None
    decoder = PhysicalBatchDecoder(device, decode_image_chunk_size=64, dali_num_threads=4, dali_prefetch_queue_depth=2)
    diagnostics = _MicrobatchGradientDiagnostics(model.parameters(), enabled=True, sketch_size=args.sketch_size, sketch_seed=args.sketch_seed)
    report: dict[str, Any] = {
        "schema": "mvtracker-recipe-optimizer-step-audit-v1",
        "optimizer_steps": list(args.optimizer_steps),
        "recipe_steps": [int(step) - 1 for step in args.optimizer_steps],
        "checkpoint": str(checkpoint),
        "recipe": str(recipe_path),
        "depth_mode": args.depth_mode,
        "gradient_sketch_size": int(args.sketch_size),
        "gradient_sketch_seed": int(args.sketch_seed),
        "samples": [],
    }
    try:
        for optimizer_step in args.optimizer_steps:
            recipe_step = int(optimizer_step) - 1
            if recipe_step < 0:
                raise ValueError("optimizer steps are 1-indexed and must be positive")
            records = _load_recipe_step(reader, recipe_step)
            print(f"AUDIT event=step_start optimizer_step={optimizer_step} recipe_step={recipe_step} samples={len(records)}", flush=True)
            step_rows = []
            for position, record in enumerate(records):
                request = record.replay_request(ScheduledSampleRequest)
                plan = datasets[record.source].plan_sample(request)
                if plan is None:
                    raise RuntimeError(f"recipe sample replayed invalid: {record.source} {record.logical_index}")
                _validate_plan(record, plan)
                sample = _materialize(datasets[record.source], record, plan, args, runtime_store)
                batch = _decode(sample, decoder)
                output, gradient = _run_forward(model, batch, cfg, recipe_step, diagnostics)
                row = {
                    "optimizer_step": optimizer_step, "recipe_step": recipe_step,
                    "logical_index": int(record.logical_index), "source": record.source,
                    "scene": record.scene, "virtual_index": int(record.request["virtual_index"]),
                    "seed": int(record.seed), "frames": list(record.frames), "views": list(record.views),
                    "track_count": int(record.track_count), "gradient_norm": float(gradient["norm"]),
                    "gradient_sketch": gradient["sketch"].tolist(),
                    "scene_losses": _scene_loss_report(output),
                    "trajectory_concentration": _concentration(batch, output, cfg),
                }
                counterfactual, removed = (None, 0)
                if record.source == "diegesis":
                    counterfactual, removed = _counterfactual_batch(
                        batch, plan, datasets[record.source]
                    )
                if counterfactual is not None:
                    cf_output, cf_gradient = _run_forward(model, counterfactual, cfg, recipe_step, diagnostics)
                    row["counterfactual_radius_30m"] = {
                        "removed_tracks": removed,
                        "remaining_tracks": int(counterfactual.trajectory_3d.shape[1]),
                        "gradient_norm": float(cf_gradient["norm"]),
                        "scene_losses": _scene_loss_report(cf_output),
                        "trajectory_concentration": _concentration(counterfactual, cf_output, cfg),
                    }
                step_rows.append(row)
                report["samples"].append(row)
                print(f"AUDIT event=sample_complete optimizer_step={optimizer_step} progress={position + 1}/{len(records)} source={record.source} scene={record.scene} gradient_norm={row['gradient_norm']:.6g}", flush=True)
            report.setdefault("steps", []).append({
                "optimizer_step": optimizer_step, "recipe_step": recipe_step,
                "samples": step_rows, "gradient_relations": _gradient_relations(step_rows),
            })
            print(f"AUDIT event=step_complete optimizer_step={optimizer_step} recipe_step={recipe_step}", flush=True)
    finally:
        diagnostics.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.write_text(json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(f"AUDIT event=report_complete path={output_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--diegesis-root", type=Path, required=True)
    parser.add_argument("--syn4d-root", type=Path, required=True)
    parser.add_argument("--mvkubric-root", type=Path, required=True)
    parser.add_argument("--mvkubric-index-root", type=Path, required=True)
    parser.add_argument("--runtime-depth-root", type=Path)
    parser.add_argument("--depth-mode", choices=("gt", "runtime"), default="gt")
    parser.add_argument("--optimizer-steps", type=int, nargs="+", default=DEFAULT_OPTIMIZER_STEPS)
    parser.add_argument("--sketch-size", type=int, default=DEFAULT_SKETCH_SIZE)
    parser.add_argument("--sketch-seed", type=int, default=DEFAULT_SKETCH_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(step < 1 for step in args.optimizer_steps):
        raise ValueError("optimizer steps are 1-indexed and must be positive")
    if args.depth_mode == "runtime" and args.runtime_depth_root is None:
        raise ValueError("--depth-mode runtime requires --runtime-depth-root")
    run(args)


if __name__ == "__main__":
    main()
