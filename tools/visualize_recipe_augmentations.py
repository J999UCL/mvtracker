"""Export exact before/after RGB grids for selected training-recipe samples."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from tools.audit_recipe_optimizer_steps import (
    SOURCE_NAMES,
    _build_config,
    _build_datasets,
    _decode,
    _load_recipe_step,
    _validate_plan,
)


TARGETS = {
    515: ("mvkubric", "2922"),
    487: ("mvkubric", "1890"),
    750: ("mvkubric", "1155"),
    675: ("mvkubric", "2662"),
    382: ("syn4d", "hospital__seq_000003"),
}


def _tile(video: torch.Tensor, title: str, frames: int = 6) -> Image.Image:
    video = video.detach().cpu().clamp(0, 255).byte()
    view_count, frame_count = video.shape[:2]
    selected = np.linspace(0, frame_count - 1, min(frames, frame_count), dtype=int)
    thumb_w = 256
    first = Image.fromarray(video[0, 0].permute(1, 2, 0).numpy())
    thumb_h = round(first.height * thumb_w / first.width)
    header = 42
    canvas = Image.new(
        "RGB",
        (len(selected) * thumb_w, header + view_count * thumb_h),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill="white")
    for row in range(view_count):
        for column, frame in enumerate(selected):
            image = Image.fromarray(video[row, frame].permute(1, 2, 0).numpy())
            image = image.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
            canvas.paste(image, (column * thumb_w, header + row * thumb_h))
            draw.text(
                (column * thumb_w + 5, header + row * thumb_h + 4),
                f"v{row} f{frame}",
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
    return canvas


def _save_comparison(raw: torch.Tensor, augmented: torch.Tensor, path: Path, title: str):
    top = _tile(raw, f"RAW (same crop) | {title}")
    bottom = _tile(augmented, f"AUGMENTED | {title}")
    canvas = Image.new("RGB", (max(top.width, bottom.width), top.height + bottom.height), "black")
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, top.height))
    canvas.save(path, quality=92)


def run(args) -> dict:
    from mvtracker.datasets.mixed_source_schedule import ScheduledSampleRequest
    from mvtracker.datasets.mixed_physical_loader import PhysicalBatchDecoder
    from mvtracker.datasets.training_recipe import RecipeReader

    reader = RecipeReader(args.recipe)
    records_by_step = {
        step: _load_recipe_step(reader, step - 1) for step in TARGETS
    }
    selected_scenes = {
        source: tuple(
            sorted(
                {
                    record.scene
                    for records in records_by_step.values()
                    for record in records
                    if record.source == source
                }
            )
        )
        for source in SOURCE_NAMES
    }
    cfg = _build_config(args, reader.manifest, selected_scenes)
    datasets = _build_datasets(cfg, args, reader.manifest, selected_scenes)
    decoder = PhysicalBatchDecoder(
        torch.device("cuda"),
        decode_image_chunk_size=64,
        dali_num_threads=4,
        dali_prefetch_queue_depth=2,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for position, (step, identity) in enumerate(TARGETS.items(), start=1):
        source, scene = identity
        record = next(
            record
            for record in records_by_step[step]
            if record.source == source and record.scene == scene
        )
        request = record.replay_request(ScheduledSampleRequest)
        request = dataclasses.replace(
            request,
            scene_index=datasets[source].seq_names.index(record.scene),
        )
        plan = datasets[source].plan_sample(request)
        if plan is None:
            raise RuntimeError(f"sample became invalid: step={step} source={source} scene={scene}")
        _validate_plan(record, plan)
        # RGB augmentation is independent of depth. Use GT depth here so this
        # visualization does not need to run a depth-estimation model.
        rgb_plan = dataclasses.replace(plan, depth_source="gt")
        sample, gotit = datasets[source].materialize_sample(rgb_plan)
        if not gotit or sample is None:
            raise RuntimeError(f"materialization failed: step={step} source={source} scene={scene}")
        raw_sample = dataclasses.replace(
            sample,
            apply_rgb_aug=False,
            apply_depth_aug=False,
        )
        raw = _decode(raw_sample, decoder, source, rgb_plan)
        augmented = _decode(sample, decoder, source, rgb_plan)
        name = f"step-{step:04d}_{source}_{scene}.jpg"
        title = (
            f"step {step} | {source}/{scene} | views {list(record.views)} | "
            f"tracks {record.track_count} | planned depth {record.depth_source}"
        )
        _save_comparison(raw.video[0], augmented.video[0], output / name, title)
        rows.append(
            {
                "step": step,
                "source": source,
                "scene": scene,
                "path": name,
                "augmentation": record.augmentation,
                "planned_depth_source": record.depth_source,
            }
        )
        print(
            f"AUGMENTATION_VIS progress={position}/{len(TARGETS)} "
            f"step={step} source={source} scene={scene} output={name}",
            flush=True,
        )
    return {"output": str(output), "samples": rows}

