"""T4-only native versus DALI MV-Kubric loader measurements."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Callable, Mapping, Sequence


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))]


def _native_encoded_bytes(data_root: Path, scene_ids: Sequence[str], view_count: int) -> float:
    """Estimate selected-view encoded input bytes from native file sizes.

    The native profile chooses views inside worker processes, so its exact
    selected paths are not exposed by the existing profiling helper.  The
    uniform view sampler makes the ``view_count / source_view_count`` estimate unbiased.
    """
    totals = []
    for scene_id in scene_ids:
        scene = data_root / "datasets/kubric-multiview/train" / str(scene_id)
        view_bytes = []
        for view in sorted(scene.glob("view_*")):
            view_bytes.append(
                sum(path.stat().st_size for pattern in ("rgba_*", "depth_*") for path in view.glob(pattern))
            )
        if len(view_bytes) < view_count:
            raise ValueError(f"{scene}: only {len(view_bytes)} native views for request {view_count}")
        totals.append(sum(view_bytes) * float(view_count) / len(view_bytes))
    return sum(totals) / len(totals) if totals else 0.0


def benchmark_native_case(
    data_root: Path,
    scene_ids: Sequence[str],
    *,
    view_count: int,
    warmup: int,
    measured: int,
    workers: int,
    hardware_sampler: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    from mvtracker.profiling.modal_continual_data import profile_encoded_loader

    started = time.perf_counter()
    profile = profile_encoded_loader(
        data_root,
        source="mvkubric",
        warmup=warmup,
        measured=measured,
        workers=workers,
        use_cuda=True,
        mvkubric_scene_ids=tuple(scene_ids),
        view_count=view_count,
        hardware_sampler=hardware_sampler,
        mvkubric_storage="native",
    )
    estimated_bytes = _native_encoded_bytes(data_root, scene_ids, view_count) * measured
    wall_seconds = float(profile["wall_elapsed_seconds"])
    return {
        "path": "native",
        "view_count": view_count,
        "startup_seconds": float(profile["first_sample_seconds"]),
        "cold_first_sample_seconds": float(profile["first_sample_seconds"]),
        "read_unpack_seconds_p50": float(profile["worker_prepare_seconds_p50"]),
        "read_unpack_seconds_p95": float(profile["worker_prepare_seconds_p95"]),
        "exposed_wait_seconds_p50": float(profile["exposed_wait_seconds_p50"]),
        "exposed_wait_seconds_p95": float(profile["exposed_wait_seconds_p95"]),
        "samples_per_second": float(profile["samples_per_second"]),
        "encoded_bytes_per_second": estimated_bytes / wall_seconds if wall_seconds else 0.0,
        "estimated_encoded_bytes": estimated_bytes,
        "wall_elapsed_seconds": wall_seconds,
        "hardware_samples": profile.get("hardware_samples", []),
        "source_profile": profile,
        "benchmark_seconds": time.perf_counter() - started,
    }


def _encoded_sample(record, view_count: int):
    import torch
    from mvtracker.datasets.tapvid3d_multiview_dataset import EncodedTapVid3DSample

    views = tuple(range(view_count))
    frames = record.frame_count
    jpeg = tuple(record.rgb_frames[view][frame] for view in views for frame in range(frames))
    depth = tuple(record.depth_frames[view][frame] for view in views for frame in range(frames))
    theta = torch.zeros((view_count, frames, 2, 3), device="cpu")
    theta[..., 0, 0] = 1
    theta[..., 1, 1] = 1
    tracks = torch.zeros((frames, 1, 3), dtype=torch.float32)
    visibility = torch.ones((view_count, frames, 1), dtype=torch.bool)
    return EncodedTapVid3DSample(
        jpeg_bytes=jpeg,
        depth=None,
        theta=theta,
        intrs=torch.from_numpy(record.intrinsics[list(views), None].repeat(frames, axis=1)),
        extrs=torch.from_numpy(record.extrinsics[list(views)]),
        trajectory=torch.zeros((view_count, frames, 1, 3), dtype=torch.float32),
        trajectory_3d=tracks,
        visibility=visibility,
        valid=torch.ones((frames, 1), dtype=torch.float32),
        query_points_3d=torch.zeros((1, 3), dtype=torch.float32),
        seq_name=record.name,
        metadata={"gotit": True},
        output_size=record.resolution_hw,
        apply_rgb_aug=False,
        rgb_augmentation=None,
        apply_depth_aug=False,
        augmentation_seed=0,
        depth_scale=1.0,
        track_upscaling_factor=1.0,
        max_depth=100.0,
        depth_patch_operations=(),
        image_codec="nvimagecodec",
        depth_bytes=depth,
        depth_sensor_widths=tuple(float(value) for value in record.sensor_widths[list(views)]),
        depth_focal_lengths=tuple(float(value) for value in record.focal_lengths[list(views)]),
    )


def benchmark_dali_case(
    webdataset_root: Path,
    scene_ids: Sequence[str],
    *,
    view_count: int,
    warmup: int,
    measured: int,
    hardware_sampler: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    import torch
    from nvidia import nvimgcodec
    from mvtracker.datasets.kubric_dali_dataset import DaliKubricSceneStream
    from mvtracker.datasets.tapvid3d_multiview_dataset import EncodedTapVid3DBatch, decode_tapvid3d_batch

    build_started = time.perf_counter()
    stream = DaliKubricSceneStream(
        webdataset_root,
        split="train",
        num_threads=4,
        prefetch_queue_depth=2,
        initial_fill=32,
        random_shuffle=False,
    )
    startup_seconds = time.perf_counter() - build_started
    device = torch.device("cuda", torch.cuda.current_device())
    rgb_decoder = nvimgcodec.Decoder(device_id=device.index or 0)
    depth_decoder = nvimgcodec.Decoder(device_id=device.index or 0)
    rgb_stream = torch.cuda.Stream(device=device)
    depth_stream = torch.cuda.Stream(device=device)
    prepare_stream = torch.cuda.Stream(device=device)

    def consume() -> tuple[object, float, float, int, int, list[Mapping[str, object]]]:
        read_started = time.perf_counter()
        record = next(stream)
        read_seconds = time.perf_counter() - read_started
        sample = _encoded_sample(record, view_count)
        encoded_bytes = sum(len(value) for value in sample.jpeg_bytes) + sum(len(value) for value in sample.depth_bytes)
        decode_started = time.perf_counter()
        with torch.no_grad():
            decode_tapvid3d_batch(
                EncodedTapVid3DBatch([sample]),
                device,
                nvimagecodec_rgb_decoder=rgb_decoder,
                nvimagecodec_depth_decoder=depth_decoder,
                rgb_stream=rgb_stream,
                depth_stream=depth_stream,
                prepare_stream=prepare_stream,
            )
        torch.cuda.synchronize(device)
        return record, read_seconds, time.perf_counter() - decode_started, encoded_bytes, len(sample.jpeg_bytes), []

    for _ in range(warmup):
        consume()
    read_times: list[float] = []
    decode_times: list[float] = []
    hardware_samples = []
    encoded_bytes = 0
    measured_started = time.perf_counter()
    for _ in range(measured):
        _, read_seconds, decode_seconds, sample_bytes, _, _ = consume()
        read_times.append(read_seconds)
        decode_times.append(decode_seconds)
        encoded_bytes += sample_bytes
        if hardware_sampler is not None:
            hardware_samples.append(hardware_sampler())
    wall_seconds = time.perf_counter() - measured_started
    return {
        "path": "dali",
        "view_count": view_count,
        "startup_seconds": startup_seconds,
        "cold_first_sample_seconds": startup_seconds + (read_times[0] if read_times else 0.0),
        "read_unpack_seconds_p50": _percentile(read_times, 0.50),
        "read_unpack_seconds_p95": _percentile(read_times, 0.95),
        "gpu_decode_seconds_p50": _percentile(decode_times, 0.50),
        "gpu_decode_seconds_p95": _percentile(decode_times, 0.95),
        "exposed_wait_seconds_p50": _percentile(read_times, 0.50),
        "exposed_wait_seconds_p95": _percentile(read_times, 0.95),
        "samples_per_second": measured / wall_seconds if wall_seconds else 0.0,
        "encoded_bytes_per_second": encoded_bytes / wall_seconds if wall_seconds else 0.0,
        "encoded_bytes": encoded_bytes,
        "wall_elapsed_seconds": wall_seconds,
        "hardware_samples": hardware_samples,
        "warmup": warmup,
        "measured": measured,
    }


def benchmark_matrix(
    data_root: Path,
    webdataset_root: Path,
    scene_ids: Sequence[str],
    *,
    warmup: int = 4,
    measured: int = 16,
    workers: int = 8,
    hardware_sampler: Callable[[], Mapping[str, object]] | None = None,
    progress_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    results = {}
    for view_count in (1, 2, 4, 6):
        native = benchmark_native_case(
            data_root, scene_ids, view_count=view_count, warmup=warmup,
            measured=measured, workers=workers, hardware_sampler=hardware_sampler,
        )
        dali = benchmark_dali_case(
            webdataset_root, scene_ids, view_count=view_count, warmup=warmup,
            measured=measured, hardware_sampler=hardware_sampler,
        )
        results[f"views{view_count}"] = {"native": native, "dali": dali}
        if progress_callback is not None:
            progress_callback(f"views{view_count}", results[f"views{view_count}"])
    return {
        "scene_ids": list(scene_ids),
        "view_counts": [1, 2, 4, 6],
        "warmup": warmup,
        "measured": measured,
        "workers": workers,
        "cases": results,
    }
