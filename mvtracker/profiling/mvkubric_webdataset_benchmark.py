"""T4-only native versus indexed-WIDS MV-Kubric loader measurements."""

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
    from mvtracker.datasets.kubric_dali_dataset import DaliKubricMultiViewDataset
    from mvtracker.datasets.tapvid3d_multiview_dataset import (
        DaliEncodedImageDecoder,
        EncodedTapVid3DBatch,
        decode_tapvid3d_batch,
    )
    from mvtracker.datasets.utils import SampleRequest

    build_started = time.perf_counter()
    dataset = DaliKubricMultiViewDataset(
        data_root=str(webdataset_root),
        webdataset_root=str(webdataset_root),
        webdataset_split="train",
        seq_len=24,
        num_views=view_count,
        traj_per_sample=512,
        max_depth=24,
        seed=72,
        include_scene_ids=list(scene_ids),
        augmentation_probability=0.0,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    dali_decoder = DaliEncodedImageDecoder(device, num_threads=4, prefetch_queue_depth=2)
    rgb_stream = torch.cuda.Stream(device=device)
    depth_stream = torch.cuda.Stream(device=device)
    prepare_stream = torch.cuda.Stream(device=device)
    startup_seconds = time.perf_counter() - build_started

    def consume(virtual_index: int) -> tuple[object, float, float, float, int, int, Mapping[str, object]]:
        plan_started = time.perf_counter()
        request = SampleRequest(
            virtual_index=virtual_index,
            view_count=view_count,
            scene_index=virtual_index % dataset.real_len,
        )
        plan = dataset.plan_sample(request)
        if plan is None:
            raise RuntimeError(f"sample planning rejected scene index {request.scene_index}")
        plan_seconds = time.perf_counter() - plan_started
        materialize_started = time.perf_counter()
        sample, gotit = dataset.materialize_sample(plan)
        if not gotit:
            raise RuntimeError("indexed MV-Kubric materialization returned gotit=False")
        media_seconds = time.perf_counter() - materialize_started
        encoded_bytes = sum(len(value) for value in sample.jpeg_bytes) + sum(len(value) for value in sample.depth_bytes)
        decode_started = time.perf_counter()
        with torch.no_grad():
            decode_tapvid3d_batch(
                EncodedTapVid3DBatch([sample]),
                device,
                dali_decoder=dali_decoder,
                rgb_stream=rgb_stream,
                depth_stream=depth_stream,
                prepare_stream=prepare_stream,
            )
        torch.cuda.synchronize(device)
        return sample, plan_seconds, media_seconds, time.perf_counter() - decode_started, encoded_bytes, len(sample.jpeg_bytes), sample.metadata

    next_virtual_index = 0
    for _ in range(warmup):
        consume(next_virtual_index)
        next_virtual_index += 1
    plan_times: list[float] = []
    media_times: list[float] = []
    decode_times: list[float] = []
    hardware_samples = []
    encoded_bytes = 0
    media_record_counts: list[int] = []
    measured_started = time.perf_counter()
    for _ in range(measured):
        _, plan_seconds, media_seconds, decode_seconds, sample_bytes, _, metadata = consume(next_virtual_index)
        next_virtual_index += 1
        plan_times.append(plan_seconds)
        media_times.append(media_seconds)
        decode_times.append(decode_seconds)
        encoded_bytes += sample_bytes
        media_record_counts.append(int(metadata.get("media_record_count", view_count)))
        if hardware_sampler is not None:
            hardware_samples.append(hardware_sampler())
    wall_seconds = time.perf_counter() - measured_started
    return {
        "path": "dali",
        "view_count": view_count,
        "startup_seconds": startup_seconds,
        "cold_first_sample_seconds": startup_seconds + (plan_times[0] + media_times[0] + decode_times[0] if plan_times else 0.0),
        "metadata_plan_seconds_p50": _percentile(plan_times, 0.50),
        "metadata_plan_seconds_p95": _percentile(plan_times, 0.95),
        "media_read_seconds_p50": _percentile(media_times, 0.50),
        "media_read_seconds_p95": _percentile(media_times, 0.95),
        "read_unpack_seconds_p50": _percentile(media_times, 0.50),
        "read_unpack_seconds_p95": _percentile(media_times, 0.95),
        "gpu_decode_seconds_p50": _percentile(decode_times, 0.50),
        "gpu_decode_seconds_p95": _percentile(decode_times, 0.95),
        "exposed_wait_seconds_p50": _percentile(media_times, 0.50),
        "exposed_wait_seconds_p95": _percentile(media_times, 0.95),
        "samples_per_second": measured / wall_seconds if wall_seconds else 0.0,
        "encoded_bytes_per_second": encoded_bytes / wall_seconds if wall_seconds else 0.0,
        "encoded_bytes": encoded_bytes,
        "media_record_count_p50": _percentile(media_record_counts, 0.50),
        "media_record_count_p95": _percentile(media_record_counts, 0.95),
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
