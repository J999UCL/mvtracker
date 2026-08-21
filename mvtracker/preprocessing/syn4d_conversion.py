"""Convert one Syn4D sequence into the DIEGESIS-style training cache."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from fractions import Fraction
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mvtracker.preprocessing.syn4d import (
    CACHE_HEIGHT,
    CACHE_WIDTH,
    TRACK_COUNT,
    VIEW_COUNT,
    camera_from_syn4d_row,
    compute_depth_visibility,
    create_sequence_cache,
    depth_centimetres_to_metres,
    finalize_sequence_cache,
    motion_path_length,
    resize_depth_validity_weighted,
    resize_intrinsics,
)


ProgressCallback = Callable[[dict[str, object]], None]
TRACK_SOURCE_HEIGHT = 494
TRACK_SOURCE_WIDTH = 878
JPEG_QUALITY = 95


def _emit(progress: ProgressCallback | None, **event: object) -> None:
    if progress is not None:
        progress(dict(event))


def _camera_rows(scene_root: Path, sequence_base: str, view: int) -> list[dict[str, str]]:
    path = (
        scene_root
        / "ground_truth"
        / "meta_exr_csv"
        / f"{sequence_base}_{view}_camera.csv"
    )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if str(row.get("name", "")).endswith(".png")
        ]
    if not rows:
        raise ValueError(f"camera CSV contains no PNG frame rows: {path}")
    required = {
        "name",
        "focal_length",
        "sensor_width",
        "sensor_height",
        "yaw",
        "pitch",
        "roll",
        "x",
        "y",
        "z",
    }
    if not required.issubset(rows[0]):
        raise ValueError(f"camera CSV lacks required columns: {path}")
    return rows


def _frame_number(name: str) -> int:
    match = re.search(r"(\d+)$", Path(name).stem)
    if match is None:
        raise ValueError(f"cannot parse Syn4D frame number from {name!r}")
    return int(match.group(1))


def _sequence_rows(
    scene_root: Path, sequence_base: str
) -> tuple[tuple[dict[str, str], ...], ...]:
    rows = tuple(
        tuple(_camera_rows(scene_root, sequence_base, view))
        for view in range(VIEW_COUNT)
    )
    frame_numbers = tuple(_frame_number(row["name"]) for row in rows[0])
    for view, view_rows in enumerate(rows):
        current = tuple(_frame_number(row["name"]) for row in view_rows)
        if current != frame_numbers:
            raise ValueError(f"camera {view} frame numbering differs for {sequence_base}")
    return rows


def _probe_video(path: Path) -> dict[str, float | int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"ffprobe found {len(streams)} video streams in {path}")
    stream = streams[0]
    frame_count = int(stream["nb_frames"])
    fps = float(Fraction(stream["avg_frame_rate"]))
    if frame_count <= 0 or fps <= 0.0:
        raise ValueError(f"invalid CFR video metadata for {path}")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": frame_count,
        "fps": fps,
    }


def _sequence_video_metadata(
    scene_root: Path, sequence_base: str, *, frame_count: int
) -> tuple[tuple[Path, ...], dict[str, float | int]]:
    videos = tuple(
        scene_root / "mp4" / f"{sequence_base}_{view}.mp4"
        for view in range(VIEW_COUNT)
    )
    probes = []
    for path in videos:
        if not path.is_file():
            raise FileNotFoundError(path)
        probes.append(_probe_video(path))
    first = probes[0]
    for view, probe in enumerate(probes):
        if probe != first:
            raise ValueError(f"MP4 metadata differs for {sequence_base} view {view}")
    if int(first["frames"]) != frame_count:
        raise ValueError(
            f"camera CSV has {frame_count} frames but MP4 has {first['frames']}"
        )
    return videos, first


def _import_official_syn4d(official_visualizer_root: Path) -> Any:
    root = Path(official_visualizer_root).resolve()
    source = root / "syn4d_track.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    root_text = str(root)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    for name in ("syn4d_track", "utils", "viser_visualizer_track"):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    module = importlib.import_module("syn4d_track")
    if Path(module.__file__).resolve() != source:
        raise ImportError(f"official syn4d_track imported from {module.__file__}, expected {source}")
    if not hasattr(module, "Syn4D_Track") or not hasattr(module, "exr_to_array"):
        raise ImportError("official Syn4D module lacks Syn4D_Track or exr_to_array")
    return module


def _camera_sequence_name(annotation_path: str | Path) -> str:
    return Path(annotation_path).name.removesuffix("_camera.csv")


def _official_sequence_items(
    official_module: Any,
    *,
    scene_root: Path,
    metadata_root: Path,
    sequence_base: str,
    frame_count: int,
) -> list[Mapping[str, Any]]:
    """Load one dense query-camera trajectory using the pinned reader."""

    dataset = official_module.Syn4D_Track(
        dataset_root=str(scene_root.parent),
        metadata_root=str(metadata_root),
        fallback_metadata_root=None,
        scene_name_list=[scene_root.name],
        track_query_idx=0,
        use_augs=False,
        S=frame_count,
        N=TRACK_COUNT,
        strides=[1],
        min_interval=1,
        max_interval=1,
        rgb_source="mp4",
        tracking_format="safetensor",
        resolution=((TRACK_SOURCE_WIDTH, TRACK_SOURCE_HEIGHT), frame_count),
        seed=int(sequence_base.removeprefix("seq_")),
        allow_repeat=False,
    )
    wanted = f"{sequence_base}_0"
    matching = [
        index
        for index, path in enumerate(dataset.annotation_paths)
        if _camera_sequence_name(path) == wanted
    ]
    if len(matching) != 1:
        raise ValueError(f"official reader found {len(matching)} entries for {wanted}")
    requested_index = matching[0]
    items = dataset[requested_index]
    if len(items) != frame_count:
        raise ValueError(
            f"official reader returned {len(items)} frames for {wanted}; expected {frame_count}"
        )
    returned_index = int(np.asarray(items[0]["idx"]).reshape(-1)[0])
    if returned_index != requested_index:
        raise RuntimeError(
            f"official reader skipped {wanted}: requested index {requested_index}, got {returned_index}"
        )
    return list(items)


def _quarter_query_pixels(
    query_valid: np.ndarray,
    query_track: np.ndarray,
    *,
    seed: int,
    cap: int = TRACK_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    valid = np.asarray(query_valid, dtype=bool)
    track = np.asarray(query_track, dtype=np.float32)
    if track.shape != (*valid.shape, 3):
        raise ValueError("query track and validity mask dimensions disagree")
    candidates = np.flatnonzero(valid.reshape(-1) & np.isfinite(track).all(axis=-1).reshape(-1))
    quarter_count = candidates.size // 4
    if quarter_count < cap:
        raise ValueError(
            f"25% of query-valid pixels yields {quarter_count} tracks; need {cap}"
        )
    retained = np.random.default_rng(seed).choice(
        candidates, size=quarter_count, replace=False
    )
    selected = retained[:cap]
    ys, xs = np.divmod(selected, valid.shape[1])
    return selected, xs.astype(np.int32), ys.astype(np.int32), quarter_count


def _explicit_world_tracks(
    items: Sequence[Mapping[str, Any]], *, sequence_base: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    first_track = np.asarray(items[0]["track"], dtype=np.float32)
    first_valid = np.asarray(items[0]["track_valid_mask"], dtype=bool)
    if first_track.shape != (TRACK_SOURCE_HEIGHT, TRACK_SOURCE_WIDTH, 3):
        raise ValueError(
            f"official query track has shape {first_track.shape}; "
            f"expected {(TRACK_SOURCE_HEIGHT, TRACK_SOURCE_WIDTH, 3)}"
        )
    if first_valid.shape != (TRACK_SOURCE_HEIGHT, TRACK_SOURCE_WIDTH):
        raise ValueError("official query validity mask has wrong resolution")
    selected, xs, ys, quarter_count = _quarter_query_pixels(
        first_valid,
        first_track,
        seed=int(sequence_base.removeprefix("seq_")),
    )
    tracks = np.zeros((len(items), TRACK_COUNT, 3), dtype=np.float32)
    valid = np.zeros((len(items), TRACK_COUNT), dtype=bool)
    for frame, item in enumerate(items):
        camera_track = np.asarray(item["track"], dtype=np.float32)
        frame_valid = np.asarray(item["track_valid_mask"], dtype=bool)
        if camera_track.shape != first_track.shape or frame_valid.shape != first_valid.shape:
            raise ValueError(f"official dense-track shape changed at frame {frame}")
        # BaseStereoDynamicViewDataset.__getitem__ converts the internal
        # camera-frame track to world coordinates before returning it.  The
        # public reader output must not be transformed a second time.
        selected_world = camera_track.reshape(-1, 3)[selected]
        selected_valid = frame_valid.reshape(-1)[selected] & np.isfinite(selected_world).all(axis=-1)
        selected_valid &= np.isfinite(selected_world).all(axis=-1)
        tracks[frame] = selected_world
        tracks[frame, ~selected_valid] = 0.0
        valid[frame] = selected_valid
    queries = np.stack(
        [
            xs.astype(np.float32),
            ys.astype(np.float32),
            np.zeros(TRACK_COUNT, dtype=np.float32),
            np.zeros(TRACK_COUNT, dtype=np.float32),
        ],
        axis=-1,
    )
    return tracks, valid, queries, quarter_count


def _decode_rgb_video_dali(
    path: Path,
    *,
    frame_count: int,
    device: str,
) -> np.ndarray:
    """Decode one complete CFR MP4 with DALI/NVDEC and fused resize."""

    import torch
    from nvidia.dali import fn, pipeline_def, types

    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise ValueError("DALI video conversion requires a CUDA device")
    device_id = 0 if cuda_device.index is None else cuda_device.index

    @pipeline_def(batch_size=1, num_threads=4, device_id=device_id)
    def video_pipeline():
        frames, _labels = fn.readers.video_resize(
            device="gpu",
            filenames=[str(path)],
            labels=[0],
            sequence_length=frame_count,
            step=frame_count,
            stride=1,
            random_shuffle=False,
            pad_sequences=False,
            resize_x=CACHE_WIDTH,
            resize_y=CACHE_HEIGHT,
            dtype=types.UINT8,
            image_type=types.RGB,
            name="syn4d_video",
        )
        return frames

    pipeline = video_pipeline()
    pipeline.build()
    output = pipeline.run()[0].as_cpu().as_array()
    if output.shape != (1, frame_count, CACHE_HEIGHT, CACHE_WIDTH, 3):
        raise ValueError(f"DALI returned unexpected video shape {output.shape} for {path}")
    return np.ascontiguousarray(output[0], dtype=np.uint8)


def _write_jpeg_store(frames_rgb: np.ndarray, destination: Path, *, workers: int) -> None:
    """Encode RGB frames concurrently and write the DIEGESIS byte-range layout."""

    import cv2

    frames = np.asarray(frames_rgb, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("RGB frames must have shape [F,H,W,3]")

    def encode(frame: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(frame[..., ::-1]),
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return encoded.tobytes()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        encoded_frames = executor.map(encode, frames)
        offsets = np.zeros(len(frames) + 1, dtype=np.int64)
        with (destination / "jpeg_bytes.bin").open("wb") as handle:
            for index, encoded in enumerate(encoded_frames):
                handle.write(encoded)
                offsets[index + 1] = offsets[index] + len(encoded)
    np.save(destination / "jpeg_offsets.npy", offsets)


def _depth_paths(
    scene_root: Path,
    sequence_base: str,
    view: int,
    rows: Sequence[Mapping[str, str]],
) -> tuple[Path, ...]:
    root = scene_root / "exr_layers" / "depth" / f"{sequence_base}_{view}"
    paths = tuple(root / f"{Path(row['name']).stem}_depth.exr" for row in rows)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def _read_view_depths_cm(
    paths: Sequence[Path],
    *,
    official_module: Any,
    workers: int,
) -> np.ndarray:
    """Read independent EXRs concurrently through the official OpenEXR decoder."""

    import OpenEXR  # noqa: F401 - explicit runtime requirement

    decoder = official_module.exr_to_array

    def read(path: Path) -> np.ndarray:
        return np.asarray(decoder(path), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        frames = list(executor.map(read, paths))
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"EXR depth shapes differ: {sorted(shapes)}")
    return np.stack(frames, axis=0)


def _write_memmap(array: np.memmap, value: np.ndarray) -> None:
    array[:] = value
    array.flush()
    del array


def _convert_sequence(
    *,
    official_module: Any,
    scene_root: Path,
    metadata_root: Path,
    output_root: Path,
    sequence_base: str,
    device: str,
    progress: ProgressCallback | None,
) -> Path:
    _emit(progress, stage="sequence_started", sequence=sequence_base)
    scene = scene_root.name
    destination = output_root / f"{scene}__{sequence_base}"
    if (destination / "manifest.json").is_file():
        _emit(progress, stage="sequence_reused", sequence=sequence_base)
        return destination
    if destination.exists():
        raise FileExistsError(
            f"incomplete Syn4D cache exists without a manifest: {destination}"
        )
    rows_by_view = _sequence_rows(scene_root, sequence_base)
    frame_count = len(rows_by_view[0])
    videos, video_metadata = _sequence_video_metadata(
        scene_root, sequence_base, frame_count=frame_count
    )
    source_width = int(video_metadata["width"])
    source_height = int(video_metadata["height"])

    items = _official_sequence_items(
        official_module,
        scene_root=scene_root,
        metadata_root=metadata_root,
        sequence_base=sequence_base,
        frame_count=frame_count,
    )
    tracks, track_valid, queries, quarter_count = _explicit_world_tracks(
        items, sequence_base=sequence_base
    )
    queries[:, 0] *= np.float32(CACHE_WIDTH / TRACK_SOURCE_WIDTH)
    queries[:, 1] *= np.float32(CACHE_HEIGHT / TRACK_SOURCE_HEIGHT)
    del items
    path_length = motion_path_length(tracks, track_valid)
    _emit(
        progress,
        stage="tracks_ready",
        sequence=sequence_base,
        frames=frame_count,
        query_valid_quarter=quarter_count,
        tracks=TRACK_COUNT,
    )

    writer = create_sequence_cache(
        destination,
        scene=scene,
        sequence_base=sequence_base,
        frame_count=frame_count,
    )
    _write_memmap(writer.array("tracks_xyz"), tracks)
    _write_memmap(writer.array("track_valid"), track_valid)
    _write_memmap(writer.array("queries_xytv"), queries)
    _write_memmap(writer.array("motion_path_length"), path_length)

    cpu_stage_workers = max(1, min(3, (os.cpu_count() or 1) // 2))

    def read_depth_view(view: int) -> np.ndarray:
        return _read_view_depths_cm(
            _depth_paths(scene_root, sequence_base, view, rows_by_view[view]),
            official_module=official_module,
            workers=cpu_stage_workers,
        )

    # The T4 stages remain serial. CPU work rolls around them: one future holds
    # the next native-depth view and one writes the preceding view's JPEGs.
    with (
        ThreadPoolExecutor(max_workers=1) as depth_prefetch,
        ThreadPoolExecutor(max_workers=1) as jpeg_writer,
    ):
        depth_future = depth_prefetch.submit(read_depth_view, 0)
        jpeg_future = None
        jpeg_view = None
        for view in range(VIEW_COUNT):
            rows = rows_by_view[view]
            depths_cm = depth_future.result()
            if view + 1 < VIEW_COUNT:
                depth_future = depth_prefetch.submit(read_depth_view, view + 1)
            if depths_cm.shape != (frame_count, source_height, source_width):
                raise ValueError(
                    f"view {view} EXR shape {depths_cm.shape} disagrees with MP4 "
                    f"{(frame_count, source_height, source_width)}"
                )
            native_depths_m = depth_centimetres_to_metres(depths_cm)
            del depths_cm
            native_intrinsics = np.empty((frame_count, 3, 3), dtype=np.float32)
            extrinsics = np.empty((frame_count, 4, 4), dtype=np.float32)
            for frame, row in enumerate(rows):
                native_intrinsics[frame], extrinsics[frame] = camera_from_syn4d_row(
                    row, source_width=source_width, source_height=source_height
                )
            visibility = compute_depth_visibility(
                tracks,
                track_valid,
                native_depths_m[None],
                native_intrinsics[None],
                extrinsics[None],
                device=device,
            )[0]
            _write_memmap(writer.view_array(view, "visibility"), visibility)
            _write_memmap(
                writer.view_array(view, "intrinsics"),
                resize_intrinsics(
                    native_intrinsics,
                    source_width=source_width,
                    source_height=source_height,
                ),
            )
            _write_memmap(writer.view_array(view, "extrinsics_w2c"), extrinsics)

            depth_output = writer.view_array(view, "depth")
            for start in range(0, frame_count, 16):
                stop = min(start + 16, frame_count)
                depth_output[start:stop] = resize_depth_validity_weighted(
                    native_depths_m[start:stop], device=device
                )
            depth_output.flush()
            del depth_output, native_depths_m, visibility

            # Do not decode another raw RGB view until the previous one has
            # been encoded and released.
            if jpeg_future is not None:
                jpeg_future.result()
                _emit(
                    progress,
                    stage="view_ready",
                    sequence=sequence_base,
                    view=jpeg_view,
                    frames=frame_count,
                )
            rgb = _decode_rgb_video_dali(
                videos[view], frame_count=frame_count, device=device
            )
            jpeg_future = jpeg_writer.submit(
                _write_jpeg_store,
                rgb,
                destination / f"view_{view}",
                workers=cpu_stage_workers,
            )
            jpeg_view = view
        if jpeg_future is not None:
            jpeg_future.result()
            _emit(
                progress,
                stage="view_ready",
                sequence=sequence_base,
                view=jpeg_view,
                frames=frame_count,
            )

    manifest = finalize_sequence_cache(
        writer,
        fps=float(video_metadata["fps"]),
        source_width=source_width,
        source_height=source_height,
    )
    _emit(progress, stage="sequence_complete", sequence=sequence_base, path=str(manifest))
    return destination


def convert_syn4d_sequence(
    scene_root: Path,
    metadata_root: Path,
    output_root: Path,
    *,
    official_visualizer_root: Path,
    sequence: str,
    device: str = "cuda",
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Convert exactly one sequence from the scene at ``scene_root``.

    A sequence is complete only when ``manifest.json`` exists.  The function
    has no decoder, metadata, or storage fallbacks.
    """

    scene_root = Path(scene_root).resolve()
    metadata_root = Path(metadata_root).resolve()
    output_root = Path(output_root).resolve()
    if re.fullmatch(r"seq_\d{6}", sequence) is None:
        raise ValueError(f"invalid Syn4D sequence {sequence!r}")
    output_root.mkdir(parents=True, exist_ok=True)
    official_module = _import_official_syn4d(official_visualizer_root)
    destination = _convert_sequence(
        official_module=official_module,
        scene_root=scene_root,
        metadata_root=metadata_root,
        output_root=output_root,
        sequence_base=sequence,
        device=device,
        progress=progress,
    )
    return {
        "scene": scene_root.name,
        "sequence": sequence,
        "output_path": str(destination),
    }


__all__ = ["convert_syn4d_sequence"]
