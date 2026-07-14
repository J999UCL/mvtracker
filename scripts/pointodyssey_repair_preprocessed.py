#!/usr/bin/env python3
"""Repair a schema-v4 PointOdyssey tree without re-encoding RGB or depth.

The command publishes a new schema-v5 tree.  Immutable prepared assets are
hard-linked from the v4 tree; visibility and metadata are regenerated from the
original source visibility and the already prepared geometry/depth.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__:
    from . import pointodyssey_preprocessing as preprocessing
else:
    import pointodyssey_preprocessing as preprocessing


SOURCE_SCHEMA_VERSION = 4
OUTPUT_SCHEMA_VERSION = preprocessing.SCHEMA_VERSION
REPORT_FORMAT = "pointodyssey_mvtracker_preprocessed_validation"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _require_array(
    path: Path,
    shape: tuple[int, ...],
    dtype: np.dtype,
    *,
    mmap_mode: str | None = "r",
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required array is missing: {path}")
    try:
        value = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Could not read array: {path}") from exc
    if value.shape != shape or value.dtype != np.dtype(dtype):
        raise ValueError(
            f"{path} has shape/dtype {(value.shape, value.dtype)}, expected "
            f"{(shape, np.dtype(dtype))}"
        )
    return value


def _hard_link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required immutable prepared file is missing: {source}")
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as exc:
        raise OSError(
            f"Hard-linking is required and failed for {source} -> {destination}"
        ) from exc
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (source_stat.st_dev, source_stat.st_ino) != (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        raise AssertionError(f"Created file is not a hard link of {source}: {destination}")


def _require_same_filesystem(prepared_root: Path, output_parent: Path) -> None:
    if prepared_root.stat().st_dev != output_parent.stat().st_dev:
        raise OSError(
            "Prepared root and output parent must be on the same filesystem for hard links"
        )


def _scene_path(root: Path, spec: preprocessing.SceneSpec) -> Path:
    return root / spec.split / spec.scene_id


def _source_visibility_path(
    source_root: Path,
    spec: preprocessing.SceneSpec,
    view: int,
) -> Path:
    return (
        source_root
        / preprocessing.SOURCE_SUBROOTS[spec.layout]
        / spec.source_sequence
        / str(view)
        / "visibility.npy"
    )


def _require_exact_scenes(root: Path, specs: Sequence[preprocessing.SceneSpec]) -> None:
    expected = {(spec.split, spec.scene_id) for spec in specs}
    actual: set[tuple[str, str]] = set()
    for split in ("train", "validation", "test"):
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Required prepared split is missing: {split_root}")
        actual.update(
            (split, path.name)
            for path in split_root.iterdir()
            if path.is_dir()
        )
    if actual != expected:
        raise ValueError("Prepared scene set does not match the fixed PointOdyssey split")


def _validate_v4_root(
    prepared_root: Path,
    specs: Sequence[preprocessing.SceneSpec],
) -> None:
    report = _read_json_object(prepared_root / "validation_report.json")
    if report.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("Targeted repair requires a schema-v4 prepared root")
    if report.get("format") != REPORT_FORMAT:
        raise ValueError("Prepared validation report format is not PointOdyssey")
    if report.get("status") != "completed" or report.get("failures") != []:
        raise ValueError("Targeted repair requires a clean completed v4 report")
    _require_exact_scenes(prepared_root, specs)
    for spec in specs:
        metadata_path = _scene_path(prepared_root, spec) / "scene.json"
        metadata = _read_json_object(metadata_path)
        if metadata.get("schema_version") != SOURCE_SCHEMA_VERSION:
            raise ValueError(f"Targeted repair requires schema v4: {metadata_path}")
        if metadata.get("format") != "pointodyssey_mvtracker_preprocessed":
            raise ValueError(f"Unexpected scene format: {metadata_path}")
        if metadata.get("split") != spec.split or metadata.get("scene_id") != spec.scene_id:
            raise ValueError(f"Scene identity does not match the fixed split: {metadata_path}")


def _link_scene_assets(
    prepared_scene: Path,
    output_scene: Path,
    frame_count: int,
) -> int:
    output_scene.mkdir(parents=True, exist_ok=False)
    _hard_link(prepared_scene / "tracks_3d.npy", output_scene / "tracks_3d.npy")
    linked = 1
    expected_rgb_names = [f"rgba_{frame:05d}.jpg" for frame in range(frame_count)]
    for view in preprocessing.VIEW_IDS:
        prepared_view = prepared_scene / f"view_{view}"
        output_view = output_scene / f"view_{view}"
        output_view.mkdir(exist_ok=False)
        actual_rgb_names = sorted(path.name for path in prepared_view.glob("rgba_*.jpg"))
        if actual_rgb_names != expected_rgb_names:
            raise ValueError(f"Prepared RGB sequence is not contiguous: {prepared_view}")
        for name in expected_rgb_names:
            _hard_link(prepared_view / name, output_view / name)
            linked += 1
        for name in ("depth.npy", "intrinsics.npy", "extrinsics_w2c.npy"):
            _hard_link(prepared_view / name, output_view / name)
            linked += 1
    return linked


def _repair_scene(
    source_root: Path,
    prepared_root: Path,
    build_root: Path,
    spec: preprocessing.SceneSpec,
) -> tuple[dict[str, Any], int]:
    prepared_scene = _scene_path(prepared_root, spec)
    output_scene = _scene_path(build_root, spec)
    old_metadata = _read_json_object(prepared_scene / "scene.json")
    linked_file_count = _link_scene_assets(
        prepared_scene,
        output_scene,
        spec.frame_count,
    )

    tracks = _require_array(
        prepared_scene / "tracks_3d.npy",
        (spec.frame_count, preprocessing.POINT_COUNT, 3),
        np.float32,
    )
    finite_tracks = np.isfinite(tracks).all(axis=-1)
    candidate_counts = np.zeros(spec.frame_count, dtype=np.int64)
    failure_counts = np.zeros(spec.frame_count, dtype=np.int64)
    repaired_view_statistics: dict[str, dict[str, Any]] = {}

    for view in preprocessing.VIEW_IDS:
        prepared_view = prepared_scene / f"view_{view}"
        output_view = output_scene / f"view_{view}"
        depth = _require_array(
            prepared_view / "depth.npy",
            (spec.frame_count, preprocessing.OUTPUT_HEIGHT, preprocessing.OUTPUT_WIDTH),
            np.float32,
        )
        intrinsics = _require_array(prepared_view / "intrinsics.npy", (3, 3), np.float32)
        extrinsics = _require_array(
            prepared_view / "extrinsics_w2c.npy",
            (spec.frame_count, 3, 4),
            np.float32,
        )
        source_visibility_all = _require_array(
            _source_visibility_path(source_root, spec, view),
            (spec.source_frame_count, preprocessing.POINT_COUNT),
            np.bool_,
        )
        source_visibility = np.asarray(
            source_visibility_all[spec.source_frame_start : spec.source_frame_end],
            dtype=bool,
        )
        projected_xy, camera_z = preprocessing._project_points(
            np.asarray(tracks),
            np.asarray(extrinsics),
            np.asarray(intrinsics),
        )
        finite_projection = np.isfinite(projected_xy).all(axis=-1) & np.isfinite(camera_z)
        inside = (
            (projected_xy[..., 0] >= -0.5)
            & (projected_xy[..., 0] < preprocessing.OUTPUT_WIDTH - 0.5)
            & (projected_xy[..., 1] >= -0.5)
            & (projected_xy[..., 1] < preprocessing.OUTPUT_HEIGHT - 0.5)
        )
        geometric_visibility = (
            source_visibility
            & finite_tracks
            & finite_projection
            & (camera_z > 0.0)
            & inside
        )

        view_candidate_count_by_frame = [0] * spec.frame_count
        view_failure_count_by_frame = [0] * spec.frame_count
        view_failures = 0
        view_no_valid_depth = 0
        for frame in range(spec.frame_count):
            consistent, any_valid_depth = preprocessing._depth_track_consistency_masks(
                np.asarray(depth[frame]),
                projected_xy[frame],
                camera_z[frame],
                geometric_visibility[frame],
            )
            candidates = geometric_visibility[frame]
            no_valid_depth = int((candidates & ~any_valid_depth).sum())
            residual_failure = int(
                (candidates & any_valid_depth & ~consistent).sum()
            )
            failures = no_valid_depth + residual_failure
            frame_candidate_count = int(candidates.sum())
            view_candidate_count_by_frame[frame] = frame_candidate_count
            view_failure_count_by_frame[frame] = failures
            candidate_counts[frame] += frame_candidate_count
            failure_counts[frame] += failures
            view_failures += failures
            view_no_valid_depth += no_valid_depth

        np.save(output_view / "visibility.npy", np.ascontiguousarray(geometric_visibility))
        persisted_visibility = _require_array(
            output_view / "visibility.npy",
            geometric_visibility.shape,
            np.bool_,
            mmap_mode=None,
        )
        if not np.array_equal(persisted_visibility, geometric_visibility):
            raise AssertionError(f"Repaired visibility did not round-trip: {output_view}")

        old_view_stats = copy.deepcopy(
            old_metadata.get("statistics", {}).get("views", {}).get(str(view), {})
        )
        for old_key in (
            "visibility_true_after_depth_consistency_gating",
            "visibility_removed_by_depth_consistency_gating",
            "visibility_true_after_gating",
            "visibility_removed_by_gating",
            "visibility_rejected_no_valid_depth",
            "visibility_rejected_residual_over_tolerance",
        ):
            old_view_stats.pop(old_key, None)
        before = int(source_visibility.sum())
        after_geometry = int(geometric_visibility.sum())
        old_view_stats.update(
            {
                "visibility_true_before_gating": before,
                "visibility_true_after_geometric_gating": after_geometry,
                "visibility_removed_by_geometric_gating": before - after_geometry,
                "depth_consistency_candidate_count": after_geometry,
                "depth_consistency_failure_count": view_failures,
                "depth_consistency_candidate_count_by_frame": (
                    view_candidate_count_by_frame
                ),
                "depth_consistency_failure_count_by_frame": (
                    view_failure_count_by_frame
                ),
                "depth_consistency_no_valid_depth_count": view_no_valid_depth,
                "depth_consistency_residual_over_tolerance_count": (
                    view_failures - view_no_valid_depth
                ),
                "visibility_true_saved": after_geometry,
            }
        )
        repaired_view_statistics[str(view)] = old_view_stats

    candidate_count_list = [int(value) for value in candidate_counts]
    failure_count_list = [int(value) for value in failure_counts]
    depth_invalid_frames = preprocessing._depth_invalid_frame_indices(
        candidate_count_list,
        failure_count_list,
    )
    rgb_invalid_frames = preprocessing._sorted_unique_frame_indices(
        old_metadata["output"]["rgb"]["invalid_frame_indices"],
        spec.frame_count,
        label="schema-v4 RGB invalid frame indices",
    )
    invalid_frames = sorted(set(rgb_invalid_frames) | set(depth_invalid_frames))
    total_starts, excluded_starts, legal_starts = preprocessing._window_start_counts(
        spec.frame_count,
        invalid_frames,
    )
    if legal_starts == 0:
        raise ValueError(
            f"{spec.split}/{spec.scene_id} has no legal "
            f"{preprocessing.WINDOW_LENGTH}-frame windows"
        )
    per_frame = [
        {
            "frame": frame,
            "candidate_count": candidates,
            "failure_count": failures,
            "failure_fraction": failures / candidates if candidates else None,
        }
        for frame, (candidates, failures) in enumerate(
            zip(candidate_count_list, failure_count_list)
        )
    ]
    depth_statistics = {
        "candidate_count": sum(candidate_count_list),
        "failure_count": sum(failure_count_list),
        "invalid_frame_count": len(depth_invalid_frames),
        "invalid_frame_indices": depth_invalid_frames,
        "per_frame": per_frame,
    }
    window_exclusion = {
        "window_length": preprocessing.WINDOW_LENGTH,
        "invalid_frame_indices": invalid_frames,
        "reasons": {
            "rgb_decode": rgb_invalid_frames,
            "depth_track_majority_mismatch": depth_invalid_frames,
        },
        "total_start_count": total_starts,
        "excluded_start_count": excluded_starts,
        "legal_start_count": legal_starts,
    }

    metadata = copy.deepcopy(old_metadata)
    metadata["schema_version"] = OUTPUT_SCHEMA_VERSION
    output = metadata["output"]
    output["visibility"] = {
        "format": "npy",
        "dtype": "bool",
        "geometric_gate": (
            "source-visible, finite track and projection, positive camera-Z, "
            "inside resized pixel-center bounds"
        ),
        "depth_gated": False,
    }
    output["depth_track_consistency"] = {
        "depth_source": "exact resized float32 optical-Z frame written to depth.npy",
        "nearest_pixel_rule": "floor(coordinate + 0.5)",
        "neighborhood": (
            "in-bounds finite positive samples from the 3x3 around nearest pixel; "
            "out-of-bounds offsets are skipped"
        ),
        "observation_failure": (
            "no valid positive depth in the 3x3 neighborhood or no sample with "
            "abs(depth-camera_z) <= tolerance_metres"
        ),
        "tolerance_metres": preprocessing.DEPTH_TRACK_TOLERANCE_METRES,
        "frame_failure_fraction_threshold": preprocessing.DEPTH_FRAME_FAILURE_FRACTION,
        "frame_failure_rule": (
            "candidate_count > 0 and failure_count / candidate_count > "
            "frame_failure_fraction_threshold"
        ),
        "zero_candidate_failure_fraction": None,
        "invalid_frame_indices": depth_invalid_frames,
        "per_frame": per_frame,
    }
    output["window_exclusion"] = window_exclusion
    statistics = metadata.setdefault("statistics", {})
    statistics["views"] = repaired_view_statistics
    statistics["depth_track_consistency"] = depth_statistics
    statistics["window_exclusion"] = window_exclusion
    metadata.setdefault("validation", {"failure_count": 0, "failures": []})
    preprocessing._validate_scene_metadata_contract(metadata, spec)
    _write_json(output_scene / "scene.json", metadata)
    return metadata, linked_file_count


def _validate_repaired_tree(
    prepared_root: Path,
    build_root: Path,
    specs: Sequence[preprocessing.SceneSpec],
) -> int:
    _require_exact_scenes(build_root, specs)
    validated_hard_links = 0
    for spec in specs:
        prepared_scene = _scene_path(prepared_root, spec)
        output_scene = _scene_path(build_root, spec)
        metadata = _read_json_object(output_scene / "scene.json")
        preprocessing._validate_scene_metadata_contract(metadata, spec)
        immutable_relative_paths = [Path("tracks_3d.npy")]
        for view in preprocessing.VIEW_IDS:
            view_relative = Path(f"view_{view}")
            immutable_relative_paths.extend(
                view_relative / f"rgba_{frame:05d}.jpg"
                for frame in range(spec.frame_count)
            )
            immutable_relative_paths.extend(
                view_relative / name
                for name in ("depth.npy", "intrinsics.npy", "extrinsics_w2c.npy")
            )
            _require_array(
                output_scene / view_relative / "visibility.npy",
                (spec.frame_count, preprocessing.POINT_COUNT),
                np.bool_,
            )
        for relative_path in immutable_relative_paths:
            source_stat = (prepared_scene / relative_path).stat()
            output_stat = (output_scene / relative_path).stat()
            if (source_stat.st_dev, source_stat.st_ino) != (
                output_stat.st_dev,
                output_stat.st_ino,
            ):
                raise ValueError(f"Immutable artifact was not hard-linked: {relative_path}")
            validated_hard_links += 1
    return validated_hard_links


def _repair_report(
    specs: Sequence[preprocessing.SceneSpec],
    scene_metadata: dict[tuple[str, str], dict[str, Any]],
    *,
    prepared_root: Path,
    validated_hard_links: int,
) -> dict[str, Any]:
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        split_specs = [spec for spec in specs if spec.split == split]
        split_counts[split] = {
            "prepared_scenes": len(split_specs),
            "frames": sum(spec.frame_count for spec in split_specs),
            "source_sequences": len({spec.source_key for spec in split_specs}),
        }
    depth_invalid_frames = 0
    total_starts = 0
    excluded_starts = 0
    legal_starts = 0
    scenes = []
    for spec in specs:
        output = scene_metadata[(spec.split, spec.scene_id)]["output"]
        depth = output["depth_track_consistency"]
        exclusion = output["window_exclusion"]
        depth_invalid_frames += len(depth["invalid_frame_indices"])
        total_starts += int(exclusion["total_start_count"])
        excluded_starts += int(exclusion["excluded_start_count"])
        legal_starts += int(exclusion["legal_start_count"])
        scenes.append(
            {
                "split": spec.split,
                "scene_id": spec.scene_id,
                "invalid_frame_indices": exclusion["invalid_frame_indices"],
                "reasons": exclusion["reasons"],
                "total_start_count": exclusion["total_start_count"],
                "excluded_start_count": exclusion["excluded_start_count"],
                "legal_start_count": exclusion["legal_start_count"],
            }
        )
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "format": REPORT_FORMAT,
        "status": "completed",
        "ignore_validation_failures": False,
        "counts": split_counts,
        "totals": {
            "prepared_scenes": len(specs),
            "frames": sum(spec.frame_count for spec in specs),
            "source_sequences": len({spec.source_key for spec in specs}),
            "semantic_validation_failures": 0,
        },
        "failures": [],
        "repair": {
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "prepared_root": str(prepared_root),
            "immutable_artifact_policy": "hard_links_only_no_copy_fallback",
            "validated_hard_link_count": validated_hard_links,
            "rewritten_artifacts": ["view_*/visibility.npy", "scene.json"],
        },
        "statistics": {
            "depth_track_consistency": {
                "invalid_frame_count": depth_invalid_frames,
                "tolerance_metres": preprocessing.DEPTH_TRACK_TOLERANCE_METRES,
                "frame_failure_fraction_threshold": (
                    preprocessing.DEPTH_FRAME_FAILURE_FRACTION
                ),
            },
            "window_exclusion": {
                "window_length": preprocessing.WINDOW_LENGTH,
                "total_start_count": total_starts,
                "excluded_start_count": excluded_starts,
                "legal_start_count": legal_starts,
                "scenes": scenes,
            },
        },
        "split_plan": [
            {
                field: getattr(spec, field)
                for field in spec.__dataclass_fields__
            }
            for spec in specs
        ],
    }


def repair_preprocessed(
    source_root: Path,
    prepared_root: Path,
    output_root: Path,
) -> None:
    source_root = Path(source_root).expanduser().resolve()
    prepared_root = Path(prepared_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root is missing: {source_root}")
    if not prepared_root.is_dir():
        raise FileNotFoundError(f"Prepared root is missing: {prepared_root}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output_root}")
    if output_root == prepared_root or prepared_root in output_root.parents:
        raise ValueError("Output root must be a separate sibling tree, not inside v4")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    _require_same_filesystem(prepared_root, output_root.parent)

    specs = preprocessing.build_scene_specs()
    _validate_v4_root(prepared_root, specs)
    build_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        for split in ("train", "validation", "test"):
            (build_root / split).mkdir(exist_ok=False)
        repaired_metadata: dict[tuple[str, str], dict[str, Any]] = {}
        linked_file_count = 0
        for spec in specs:
            metadata, scene_linked_files = _repair_scene(
                source_root,
                prepared_root,
                build_root,
                spec,
            )
            repaired_metadata[(spec.split, spec.scene_id)] = metadata
            linked_file_count += scene_linked_files
        validated_hard_links = _validate_repaired_tree(
            prepared_root,
            build_root,
            specs,
        )
        if validated_hard_links != linked_file_count:
            raise AssertionError("Hard-link conversion and validation counts differ")
        report = _repair_report(
            specs,
            repaired_metadata,
            prepared_root=prepared_root,
            validated_hard_links=validated_hard_links,
        )
        _write_json(build_root / "validation_report.json", report)
        if output_root.exists():
            raise FileExistsError(f"Output root appeared during repair: {output_root}")
        os.replace(build_root, output_root)
    except BaseException:
        shutil.rmtree(build_root, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repair_preprocessed(args.source_root, args.prepared_root, args.output_root)
    print(f"POINTODYSSEY_REPAIR_DONE output={args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
