"""Scene-scoped staging contracts for the Syn4D lab_bald pilot."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
from typing import Any


SYN4D_REPO_ID = "Syn4D/Syn4D"
SYN4D_REVISION = "181c6a2da735b216826ab9411b08e0d1d225aced"
SYN4D_SCENE = "lab_bald"
SYN4D_SEQUENCE = "seq_000000"
SYN4D_SEQUENCES = tuple(f"seq_{index:06d}" for index in range(20))
SYN4D_REMAINING_SEQUENCES = SYN4D_SEQUENCES[1:]
SYN4D_HF_MAPPING = "data/syn4d_v1_stride_1/sequence_to_asset_mapping.csv"
SYN4D_HF_SOURCE = "data/syn4d_v1_stride_1/lab_bald.tar.zst"
SYN4D_SOURCE_BYTES = 16_004_849_235
SYN4D_OBJECT_ROOT = "data/metadata/new_weight_bone"
SYN4D_ROOT = Path("datasets/syn4d/lab_bald")
SYN4D_SOURCE_ROOT = SYN4D_ROOT / "source"
SYN4D_METADATA_ROOT = SYN4D_ROOT / "metadata"
SYN4D_OBJECT_ROOT_LOCAL = SYN4D_METADATA_ROOT / "new_weight_bone"
SYN4D_MAPPING = SYN4D_ROOT / "sequence_to_asset_mapping.csv"
SYN4D_BODY_ROOT = SYN4D_METADATA_ROOT / "bedlam2_smpl_npz"
SYN4D_CACHE_ROOT = Path("datasets/syn4d-mvtracker/train")
SELECTIVE_BEDLAM_ROOT = Path(
    "datasets/syn4d/v1-stride1-12train-4validation/metadata"
)
SYN4D_OBJECT_VERTEX_FILENAME = "vertices_sequence.npz"
MANIFEST_NAME = "lab_bald_manifest.json"


def _mapping_text(mapping_csv: str | Path) -> str:
    if isinstance(mapping_csv, Path):
        return mapping_csv.read_text(encoding="utf-8-sig")
    return mapping_csv


def sequence_dependencies(
    mapping_csv: str | Path, sequence: str = SYN4D_SEQUENCE
) -> dict[str, Any]:
    """Resolve one lab_bald sequence and its shared assets."""

    views: set[int] = set()
    bodies: set[str] = set()
    objects: set[tuple[str, str]] = set()
    pattern = re.compile(r"(seq_\d{6})_([0-7])$")
    for row in csv.DictReader(io.StringIO(_mapping_text(mapping_csv))):
        if row.get("scene") != SYN4D_SCENE:
            continue
        match = pattern.fullmatch(str(row.get("sequence_name", "")))
        if match is None or match.group(1) != sequence:
            continue
        views.add(int(match.group(2)))
        asset_type = row.get("asset_type")
        asset = str(row.get("asset", ""))
        if asset_type == "bedlam2_body":
            bodies.add(asset)
        elif asset_type == "objaverse_object":
            parts = asset.rsplit("_", 1)
            if len(parts) != 2:
                raise ValueError(f"invalid Syn4D object asset {asset!r}")
            objects.add((parts[0], parts[1]))

    if views != set(range(8)):
        raise ValueError(
            f"{SYN4D_SCENE}/{sequence} must cover views 0 through 7; "
            f"got {sorted(views)}"
        )
    if len(bodies) != 1:
        raise ValueError(
            f"{SYN4D_SCENE}/{sequence} must resolve to one BEDLAM2 body"
        )
    if not objects:
        raise ValueError(
            f"{SYN4D_SCENE}/{sequence} has no referenced objects"
        )
    body_motion = next(iter(bodies))
    subject, animation = body_motion.rsplit("_", 1)
    clothing_member = f"{subject}/{animation}/{animation}.npz"
    return {
        "scene": SYN4D_SCENE,
        "sequence": sequence,
        "views": sorted(views),
        "body_motion": body_motion,
        "motions": [body_motion],
        "clothing_member": clothing_member,
        "objects": sorted(objects),
        "body_count": 1,
        "clothing_count": 1,
        "object_count": len(objects),
    }


def sequence_object_paths(
    mapping_csv: str | Path, sequence: str = SYN4D_SEQUENCE
) -> tuple[Path, ...]:
    """Return only public vertex files referenced by the pilot sequence."""

    plan = sequence_dependencies(mapping_csv, sequence)
    return tuple(
        Path(SYN4D_OBJECT_ROOT)
        / group
        / object_id
        / SYN4D_OBJECT_VERTEX_FILENAME
        for group, object_id in plan["objects"]
    )


def scene_dependencies(mapping_csv: str | Path) -> tuple[dict[str, Any], ...]:
    """Resolve all twenty lab_bald sequences in deterministic order."""

    return tuple(
        sequence_dependencies(mapping_csv, sequence)
        for sequence in SYN4D_SEQUENCES
    )


def scene_object_paths(mapping_csv: str | Path) -> tuple[Path, ...]:
    """Return the unique public vertex files referenced by all sequences."""

    return tuple(
        dict.fromkeys(
            path
            for sequence in SYN4D_SEQUENCES
            for path in sequence_object_paths(mapping_csv, sequence)
        )
    )


def sequence_bedlam_plan(
    mapping_csv: str, archive_map: dict[str, list[str]]
) -> dict[str, Any]:
    """Join the pilot mapping to one cached BEDLAM clothing archive."""

    plan = sequence_dependencies(mapping_csv)
    member_to_archive = {
        member: archive for archive, members in archive_map.items() for member in members
    }
    archive = member_to_archive.get(plan["clothing_member"])
    if archive is None:
        raise RuntimeError(
            f"BEDLAM archive map has no clothing member {plan['clothing_member']}"
        )
    return {**plan, "required_members": {archive: [plan["clothing_member"]]}}


def scene_bedlam_plan(
    mapping_csv: str | Path, archive_map: dict[str, list[str]]
) -> dict[str, Any]:
    """Join all twenty sequence clothing members to selective BEDLAM archives."""

    plans = scene_dependencies(mapping_csv)
    member_to_archive = {
        member: archive
        for archive, members in archive_map.items()
        for member in members
    }
    required_members: dict[str, list[str]] = {}
    for plan in plans:
        archive = member_to_archive.get(plan["clothing_member"])
        if archive is None:
            raise RuntimeError(
                f"BEDLAM archive map has no clothing member {plan['clothing_member']}"
            )
        required_members.setdefault(archive, []).append(plan["clothing_member"])
    return {
        "scene": SYN4D_SCENE,
        "sequences": [plan["sequence"] for plan in plans],
        "motions": [plan["body_motion"] for plan in plans],
        "required_members": {
            archive: sorted(members)
            for archive, members in sorted(required_members.items())
        },
        "sequence_count": len(plans),
        "body_count": len({plan["body_motion"] for plan in plans}),
        "clothing_count": len({plan["clothing_member"] for plan in plans}),
        "object_count": len(scene_object_paths(mapping_csv)),
    }


def write_sequence_manifest(
    destination: Path,
    *,
    source_archive: Path,
    mapping: Path,
    object_files: tuple[Path, ...],
    bedlam: dict[str, Any],
    bedlam_root: Path,
) -> Path:
    """Publish the completion marker only after every pilot dependency exists."""

    dependencies = sequence_dependencies(mapping)
    expected = len(dependencies["objects"])
    if len(object_files) != expected:
        raise RuntimeError(f"expected {expected} object vertices, got {len(object_files)}")
    if not source_archive.is_file() or source_archive.stat().st_size != SYN4D_SOURCE_BYTES:
        raise RuntimeError(f"incomplete {SYN4D_SCENE} source archive: {source_archive}")
    missing = [str(path) for path in object_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing object vertices: {missing[:3]}")
    body_source = Path(bedlam_root) / "b2_motions_npz_training/motions_npz_training"
    clothing_source = Path(bedlam_root) / "b2_assetdata_download/clothing/npz"
    missing_body = [
        motion for motion in dependencies["motions"]
        if not (body_source / f"{motion}.npz").is_file()
    ]
    missing_clothing = [
        archive for archive in bedlam["required_members"]
        if not (clothing_source / f"{archive}.tar").is_file()
    ]
    if missing_body or missing_clothing:
        raise FileNotFoundError(
            f"verified BEDLAM2 cache is incomplete for {SYN4D_SCENE}: "
            f"bodies={missing_body} clothing={missing_clothing}"
        )
    manifest = {
        "format": "mvtracker-syn4d-lab-bald",
        "scene": SYN4D_SCENE,
        "sequence": SYN4D_SEQUENCE,
        "source_revision": SYN4D_REVISION,
        "source_archive": str(source_archive),
        "mapping": str(mapping),
        "body_count": 1,
        "clothing_count": 1,
        "object_count": expected,
        "bedlam_root": str(bedlam_root),
        "bedlam": bedlam,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def write_scene_manifest(
    destination: Path,
    *,
    source_archive: Path,
    mapping: Path,
    object_files: tuple[Path, ...],
    bedlam: dict[str, Any],
    bedlam_root: Path,
) -> Path:
    """Publish the all-sequence staging marker after selective dependencies exist."""

    plans = scene_dependencies(mapping)
    expected_objects = len(scene_object_paths(mapping))
    if len(object_files) != expected_objects:
        raise RuntimeError(
            f"expected {expected_objects} object vertices, got {len(object_files)}"
        )
    if not source_archive.is_file() or source_archive.stat().st_size != SYN4D_SOURCE_BYTES:
        raise RuntimeError(f"incomplete {SYN4D_SCENE} source archive: {source_archive}")
    missing_objects = [str(path) for path in object_files if not path.is_file()]
    if missing_objects:
        raise FileNotFoundError(f"missing object vertices: {missing_objects[:3]}")
    body_source = Path(bedlam_root) / "b2_motions_npz_training/motions_npz_training"
    clothing_source = Path(bedlam_root) / "b2_assetdata_download/clothing/npz"
    missing_bodies = [
        plan["body_motion"] for plan in plans
        if not (body_source / f"{plan['body_motion']}.npz").is_file()
    ]
    missing_clothing = [
        archive for archive in bedlam["required_members"]
        if not (clothing_source / f"{archive}.tar").is_file()
    ]
    if missing_bodies or missing_clothing:
        raise FileNotFoundError(
            f"verified BEDLAM2 cache is incomplete for {SYN4D_SCENE}: "
            f"bodies={missing_bodies[:3]} clothing={missing_clothing[:3]}"
        )
    manifest = {
        "format": "mvtracker-syn4d-lab-bald",
        "scene": SYN4D_SCENE,
        "sequences": [plan["sequence"] for plan in plans],
        "source_revision": SYN4D_REVISION,
        "source_archive": str(source_archive),
        "mapping": str(mapping),
        "sequence_count": len(plans),
        "body_count": len({plan["body_motion"] for plan in plans}),
        "clothing_count": len({plan["clothing_member"] for plan in plans}),
        "object_count": expected_objects,
        "bedlam_root": str(bedlam_root),
        "bedlam": bedlam,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


__all__ = [
    "MANIFEST_NAME", "SELECTIVE_BEDLAM_ROOT", "SYN4D_BODY_ROOT", "SYN4D_CACHE_ROOT",
    "SYN4D_HF_MAPPING", "SYN4D_HF_SOURCE", "SYN4D_MAPPING", "SYN4D_METADATA_ROOT",
    "SYN4D_OBJECT_ROOT", "SYN4D_OBJECT_ROOT_LOCAL", "SYN4D_REPO_ID", "SYN4D_REVISION",
    "SYN4D_ROOT", "SYN4D_SCENE", "SYN4D_SEQUENCE", "SYN4D_SEQUENCES",
    "SYN4D_REMAINING_SEQUENCES", "SYN4D_SOURCE_BYTES",
    "SYN4D_SOURCE_ROOT", "sequence_bedlam_plan", "sequence_dependencies",
    "scene_bedlam_plan", "scene_dependencies", "scene_object_paths",
    "sequence_object_paths", "write_scene_manifest", "write_sequence_manifest",
]
