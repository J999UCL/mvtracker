"""Small, scene-scoped staging helpers for the Syn4D temple_group profile."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import shutil
from typing import Any


SYN4D_REPO_ID = "Syn4D/Syn4D"
SYN4D_REVISION = "181c6a2da735b216826ab9411b08e0d1d225aced"
TEMPLE_GROUP_SCENE = "temple_group"
TEMPLE_GROUP_HF_MAPPING = "data/syn4d_v1_stride_1/sequence_to_asset_mapping.csv"
TEMPLE_GROUP_HF_SOURCE = "data/syn4d_v1_stride_1/temple_group.tar.zst"
TEMPLE_GROUP_OBJECT_ROOT = "data/metadata/new_weight_bone"
TEMPLE_GROUP_SOURCE_BYTES = 13_287_559_476
TEMPLE_GROUP_ROOT = Path("datasets/syn4d/temple_group")
TEMPLE_GROUP_SOURCE_ROOT = TEMPLE_GROUP_ROOT / "source"
TEMPLE_GROUP_OBJECT_ROOT_LOCAL = TEMPLE_GROUP_ROOT / "objects"
TEMPLE_GROUP_MAPPING = TEMPLE_GROUP_ROOT / "sequence_to_asset_mapping.csv"
TEMPLE_GROUP_BEDLAM_ROOT = TEMPLE_GROUP_ROOT / "bedlam2"
TEMPLE_GROUP_CACHE_ROOT = TEMPLE_GROUP_ROOT / "cache"
TEMPLE_GROUP_OBJECT_VERTEX_FILENAME = "vertices_sequence.npz"
MANIFEST_NAME = "temple_group_manifest.json"


def temple_group_object_paths(mapping_csv: Path) -> tuple[Path, ...]:
    """Return the 60 public HF vertex files referenced by temple_group."""

    from mvtracker.preprocessing.syn4d import temple_group_dependencies

    plan = temple_group_dependencies(Path(mapping_csv))
    return tuple(
        Path(TEMPLE_GROUP_OBJECT_ROOT)
        / group
        / object_id
        / TEMPLE_GROUP_OBJECT_VERTEX_FILENAME
        for group, object_id in plan.objects
    )


def stage_hf_file(
    *, repo_id: str, revision: str, filename: str, token: str,
    destination: Path, local_dir: Path = Path("/tmp/syn4d-hf"),
) -> Path:
    """Copy one pinned HF file into the Volume, without a second full hash pass."""

    from huggingface_hub import hf_hub_download

    source = Path(hf_hub_download(
        repo_id=repo_id, repo_type="dataset", revision=revision,
        filename=filename, token=token, local_dir=str(local_dir),
    ))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def temple_group_bedlam_plan(
    mapping_csv: str, archive_map: dict[str, list[str]],
) -> dict[str, Any]:
    """Join the one-scene mapping to private BEDLAM clothing archives."""

    motions: set[str] = set()
    sequence_names: set[str] = set()
    for row in csv.DictReader(io.StringIO(mapping_csv)):
        if row.get("scene") != TEMPLE_GROUP_SCENE:
            continue
        sequence_names.add(str(row["sequence_name"]).rsplit("_", 1)[0])
        if row.get("asset_type") == "bedlam2_body":
            motions.add(str(row["asset"]))
    if len(sequence_names) != 20 or len(motions) != 20:
        raise RuntimeError(
            "unexpected temple_group BEDLAM plan: "
            f"sequences={len(sequence_names)} motions={len(motions)}"
        )
    member_to_archive = {
        member: archive for archive, members in archive_map.items() for member in members
    }
    required_members: dict[str, list[str]] = {}
    for motion in sorted(motions):
        subject, animation = motion.rsplit("_", 1)
        member = f"{subject}/{animation}/{animation}.npz"
        archive = member_to_archive.get(member)
        if archive is None:
            raise RuntimeError(f"BEDLAM archive map has no clothing member {member}")
        required_members.setdefault(archive, []).append(member)
    return {
        "scene": TEMPLE_GROUP_SCENE,
        "sequence_count": len(sequence_names),
        "motions": sorted(motions),
        "required_members": {
            archive: sorted(members)
            for archive, members in sorted(required_members.items())
        },
        "body_count": len(motions),
        "clothing_count": len(motions),
    }


def write_temple_group_manifest(
    destination: Path, *, source_archive: Path, mapping: Path,
    object_files: tuple[Path, ...], bedlam: dict[str, Any],
) -> Path:
    """Write the completion marker last, after selective dependencies exist."""

    from mvtracker.preprocessing.syn4d import temple_group_dependencies

    dependencies = temple_group_dependencies(mapping)
    expected = len(dependencies.objects)
    if len(object_files) != expected:
        raise RuntimeError(f"expected {expected} object vertices, got {len(object_files)}")
    if not source_archive.is_file() or source_archive.stat().st_size != TEMPLE_GROUP_SOURCE_BYTES:
        raise RuntimeError(f"incomplete temple_group source archive: {source_archive}")
    missing = [str(path) for path in object_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing object vertices: {missing[:3]}")
    manifest = {
        "format": "mvtracker-syn4d-temple-group",
        "scene": TEMPLE_GROUP_SCENE,
        "source_revision": SYN4D_REVISION,
        "source_archive": str(source_archive),
        "mapping": str(mapping),
        "sequence_count": len(dependencies.sequences),
        "body_count": len(dependencies.body_motions),
        "clothing_count": len(dependencies.clothing_members),
        "object_count": expected,
        "bedlam": bedlam,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


__all__ = [
    "MANIFEST_NAME", "SYN4D_REPO_ID", "SYN4D_REVISION", "TEMPLE_GROUP_BEDLAM_ROOT",
    "TEMPLE_GROUP_CACHE_ROOT", "TEMPLE_GROUP_HF_MAPPING", "TEMPLE_GROUP_HF_SOURCE",
    "TEMPLE_GROUP_MAPPING", "TEMPLE_GROUP_OBJECT_ROOT", "TEMPLE_GROUP_OBJECT_ROOT_LOCAL",
    "TEMPLE_GROUP_ROOT", "TEMPLE_GROUP_SCENE", "TEMPLE_GROUP_SOURCE_BYTES",
    "TEMPLE_GROUP_SOURCE_ROOT", "stage_hf_file", "temple_group_bedlam_plan",
    "temple_group_object_paths", "write_temple_group_manifest",
]
