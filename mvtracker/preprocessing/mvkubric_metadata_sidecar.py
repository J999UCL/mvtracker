"""Metadata-only MV-Kubric WebDataset sidecars.

The media WebDataset contains one ``meta.npz`` record per scene alongside the
RGB and depth records.  This module copies those metadata records byte for
byte into a small, fixed number of uncompressed TARs.  Recipe planning can
then stage only the shards it needs and never touches the media archives.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import tarfile
import time
from typing import Iterable, Mapping, Sequence

from .mvkubric_webdataset import (
    META_COMPONENT,
    RECORD_LOCATOR,
    build_record_locator,
    parse_dali_index,
    read_component,
)


METADATA_SIDECAR_FORMAT = "mvtracker-kubric-metadata-sidecar"
METADATA_SIDECAR_VERSION = 1
METADATA_SHARD_COUNT = 16
_COPY_CHUNK_BYTES = 8 * 1024 * 1024


def _manifest_path(root: str | Path) -> Path:
    path = Path(root)
    if path.is_file():
        return path
    manifest = path / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"MV-Kubric WebDataset manifest is missing: {manifest}")
    return manifest


def _archive_path(manifest_path: Path, shard: Mapping[str, object]) -> Path:
    value = shard.get("tar", shard.get("url"))
    if value is None:
        raise ValueError("source shard is missing tar/url")
    path = Path(str(value))
    return path if path.is_absolute() else manifest_path.parent / path


def _index_path(manifest_path: Path, shard: Mapping[str, object]) -> Path:
    value = shard.get("index_path")
    if value is not None:
        path = Path(str(value))
        return path if path.is_absolute() else manifest_path.parent / path
    return _archive_path(manifest_path, shard).with_suffix(".idx")


def _ordered_scene_ids(scene_ids: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(scene) for scene in scene_ids)
    if not values:
        raise ValueError("scene_ids must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("scene_ids must be unique")
    return tuple(sorted(values, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)))


def _indexed_metadata_locations(
    manifest_path: Path,
    scene_ids: Sequence[str],
) -> dict[str, tuple[Path, int, int]]:
    wanted = {f"scene-{scene}" : str(scene) for scene in scene_ids}
    found: dict[str, tuple[Path, int, int]] = {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"{manifest_path}: manifest has no shards")
    started = time.perf_counter()
    print(
        "MVKUBRIC_METADATA event=index_scan_start "
        f"source_shards={len(shards)} scenes={len(scene_ids)}",
        flush=True,
    )
    for shard_number, shard in enumerate(shards, start=1):
        if not isinstance(shard, dict):
            raise ValueError(f"{manifest_path}: malformed shard entry")
        archive_path = _archive_path(manifest_path, shard)
        index_path = _index_path(manifest_path, shard)
        if not archive_path.is_file():
            raise FileNotFoundError(f"source WebDataset TAR is missing: {archive_path}")
        records = parse_dali_index(index_path)
        for record in records:
            key = record.key
            scene = wanted.get(key)
            if scene is None:
                continue
            components = [
                component
                for component in record.components
                if component.extension == META_COMPONENT
            ]
            if len(components) != 1:
                raise ValueError(
                    f"{index_path}: {key} must have exactly one {META_COMPONENT} component"
                )
            component = components[0]
            if key in found:
                raise ValueError(f"scene metadata is duplicated: {scene}")
            found[key] = (archive_path, component.offset, component.size)
        if shard_number % 50 == 0 or shard_number == len(shards):
            print(
                "MVKUBRIC_METADATA event=index_scan_progress "
                f"source_shards={shard_number}/{len(shards)} "
                f"scenes={len(found)}/{len(scene_ids)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    missing = sorted(set(wanted).difference(found), key=str)
    if missing:
        raise FileNotFoundError(f"metadata records are missing from indexed WebDataset: {missing}")
    return {scene: found[key] for key, scene in wanted.items()}


def _tar_add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def _write_dali_index(archive_path: Path, index_path: Path) -> Path:
    records: list[tuple[str, int, int, str]] = []
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            key, extension = member.name.split(".", 1)
            records.append((extension, int(member.offset_data), int(member.size), member.name))
    if any(extension != META_COMPONENT for extension, *_ in records):
        raise ValueError(f"metadata sidecar contains a non-metadata component: {archive_path}")
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".partial")
    lines = [f"v1.2 {len(records)}"]
    lines.extend(f"{extension} {offset} {size} {name}" for extension, offset, size, name in records)
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(index_path)
    return index_path


def _partition(values: Sequence[str], count: int) -> tuple[tuple[str, ...], ...]:
    quotient, remainder = divmod(len(values), count)
    groups: list[tuple[str, ...]] = []
    start = 0
    for index in range(count):
        size = quotient + (index < remainder)
        groups.append(tuple(values[start : start + size]))
        start += size
    return tuple(groups)


def _write_metadata_shard(
    output_root: Path,
    shard_index: int,
    scene_ids: Sequence[str],
    locations: Mapping[str, tuple[Path, int, int]],
) -> dict[str, object]:
    name = f"mvkubric-meta-{shard_index:02d}"
    archive_path = output_root / f"{name}.tar"
    partial = archive_path.with_suffix(archive_path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    written_scenes = []
    by_archive: dict[Path, list[tuple[str, int, int]]] = {}
    for scene in scene_ids:
        source_archive_path, offset, size = locations[scene]
        by_archive.setdefault(source_archive_path, []).append((scene, offset, size))
    with tarfile.open(partial, "w", format=tarfile.USTAR_FORMAT) as archive:
        for source_archive, records in by_archive.items():
            with source_archive.open("rb") as source:
                for scene, offset, size in sorted(records, key=lambda item: item[1]):
                    source.seek(offset)
                    payload = source.read(size)
                    if len(payload) != size:
                        raise OSError(
                            f"short indexed read from {source_archive} at {offset}"
                        )
                    _tar_add_bytes(
                        archive,
                        f"scene-{scene}.{META_COMPONENT}",
                        payload,
                    )
                    written_scenes.append(scene)
    partial.replace(archive_path)
    index_path = _write_dali_index(archive_path, archive_path.with_suffix(".idx"))
    return {
        "name": name,
        "index": shard_index,
        "tar": archive_path.name,
        "index_path": index_path.name,
        "scene_ids": written_scenes,
        "nsamples": len(written_scenes),
        "bytes": archive_path.stat().st_size,
    }


def build_metadata_sidecar(
    source_root: str | Path,
    output_root: str | Path,
    scene_ids: Sequence[str] | None = None,
    *,
    shard_count: int = METADATA_SHARD_COUNT,
) -> dict[str, object]:
    """Build exactly ``shard_count`` metadata-only indexed TARs.

    Metadata bytes are read through the source DALI indexes and copied without
    decoding or reserializing.  The source archives are never written.
    """
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    source_manifest_path = _manifest_path(source_root)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    requested = scene_ids if scene_ids is not None else source_manifest.get("scene_ids")
    if requested is None:
        raise ValueError("scene_ids are required when the source manifest has no scene_ids")
    ordered = _ordered_scene_ids(requested)
    locations = _indexed_metadata_locations(source_manifest_path, ordered)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    groups = _partition(ordered, int(shard_count))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(shard_count, len(groups))) as executor:
        futures = {
            executor.submit(
                _write_metadata_shard,
                output_root,
                index,
                group,
                locations,
            ): index
            for index, group in enumerate(groups)
        }
        results = []
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                "MVKUBRIC_METADATA event=write_progress "
                f"shards={completed}/{len(futures)} "
                f"latest={result['name']} bytes={result['bytes']} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    results.sort(key=lambda item: int(item["index"]))
    shards = [{key: value for key, value in result.items() if key != "index_path"} for result in results]
    locator = build_record_locator(shards, output_root / RECORD_LOCATOR)
    scenes = {}
    global_index = 0
    for shard in results:
        for local_index, scene in enumerate(shard["scene_ids"]):
            scenes[str(scene)] = {
                "metadata_index": global_index,
                "shard": shard["tar"],
                "shard_index": int(shard["index"]),
                "record_index": local_index,
            }
            global_index += 1
    manifest = {
        "format": METADATA_SIDECAR_FORMAT,
        "version": METADATA_SIDECAR_VERSION,
        "scene_ids": list(ordered),
        "shard_count": int(shard_count),
        "record_locator": locator.name,
        "shards": shards,
        "scenes": scenes,
        "source_manifest": str(source_manifest_path),
    }
    manifest_path = output_root / "manifest.json"
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(
        f"MVKUBRIC_METADATA event=complete scenes={len(ordered)} shards={shard_count} "
        f"seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return manifest


def _copy_sequential(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with source.open("rb") as input_handle, partial.open("wb") as output_handle:
        while True:
            chunk = input_handle.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            output_handle.write(chunk)
    partial.replace(destination)


class KubricMetadataSidecar:
    """Stage selected metadata shards and decode scene metadata on demand."""

    def __init__(self, root: str | Path):
        self.manifest_path = _manifest_path(root)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != METADATA_SIDECAR_FORMAT:
            raise ValueError(f"{self.manifest_path}: unsupported metadata sidecar format")
        if int(manifest.get("version", -1)) != METADATA_SIDECAR_VERSION:
            raise ValueError(f"{self.manifest_path}: unsupported metadata sidecar version")
        self.root = self.manifest_path.parent
        self.manifest = manifest
        self.scenes = {str(name): dict(entry) for name, entry in manifest.get("scenes", {}).items()}
        if not self.scenes:
            raise ValueError(f"{self.manifest_path}: sidecar has no scenes")

    def required_shards(self, scene_ids: Iterable[str]) -> tuple[str, ...]:
        requested = tuple(str(scene) for scene in scene_ids)
        missing = sorted(set(requested).difference(self.scenes), key=str)
        if missing:
            raise KeyError(f"scenes are absent from metadata sidecar: {missing}")
        return tuple(sorted({str(self.scenes[scene]["shard"]) for scene in requested}))

    def stage(
        self,
        scene_ids: Iterable[str],
        local_root: str | Path,
        *,
        workers: int = 8,
    ) -> Path:
        """Copy only required TAR/index pairs using bounded sequential copies."""
        if workers < 1:
            raise ValueError("workers must be positive")
        local_root = Path(local_root)
        shard_names = self.required_shards(scene_ids)
        jobs = []
        for shard_name in shard_names:
            source_tar = self.root / shard_name
            source_index = source_tar.with_suffix(".idx")
            if not source_tar.is_file() or not source_index.is_file():
                raise FileNotFoundError(f"metadata shard pair is incomplete: {source_tar}")
            jobs.extend(((source_tar, local_root / source_tar.name), (source_index, local_root / source_index.name)))
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(int(workers), max(1, len(jobs)))) as executor:
            futures = {executor.submit(_copy_sequential, source, destination): destination for source, destination in jobs}
            for completed, future in enumerate(as_completed(futures), start=1):
                future.result()
                print(f"MVKUBRIC_METADATA event=stage progress={completed}/{len(jobs)}", flush=True)
        print(
            f"MVKUBRIC_METADATA event=stage_complete shards={len(shard_names)} "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
        return local_root

    def _payload(self, scene: str, staged_root: str | Path | None = None) -> bytes:
        entry = self.scenes.get(str(scene))
        if entry is None:
            raise KeyError(f"scene is absent from metadata sidecar: {scene}")
        archive_path = (Path(staged_root) if staged_root is not None else self.root) / str(entry["shard"])
        index_path = archive_path.with_suffix(".idx")
        records = parse_dali_index(index_path)
        key = f"scene-{scene}"
        matches = [record for record in records if record.key == key]
        if len(matches) != 1:
            raise ValueError(f"{index_path}: expected one record for {key}")
        components = [component for component in matches[0].components if component.extension == META_COMPONENT]
        if len(components) != 1:
            raise ValueError(f"{index_path}: expected one {META_COMPONENT} component for {key}")
        component = components[0]
        with archive_path.open("rb") as handle:
            handle.seek(component.offset)
            payload = handle.read(component.size)
        if len(payload) != component.size:
            raise OSError(f"short indexed read from metadata sidecar: {archive_path}")
        return payload

    def load(self, scene: str, staged_root: str | Path | None = None):
        """Load one :class:`KubricSceneMetadata` from a staged or source shard."""
        return _decode_metadata(str(scene), self._payload(str(scene), staged_root))

    def load_many(
        self,
        scene_ids: Iterable[str],
        *,
        staged_root: str | Path | None = None,
        workers: int = 16,
    ) -> dict[str, object]:
        """Load selected scenes while opening and indexing each shard once."""
        if workers < 1:
            raise ValueError("workers must be positive")
        requested = tuple(dict.fromkeys(str(scene) for scene in scene_ids))
        self.required_shards(requested)
        by_shard: dict[str, list[str]] = {}
        for scene in requested:
            by_shard.setdefault(str(self.scenes[scene]["shard"]), []).append(scene)

        root = Path(staged_root) if staged_root is not None else self.root

        def load_shard(shard_name: str, scenes: Sequence[str]):
            archive_path = root / shard_name
            records = {
                record.key: record for record in parse_dali_index(archive_path.with_suffix(".idx"))
            }
            loaded = {}
            with archive_path.open("rb") as archive:
                for scene in scenes:
                    record = records[f"scene-{scene}"]
                    component = next(
                        item
                        for item in record.components
                        if item.extension == META_COMPONENT
                    )
                    archive.seek(component.offset)
                    payload = archive.read(component.size)
                    loaded[scene] = _decode_metadata(scene, payload)
            return loaded

        loaded = {}
        with ThreadPoolExecutor(
            max_workers=min(int(workers), max(1, len(by_shard)))
        ) as executor:
            futures = [
                executor.submit(load_shard, shard, scenes)
                for shard, scenes in by_shard.items()
            ]
            for future in as_completed(futures):
                loaded.update(future.result())
        return loaded


def _decode_metadata(scene: str, payload: bytes):
    """Decode one copied ``meta.npz`` payload into planner metadata."""
    from mvtracker.datasets.kubric_dali_dataset import KubricSceneMetadata

    arrays = read_component(payload)
    return KubricSceneMetadata(
        name=str(scene),
        tracks_3d=arrays["tracks_3d"].astype("float32", copy=False),
        visibility=arrays["visibility"].astype("bool", copy=False),
        intrinsics=arrays["intrinsics"].astype("float32", copy=False),
        extrinsics=arrays["extrinsics"].astype("float32", copy=False),
        sensor_widths=arrays["sensor_widths"].astype("float32", copy=False),
        focal_lengths=arrays["focal_lengths"].astype("float32", copy=False),
        invalid_frame_indices=tuple(
            int(value)
            for value in arrays.get("invalid_frame_indices", ()).reshape(-1)
        ),
        resolution_hw=tuple(
            int(value) for value in arrays["resolution_hw"].reshape(-1)
        ),
    )


def stage_metadata_shards(
    sidecar_root: str | Path,
    scene_ids: Iterable[str],
    local_root: str | Path,
    *,
    workers: int = 8,
) -> Path:
    """Functional wrapper around :meth:`KubricMetadataSidecar.stage`."""
    return KubricMetadataSidecar(sidecar_root).stage(scene_ids, local_root, workers=workers)


def load_scene_metadata(
    sidecar_root: str | Path,
    scene_id: str,
    *,
    staged_root: str | Path | None = None,
):
    """Functional wrapper returning a :class:`KubricSceneMetadata`."""
    return KubricMetadataSidecar(sidecar_root).load(scene_id, staged_root=staged_root)


__all__ = [
    "METADATA_SIDECAR_FORMAT",
    "METADATA_SIDECAR_VERSION",
    "METADATA_SHARD_COUNT",
    "KubricMetadataSidecar",
    "build_metadata_sidecar",
    "load_scene_metadata",
    "stage_metadata_shards",
]
