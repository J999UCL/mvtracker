"""Immutable Modal data materialization for continual training."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

from mvtracker.profiling.modal_mvkubric2000 import (
    ARCHIVE_ROOT_RELATIVE,
    TRAIN_ARCHIVES,
    TRAIN_SCENES,
    VALIDATION_SCENES,
)

CHECKPOINT_REPO = "ethz-vlg/mvtracker"
CHECKPOINT_REVISION = "010d5d114e860aae6b2568104927b636cdca01bc"
CHECKPOINT_FILE = "mvtracker_200000_june2025.pth"
CHECKPOINT_SHA256 = "a7fa86f2a7223e3e0aa4c1d3eff0dec5fe8a9227a48572ce943b8e49d8a4f8e6"
MANIFEST_VERSION = 1
EXPECTED_DIEGESIS_SPLITS = {"train": 17, "validation": 2, "test": 2}
EXPECTED_MVKUBRIC_TRAIN_SCENES = set(TRAIN_SCENES)
EXPECTED_MVKUBRIC_POOL_SCENES = set(TRAIN_SCENES) | set(VALIDATION_SCENES)
EXPECTED_MVKUBRIC_SCENES = set(TRAIN_SCENES)
MVKUBRIC_VALIDATION_SCENES = set(VALIDATION_SCENES)
MVKUBRIC_INDEX_RELATIVE = Path("datasets/kubric-multiview/train/MVTracker_index")
DIEGESIS_ARCHIVE_RELATIVE = Path(
    "archives/diegesis/"
    "diegesis-81389015a6d713a848a120e34850f360621bcdce.tar.zst"
)
MVKUBRIC_SHARDS = tuple(ARCHIVE_ROOT_RELATIVE / item["filename"] for item in TRAIN_ARCHIVES)
LOCAL_STAGING_SIDECARS = (
    Path("profile-data-manifest.json"),
    Path("continual-training-data-manifest.json"),
    Path("checkpoints"),
    Path("datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache"),
    MVKUBRIC_INDEX_RELATIVE,
)


class _DeterministicRequestSampler:
    """Finite production-style requests with an explicit view count."""

    def __init__(self, dataset, total: int, view_count: int | None):
        self.dataset = dataset
        self.total = int(total)
        self.view_count = view_count

    def __iter__(self):
        from mvtracker.datasets.utils import SampleRequest

        for virtual_index in range(self.total):
            yield SampleRequest(
                virtual_index=virtual_index,
                view_count=self.view_count,
                scene_index=virtual_index % self.dataset.real_len,
            )

    def __len__(self):
        return self.total


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _materialize_checkpoint(data_root: Path, token: str) -> dict:
    checkpoint_root = data_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            revision=CHECKPOINT_REVISION,
            filename=CHECKPOINT_FILE,
            token=token,
            local_dir=checkpoint_root,
        )
    )
    observed = sha256(path)
    if observed != CHECKPOINT_SHA256:
        raise RuntimeError(f"mixed-depth checkpoint checksum mismatch: {observed}")
    return {
        "repo_id": CHECKPOINT_REPO,
        "revision": CHECKPOINT_REVISION,
        "filename": CHECKPOINT_FILE,
        "sha256": observed,
        "size_bytes": path.stat().st_size,
    }


def _require_existing_profile_data(data_root: Path) -> dict:
    manifest_path = data_root / "profile-data-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "profile-data-manifest.json is required; continual setup does not "
            "download or rebuild dataset inputs"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("diegesis", {}).get("splits") != EXPECTED_DIEGESIS_SPLITS:
        raise RuntimeError("existing DIEGESIS split manifest is incompatible")

    raw_root = data_root / "datasets/diegesis-mvtracker/TAPVid3D_raw"
    cache_root = data_root / "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache"
    for split, expected_count in EXPECTED_DIEGESIS_SPLITS.items():
        raw = {path.name for path in (raw_root / split).iterdir() if path.is_dir()}
        cached = {path.name for path in (cache_root / split).iterdir() if path.is_dir()}
        if len(raw) != expected_count or raw != cached:
            raise RuntimeError(f"existing DIEGESIS {split} data is incomplete")

    mvkubric_root = data_root / "datasets/kubric-multiview/train"
    mvkubric = {
        path.name
        for path in mvkubric_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    }
    if mvkubric != EXPECTED_MVKUBRIC_POOL_SCENES:
        raise RuntimeError(
            "existing MV-Kubric pool must contain scenes 1001..3000 plus validation 101..127"
        )
    _validate_mvkubric_index(data_root, EXPECTED_MVKUBRIC_POOL_SCENES)
    return manifest


def _validate_mvkubric_index(
    data_root: Path,
    scene_ids: set[str] = EXPECTED_MVKUBRIC_SCENES,
) -> dict:
    """Validate the declared MV-Kubric metadata index contract.

    The index is deliberately separate from native scene directories so the
    loader can open metadata without scanning the 78-GB scene pool.  The
    manifest schema is owned by the loader; this check only enforces the
    stable filesystem contract and, when present, its scene allowlist.
    """
    index_root = data_root / MVKUBRIC_INDEX_RELATIVE
    manifest_path = index_root / "manifest.json"
    scenes_root = index_root / "scenes"
    if not manifest_path.is_file() or not scenes_root.is_dir():
        raise FileNotFoundError(
            "MV-Kubric metadata index is required at "
            f"{MVKUBRIC_INDEX_RELATIVE}/manifest.json"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid MV-Kubric index manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("MV-Kubric index manifest must be a JSON object")
    observed_scene_ids = {
        path.stem for path in scenes_root.glob("*.npz") if path.is_file()
    }
    if observed_scene_ids != set(scene_ids):
        raise RuntimeError(
            "MV-Kubric metadata index scenes do not match native scene pool "
            f"({len(observed_scene_ids)} indexed, {len(scene_ids)} expected)"
        )
    if manifest.get("version") != 1 or not isinstance(manifest.get("scenes"), dict):
        raise RuntimeError("MV-Kubric metadata index manifest has an unsupported schema")
    if set(map(str, manifest["scenes"])) != set(scene_ids):
        raise RuntimeError("MV-Kubric metadata index manifest has an incompatible scene allowlist")
    for scene_id, entry in manifest["scenes"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("arrays"), str):
            raise RuntimeError(f"MV-Kubric metadata index entry is invalid for scene {scene_id}")
        arrays_path = index_root / entry["arrays"]
        if not arrays_path.is_file():
            raise FileNotFoundError(f"MV-Kubric indexed arrays are missing: {arrays_path}")
    index_module_path = Path(__file__).resolve().parents[1] / "datasets/kubric_metadata_index.py"
    spec = importlib.util.spec_from_file_location("mvtracker_kubric_metadata_index", index_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load MV-Kubric index validator: {index_module_path}")
    index_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(index_module)

    index_module.KubricMetadataIndex(index_root).validate_source(
        data_root / "datasets/kubric-multiview/train"
    )
    return {
        "relative_path": str(MVKUBRIC_INDEX_RELATIVE),
        "manifest": manifest,
        "scene_count": len(observed_scene_ids),
        "manifest_sha256": sha256(manifest_path),
        "size_bytes": _tree_stats(index_root)["size_bytes"],
    }


def _tree_stats(root: Path) -> dict[str, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
    }


def stage_continual_training_data(
    data_root: Path,
    *,
    local_data_root: Path,
) -> dict:
    """Legacy staging is disabled; use the immutable dataset image."""
    raise RuntimeError(
        "legacy archive staging is disabled for the 2,000-scene pool; "
        "use the cached dataset image"
    )
    data_root = Path(data_root)
    local_data_root = Path(local_data_root)
    inputs = (DIEGESIS_ARCHIVE_RELATIVE, *MVKUBRIC_SHARDS, *LOCAL_STAGING_SIDECARS)
    missing = [relative for relative in inputs if not (data_root / relative).exists()]
    if missing:
        raise FileNotFoundError(f"local staging inputs are missing: {', '.join(map(str, missing))}")
    if local_data_root.exists():
        shutil.rmtree(local_data_root)
    local_data_root.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    archive_root = local_data_root / "archives"
    archive_root.mkdir()
    archive_sources = [data_root / DIEGESIS_ARCHIVE_RELATIVE, *(
        data_root / relative for relative in MVKUBRIC_SHARDS
    )]
    archive_paths = [archive_root / source.name for source in archive_sources]
    with ThreadPoolExecutor(max_workers=len(archive_sources)) as executor:
        list(executor.map(shutil.copyfile, archive_sources, archive_paths))

    diegesis_root = local_data_root / "source/diegesis"
    mvkubric_root = local_data_root / "datasets/kubric-multiview/train"
    diegesis_root.mkdir(parents=True)
    mvkubric_root.mkdir(parents=True)

    def extract_mvkubric(path: Path) -> None:
        subprocess.run(
            [
                "tar", "--extract", "--zstd", "--strip-components=3",
                "--file", str(path), "--directory", str(mvkubric_root),
            ],
            check=True,
        )

    def extract_diegesis() -> None:
        subprocess.run(
            [
                "tar", "--extract", "--zstd", "--file", str(archive_paths[0]),
                "--directory", str(diegesis_root),
            ],
            check=True,
        )

    with ThreadPoolExecutor(max_workers=len(archive_paths)) as executor:
        tasks = [executor.submit(extract_diegesis)]
        tasks.extend(executor.submit(extract_mvkubric, path) for path in archive_paths[1:])
        for task in tasks:
            task.result()

    for scene_id in sorted(MVKUBRIC_VALIDATION_SCENES):
        source = data_root / "datasets/kubric-multiview/train" / scene_id
        destination = mvkubric_root / scene_id
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, destination)

    for relative in LOCAL_STAGING_SIDECARS:
        source = data_root / relative
        destination = local_data_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=True)
        else:
            shutil.copy2(source, destination)

    split_document = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/diegesis_split_v1.json").read_text(
            encoding="utf-8"
        )
    )
    raw_root = local_data_root / "datasets/diegesis-mvtracker/TAPVid3D_raw"
    for split, scenes in split_document["splits"].items():
        split_root = raw_root / split
        split_root.mkdir(parents=True)
        for scene in scenes:
            sequence = diegesis_root / "scenes" / scene / "tracking/sequence"
            if not sequence.is_dir():
                raise FileNotFoundError(sequence)
            (split_root / scene).symlink_to(
                os.path.relpath(sequence, split_root), target_is_directory=True
            )

    observed_mvkubric = {
        path.name for path in mvkubric_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    }
    if observed_mvkubric != EXPECTED_MVKUBRIC_POOL_SCENES:
        raise RuntimeError(
            "staged MV-Kubric pool must contain scenes 1001..3000 plus validation 101..127"
        )
    return {
        "local_data_root": str(local_data_root),
        "copied_size_bytes": _tree_stats(local_data_root)["size_bytes"],
        "elapsed_seconds": time.perf_counter() - started,
        "mvkubric_index": str(local_data_root / MVKUBRIC_INDEX_RELATIVE),
    }


def stage_mvkubric_profile_shard(
    data_root: Path,
    *,
    local_data_root: Path,
    shard_index: int = 0,
) -> dict:
    """Legacy shard staging is disabled; profile the cached dataset image."""
    raise RuntimeError(
        "legacy MV-Kubric shard staging is disabled; use the cached dataset image"
    )
    if not 0 <= shard_index < len(MVKUBRIC_SHARDS):
        raise ValueError("shard_index must be in [0, 4)")
    data_root = Path(data_root)
    local_data_root = Path(local_data_root)
    shard = data_root / MVKUBRIC_SHARDS[shard_index]
    index = data_root / MVKUBRIC_INDEX_RELATIVE
    if not shard.is_file() or not index.is_dir():
        raise FileNotFoundError("MV-Kubric shard or metadata index is missing")
    if local_data_root.exists():
        shutil.rmtree(local_data_root)
    archive = local_data_root / "archives" / shard.name
    archive.parent.mkdir(parents=True)
    started = time.perf_counter()
    shutil.copyfile(shard, archive)
    train_root = local_data_root / "datasets/kubric-multiview/train"
    train_root.mkdir(parents=True)
    subprocess.run(
        [
            "tar", "--extract", "--zstd", "--strip-components=3",
            "--file", str(archive), "--directory", str(train_root),
        ],
        check=True,
    )
    shutil.copytree(index, train_root / "MVTracker_index")
    first_scene = 900 + shard_index * 25
    scenes = tuple(str(scene) for scene in range(first_scene, first_scene + 25))
    observed = tuple(sorted(
        (path.name for path in train_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=int,
    ))
    if observed != scenes:
        raise RuntimeError(f"MV-Kubric shard inventory mismatch: {observed}")
    return {
        "local_data_root": str(local_data_root),
        "copied_size_bytes": archive.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "mvkubric_index": str(train_root / "MVTracker_index"),
        "scene_ids": list(scenes),
    }


def profile_encoded_loader(
    data_root: Path,
    *,
    source: str = "diegesis",
    warmup: int = 4,
    measured: int = 32,
    workers: int = 0,
    use_cuda: bool = False,
    mvkubric_scene_ids=None,
    view_count: int | None = None,
    source_schedule=None,
    simulated_compute_seconds: float = 0.0,
    hardware_sampler=None,
) -> dict:
    """Measure encoded TAPVid-3D samples after local extraction.

    This intentionally profiles loader/decode work only; model forward and
    optimizer work are outside the measurement.  The GPU path uses the
    production CUDA prefetch wrapper, while the CPU lane iterates samples
    directly to isolate encoded-cache and host-side work.
    """
    if source not in {"diegesis", "mvkubric"}:
        raise ValueError("source must be diegesis or mvkubric")
    if warmup < 0 or measured <= 0 or workers < 0:
        raise ValueError("warmup must be non-negative, measured and workers must be positive")
    if use_cuda and workers < 1:
        raise ValueError("the CUDA encoded-loader profile requires at least one worker")
    if view_count is not None and not 1 <= int(view_count) <= 6:
        raise ValueError("view_count must be between one and six")
    if simulated_compute_seconds < 0:
        raise ValueError("simulated_compute_seconds must be non-negative")
    if source_schedule is None:
        source_schedule = (source,)
    else:
        source_schedule = tuple(source_schedule)
        if not source_schedule or any(item not in {"diegesis", "mvkubric"} for item in source_schedule):
            raise ValueError("source_schedule must contain diegesis or mvkubric")
    if source_schedule != (source,) and not use_cuda:
        raise ValueError("source_schedule requires the CUDA production loader")
    import itertools
    from omegaconf import OmegaConf
    from types import SimpleNamespace

    import torch
    from mvtracker.datasets.tapvid3d_multiview_dataset import (
        CudaPrefetchLoader,
        TapVid3DMultiViewDataset,
    )

    repo_root = Path(__file__).resolve().parents[2]
    config = OmegaConf.merge(
        OmegaConf.load(repo_root / "configs/train.yaml"),
        OmegaConf.load(repo_root / "configs/experiment/diegesis_mvkubric_gt_ddp.yaml"),
    )
    datasets_root = data_root / "datasets"
    config.datasets.root = str(datasets_root)
    config.datasets.train.kubric_metadata_index_root = str(
        datasets_root / "kubric-multiview/train/MVTracker_index"
    )
    def build_dataset(dataset_source):
        if dataset_source == "diegesis":
            return TapVid3DMultiViewDataset.from_name(
                config.datasets.train.sources.diegesis.name,
                str(datasets_root / "diegesis-mvtracker"),
                training_args=config,
                fabric=SimpleNamespace(world_size=1),
                include_scene_ids=list((
                    "bathroom01", "bathroom02", "bathroom03", "bedroom02", "bedroom03",
                    "bedroom04", "diningroom01", "diningroom03", "diningroom04", "kitchen01",
                    "kitchen02", "kitchen03", "kitchen04", "livingroom01", "livingroom03",
                    "livingroom04", "livingroom05",
                )),
            )
        from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset

        return KubricMultiViewDataset.from_name(
            config.datasets.train.sources.mvkubric.name,
            str(datasets_root),
            training_args=config,
            fabric=SimpleNamespace(world_size=1),
            include_scene_ids=(
                list(mvkubric_scene_ids)
                if mvkubric_scene_ids is not None
                else [str(scene) for scene in range(1001, 3001)]
            ),
        )

    datasets = {dataset_source: build_dataset(dataset_source) for dataset_source in set(source_schedule)}
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA encoded-loader profile requires a visible GPU")

    iterators = {}
    for dataset_source, dataset in datasets.items():
        if use_cuda:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                sampler=_DeterministicRequestSampler(
                    dataset, warmup + measured + 8, view_count
                ),
                num_workers=workers,
                pin_memory=True,
                persistent_workers=workers > 0,
                prefetch_factor=2 if workers > 0 else None,
                multiprocessing_context="spawn" if workers > 0 else None,
                collate_fn=dataset.collate_fn,
                drop_last=True,
            )
            iterators[dataset_source] = iter(CudaPrefetchLoader(loader))
        else:
            iterators[dataset_source] = itertools.cycle(range(dataset.real_len))

    source_cursor = 0

    def consume() -> tuple[str, int | None, dict | None]:
        nonlocal source_cursor
        dataset_source = source_schedule[source_cursor % len(source_schedule)]
        source_cursor += 1
        iterator = iterators[dataset_source]
        if use_cuda:
            batch, gotit = next(iterator)
            if not all(gotit):
                return dataset_source, None, None
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream())
            ready.synchronize()
            return (
                dataset_source,
                int(batch.video.shape[0] * batch.video.shape[1]),
                batch.sample_metadata[0],
            )
        index = next(iterator)
        sample, gotit = datasets[dataset_source][index]
        if not gotit:
            return dataset_source, None, None
        if hasattr(sample, "jpeg_bytes"):
            return dataset_source, len(sample.jpeg_bytes), sample.metadata
        return dataset_source, int(sample.video.shape[0]), sample.metadata

    warmup_done = 0
    rejected = 0
    while warmup_done < warmup:
        _, frame_count, _ = consume()
        if frame_count is None:
            rejected += 1
        else:
            warmup_done += 1
    sample_seconds = []
    exposed_wait_seconds = []
    measured_sources = []
    worker_prepare_seconds = []
    hardware_samples = []
    encoded_frames = 0
    started = time.perf_counter()
    while len(sample_seconds) < measured:
        sample_started = time.perf_counter()
        measured_source, frame_count, metadata = consume()
        if frame_count is None:
            rejected += 1
            continue
        sample_seconds.append(time.perf_counter() - sample_started)
        exposed_wait_seconds.append(sample_seconds[-1])
        measured_sources.append(measured_source)
        worker_prepare_seconds.append(float(metadata["worker_prepare_seconds"]))
        encoded_frames += frame_count
        if hardware_sampler is not None:
            hardware_samples.append(hardware_sampler())
        if simulated_compute_seconds:
            time.sleep(simulated_compute_seconds)
            if hardware_sampler is not None:
                hardware_samples.append(hardware_sampler())
    if use_cuda:
        torch.cuda.synchronize()
    wall_elapsed = time.perf_counter() - started
    elapsed = sum(sample_seconds)
    return {
        "warmup": warmup,
        "measured": measured,
        "workers": workers,
        "use_cuda": use_cuda,
        "rejected": rejected,
        "elapsed_seconds": elapsed,
        "wall_elapsed_seconds": wall_elapsed,
        "samples_per_second": measured / elapsed,
        "sample_seconds_median": sorted(sample_seconds)[len(sample_seconds) // 2],
        "sample_seconds_p50": sorted(sample_seconds)[len(sample_seconds) // 2],
        "sample_seconds_p95": sorted(sample_seconds)[max(0, int(len(sample_seconds) * 0.95) - 1)],
        "first_sample_seconds": sample_seconds[0],
        "exposed_wait_seconds_p50": sorted(exposed_wait_seconds)[len(exposed_wait_seconds) // 2],
        "exposed_wait_seconds_p95": sorted(exposed_wait_seconds)[max(0, int(len(exposed_wait_seconds) * 0.95) - 1)],
        "max_exposed_wait_seconds": max(exposed_wait_seconds),
        "simulated_compute_seconds": simulated_compute_seconds,
        "encoded_frames": encoded_frames,
        "view_count": view_count,
        "source_schedule": list(source_schedule),
        "measured_sources": measured_sources,
        "worker_prepare_seconds_p50": sorted(worker_prepare_seconds)[len(worker_prepare_seconds) // 2],
        "worker_prepare_seconds_p95": sorted(worker_prepare_seconds)[max(0, int(len(worker_prepare_seconds) * 0.95) - 1)],
        "hardware_samples": hardware_samples,
        "index_root": str(datasets_root / "kubric-multiview/train/MVTracker_index"),
    }


def materialize_continual_training_data(data_root: Path) -> dict:
    token = os.environ["HF_TOKEN"]
    index_root = data_root / MVKUBRIC_INDEX_RELATIVE
    if not (index_root / "manifest.json").is_file():
        from mvtracker.datasets.kubric_metadata_index import build_kubric_metadata_index

        build_kubric_metadata_index(
            data_root / "datasets/kubric-multiview/train",
            index_root=index_root,
        )
    profile_manifest = _require_existing_profile_data(data_root)
    manifest_path = data_root / "continual-training-data-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != MANIFEST_VERSION
            or manifest.get("mvkubric_revision") != profile_manifest["mvkubric_revision"]
            or manifest.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256
        ):
            raise RuntimeError("existing continual-training data manifest is incompatible")
        _materialize_checkpoint(data_root, token)
        return manifest
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "mvkubric_revision": profile_manifest["mvkubric_revision"],
        "diegesis": profile_manifest["diegesis"],
        "mvkubric": profile_manifest["mvkubric"],
        "checkpoint": _materialize_checkpoint(data_root, token),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def materialize_expanded_continual_training_data(data_root: Path) -> dict:
    """Prepare the versioned 2,000-scene pool for the cached dataset image."""

    token = os.environ["HF_TOKEN"]
    from mvtracker.profiling.modal_mvkubric2000 import materialize_mvkubric2000

    data_root = Path(data_root)
    profile_manifest_path = data_root / "profile-data-manifest.json"
    if not profile_manifest_path.is_file():
        raise FileNotFoundError(profile_manifest_path)
    profile_manifest = json.loads(profile_manifest_path.read_text(encoding="utf-8"))
    if profile_manifest.get("diegesis", {}).get("splits") != EXPECTED_DIEGESIS_SPLITS:
        raise RuntimeError("existing DIEGESIS split manifest is incompatible")
    for split, expected_count in EXPECTED_DIEGESIS_SPLITS.items():
        raw_root = data_root / "datasets/diegesis-mvtracker/TAPVid3D_raw" / split
        cache_root = data_root / "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache" / split
        raw = {path.name for path in raw_root.iterdir() if path.is_dir()}
        cached = {path.name for path in cache_root.iterdir() if path.is_dir()}
        if len(raw) != expected_count or raw != cached:
            raise RuntimeError(f"existing DIEGESIS {split} data is incomplete")

    mvkubric_manifest = materialize_mvkubric2000(data_root, token)
    checkpoint = _materialize_checkpoint(data_root, token)
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "mvkubric_revision": mvkubric_manifest["revision"],
        "diegesis": profile_manifest["diegesis"],
        "mvkubric": mvkubric_manifest,
        "checkpoint": checkpoint,
    }
    manifest_path = data_root / "continual-training-data-manifest.json"
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest
