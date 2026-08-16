"""Immutable Modal data materialization for continual training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from huggingface_hub import hf_hub_download

CHECKPOINT_REPO = "ethz-vlg/mvtracker"
CHECKPOINT_REVISION = "010d5d114e860aae6b2568104927b636cdca01bc"
CHECKPOINT_FILE = "mvtracker_200000_june2025.pth"
CHECKPOINT_SHA256 = "a7fa86f2a7223e3e0aa4c1d3eff0dec5fe8a9227a48572ce943b8e49d8a4f8e6"
MANIFEST_VERSION = 1
EXPECTED_DIEGESIS_SPLITS = {"train": 17, "validation": 2, "test": 2}
EXPECTED_MVKUBRIC_SCENES = {str(scene) for scene in range(900, 1000)}
MVKUBRIC_INDEX_RELATIVE = Path("datasets/kubric-multiview/train/MVTracker_index")
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_RELATIVE_PATH = Path("bundles/continual-training-data.tar")
BUNDLE_MANIFEST_RELATIVE_PATH = Path("bundles/continual-training-data-manifest.json")


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
    if mvkubric != EXPECTED_MVKUBRIC_SCENES:
        raise RuntimeError("existing MV-Kubric micro pool must be exactly scenes 900..999")
    _validate_mvkubric_index(data_root, EXPECTED_MVKUBRIC_SCENES)
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


def _bundle_inputs(data_root: Path, index_relative: Path) -> tuple[Path, ...]:
    """Return archive members, relative to ``data_root``.

    Raw DIEGESIS scenes are symlinks into ``source/diegesis``.  Keeping both
    trees in the archive preserves the existing cache without dereferencing or
    rebuilding it during setup.
    """
    return (
        Path("profile-data-manifest.json"),
        Path("continual-training-data-manifest.json"),
        Path("source/diegesis"),
        Path("datasets/diegesis-mvtracker"),
        Path("datasets/kubric-multiview/train"),
        index_relative,
        Path("checkpoints/mvtracker_200000_june2025.pth"),
    )


def prepare_continual_training_bundle(
    data_root: Path,
    *,
    bundle_path: Path | None = None,
    index_relative: Path = MVKUBRIC_INDEX_RELATIVE,
) -> dict:
    """Create an immutable tar bundle for local-SSD extraction in GPU jobs.

    ``data_root`` is the existing Modal Volume tree.  No source data is
    downloaded or regenerated here; the bundle is a transport representation
    of the already validated DIEGESIS cache and MV-Kubric pool.
    """
    data_root = Path(data_root)
    index_relative = Path(index_relative)
    if index_relative != MVKUBRIC_INDEX_RELATIVE:
        raise ValueError(
            f"index_relative must be {MVKUBRIC_INDEX_RELATIVE}, got {index_relative}"
        )
    profile_manifest = _require_existing_profile_data(data_root)
    index = _validate_mvkubric_index(data_root)
    members = _bundle_inputs(data_root, index_relative)
    missing = [member for member in members if not (data_root / member).exists()]
    if missing:
        raise FileNotFoundError(f"bundle inputs are missing: {', '.join(map(str, missing))}")

    archive = Path(bundle_path) if bundle_path is not None else data_root / BUNDLE_RELATIVE_PATH
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    started = time.perf_counter()
    subprocess.run(
        [
            "tar",
            "--create",
            "--file",
            str(temporary),
            "--directory",
            str(data_root),
            *(str(member) for member in members),
        ],
        check=True,
    )
    os.replace(temporary, archive)
    archive_size = archive.stat().st_size
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "format": "tar",
        "archive": {
            "relative_path": str(archive.relative_to(data_root))
            if archive.is_relative_to(data_root)
            else str(archive),
            "size_bytes": archive_size,
            "sha256": sha256(archive),
        },
        "source_profile_manifest": profile_manifest,
        "inputs": [str(member) for member in members],
        "diegesis": profile_manifest["diegesis"],
        "mvkubric": profile_manifest["mvkubric"],
        "mvkubric_index": index,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = data_root / BUNDLE_MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def extract_continual_training_bundle(
    bundle_root: Path,
    *,
    local_data_root: Path,
    bundle_path: Path | None = None,
    bundle_manifest: dict | None = None,
) -> dict:
    """Extract the validated bundle into a container's local ephemeral disk."""
    bundle_root = Path(bundle_root)
    local_data_root = Path(local_data_root)
    manifest = bundle_manifest
    if manifest is None:
        manifest_path = bundle_root / BUNDLE_MANIFEST_RELATIVE_PATH
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION or manifest.get("format") != "tar":
        raise RuntimeError("unsupported continual-training bundle manifest")
    archive = Path(bundle_path) if bundle_path is not None else bundle_root / manifest["archive"]["relative_path"]
    if not archive.is_file():
        raise FileNotFoundError(f"continual-training bundle archive is missing: {archive}")
    expected_size = int(manifest["archive"]["size_bytes"])
    if archive.stat().st_size != expected_size:
        raise RuntimeError("continual-training bundle archive size mismatch")
    if local_data_root.exists():
        shutil.rmtree(local_data_root)
    local_data_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    subprocess.run(
        ["tar", "--extract", "--file", str(archive), "--directory", str(local_data_root), "--no-same-owner"],
        check=True,
    )
    _require_existing_profile_data(local_data_root)
    return {
        "local_data_root": str(local_data_root),
        "archive_size_bytes": expected_size,
        "extracted_size_bytes": _tree_stats(local_data_root)["size_bytes"],
        "elapsed_seconds": time.perf_counter() - started,
        "mvkubric_index": str(local_data_root / MVKUBRIC_INDEX_RELATIVE),
    }


def profile_encoded_loader(
    data_root: Path,
    *,
    source: str = "diegesis",
    warmup: int = 4,
    measured: int = 32,
    workers: int = 0,
    use_cuda: bool = False,
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
    import itertools
    from omegaconf import OmegaConf
    from types import SimpleNamespace

    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader

    from mvtracker.datasets.tapvid3d_multiview_dataset import (
        CudaPrefetchLoader,
        TapVid3DMultiViewDataset,
    )

    repo_root = Path(__file__).resolve().parents[2]
    config = OmegaConf.merge(
        OmegaConf.load(repo_root / "configs/train.yaml"),
        OmegaConf.load(repo_root / "configs/experiment/diegesis_mvkubric_gt_ddp.yaml"),
    )
    config.datasets.root = str(data_root)
    config.datasets.train.kubric_metadata_index_root = str(
        data_root / "kubric-multiview/train/MVTracker_index"
    )
    if source == "diegesis":
        dataset = TapVid3DMultiViewDataset.from_name(
            config.datasets.train.sources.diegesis.name,
            str(data_root / "diegesis-mvtracker"),
            training_args=config,
            fabric=SimpleNamespace(world_size=1),
            include_scene_ids=list((
                "bathroom01", "bathroom02", "bathroom03", "bedroom02", "bedroom03",
                "bedroom04", "diningroom01", "diningroom03", "diningroom04", "kitchen01",
                "kitchen02", "kitchen03", "kitchen04", "livingroom01", "livingroom03",
                "livingroom04", "livingroom05",
            )),
        )
    else:
        from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset

        dataset = KubricMultiViewDataset.from_name(
            config.datasets.train.sources.mvkubric.name,
            str(data_root),
            training_args=config,
            fabric=SimpleNamespace(world_size=1),
            include_scene_ids=[str(scene) for scene in range(900, 998)],
        )
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA encoded-loader profile requires a visible GPU")
        loader = StatefulDataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=dataset.collate_fn,
            drop_last=True,
            in_order=False,
        )
        iterator = iter(CudaPrefetchLoader(loader))
    else:
        iterator = itertools.cycle(range(dataset.real_len))

    def consume() -> int:
        if use_cuda:
            batch, gotit = next(iterator)
            if not all(gotit):
                raise RuntimeError("encoded-loader profile produced an invalid sample")
            torch.cuda.synchronize()
            return sum(len(sample.jpeg_bytes) for sample in batch.samples)
        index = next(iterator)
        sample, gotit = dataset[index]
        if not gotit:
            raise RuntimeError("encoded-loader profile produced an invalid sample")
        if hasattr(sample, "jpeg_bytes"):
            return len(sample.jpeg_bytes)
        return int(sample.video.shape[0])

    for _ in range(warmup):
        consume()
    sample_seconds = []
    encoded_frames = 0
    started = time.perf_counter()
    for _ in range(measured):
        sample_started = time.perf_counter()
        encoded_frames += consume()
        sample_seconds.append(time.perf_counter() - sample_started)
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "warmup": warmup,
        "measured": measured,
        "workers": workers,
        "use_cuda": use_cuda,
        "elapsed_seconds": elapsed,
        "samples_per_second": measured / elapsed,
        "sample_seconds_median": sorted(sample_seconds)[len(sample_seconds) // 2],
        "sample_seconds_p95": sorted(sample_seconds)[max(0, int(len(sample_seconds) * 0.95) - 1)],
        "encoded_frames": encoded_frames,
        "index_root": str(data_root / "kubric-multiview/train/MVTracker_index"),
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
