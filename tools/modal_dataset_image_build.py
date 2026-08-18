"""CPU-only filesystem stages for the cached Modal dataset image."""

from __future__ import annotations

from pathlib import Path


DATA_ROOT = "/mnt/mvtracker-data"
IMAGE_ROOT = "/opt/mvtracker-data"
ARCHIVE_ROOT = "archives/mvkubric/2000-scenes-v1"
VALIDATION_ROOT = "datasets/kubric-multiview/2000-scenes-v1/validation"
TRAIN_ROOT = "datasets/kubric-multiview/train"
TRAIN_ARCHIVES = (
    "kubric-multiview--train.full.1001-2000.tar.gz",
    "kubric-multiview--train.full.2001-3000.tar.gz",
)
TRAIN_ARCHIVE_STAGES = (
    (TRAIN_ARCHIVES[0], 1001, 1500),
    (TRAIN_ARCHIVES[0], 1501, 2000),
    (TRAIN_ARCHIVES[1], 2001, 2500),
    (TRAIN_ARCHIVES[1], 2501, 3000),
)
TRAIN_SCENES = tuple(str(scene) for scene in range(1001, 3001))
VALIDATION_SCENES = tuple(str(scene) for scene in range(101, 128))


def _copy_tree(source, destination) -> None:
    import shutil

    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination)


def install_dataset_base(dataset_version: str) -> None:
    """Install DIEGESIS, its encoded cache, and the existing checkpoint."""

    import json
    import os
    import subprocess

    source = Path(DATA_ROOT)
    target = Path(IMAGE_ROOT)
    if target.exists():
        raise RuntimeError(f"dataset image root already exists: {target}")
    target.mkdir(parents=True)
    diegesis_archive = source / (
        "archives/diegesis/"
        "diegesis-81389015a6d713a848a120e34850f360621bcdce.tar.zst"
    )
    if not diegesis_archive.is_file():
        raise FileNotFoundError(diegesis_archive)
    diegesis_root = target / "source/diegesis"
    diegesis_root.mkdir(parents=True)
    subprocess.run(
        [
            "tar", "--extract", "--zstd", "--file", str(diegesis_archive),
            "--directory", str(diegesis_root),
        ],
        check=True,
    )
    for relative in (
        "profile-data-manifest.json",
        "continual-training-data-manifest.json",
        "mvkubric2000-data-manifest.json",
        "checkpoints",
        "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache",
    ):
        _copy_tree(source / relative, target / relative)

    split_document = {
        "train": [
            "bathroom01", "bathroom02", "bathroom03", "bedroom02", "bedroom03",
            "bedroom04", "diningroom01", "diningroom03", "diningroom04", "kitchen01",
            "kitchen02", "kitchen03", "kitchen04", "livingroom01", "livingroom03",
            "livingroom04", "livingroom05",
        ],
        "validation": ["bedroom01", "diningroom02"],
        "test": ["bathroom04", "livingroom02"],
    }
    raw_root = target / "datasets/diegesis-mvtracker/TAPVid3D_raw"
    for split, scenes in split_document.items():
        split_root = raw_root / split
        split_root.mkdir(parents=True)
        for scene in scenes:
            sequence = diegesis_root / "scenes" / scene / "tracking/sequence"
            if not sequence.is_dir():
                raise FileNotFoundError(sequence)
            split_root.joinpath(scene).symlink_to(
                os.path.relpath(sequence, split_root), target_is_directory=True
            )
    (target / "dataset-image-base.json").write_text(
        json.dumps({"version": dataset_version, "stage": "base"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def install_mvkubric_archive(
    dataset_version: str,
    archive_name: str,
    scene_start: int,
    scene_end: int,
    archive_size_bytes: int,
    archive_sha256: str,
) -> None:
    """Extract one <=500-scene archive range into the image."""

    import json
    import subprocess

    if (archive_name, scene_start, scene_end) not in TRAIN_ARCHIVE_STAGES:
        raise ValueError(f"unsupported MV-Kubric archive stage: {archive_name} {scene_start}-{scene_end}")
    source = Path(DATA_ROOT) / ARCHIVE_ROOT / archive_name
    target = Path(IMAGE_ROOT) / TRAIN_ROOT
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size != archive_size_bytes:
        raise RuntimeError(
            f"{source}: size {source.stat().st_size} does not match pinned "
            f"size {archive_size_bytes}"
        )
    source_manifest = Path(DATA_ROOT) / "mvkubric2000-data-manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    archive_records = __import__("json").loads(source_manifest.read_text(encoding="utf-8"))["archives"]
    record = next((item for item in archive_records if item["filename"] == archive_name), None)
    if record is None or record.get("sha256") != archive_sha256:
        raise RuntimeError(f"{source}: pinned archive SHA-256 metadata does not match build kwargs")
    target.mkdir(parents=True, exist_ok=True)
    scene_paths = [f"kubric-multiview/train/{scene}" for scene in range(scene_start, scene_end + 1)]
    subprocess.run(
        [
            "tar", "--extract", "--gzip", "--strip-components=3",
            "--file", str(source), "--directory", str(target), "--",
            *scene_paths,
        ],
        check=True,
    )
    expected = set(str(scene) for scene in range(1001, scene_end + 1))
    present = {path.name for path in target.iterdir() if path.is_dir() and path.name.isdigit()}
    if present != expected:
        raise RuntimeError(
            f"{archive_name} stage {scene_start}-{scene_end} produced {len(present)} "
            f"scenes; expected cumulative inventory size {len(expected)}"
        )
    Path(IMAGE_ROOT, f"dataset-image-{scene_start}-{scene_end}.json").write_text(
        json.dumps(
            {
                "version": dataset_version,
                "stage": "mvkubric-archive",
                "archive": archive_name,
                "scene_start": scene_start,
                "scene_end": scene_end,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def install_mvkubric_validation_and_index(dataset_version: str) -> None:
    """Copy validation scenes and build the canonical relocatable index."""

    import importlib.util
    import json
    import shutil

    source = Path(DATA_ROOT) / VALIDATION_ROOT
    train = Path(IMAGE_ROOT) / TRAIN_ROOT
    if not source.is_dir():
        raise FileNotFoundError(source)
    train.mkdir(parents=True, exist_ok=True)
    for scene in VALIDATION_SCENES:
        source_scene = source / scene
        if not source_scene.is_dir():
            raise FileNotFoundError(source_scene)
        shutil.copytree(source_scene, train / scene)

    all_scenes = set(TRAIN_SCENES) | set(VALIDATION_SCENES)
    observed = {path.name for path in train.iterdir() if path.is_dir() and path.name.isdigit()}
    if observed != all_scenes:
        raise RuntimeError("final MV-Kubric inventory is not exactly train 1001-3000 plus validation 101-127")

    index_module_path = Path("/opt/mvtracker/mvtracker/datasets/kubric_metadata_index.py")
    spec = importlib.util.spec_from_file_location("kubric_metadata_index", index_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(index_module_path)
    index_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(index_module)
    index_root = train / "MVTracker_index"
    index_module.build_kubric_metadata_index(train, index_root=index_root, overwrite=True)
    index = index_module.KubricMetadataIndex(index_root)
    if set(index.scenes) != all_scenes:
        raise RuntimeError("canonical MV-Kubric metadata index does not cover final inventory")
    index.validate_source(train)
    (Path(IMAGE_ROOT) / "dataset-image.json").write_text(
        json.dumps(
            {
                "version": dataset_version,
                "train_scene_ids": list(TRAIN_SCENES),
                "validation_scene_ids": list(VALIDATION_SCENES),
                "scene_count": len(all_scenes),
                "index": str(Path(TRAIN_ROOT) / "MVTracker_index"),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def install_dataset_image(dataset_version: str) -> None:
    """Compatibility entrypoint for callers that have not adopted layering."""

    import json

    install_dataset_base(dataset_version)
    records = json.loads(
        (Path(DATA_ROOT) / "mvkubric2000-data-manifest.json").read_text(encoding="utf-8")
    )["archives"]
    for archive_name, scene_start, scene_end in TRAIN_ARCHIVE_STAGES:
        record = next(item for item in records if item["filename"] == archive_name)
        install_mvkubric_archive(
            dataset_version,
            archive_name,
            scene_start,
            scene_end,
            record["size_bytes"],
            record["sha256"],
        )
    install_mvkubric_validation_and_index(dataset_version)
