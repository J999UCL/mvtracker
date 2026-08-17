"""Filesystem build step for the cached Modal training dataset image."""


def install_dataset_image(dataset_version: str) -> None:
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import shutil
    import subprocess

    source = Path("/mnt/mvtracker-data")
    target = Path("/opt/mvtracker-data")
    diegesis_archive = source / (
        "archives/diegesis/"
        "diegesis-81389015a6d713a848a120e34850f360621bcdce.tar.zst"
    )
    mvkubric_archives = [
        source / f"archives/mvkubric/zstd/mvkubric-train-{index:02d}.tar.zst"
        for index in range(4)
    ]
    diegesis_root = target / "source/diegesis"
    mvkubric_root = target / "datasets/kubric-multiview/train"
    diegesis_root.mkdir(parents=True)
    mvkubric_root.mkdir(parents=True)

    def extract_diegesis() -> None:
        subprocess.run(
            [
                "tar", "--extract", "--zstd", "--file", str(diegesis_archive),
                "--directory", str(diegesis_root),
            ],
            check=True,
        )

    def extract_mvkubric(archive: Path) -> None:
        subprocess.run(
            [
                "tar", "--extract", "--zstd", "--strip-components=3",
                "--file", str(archive), "--directory", str(mvkubric_root),
            ],
            check=True,
        )

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(extract_diegesis)]
        futures.extend(executor.submit(extract_mvkubric, path) for path in mvkubric_archives)
        for future in futures:
            future.result()

    for scene in ("101", "102"):
        shutil.copytree(
            source / "datasets/kubric-multiview/train" / scene,
            mvkubric_root / scene,
        )
    for relative in (
        "profile-data-manifest.json",
        "continual-training-data-manifest.json",
        "checkpoints",
        "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache",
        "datasets/kubric-multiview/train/MVTracker_index",
    ):
        src = source / relative
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)

    splits = {
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
    for split, scenes in splits.items():
        split_root = raw_root / split
        split_root.mkdir(parents=True)
        for scene in scenes:
            sequence = diegesis_root / "scenes" / scene / "tracking/sequence"
            (split_root / scene).symlink_to(
                os.path.relpath(sequence, split_root), target_is_directory=True
            )
    (target / "dataset-image.json").write_text(
        json.dumps({"version": dataset_version}) + "\n", encoding="utf-8"
    )
