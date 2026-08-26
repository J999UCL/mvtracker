"""Build one interactive Waymo long-horizon trajectory Rerun recording."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

import modal


APP_NAME = "jeet-mvtracker-waymo-rerun"
DATA_VOLUME_NAME = "jeet-mvtracker-data-v2"
RUN_VOLUME_NAME = "jeet-mvtracker-runs-v2"
WANDB_SECRET_NAME = "jeet-mvtracker-wandb"
SOURCE_ROOT = Path("/opt/mvtracker")
DATA_ROOT = Path("/mnt/mvtracker-data")
RUN_ROOT = Path("/mnt/mvtracker-runs")
SEGMENT = "9142545919543484617_86_000_106_000"
TFRECORD_NAME = f"segment-{SEGMENT}_with_camera_labels.tfrecord"
ANNOTATION_NAME = f"tapvid3d_{SEGMENT}_2_5AKc-TYQochsSWXpv376cA.npz"
TFRECORD_PATH = DATA_ROOT / "datasets/waymo-visualization/source" / TFRECORD_NAME
OUTPUT_PATH = RUN_ROOT / "waymo-visualization" / f"{SEGMENT}.rrd"
ANNOTATION_URL = (
    "https://storage.googleapis.com/dm-tapnet/tapvid3d/release_files/v1.0/"
    f"drivetrack/{ANNOTATION_NAME}"
)
TAGS = {
    "owner": "jeet",
    "project": "mvtracker",
    "purpose": "profiling",
    "experiment": "waymo-rerun",
    "gpu": "cpu",
}
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _source_commit() -> str:
    commit = os.environ.get("MVTRACKER_MODAL_COMMIT", "")
    if _COMMIT.fullmatch(commit) is None:
        raise RuntimeError("MVTRACKER_MODAL_COMMIT must be one full lowercase Git commit")
    return commit


def _image() -> modal.Image:
    commit = _source_commit()
    clone = (
        f"git init {SOURCE_ROOT} && "
        f"git -C {SOURCE_ROOT} remote add origin https://github.com/J999UCL/mvtracker.git && "
        f"git -C {SOURCE_ROOT} fetch --depth=1 origin {commit} && "
        f"git -C {SOURCE_ROOT} checkout --detach FETCH_HEAD && "
        f'test "$(git -C {SOURCE_ROOT} rev-parse HEAD)" = "{commit}"'
    )
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("ca-certificates", "git", "libgl1", "libglib2.0-0")
        .run_commands(
            "python -m pip install --find-links "
            "https://storage.googleapis.com/jax-releases/jax_releases.html jaxlib==0.4.13"
        )
        .pip_install(
            "waymo-open-dataset-tf-2-12-0==1.6.7",
            "rerun-sdk==0.21.0",
            "requests==2.32.3",
            "wandb==0.19.9",
        )
        .run_commands(clone)
        .env({"MVTRACKER_MODAL_COMMIT": commit, "PYTHONPATH": str(SOURCE_ROOT)})
    )


app = modal.App(APP_NAME, tags=TAGS)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True, version=2)
run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True, version=2)
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME, required_keys=["WANDB_API_KEY"])


@app.function(
    image=_image(),
    cpu=8,
    memory=16 * 1024,
    ephemeral_disk=32 * 1024,
    timeout=2 * 60 * 60,
    retries=0,
    max_containers=1,
    secrets=[wandb_secret],
    volumes={DATA_ROOT: data_volume, RUN_ROOT: run_volume},
)
def build_recording() -> dict[str, object]:
    import requests
    import wandb

    from mvtracker.preprocessing.waymo_rerun import build_waymo_rerun

    if not TFRECORD_PATH.is_file():
        raise FileNotFoundError(TFRECORD_PATH)
    annotation_path = Path("/tmp") / ANNOTATION_NAME
    response = requests.get(ANNOTATION_URL, timeout=120)
    response.raise_for_status()
    annotation_path.write_bytes(response.content)
    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-modal-profiling",
        group="waymo-rerun",
        job_type="visualization",
        name="waymo-rerun-first-scene",
        tags=["modal", "waymo", "rerun", "cpu"],
        config=TAGS,
    )
    result = build_waymo_rerun(TFRECORD_PATH, annotation_path, OUTPUT_PATH)
    run.log({key: value for key, value in result.items() if isinstance(value, (int, float))})
    run.summary.update(result)
    run.finish()
    run_volume.commit()
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def render() -> None:
    print(json.dumps(build_recording.remote(), indent=2, sort_keys=True))
