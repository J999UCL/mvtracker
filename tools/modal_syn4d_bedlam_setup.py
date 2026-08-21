"""Fetch only the BEDLAM2 members required by the Syn4D selection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from modal_training_profile import (
    BASE_TAGS,
    DATA_ROOT,
    HF_SECRET_NAME,
    WANDB_SECRET_NAME,
    _source_commit,
    data_volume,
)
from mvtracker.profiling.modal_continual_training import (
    WANDB_ENTITY,
    preflight_active_containers,
    require_pushed_main_commit,
)


APP_NAME = "jeet-mvtracker-syn4d-bedlam-selective"
MODAL_TAGS = {
    **BASE_TAGS,
    "experiment": "syn4d-bedlam-selective-dependencies",
    "gpu": "cpu",
}
WANDB_PROJECT = "mvtracker-modal-profiling"
WANDB_GROUP = "syn4d-stride1-12train-4validation"


def _bedlam_credentials() -> dict[str, str]:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[7:]
        key, separator, value = line.partition("=")
        if separator and key in {"BEDLAM_EMAIL", "BEDLAM_PASSWORD"}:
            values[key] = value.strip().strip("\"'")
    missing = {"BEDLAM_EMAIL", "BEDLAM_PASSWORD"}.difference(values)
    if missing:
        raise RuntimeError(f"missing BEDLAM variables in {env_path}: {sorted(missing)}")
    return values


def _image() -> modal.Image:
    commit = _source_commit()
    clone = (
        "git init /opt/mvtracker && "
        "git -C /opt/mvtracker remote add origin https://github.com/J999UCL/mvtracker.git && "
        f"git -C /opt/mvtracker fetch --depth=1 origin {commit} && "
        "git -C /opt/mvtracker checkout --detach FETCH_HEAD && "
        f'test "$(git -C /opt/mvtracker rev-parse HEAD)" = "{commit}"'
    )
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ca-certificates", "git")
        .pip_install(
            "hf-xet==1.1.8",
            "huggingface-hub==0.30.2",
            "requests==2.32.3",
            "wandb==0.19.9",
        )
        .run_commands(clone)
        .env(
            {
                "HF_HOME": "/tmp/huggingface",
                "HF_XET_CACHE": "/tmp/huggingface/xet",
                "MVTRACKER_MODAL_COMMIT": commit,
                "PYTHONPATH": "/opt/mvtracker:/opt/mvtracker/tools",
            }
        )
    )


app = modal.App(APP_NAME, tags=MODAL_TAGS)
image = _image()
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])
wandb_secret = modal.Secret.from_name(
    WANDB_SECRET_NAME, required_keys=["WANDB_API_KEY"]
)
bedlam_secret = modal.Secret.from_dict(_bedlam_credentials())


def _inputs() -> tuple[str, dict[str, list[str]], str]:
    import requests
    from huggingface_hub import hf_hub_download

    from mvtracker.profiling.modal_syn4d_bedlam import (
        CLOTHING_SOURCE_ROOT,
        SYN4D_MAPPING_PATH,
        SYN4D_REPO_ID,
        SYN4D_REVISION,
        download_text,
    )

    mapping_path = hf_hub_download(
        repo_id=SYN4D_REPO_ID,
        repo_type="dataset",
        revision=SYN4D_REVISION,
        filename=SYN4D_MAPPING_PATH,
        token=os.environ["HF_TOKEN"],
    )
    with requests.Session() as session:
        archive_map = json.loads(
            download_text(
                session,
                f"{CLOTHING_SOURCE_ROOT}/archive_map.json",
                os.environ["BEDLAM_EMAIL"],
                os.environ["BEDLAM_PASSWORD"],
            )
        )
        checksums = download_text(
            session,
            f"{CLOTHING_SOURCE_ROOT}/checksum.xxh128",
            os.environ["BEDLAM_EMAIL"],
            os.environ["BEDLAM_PASSWORD"],
        )
    return Path(mapping_path).read_text(encoding="utf-8-sig"), archive_map, checksums


@app.function(
    image=image,
    secrets=[hf_secret, wandb_secret, bedlam_secret],
    cpu=2,
    memory=4096,
    ephemeral_disk=8 * 1024,
    timeout=30 * 60,
    retries=0,
    max_containers=1,
    include_source=False,
)
def probe_remote() -> dict[str, object]:
    import wandb

    from mvtracker.profiling.modal_syn4d_bedlam import probe_dependencies

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="bedlam-range-probe",
        name="syn4d-bedlam-selective-range-probe",
        tags=["modal", "cpu", "syn4d", "bedlam2", "range", *BASE_TAGS.values()],
        config={"source_commit": _source_commit(), **MODAL_TAGS},
    )
    try:
        mapping, archive_map, _ = _inputs()
        result = probe_dependencies(
            mapping,
            archive_map,
            os.environ["BEDLAM_EMAIL"],
            os.environ["BEDLAM_PASSWORD"],
        )
        run.log(result)
        run.summary.update(result)
        run.finish()
        print(f"BEDLAM_PROBE {json.dumps(result, sort_keys=True)}", flush=True)
        return result
    except BaseException:
        run.finish(exit_code=1)
        raise


@app.function(
    image=image,
    secrets=[hf_secret, wandb_secret, bedlam_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=8,
    memory=16384,
    ephemeral_disk=128 * 1024,
    timeout=12 * 60 * 60,
    retries=1,
    max_containers=1,
    include_source=False,
)
def download_remote(workers: int = 4) -> dict[str, object]:
    import wandb

    from mvtracker.profiling.modal_syn4d_bedlam import materialize_dependencies

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="bedlam-selective-download",
        name="syn4d-bedlam-selective-dependencies",
        tags=["modal", "cpu", "syn4d", "bedlam2", "selective", *BASE_TAGS.values()],
        config={
            "source_commit": _source_commit(),
            "workers": workers,
            **MODAL_TAGS,
        },
    )

    completed = 0

    def progress(payload: dict[str, object]) -> None:
        nonlocal completed
        completed += 1
        metrics = {
            "dependencies/completed_units": completed,
            "dependencies/latest_size_bytes": int(payload.get("size_bytes", 0)),
        }
        run.log(metrics)
        printable = {key: value for key, value in payload.items() if key != "member_sizes"}
        print(f"BEDLAM_DOWNLOAD {json.dumps(printable, sort_keys=True)}", flush=True)

    try:
        mapping, archive_map, checksums = _inputs()
        manifest = materialize_dependencies(
            DATA_ROOT,
            mapping,
            archive_map,
            checksums,
            os.environ["BEDLAM_EMAIL"],
            os.environ["BEDLAM_PASSWORD"],
            workers=workers,
            progress=progress,
            commit=data_volume.commit,
        )
        run.summary.update(
            {
                "status": "complete",
                "sequence_count": manifest["sequence_count"],
                "body_motion_count": manifest["body_motion_count"],
                "clothing_member_count": manifest["clothing_member_count"],
                "clothing_archive_count": manifest["clothing_archive_count"],
                "elapsed_seconds": manifest["elapsed_seconds"],
            }
        )
        run.finish()
        return manifest
    except BaseException:
        run.finish(exit_code=1)
        raise


def _preflight() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    app.set_tags(MODAL_TAGS)


@app.local_entrypoint(name="probe")
def probe() -> None:
    _preflight()
    print(json.dumps(probe_remote.remote(), indent=2, sort_keys=True))


@app.local_entrypoint(name="download")
def download(workers: int = 4) -> None:
    _preflight()
    print(json.dumps(download_remote.remote(workers), indent=2, sort_keys=True))
