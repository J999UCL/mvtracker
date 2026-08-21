"""Modal launcher for the bounded Syn4D ``temple_group`` conversion.

The app has three explicit stages: selective source/dependency staging, the
official Blender 4.5 BEDLAM2 conversion, and the T4 sequence-cache converter.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import modal

from modal_training_profile import (
    BASE_TAGS,
    DATA_ROOT,
    DATA_VOLUME_NAME,
    HF_SECRET_NAME,
    RUN_ROOT,
    RUN_VOLUME_NAME,
    WANDB_SECRET_NAME,
    _source_commit,
)
from mvtracker.profiling.modal_continual_training import (
    WANDB_ENTITY,
    preflight_active_containers,
    require_pushed_main_commit,
)


APP_NAME = "jeet-mvtracker-syn4d-temple-group"
WANDB_PROJECT = "mvtracker-modal-profiling"
WANDB_GROUP = "syn4d-temple-group"
MAX_CONTAINERS = 1
T4_CPU = 16
T4_MEMORY_MIB = 128 * 1024
T4_EPHEMERAL_DISK_MIB = 1 * 1024 * 1024
CPU_CONVERSION_PROCESSES = 6
BLENDER_VERSION = "4.5.0"
BLENDER_ARCHIVE = f"blender-{BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL = f"https://download.blender.org/release/Blender4.5/{BLENDER_ARCHIVE}"
BLENDER_ROOT = Path(f"/opt/blender-{BLENDER_VERSION}-linux-x64")
BLENDER_BIN = BLENDER_ROOT / "blender"
SMPLX_ADDON_ZIP = DATA_ROOT / "bedlam2" / "smplx_blender_addon.zip"
BEDLAM_SCRIPTS_ROOT = Path("/opt/syn4d-bedlam2")
BEDLAM_README_URL = (
    "https://huggingface.co/datasets/Syn4D/Syn4D/resolve/"
    "181c6a2da735b216826ab9411b08e0d1d225aced/code/bedlam2/"
)
MODAL_TAGS = {**BASE_TAGS, "experiment": "temple-group-conversion", "gpu": "t4"}
CPU_TAGS = {**BASE_TAGS, "experiment": "temple-group-bedlam2", "gpu": "cpu"}

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True, version=2)
run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME, required_keys=["WANDB_API_KEY"])
bedlam_secret = modal.Secret.from_name(
    "jeet-mvtracker-bedlam", required_keys=["BEDLAM_EMAIL", "BEDLAM_PASSWORD"]
)


def _clone_source(image: modal.Image) -> modal.Image:
    commit = _source_commit()
    clone = (
        "git init /opt/mvtracker && "
        "git -C /opt/mvtracker remote add origin https://github.com/J999UCL/mvtracker.git && "
        f"git -C /opt/mvtracker fetch --depth=1 origin {commit} && "
        "git -C /opt/mvtracker checkout --detach FETCH_HEAD && "
        f'test "$(git -C /opt/mvtracker rev-parse HEAD)" = "{commit}"'
    )
    return image.run_commands(clone).env(
        {"MVTRACKER_MODAL_COMMIT": commit, "PYTHONPATH": "/opt/mvtracker:/opt/mvtracker/tools"}
    )


def _t4_image() -> modal.Image:
    """Lean CUDA image; training-only extensions are deliberately absent."""

    base = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.10"
        )
        .apt_install(
            "ca-certificates", "ffmpeg", "git", "libgl1", "libglib2.0-0",
            "libopenexr-dev", "openexr", "zstd",
        )
        .pip_install(
            "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install(
            "hf-xet==1.1.8", "huggingface-hub==0.30.2",
            "nvidia-dali-cuda120==1.53.0",
            "nvidia-nvimgcodec-cu12[nvtiff]==0.7.0.11",
            "nvidia-libnvcomp-cu12==5.1.0.21", "OpenEXR==3.3.5",
            "safetensors==0.5.3", "wandb==0.19.9",
        )
    )
    return _clone_source(base).env(
        {"DALI_DISABLE_NVML": "1", "HF_HOME": "/tmp/huggingface", "HF_XET_CACHE": "/tmp/huggingface/xet"}
    )


def _blender_image() -> modal.Image:
    """CPU image carrying the official Syn4D Blender batch scripts."""

    script_names = (
        "smplx_anim_to_alembic.py", "smplx_anim_to_alembic_batch.py",
        "smplx_anim_to_objs.py", "smplx_anim_to_objs_batch.py",
        "supervised_blender_batch.py",
    )
    download_scripts = " && ".join(
        "curl -fsSL " + shlex.quote(BEDLAM_README_URL + name) + " -o "
        + shlex.quote(str(BEDLAM_SCRIPTS_ROOT / name))
        for name in script_names
    )
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "ca-certificates", "curl", "libgl1", "libglib2.0-0", "libopenexr-dev",
            "libxi6", "libxrender1", "libxxf86vm1", "openexr", "xz-utils",
        )
        .pip_install("numpy==2.2.4", "wandb==0.19.9")
        .run_commands(
            f"mkdir -p {shlex.quote(str(BEDLAM_SCRIPTS_ROOT))}",
            f"curl -fsSL {shlex.quote(BLENDER_URL)} | tar -xJ -C /opt",
            download_scripts,
        )
        .env({"BLENDER_BIN": str(BLENDER_BIN)})
    )


app = modal.App(APP_NAME, tags=MODAL_TAGS)
t4_image = _t4_image()
blender_image = _blender_image()


def _wandb_run(*, job_type: str, name: str, tags: list[str], config: dict):
    import wandb

    return wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT, group=WANDB_GROUP,
        job_type=job_type, name=name,
        tags=["modal", "syn4d", "temple_group", *tags],
        config={**BASE_TAGS, **config},
    )


@app.function(
    image=t4_image,
    secrets=[hf_secret, wandb_secret, bedlam_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    cpu=T4_CPU, memory=T4_MEMORY_MIB, ephemeral_disk=T4_EPHEMERAL_DISK_MIB,
    timeout=24 * 60 * 60, retries=1, max_containers=MAX_CONTAINERS,
    include_source=False,
)
def download_dependencies_remote() -> dict[str, object]:
    """Stage exactly one source scene and its 20 bodies/20 clothing/60 objects."""

    from huggingface_hub import hf_hub_download
    import requests

    from mvtracker.profiling.modal_syn4d import (
        SYN4D_REPO_ID, SYN4D_REVISION, TEMPLE_GROUP_HF_MAPPING,
        TEMPLE_GROUP_HF_SOURCE, TEMPLE_GROUP_MAPPING,
        TEMPLE_GROUP_OBJECT_ROOT_LOCAL, TEMPLE_GROUP_ROOT,
        TEMPLE_GROUP_SOURCE_BYTES, TEMPLE_GROUP_SOURCE_ROOT,
        temple_group_bedlam_plan, temple_group_object_paths,
        write_temple_group_manifest,
    )
    from mvtracker.profiling.modal_syn4d_bedlam import (
        CLOTHING_SOURCE_ROOT, download_body_motions,
        download_sparse_clothing_tar, download_text,
    )

    run = _wandb_run(
        job_type="temple-group-download", name="temple-group-selective-download",
        tags=["download", "t4"], config={"source_revision": SYN4D_REVISION, **MODAL_TAGS},
    )
    started = time.perf_counter()
    root = DATA_ROOT / TEMPLE_GROUP_ROOT
    root.mkdir(parents=True, exist_ok=True)
    mapping_source = Path(hf_hub_download(
        repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION,
        filename=TEMPLE_GROUP_HF_MAPPING, token=os.environ["HF_TOKEN"],
        local_dir="/tmp/syn4d-mapping",
    ))
    mapping_destination = DATA_ROOT / TEMPLE_GROUP_MAPPING
    mapping_destination.parent.mkdir(parents=True, exist_ok=True)
    mapping_destination.write_bytes(mapping_source.read_bytes())
    source_destination = DATA_ROOT / TEMPLE_GROUP_SOURCE_ROOT / "temple_group.tar.zst"
    source_destination.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(hf_hub_download(
        repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION,
        filename=TEMPLE_GROUP_HF_SOURCE, token=os.environ["HF_TOKEN"],
        local_dir="/tmp/syn4d-source",
    ))
    if source_path.stat().st_size != TEMPLE_GROUP_SOURCE_BYTES:
        raise RuntimeError("temple_group source archive size does not match pin")
    source_destination.write_bytes(source_path.read_bytes())

    object_files = []
    for remote_path in temple_group_object_paths(mapping_destination):
        local_path = Path(hf_hub_download(
            repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION,
            filename=remote_path.as_posix(), token=os.environ["HF_TOKEN"],
            local_dir="/tmp/syn4d-object",
        ))
        relative = Path(remote_path).relative_to(Path("data/metadata/new_weight_bone"))
        destination = DATA_ROOT / TEMPLE_GROUP_OBJECT_ROOT_LOCAL / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(local_path.read_bytes())
        object_files.append(destination)
    run.log({"download/object_vertices": len(object_files)})

    with requests.Session() as session:
        archive_map = json.loads(download_text(
            session, f"{CLOTHING_SOURCE_ROOT}/archive_map.json",
            os.environ["BEDLAM_EMAIL"], os.environ["BEDLAM_PASSWORD"],
        ))
        checksums = download_text(
            session, f"{CLOTHING_SOURCE_ROOT}/checksum.xxh128",
            os.environ["BEDLAM_EMAIL"], os.environ["BEDLAM_PASSWORD"],
        )
    bedlam_plan = temple_group_bedlam_plan(
        mapping_destination.read_text(encoding="utf-8-sig"), archive_map
    )
    download_body_motions(
        DATA_ROOT / TEMPLE_GROUP_ROOT / "bedlam2", bedlam_plan["motions"],
        os.environ["BEDLAM_EMAIL"], os.environ["BEDLAM_PASSWORD"],
    )
    for archive, members in bedlam_plan["required_members"].items():
        download_sparse_clothing_tar(
            DATA_ROOT / TEMPLE_GROUP_ROOT / "bedlam2", archive, members,
            os.environ["BEDLAM_EMAIL"], os.environ["BEDLAM_PASSWORD"],
        )
    (DATA_ROOT / TEMPLE_GROUP_ROOT / "bedlam2" / "source_archives.xxh128").write_text(
        checksums, encoding="utf-8"
    )
    write_temple_group_manifest(
        DATA_ROOT / TEMPLE_GROUP_ROOT / "manifest.json",
        source_archive=source_destination,
        mapping=mapping_destination,
        object_files=tuple(object_files),
        bedlam=bedlam_plan,
    )
    data_volume.commit()
    result = {
        "scene": "temple_group", "source_archive_bytes": source_destination.stat().st_size,
        "object_vertex_count": len(object_files), "body_count": int(bedlam_plan["body_count"]),
        "clothing_count": int(bedlam_plan["clothing_count"]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    run.summary.update(result)
    run.finish()
    return result


@app.function(
    image=blender_image, secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    cpu=64, memory=128 * 1024, ephemeral_disk=T4_EPHEMERAL_DISK_MIB,
    timeout=24 * 60 * 60, retries=0, max_containers=MAX_CONTAINERS,
    include_source=False,
)
def convert_bedlam_remote(processes: int = CPU_CONVERSION_PROCESSES) -> dict[str, object]:
    """Run the official Blender 4.5 Alembic -> vertex NPZ pipeline."""

    from mvtracker.profiling.modal_syn4d import TEMPLE_GROUP_ROOT

    if processes <= 0:
        raise ValueError("processes must be positive")
    run = _wandb_run(
        job_type="bedlam2-conversion", name="temple-group-bedlam2-conversion",
        tags=["bedlam2", "blender-4.5", "cpu"],
        config={"blender_version": BLENDER_VERSION, "processes": processes, **CPU_TAGS},
    )
    addon = Path(SMPLX_ADDON_ZIP)
    if not addon.is_file():
        raise FileNotFoundError(f"private SMPL-X add-on is required on the Volume: {addon}")
    metadata_root = DATA_ROOT / TEMPLE_GROUP_ROOT / "bedlam2"
    motions_root = metadata_root / "datasets/syn4d/v1-stride1-12train-4validation/metadata/b2_motions_npz_training/motions_npz_training"
    abc_root = metadata_root / "bedlam2_smpl_abc"
    vertices_root = metadata_root / "bedlam2_smpl_npz"
    install = (
        f"{shlex.quote(str(BLENDER_BIN))} --background --python-expr "
        + shlex.quote(
            "import bpy; "
            f"bpy.ops.preferences.addon_install(filepath={str(addon)!r}, overwrite=True); "
            "bpy.ops.wm.save_userpref(); "
            "assert hasattr(bpy.context.window_manager, 'smplx_tool')"
        )
    )
    subprocess.run(install, shell=True, check=True)
    stage1 = ["python3", str(BEDLAM_SCRIPTS_ROOT / "smplx_anim_to_alembic_batch.py"),
              str(motions_root), str(abc_root), str(processes), "--blender", str(BLENDER_BIN),
              "--timeout-seconds", "600", "--retries", "1", "--skip-existing"]
    stage2 = ["python3", str(BEDLAM_SCRIPTS_ROOT / "smplx_anim_to_objs_batch.py"),
              str(abc_root), str(vertices_root), str(processes), "--blender", str(BLENDER_BIN),
              "--timeout-seconds", "600", "--retries", "1", "--skip-existing"]
    started = time.perf_counter()
    subprocess.run(stage1, check=True)
    stage1_seconds = time.perf_counter() - started
    run.log({"bedlam/stage1_seconds": stage1_seconds, "bedlam/stage1": 1})
    stage2_started = time.perf_counter()
    subprocess.run(stage2, check=True)
    stage2_seconds = time.perf_counter() - stage2_started
    run.log({"bedlam/stage2_seconds": stage2_seconds, "bedlam/stage2": 1})
    run_volume.commit()
    result = {"stage1_seconds": stage1_seconds, "stage2_seconds": stage2_seconds,
              "motion_count": len(tuple(motions_root.rglob("*.npz"))),
              "vertex_cache_count": len(tuple(vertices_root.rglob("*.npz")))}
    run.summary.update(result)
    run.finish()
    return result


@app.function(
    image=t4_image, secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    gpu="T4", cpu=T4_CPU, memory=T4_MEMORY_MIB, ephemeral_disk=T4_EPHEMERAL_DISK_MIB,
    timeout=24 * 60 * 60, retries=0, max_containers=MAX_CONTAINERS,
    include_source=False,
)
def convert_temple_group_remote(sequence: str | None = None) -> dict[str, object]:
    """Convert the whole scene or one sequence through the shared core."""

    from mvtracker.preprocessing.syn4d import convert_temple_group
    from mvtracker.profiling.modal_syn4d import TEMPLE_GROUP_ROOT

    requested = sequence or "whole-scene"
    run = _wandb_run(
        job_type="syn4d-conversion", name=f"temple-group-{requested}",
        tags=["converter", "t4"], config={"sequence": requested, **MODAL_TAGS},
    )
    started = time.perf_counter()
    source_root = DATA_ROOT / TEMPLE_GROUP_ROOT / "source"
    source_archive = source_root / "temple_group.tar.zst"
    extracted_marker = source_root / ".extracted"
    if not extracted_marker.is_file():
        subprocess.run(
            ["tar", "-I", "zstd", "-xf", str(source_archive), "-C", str(source_root)],
            check=True,
        )
        extracted_marker.write_text("complete\n", encoding="utf-8")
        run.log({"conversion/source_extract": 1})
    result = convert_temple_group(
        DATA_ROOT / TEMPLE_GROUP_ROOT, DATA_ROOT / TEMPLE_GROUP_ROOT / "cache",
        sequence=sequence, device="cuda",
        progress=lambda payload: run.log({f"conversion/{key}": value for key, value in payload.items()}),
    )
    result = {**result, "elapsed_seconds": time.perf_counter() - started}
    run.summary.update(result)
    run.finish()
    data_volume.commit()
    run_volume.commit()
    return result


def _preflight(tags: dict[str, str] = MODAL_TAGS) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    app.set_tags(tags)


@app.local_entrypoint(name="download")
def download() -> None:
    _preflight()
    print(json.dumps(download_dependencies_remote.remote(), indent=2, sort_keys=True))


@app.local_entrypoint(name="convert-bedlam")
def convert_bedlam(processes: int = CPU_CONVERSION_PROCESSES) -> None:
    _preflight(CPU_TAGS)
    print(json.dumps(convert_bedlam_remote.remote(processes), indent=2, sort_keys=True))


@app.local_entrypoint(name="convert")
def convert(sequence: str = "") -> None:
    _preflight()
    print(json.dumps(convert_temple_group_remote.remote(sequence or None), indent=2, sort_keys=True))


@app.local_entrypoint(name="temple-group")
def temple_group(processes: int = CPU_CONVERSION_PROCESSES) -> None:
    _preflight()
    download_dependencies_remote.remote()
    convert_bedlam_remote.remote(processes)
    print(json.dumps(convert_temple_group_remote.remote(), indent=2, sort_keys=True))


if __name__ == "__main__":
    print("Use: modal run tools/modal_syn4d_data_setup.py::<download|convert-bedlam|convert|temple-group>")
