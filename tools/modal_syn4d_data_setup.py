"""Modal launcher for the bounded Syn4D ``temple_group`` conversion.

The app has three explicit stages: selective source/dependency staging, the
official Blender 4.5 BEDLAM2 conversion, and the T4 sequence-cache converter.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
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
SMPLX_ADDON_PARTS = (
    DATA_ROOT
    / "datasets/syn4d/temple_group/private/smplx_addon_parts"
)
SMPLX_ADDON_BYTES = 387_473_505
BEDLAM_SCRIPTS_ROOT = Path("/opt/syn4d-bedlam2")
SYN4D_VISUALIZER_ROOT = Path("/opt/syn4d-visualizer")
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
            "ca-certificates", "curl", "ffmpeg", "git", "libgl1", "libglib2.0-0",
            "libopenexr-dev", "openexr", "zstd",
        )
        .pip_install(
            "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install(
            "hf-xet==1.1.8", "huggingface-hub==0.30.2",
            "nvidia-dali-cuda120==1.53.0",
            "OpenEXR==3.3.5", "safetensors==0.5.3", "wandb==0.19.9",
            "opencv-python-headless==4.11.0.86", "pandas==2.2.3",
            "scipy==1.15.2", "Pillow==11.1.0", "imageio==2.37.0",
            "tqdm==4.67.1", "pause==0.3", "jupyterlab==4.3.6",
        )
        .run_commands(
            f"mkdir -p {shlex.quote(str(SYN4D_VISUALIZER_ROOT))}",
            *(
                "curl -fsSL "
                + shlex.quote(
                    "https://huggingface.co/datasets/Syn4D/Syn4D/resolve/"
                    "181c6a2da735b216826ab9411b08e0d1d225aced/code/visualizer/"
                    + filename
                )
                + " -o "
                + shlex.quote(str(SYN4D_VISUALIZER_ROOT / filename))
                for filename in ("syn4d_track.py", "base_dataset.py", "utils.py")
            ),
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
    base = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "ca-certificates", "curl", "git", "libgl1", "libglib2.0-0", "libopenexr-dev",
            "libice6", "libsm6", "libx11-6", "libxfixes3", "libxi6",
            "libxkbcommon0", "libxrender1", "libxxf86vm1", "openexr", "xz-utils",
        )
        .pip_install("numpy==2.2.4", "wandb==0.19.9")
        .run_commands(
            f"mkdir -p {shlex.quote(str(BEDLAM_SCRIPTS_ROOT))}",
            f"curl -fsSL {shlex.quote(BLENDER_URL)} | tar -xJ -C /opt",
            download_scripts,
        )
        .env({"BLENDER_BIN": str(BLENDER_BIN)})
    )
    return _clone_source(base)


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
    secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    cpu=8, memory=16 * 1024, ephemeral_disk=512 * 1024,
    timeout=24 * 60 * 60, retries=1, max_containers=MAX_CONTAINERS,
    include_source=False,
)
def download_dependencies_remote() -> dict[str, object]:
    """Stage exactly one source scene and its 20 bodies/20 clothing/60 objects."""

    from huggingface_hub import hf_hub_download
    from mvtracker.profiling.modal_syn4d import (
        SELECTIVE_BEDLAM_ROOT, SYN4D_REPO_ID, SYN4D_REVISION, TEMPLE_GROUP_HF_MAPPING,
        TEMPLE_GROUP_HF_SOURCE, TEMPLE_GROUP_MAPPING,
        TEMPLE_GROUP_OBJECT_ROOT_LOCAL, TEMPLE_GROUP_ROOT,
        TEMPLE_GROUP_SOURCE_BYTES, TEMPLE_GROUP_SOURCE_ROOT,
        temple_group_bedlam_plan, temple_group_object_paths,
        write_temple_group_manifest,
    )

    run = _wandb_run(
        job_type="temple-group-download", name="temple-group-selective-download",
        tags=["download", "cpu"], config={"source_revision": SYN4D_REVISION, **CPU_TAGS},
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
    shutil.copyfile(source_path, source_destination)

    def stage_object(remote_path: Path) -> Path:
        local_path = Path(hf_hub_download(
            repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION,
            filename=remote_path.as_posix(), token=os.environ["HF_TOKEN"],
            local_dir=f"/tmp/syn4d-object-{remote_path.parent.name}",
        ))
        relative = Path(remote_path).relative_to(Path("data/metadata/new_weight_bone"))
        destination = DATA_ROOT / TEMPLE_GROUP_OBJECT_ROOT_LOCAL / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, destination)
        return destination

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        object_files = list(
            pool.map(stage_object, temple_group_object_paths(mapping_destination))
        )
    run.log({"download/object_vertices": len(object_files)})

    bedlam_root = DATA_ROOT / SELECTIVE_BEDLAM_ROOT
    archive_map = json.loads(
        (bedlam_root / "b2_assetdata_download/clothing/npz/archive_map.json").read_text()
    )
    bedlam_plan = temple_group_bedlam_plan(
        mapping_destination.read_text(encoding="utf-8-sig"), archive_map
    )
    write_temple_group_manifest(
        DATA_ROOT / TEMPLE_GROUP_ROOT / "manifest.json",
        source_archive=source_destination,
        mapping=mapping_destination,
        object_files=tuple(object_files),
        bedlam=bedlam_plan,
        bedlam_root=bedlam_root,
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

    from mvtracker.preprocessing.syn4d import temple_group_dependencies
    from mvtracker.profiling.modal_syn4d import (
        SELECTIVE_BEDLAM_ROOT,
        TEMPLE_GROUP_BODY_ROOT,
        TEMPLE_GROUP_MAPPING,
    )

    if processes <= 0:
        raise ValueError("processes must be positive")
    run = _wandb_run(
        job_type="bedlam2-conversion", name="temple-group-bedlam2-conversion",
        tags=["bedlam2", "blender-4.5", "cpu"],
        config={"blender_version": BLENDER_VERSION, "processes": processes, **CPU_TAGS},
    )
    addon_parts = sorted(Path(SMPLX_ADDON_PARTS).glob("part-*"))
    if not addon_parts:
        raise FileNotFoundError(
            f"private SMPL-X add-on parts are required on the Volume: {SMPLX_ADDON_PARTS}"
        )
    addon = Path("/tmp/smplx_blender_addon-1.0.3-20260511.zip")
    with addon.open("wb") as output:
        for part in addon_parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, 8 << 20)
    if addon.stat().st_size != SMPLX_ADDON_BYTES:
        raise RuntimeError(
            f"SMPL-X add-on parts total {addon.stat().st_size} bytes; "
            f"expected {SMPLX_ADDON_BYTES}"
        )
    dependencies = temple_group_dependencies(DATA_ROOT / TEMPLE_GROUP_MAPPING)
    source_motions = (
        DATA_ROOT
        / SELECTIVE_BEDLAM_ROOT
        / "b2_motions_npz_training/motions_npz_training"
    )
    work_root = Path("/tmp/temple-group-bedlam")
    motions_root = work_root / "motions"
    abc_root = work_root / "abc"
    vertices_root = work_root / "vertices"
    shutil.rmtree(work_root, ignore_errors=True)
    motions_root.mkdir(parents=True)
    for motion in dependencies.body_motions:
        shutil.copyfile(source_motions / f"{motion}.npz", motions_root / f"{motion}.npz")
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
    published_root = DATA_ROOT / TEMPLE_GROUP_BODY_ROOT
    for source in vertices_root.rglob("*.npz"):
        destination = published_root / source.relative_to(vertices_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    data_volume.commit()
    result = {"stage1_seconds": stage1_seconds, "stage2_seconds": stage2_seconds,
              "motion_count": len(tuple(motions_root.rglob("*.npz"))),
              "vertex_cache_count": len(tuple(published_root.rglob("*.npz")))}
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
    from mvtracker.profiling.modal_syn4d import (
        SELECTIVE_BEDLAM_ROOT,
        TEMPLE_GROUP_CACHE_ROOT,
        TEMPLE_GROUP_METADATA_ROOT,
        TEMPLE_GROUP_ROOT,
    )

    requested = sequence or "whole-scene"
    run = _wandb_run(
        job_type="syn4d-conversion", name=f"temple-group-{requested}",
        tags=["converter", "t4"], config={"sequence": requested, **MODAL_TAGS},
    )
    started = time.perf_counter()
    source_archive = DATA_ROOT / TEMPLE_GROUP_ROOT / "source/temple_group.tar.zst"
    extracted_root = Path("/tmp/syn4d-temple-group")
    shutil.rmtree(extracted_root, ignore_errors=True)
    extracted_root.mkdir(parents=True)
    subprocess.run(
        ["tar", "-I", "zstd -T0", "-xf", str(source_archive), "-C", str(extracted_root)],
        check=True,
    )
    run.log({"conversion/source_extract": 1})

    def progress(payload: dict[str, object]) -> None:
        metrics = {
            f"conversion/{key}": value
            for key, value in payload.items()
            if isinstance(value, (int, float))
        }
        if metrics:
            run.log(metrics)

    result = convert_temple_group(
        extracted_root / "temple_group",
        DATA_ROOT / TEMPLE_GROUP_METADATA_ROOT,
        DATA_ROOT / SELECTIVE_BEDLAM_ROOT,
        DATA_ROOT / TEMPLE_GROUP_CACHE_ROOT,
        sequence=sequence,
        device="cuda",
        official_visualizer_root=SYN4D_VISUALIZER_ROOT,
        progress=progress,
    )
    result = {**result, "elapsed_seconds": time.perf_counter() - started}
    run.summary.update(result)
    run.finish()
    data_volume.commit()
    run_volume.commit()
    return result


@app.function(
    image=t4_image,
    secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    gpu="T4", cpu=T4_CPU, memory=T4_MEMORY_MIB,
    ephemeral_disk=T4_EPHEMERAL_DISK_MIB,
    timeout=8 * 60 * 60, max_containers=1, include_source=False,
)
@modal.web_server(8000, startup_timeout=120)
def notebook() -> None:
    """Tagged T4 JupyterLab for inspecting and timing committed converter code."""

    subprocess.Popen(
        [
            "jupyter", "lab", "--ip=0.0.0.0", "--port=8000", "--no-browser",
            "--allow-root", "--ServerApp.token=", "--ServerApp.password=",
            "--ServerApp.root_dir=/opt/mvtracker",
        ]
    )


def _preflight(tags: dict[str, str] = MODAL_TAGS) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    app.set_tags(tags)


@app.local_entrypoint(name="download")
def download() -> None:
    _preflight(CPU_TAGS)
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
