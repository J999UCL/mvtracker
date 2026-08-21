"""Modal setup and bounded pilot profiling for Syn4D lab_bald/seq_000000.

The setup stages one Xet archive, one mapping, one BEDLAM body/clothing pair,
and only the referenced public object vertices.  Conversion is intentionally
single-sequence; the generic Syn4D converter owns the output layout.
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


APP_NAME = "jeet-mvtracker-syn4d-lab-bald"
WANDB_PROJECT = "mvtracker-modal-profiling"
WANDB_GROUP = "syn4d-lab-bald"
MAX_CONTAINERS = 1
T4_CPU = 8
T4_MEMORY_MIB = 32 * 1024
T4_EPHEMERAL_DISK_MIB = 1 * 1024 * 1024
CPU_CONVERSION_PROCESSES = 6
BLENDER_VERSION = "4.5.0"
BLENDER_ARCHIVE = f"blender-{BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL = f"https://download.blender.org/release/Blender4.5/{BLENDER_ARCHIVE}"
BLENDER_ROOT = Path(f"/opt/blender-{BLENDER_VERSION}-linux-x64")
BLENDER_BIN = BLENDER_ROOT / "blender"
SMPLX_ADDON_PARTS = (
    DATA_ROOT / "datasets/syn4d/temple_group/private/smplx_addon_parts"
)
SMPLX_ADDON_BYTES = 387_473_505
BEDLAM_SCRIPTS_ROOT = Path("/opt/syn4d-bedlam2")
SYN4D_VISUALIZER_ROOT = Path("/opt/syn4d-visualizer")
BEDLAM_README_URL = (
    "https://huggingface.co/datasets/Syn4D/Syn4D/resolve/"
    "181c6a2da735b216826ab9411b08e0d1d225aced/code/bedlam2/"
)
MODAL_TAGS = {**BASE_TAGS, "experiment": "lab-bald-conversion", "gpu": "t4"}
CPU_TAGS = {**BASE_TAGS, "experiment": "lab-bald-setup", "gpu": "cpu"}
SHARD_A_SEQUENCES = tuple(f"seq_{index:06d}" for index in range(1, 10))
SHARD_B_SEQUENCES = tuple(f"seq_{index:06d}" for index in range(10, 20))

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
            "nvidia-dali-cuda120==1.53.0", "OpenEXR==3.3.5", "safetensors==0.5.3",
            "wandb==0.19.9", "opencv-python-headless==4.11.0.86", "pandas==2.2.3",
            "scipy==1.15.2", "matplotlib==3.10.1", "Pillow==11.1.0", "imageio==2.37.0",
            "pypng==0.20220715.0", "kornia==0.7.3", "mediapy==1.2.0",
            "rerun-sdk==0.21.0", "tqdm==4.67.1", "pause==0.3", "jupyterlab==4.3.6",
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


def _cpu_image() -> modal.Image:
    base = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ca-certificates", "git", "libgl1", "libglib2.0-0", "zstd")
        .pip_install(
            "torch==2.7.1", "torchvision==0.22.1",
            index_url="https://download.pytorch.org/whl/cpu",
        )
        .pip_install(
            "numpy==2.2.4", "wandb==0.19.9", "opencv-python-headless==4.11.0.86",
            "scipy==1.15.2", "matplotlib==3.10.1", "Pillow==11.1.0",
            "pypng==0.20220715.0", "kornia==0.7.3", "mediapy==1.2.0",
            "rerun-sdk==0.21.0", "pandas==2.2.3",
        )
    )
    return _clone_source(base)


def _blender_image() -> modal.Image:
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
            "libice6", "libsm6", "libx11-6", "libxfixes3", "libxi6", "libxkbcommon0",
            "libxrender1", "libxxf86vm1", "openexr", "xz-utils",
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
cpu_image = _cpu_image()
blender_image = _blender_image()


def _wandb_run(*, job_type: str, name: str, tags: list[str], config: dict):
    import wandb

    return wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT, group=WANDB_GROUP,
        job_type=job_type, name=name,
        tags=["modal", "syn4d", "lab_bald", *tags],
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
    """Reuse seq0 staging and add only missing all-sequence dependencies."""

    from huggingface_hub import hf_hub_download
    from mvtracker.profiling.modal_syn4d import (
        SELECTIVE_BEDLAM_ROOT, SYN4D_HF_MAPPING, SYN4D_HF_SOURCE, SYN4D_MAPPING,
        SYN4D_METADATA_ROOT, SYN4D_OBJECT_ROOT_LOCAL, SYN4D_REPO_ID,
        SYN4D_REVISION, SYN4D_ROOT, SYN4D_SOURCE_BYTES, SYN4D_SOURCE_ROOT,
        scene_bedlam_plan, scene_object_paths, write_scene_manifest,
    )

    run = _wandb_run(
        job_type="lab-bald-download", name="lab-bald-all-sequences-selective-download",
        tags=["download", "cpu"], config={"source_revision": SYN4D_REVISION, **CPU_TAGS},
    )
    started = time.perf_counter()
    root = DATA_ROOT / SYN4D_ROOT
    root.mkdir(parents=True, exist_ok=True)
    mapping_destination = DATA_ROOT / SYN4D_MAPPING
    mapping_destination.parent.mkdir(parents=True, exist_ok=True)
    if not mapping_destination.is_file():
        mapping_source = Path(hf_hub_download(
            repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION,
            filename=SYN4D_HF_MAPPING, token=os.environ["HF_TOKEN"], local_dir="/tmp/syn4d-mapping",
        ))
        mapping_destination.write_bytes(mapping_source.read_bytes())
    source_destination = DATA_ROOT / SYN4D_SOURCE_ROOT / "lab_bald.tar.zst"
    source_destination.parent.mkdir(parents=True, exist_ok=True)
    if not source_destination.is_file() or source_destination.stat().st_size != SYN4D_SOURCE_BYTES:
        source_path = Path(hf_hub_download(
            repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION,
            filename=SYN4D_HF_SOURCE, token=os.environ["HF_TOKEN"], local_dir="/tmp/syn4d-source",
        ))
        if source_path.stat().st_size != SYN4D_SOURCE_BYTES:
            raise RuntimeError("lab_bald source archive size does not match pin")
        shutil.copyfile(source_path, source_destination)

    def stage_object(remote_path: Path) -> Path:
        relative = remote_path.relative_to(Path("data/metadata/new_weight_bone"))
        destination = DATA_ROOT / SYN4D_OBJECT_ROOT_LOCAL / relative
        if destination.is_file():
            return destination
        local_path = Path(hf_hub_download(
            repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION,
            filename=remote_path.as_posix(), token=os.environ["HF_TOKEN"],
            local_dir=f"/tmp/syn4d-object-{remote_path.parent.name}",
        ))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, destination)
        return destination

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        object_files = tuple(pool.map(stage_object, scene_object_paths(mapping_destination)))
    bedlam_root = DATA_ROOT / SELECTIVE_BEDLAM_ROOT
    archive_map = json.loads(
        (bedlam_root / "b2_assetdata_download/clothing/npz/archive_map.json").read_text()
    )
    bedlam_plan = scene_bedlam_plan(mapping_destination, archive_map)
    clothing_root = DATA_ROOT / SYN4D_METADATA_ROOT / "b2_assetdata_download/clothing/npz"
    clothing_root.mkdir(parents=True, exist_ok=True)
    for archive in bedlam_plan["required_members"]:
        source = bedlam_root / "b2_assetdata_download/clothing/npz" / f"{archive}.tar"
        destination = clothing_root / source.name
        if not destination.is_file():
            shutil.copyfile(source, destination)
    (clothing_root / "archive_map.json").write_text(
        json.dumps(bedlam_plan["required_members"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_scene_manifest(
        DATA_ROOT / SYN4D_ROOT / "manifest.json", source_archive=source_destination,
        mapping=mapping_destination, object_files=object_files, bedlam=bedlam_plan,
        bedlam_root=bedlam_root,
    )
    data_volume.commit()
    result = {
        "scene": "lab_bald", "sequence_count": 20,
        "source_archive_bytes": source_destination.stat().st_size,
        "object_vertex_count": len(object_files), "body_count": 20, "clothing_count": 20,
        "elapsed_seconds": time.perf_counter() - started,
    }
    run.summary.update(result)
    run.finish()
    return result


@app.function(
    image=blender_image, secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    cpu=8, memory=32 * 1024, ephemeral_disk=512 * 1024,
    timeout=24 * 60 * 60, retries=0, max_containers=MAX_CONTAINERS,
    include_source=False,
)
def convert_bedlam_remote(processes: int = CPU_CONVERSION_PROCESSES) -> dict[str, object]:
    """Convert only missing BEDLAM2 body caches; seq0 is never reconverted."""

    from mvtracker.profiling.modal_syn4d import (
        SELECTIVE_BEDLAM_ROOT, SYN4D_BODY_ROOT, SYN4D_MAPPING, SYN4D_SEQUENCE,
        scene_dependencies,
    )

    if processes <= 0:
        raise ValueError("processes must be positive")
    run = _wandb_run(
        job_type="bedlam2-conversion", name="lab-bald-remaining-bedlam2-conversion",
        tags=["bedlam2", "blender-4.5", "cpu"],
        config={"blender_version": BLENDER_VERSION, "processes": processes, **CPU_TAGS},
    )
    addon_parts = sorted(Path(SMPLX_ADDON_PARTS).glob("part-*"))
    if not addon_parts:
        raise FileNotFoundError(f"private SMPL-X add-on parts are required: {SMPLX_ADDON_PARTS}")
    addon = Path("/tmp/smplx_blender_addon-1.0.3-20260511.zip")
    with addon.open("wb") as output:
        for part in addon_parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, 8 << 20)
    if addon.stat().st_size != SMPLX_ADDON_BYTES:
        raise RuntimeError(f"SMPL-X add-on parts total {addon.stat().st_size} bytes; expected {SMPLX_ADDON_BYTES}")
    dependencies = scene_dependencies(DATA_ROOT / SYN4D_MAPPING)
    published_root = DATA_ROOT / SYN4D_BODY_ROOT
    seq0_motion = next(
        plan["body_motion"] for plan in dependencies if plan["sequence"] == SYN4D_SEQUENCE
    )
    if not tuple(published_root.rglob(f"{seq0_motion}.npz")):
        raise RuntimeError("seq_000000 body cache is required; refusing to reconvert it")
    missing_motions = [
        plan["body_motion"]
        for plan in dependencies
        if plan["sequence"] != SYN4D_SEQUENCE
        and not tuple(published_root.rglob(f"{plan['body_motion']}.npz"))
    ]
    if not missing_motions:
        result = {"motion_count": 0, "vertex_cache_count": 0, "skipped": True}
        run.summary.update(result)
        run.finish()
        return result
    source_motions = DATA_ROOT / SELECTIVE_BEDLAM_ROOT / "b2_motions_npz_training/motions_npz_training"
    work_root = Path("/tmp/lab-bald-bedlam")
    motions_root, abc_root, vertices_root = work_root / "motions", work_root / "abc", work_root / "vertices"
    shutil.rmtree(work_root, ignore_errors=True)
    motions_root.mkdir(parents=True)
    for motion in missing_motions:
        shutil.copyfile(source_motions / f"{motion}.npz", motions_root / f"{motion}.npz")
    install = (
        f"{shlex.quote(str(BLENDER_BIN))} --background --python-expr "
        + shlex.quote(
            "import bpy; bpy.ops.extensions.package_install_files("
            f"filepath={str(addon)!r}, repo='user_default', enable_on_install=True, overwrite=True); "
            "bpy.ops.wm.save_userpref(); assert hasattr(bpy.context.window_manager, 'smplx_tool')"
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
    subprocess.run(stage2, check=True)
    stage2_seconds = time.perf_counter() - started - stage1_seconds
    vertex_files = tuple(vertices_root.rglob("*.npz"))
    if len(vertex_files) != len(missing_motions):
        raise RuntimeError(
            f"BEDLAM conversion produced {len(vertex_files)} vertex caches; "
            f"expected {len(missing_motions)}"
        )
    for source in vertex_files:
        destination = published_root / source.relative_to(vertices_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    data_volume.commit()
    result = {"stage1_seconds": stage1_seconds, "stage2_seconds": stage2_seconds,
              "motion_count": len(missing_motions), "vertex_cache_count": len(vertex_files)}
    run.summary.update(result)
    run.finish()
    return result


@app.function(
    image=t4_image, secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    gpu="T4", cpu=T4_CPU, memory=T4_MEMORY_MIB, ephemeral_disk=T4_EPHEMERAL_DISK_MIB,
    timeout=24 * 60 * 60, retries=0, max_containers=1, include_source=False,
)
def convert_shard_a_remote() -> dict[str, object]:
    """Convert seq_000001..seq_000009 from one archive extraction."""

    from mvtracker.preprocessing.syn4d import convert_syn4d_sequence
    from mvtracker.profiling.modal_syn4d import (
        SYN4D_CACHE_ROOT, SYN4D_METADATA_ROOT, SYN4D_ROOT, SYN4D_SCENE,
    )

    run = _wandb_run(
        job_type="syn4d-conversion-shard", name="lab-bald-shard-a",
        tags=["converter", "t4", "shard-a"],
        config={"sequences": list(SHARD_A_SEQUENCES), **MODAL_TAGS},
    )
    source_archive = DATA_ROOT / SYN4D_ROOT / "source/lab_bald.tar.zst"
    extracted_root = Path("/tmp/syn4d-lab-bald-shard-a")
    shutil.rmtree(extracted_root, ignore_errors=True)
    extracted_root.mkdir(parents=True)
    subprocess.run(
        ["tar", "-I", "zstd -T0", "-xf", str(source_archive), "-C", str(extracted_root)],
        check=True,
    )
    started = time.perf_counter()
    results = []
    for sequence in SHARD_A_SEQUENCES:
        sequence_started = time.perf_counter()
        result = convert_syn4d_sequence(
            extracted_root / SYN4D_SCENE,
            DATA_ROOT / SYN4D_METADATA_ROOT,
            DATA_ROOT / SYN4D_CACHE_ROOT,
            sequence=sequence,
            device="cuda",
            official_visualizer_root=SYN4D_VISUALIZER_ROOT,
        )
        data_volume.commit()
        result = {**result, "elapsed_seconds": time.perf_counter() - sequence_started}
        results.append(result)
        run.log({"sequence": sequence, "sequence_elapsed_seconds": result["elapsed_seconds"]})
    result = {"shard": "a", "sequences": results, "elapsed_seconds": time.perf_counter() - started}
    run.summary.update({"sequence_count": len(results), "elapsed_seconds": result["elapsed_seconds"]})
    run.finish()
    run_volume.commit()
    return result


@app.function(
    image=t4_image, secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    gpu="T4", cpu=T4_CPU, memory=T4_MEMORY_MIB, ephemeral_disk=T4_EPHEMERAL_DISK_MIB,
    timeout=24 * 60 * 60, retries=0, max_containers=1, include_source=False,
)
def convert_shard_b_remote() -> dict[str, object]:
    """Convert seq_000010..seq_000019 from one archive extraction."""

    from mvtracker.preprocessing.syn4d import convert_syn4d_sequence
    from mvtracker.profiling.modal_syn4d import (
        SYN4D_CACHE_ROOT, SYN4D_METADATA_ROOT, SYN4D_ROOT, SYN4D_SCENE,
    )

    run = _wandb_run(
        job_type="syn4d-conversion-shard", name="lab-bald-shard-b",
        tags=["converter", "t4", "shard-b"],
        config={"sequences": list(SHARD_B_SEQUENCES), **MODAL_TAGS},
    )
    source_archive = DATA_ROOT / SYN4D_ROOT / "source/lab_bald.tar.zst"
    extracted_root = Path("/tmp/syn4d-lab-bald-shard-b")
    shutil.rmtree(extracted_root, ignore_errors=True)
    extracted_root.mkdir(parents=True)
    subprocess.run(
        ["tar", "-I", "zstd -T0", "-xf", str(source_archive), "-C", str(extracted_root)],
        check=True,
    )
    started = time.perf_counter()
    results = []
    for sequence in SHARD_B_SEQUENCES:
        sequence_started = time.perf_counter()
        result = convert_syn4d_sequence(
            extracted_root / SYN4D_SCENE,
            DATA_ROOT / SYN4D_METADATA_ROOT,
            DATA_ROOT / SYN4D_CACHE_ROOT,
            sequence=sequence,
            device="cuda",
            official_visualizer_root=SYN4D_VISUALIZER_ROOT,
        )
        data_volume.commit()
        result = {**result, "elapsed_seconds": time.perf_counter() - sequence_started}
        results.append(result)
        run.log({"sequence": sequence, "sequence_elapsed_seconds": result["elapsed_seconds"]})
    result = {"shard": "b", "sequences": results, "elapsed_seconds": time.perf_counter() - started}
    run.summary.update({"sequence_count": len(results), "elapsed_seconds": result["elapsed_seconds"]})
    run.finish()
    run_volume.commit()
    return result


@app.function(
    image=cpu_image, secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    cpu=4, memory=8 * 1024, timeout=60 * 60, max_containers=1, include_source=False,
)
def loader_smoke_remote() -> dict[str, object]:
    """Run five deterministic CPU plan/materialize samples on the pilot cache."""

    from types import SimpleNamespace

    import numpy as np
    from mvtracker.datasets.syn4d_multiview_dataset import Syn4DMultiViewDataset, _SequenceMmapCache
    from mvtracker.profiling.modal_syn4d import SYN4D_CACHE_ROOT, SYN4D_SEQUENCE, SYN4D_SCENE

    root = DATA_ROOT / SYN4D_CACHE_ROOT
    sequence = f"{SYN4D_SCENE}__{SYN4D_SEQUENCE}"
    dataset = Syn4DMultiViewDataset.__new__(Syn4DMultiViewDataset)
    dataset.data_root = str(root)
    dataset.seq_names = [sequence]
    dataset.real_len = 1
    dataset.seq_len = 24
    dataset.num_views = None
    dataset.view_count_probabilities = (1 / 6,) * 6
    dataset.traj_per_sample = 2_048
    dataset.seed = 72
    dataset.add_index_to_seed = True
    dataset.crop_size = (384, 512)
    dataset.enable_cropping_augs = True
    dataset.enable_rgb_augs = False
    dataset.enable_depth_augs = False
    dataset.enable_variable_trajpersample_augs = False
    dataset.enable_variable_num_views_augs = True
    dataset.enable_scene_transform_augs = False
    dataset.enable_camera_params_noise_augs = False
    dataset.augmentation_probability = 0.0
    dataset.ratio_dynamic = 0.5
    dataset.ratio_very_dynamic = 0.25
    dataset.max_tracks_to_preload = 18_000
    dataset.max_depth = 1_000.0
    dataset.eraser_aug_prob = 0.5
    dataset.eraser_max = 10
    dataset.eraser_bounds = [2, 100]
    dataset.replace_aug_prob = 0.5
    dataset.replace_max = 10
    dataset.replace_bounds = [2, 100]
    dataset._manifests = {sequence: dataset._load_manifest(sequence)}
    dataset._sequence_cache = _SequenceMmapCache(root, maximum=1)
    run = _wandb_run(
        job_type="syn4d-loader-smoke", name="lab-bald-seq000000-loader-smoke",
        tags=["loader", "cpu"], config={"iterations": 5, "view_count": 4, **CPU_TAGS},
    )
    rows = []
    for index in range(5):
        request = SimpleNamespace(virtual_index=index, scene_index=0, view_count=4)
        plan_started = time.perf_counter()
        plan = dataset.plan_sample(request)
        planning_seconds = time.perf_counter() - plan_started
        if plan is None:
            raise RuntimeError(f"Syn4D planner rejected deterministic sample {index}")
        if plan.track_count != 2_048 or plan.output_size != (384, 512):
            raise RuntimeError(f"unexpected plan contract: tracks={plan.track_count} output={plan.output_size}")
        materialize_started = time.perf_counter()
        sample, gotit = dataset.materialize_sample(plan)
        materialize_seconds = time.perf_counter() - materialize_started
        if not gotit or len(sample.jpeg_bytes) != 4 * 24:
            raise RuntimeError("materialized sample does not contain 96 JPEG payloads")
        if tuple(sample.depth.shape) != (4, 24, 1, 384, 683):
            raise RuntimeError(f"depth shape {tuple(sample.depth.shape)} != (4, 24, 1, 384, 683)")
        payload_bytes = sum(int(value.numel()) for value in sample.jpeg_bytes)
        rows.append({"planning_seconds": planning_seconds, "materialize_seconds": materialize_seconds,
                     "jpeg_bytes": payload_bytes, "track_count": plan.track_count})
    warm = rows[1:]
    def percentile(key: str, q: float) -> float:
        return float(np.percentile([row[key] for row in warm], q))
    result = {
        "iterations": 5, "view_count": 4, "frames": 24, "tracks": 2_048,
        "jpeg_payload_count": 96, "planned_output_size": [384, 512],
        "depth_shape": [4, 24, 1, 384, 683],
        "cold_planning_seconds": rows[0]["planning_seconds"],
        "cold_materialize_seconds": rows[0]["materialize_seconds"],
        "cold_jpeg_bytes": rows[0]["jpeg_bytes"],
        "warm_planning_p50_seconds": percentile("planning_seconds", 50),
        "warm_planning_p95_seconds": percentile("planning_seconds", 95),
        "warm_materialize_p50_seconds": percentile("materialize_seconds", 50),
        "warm_materialize_p95_seconds": percentile("materialize_seconds", 95),
        "warm_jpeg_bytes_p50": percentile("jpeg_bytes", 50),
        "warm_jpeg_bytes_p95": percentile("jpeg_bytes", 95),
        "rows": rows,
    }
    for step, row in enumerate(rows):
        run.log({f"loader/{key}": value for key, value in row.items()}, step=step)
    run.summary.update({key: value for key, value in result.items() if key != "rows"})
    run.finish()
    return result


def _preflight(tags: dict[str, str], *, required_free_slots: int = 1) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=required_free_slots)
    app.set_tags(tags)


@app.local_entrypoint(name="download")
def download() -> None:
    _preflight(CPU_TAGS)
    print(json.dumps(download_dependencies_remote.remote(), indent=2, sort_keys=True))


@app.local_entrypoint(name="convert-bedlam")
def convert_bedlam(processes: int = CPU_CONVERSION_PROCESSES) -> None:
    _preflight(CPU_TAGS)
    print(json.dumps(convert_bedlam_remote.remote(processes), indent=2, sort_keys=True))


@app.local_entrypoint(name="remaining")
def remaining() -> None:
    """Run exactly two concurrent T4 workers and wait for both results."""

    _preflight(MODAL_TAGS, required_free_slots=2)
    shard_a = convert_shard_a_remote.spawn()
    shard_b = convert_shard_b_remote.spawn()
    print(json.dumps({"shard_a": shard_a.get(), "shard_b": shard_b.get()}, indent=2, sort_keys=True))


@app.local_entrypoint(name="loader-smoke")
def loader_smoke() -> None:
    _preflight(CPU_TAGS)
    print(json.dumps(loader_smoke_remote.remote(), indent=2, sort_keys=True))


if __name__ == "__main__":
    print("Use: modal run tools/modal_syn4d_data_setup.py::<download|convert-bedlam|remaining|loader-smoke>")
