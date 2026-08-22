"""Prepare and convert the fixed 16-train/4-validation Syn4D split."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
from mvtracker.profiling.modal_syn4d_split import (
    ARCHIVE_BYTES,
    ARCHIVE_ROOT,
    BODY_ROOT,
    CLOTHING_ROOT,
    LEGACY_ARCHIVES,
    OBJECT_ROOT,
    SHARD_A_ENVIRONMENTS,
    SHARD_B_ENVIRONMENTS,
    SHARED_METADATA_ROOT,
    SPLIT_MANIFEST,
    SPLIT_ROOTS,
    SPLIT_SEED,
    SMPLX_ADDON_PARTS,
    SYN4D_REPO_ID,
    SYN4D_REVISION,
    SYN4D_SUBSET,
    dependency_plan,
    jobs_for,
)


APP_NAME = "jeet-mvtracker-syn4d-fixed-split"
WANDB_PROJECT = "mvtracker-modal-profiling"
WANDB_GROUP = "syn4d-fixed-split"
CPU_TAGS = {**BASE_TAGS, "experiment": "syn4d-fixed-split-setup", "gpu": "cpu"}
T4_TAGS = {**BASE_TAGS, "experiment": "syn4d-fixed-split-conversion", "gpu": "t4"}
MAPPING_DESTINATION = Path("datasets/syn4d/sequence_to_asset_mapping_stride1.csv")
SPLIT_MANIFEST_DESTINATION = Path("datasets/syn4d/fixed-split-manifest.json")
CONVERTED_BODY_ROOT = SHARED_METADATA_ROOT / "bedlam2_smpl_npz"
SMPLX_ADDON_BYTES = 387_473_505
BLENDER_VERSION = "4.5.0"
BLENDER_ARCHIVE = f"blender-{BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL = f"https://download.blender.org/release/Blender4.5/{BLENDER_ARCHIVE}"
BLENDER_ROOT = Path(f"/opt/blender-{BLENDER_VERSION}-linux-x64")
BLENDER_BIN = BLENDER_ROOT / "blender"
BEDLAM_SCRIPTS_ROOT = Path("/opt/syn4d-bedlam2")
VISUALIZER_ROOT = Path("/opt/syn4d-visualizer")
BEDLAM_README_URL = (
    "https://huggingface.co/datasets/Syn4D/Syn4D/resolve/"
    f"{SYN4D_REVISION}/code/bedlam2/"
)

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


def _setup_image() -> modal.Image:
    return _clone_source(
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ca-certificates", "git", "zstd")
        .pip_install(
            "hf-xet==1.1.8", "huggingface-hub==0.30.2", "numpy==2.2.4",
            "requests==2.32.3", "wandb==0.19.9",
        )
    )


def _blender_image() -> modal.Image:
    names = (
        "smplx_anim_to_alembic.py", "smplx_anim_to_alembic_batch.py",
        "smplx_anim_to_objs.py", "smplx_anim_to_objs_batch.py",
        "supervised_blender_batch.py",
    )
    downloads = " && ".join(
        "curl -fsSL " + shlex.quote(BEDLAM_README_URL + name) + " -o "
        + shlex.quote(str(BEDLAM_SCRIPTS_ROOT / name))
        for name in names
    )
    return _clone_source(
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "ca-certificates", "curl", "git", "libgl1", "libglib2.0-0",
            "libopenexr-dev", "libice6", "libsm6", "libx11-6", "libxfixes3",
            "libxi6", "libxkbcommon0", "libxrender1", "libxxf86vm1", "openexr", "xz-utils",
        )
        .pip_install("numpy==2.2.4", "wandb==0.19.9")
        .run_commands(
            f"mkdir -p {shlex.quote(str(BEDLAM_SCRIPTS_ROOT))}",
            f"curl -fsSL {shlex.quote(BLENDER_URL)} | tar -xJ -C /opt",
            downloads,
        )
        .env({"BLENDER_BIN": str(BLENDER_BIN)})
    )


def _t4_image() -> modal.Image:
    base = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.10"
        )
        .apt_install("ca-certificates", "curl", "ffmpeg", "git", "libgl1", "libglib2.0-0", "libopenexr-dev", "openexr", "zstd")
        .pip_install("torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1", index_url="https://download.pytorch.org/whl/cu128")
        .pip_install(
            "hf-xet==1.1.8", "huggingface-hub==0.30.2", "nvidia-dali-cuda120==1.53.0",
            "OpenEXR==3.3.5", "safetensors==0.5.3", "wandb==0.19.9",
            "opencv-python-headless==4.11.0.86", "pandas==2.2.3", "scipy==1.15.2",
            "matplotlib==3.10.1", "Pillow==11.1.0", "imageio==2.37.0", "pypng==0.20220715.0",
            "kornia==0.7.3", "mediapy==1.2.0", "rerun-sdk==0.21.0", "tqdm==4.67.1", "pause==0.3",
        )
        .run_commands(
            f"mkdir -p {shlex.quote(str(VISUALIZER_ROOT))}",
            *(
                "curl -fsSL "
                + shlex.quote(
                    "https://huggingface.co/datasets/Syn4D/Syn4D/resolve/"
                    f"{SYN4D_REVISION}/code/visualizer/{filename}"
                )
                + " -o " + shlex.quote(str(VISUALIZER_ROOT / filename))
                for filename in ("syn4d_track.py", "base_dataset.py", "utils.py")
            ),
        )
        .env({"DALI_DISABLE_NVML": "1", "HF_HOME": "/tmp/huggingface", "HF_XET_CACHE": "/tmp/huggingface/xet"})
    )
    return _clone_source(base)


app = modal.App(APP_NAME, tags=T4_TAGS)
setup_image = _setup_image()
blender_image = _blender_image()
t4_image = _t4_image()


def _wandb_run(*, job_type: str, name: str, tags: list[str], config: dict):
    import wandb

    return wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT, group=WANDB_GROUP,
        job_type=job_type, name=name, tags=["modal", "syn4d", "fixed-split", *tags],
        config={**BASE_TAGS, **config},
    )


def _mapping_path() -> Path:
    return DATA_ROOT / MAPPING_DESTINATION


def _archive_map(email: str, password: str) -> dict[str, list[str]]:
    from tools.modal_syn4d_bedlam_setup import _inputs

    _, archive_map, _ = _inputs()
    return archive_map


@app.function(
    image=setup_image,
    secrets=[hf_secret, wandb_secret, bedlam_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    cpu=4, memory=8 * 1024, ephemeral_disk=512 * 1024,
    timeout=24 * 60 * 60, retries=1, max_containers=1, include_source=False,
)
def stage_dependencies_remote() -> dict[str, object]:
    """Stage only dependencies referenced by the fixed 20-row manifest."""

    from huggingface_hub import hf_hub_download
    from mvtracker.profiling.modal_syn4d_bedlam import download_sparse_clothing_tar, download_body_motions

    run = _wandb_run(
        job_type="syn4d-fixed-split-dependencies", name="syn4d-fixed-split-dependencies",
        tags=["dependencies", "cpu"], config={"manifest_rows": len(SPLIT_MANIFEST), **CPU_TAGS},
    )
    started = time.perf_counter()
    mapping = _mapping_path()
    mapping.parent.mkdir(parents=True, exist_ok=True)
    if not mapping.is_file():
        source = hf_hub_download(
            repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION,
            filename=f"{SYN4D_SUBSET}/sequence_to_asset_mapping.csv",
            token=os.environ["HF_TOKEN"], local_dir="/tmp/syn4d-split-mapping",
        )
        shutil.copyfile(source, mapping)
    plans = dependency_plan(mapping)
    local_archive_map_path = DATA_ROOT / CLOTHING_ROOT / "archive_map.json"
    local_archive_map = (
        json.loads(local_archive_map_path.read_text(encoding="utf-8"))
        if local_archive_map_path.is_file()
        else {}
    )
    archive_map = _archive_map(os.environ["BEDLAM_EMAIL"], os.environ["BEDLAM_PASSWORD"])
    member_to_archive = {
        member: archive for archive, members in archive_map.items() for member in members
    }
    motions = sorted({item["dependencies"].body_motion for item in plans})
    missing_motions = [motion for motion in motions if not (DATA_ROOT / BODY_ROOT / f"{motion}.npz").is_file()]
    if missing_motions:
        download_body_motions(
            DATA_ROOT, missing_motions, os.environ["BEDLAM_EMAIL"], os.environ["BEDLAM_PASSWORD"]
        )
    required_members: dict[str, list[str]] = {}
    for item in plans:
        member = item["dependencies"].clothing_member
        archive = member_to_archive.get(member)
        if archive is None:
            raise RuntimeError(f"BEDLAM archive map has no clothing member {member}")
        required_members.setdefault(archive, []).append(member)
    required_members = {
        archive: sorted(set(local_archive_map.get(archive, [])) | set(members))
        for archive, members in required_members.items()
    }

    def refresh_archive(archive: str, members: list[str]) -> None:
        destination = DATA_ROOT / CLOTHING_ROOT / f"{archive}.tar"
        if destination.is_file() and set(local_archive_map.get(archive, [])) == set(members):
            return
        stale = destination.with_name(f".{destination.name}.stale")
        if destination.is_file():
            destination.replace(stale)
        try:
            download_sparse_clothing_tar(
                DATA_ROOT, archive, members,
                os.environ["BEDLAM_EMAIL"], os.environ["BEDLAM_PASSWORD"],
            )
        except BaseException:
            if stale.is_file():
                stale.replace(destination)
            raise
        if stale.is_file():
            stale.unlink()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(refresh_archive, archive, members)
            for archive, members in required_members.items()
        ]
        for future in futures:
            future.result()
    object_paths = set()
    for item in plans:
        for group, object_id in item["dependencies"].objects:
            relative = Path(group) / object_id / "vertices_sequence.npz"
            destination = DATA_ROOT / OBJECT_ROOT / relative
            object_paths.add((relative, destination))
    def stage_object(item: tuple[Path, Path]) -> None:
        relative, destination = item
        if destination.is_file():
            return
        source = hf_hub_download(
            repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION,
            filename=f"data/metadata/new_weight_bone/{relative.as_posix()}",
            token=os.environ["HF_TOKEN"], local_dir=f"/tmp/syn4d-object-{relative.parent.name}",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(stage_object, sorted(object_paths)))
    (DATA_ROOT / CLOTHING_ROOT).mkdir(parents=True, exist_ok=True)
    updated_archive_map = dict(local_archive_map)
    updated_archive_map.update(required_members)
    (DATA_ROOT / CLOTHING_ROOT / "archive_map.json").write_text(
        json.dumps(updated_archive_map, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "format": "mvtracker-syn4d-fixed-split",
        "split_seed": SPLIT_SEED,
        "source_revision": SYN4D_REVISION,
        "output_roots": {split: str(root) for split, root in SPLIT_ROOTS.items()},
        "rows": [
            {**row, "archive_bytes": ARCHIVE_BYTES[row["environment"]]}
            for row in SPLIT_MANIFEST
        ],
        "body_motion_count": len(motions),
        "clothing_archive_count": len(required_members),
        "object_count": len(object_paths),
    }
    manifest_path = DATA_ROOT / SPLIT_MANIFEST_DESTINATION
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    data_volume.commit()
    result = {**manifest, "elapsed_seconds": time.perf_counter() - started}
    run.summary.update({key: value for key, value in result.items() if key != "rows"})
    run.finish()
    return result


@app.function(
    image=blender_image, secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    cpu=8, memory=32 * 1024, ephemeral_disk=512 * 1024,
    timeout=24 * 60 * 60, retries=0, max_containers=1, include_source=False,
)
def convert_body_remote() -> dict[str, object]:
    """Convert missing selected body caches with the official Blender scripts."""

    from mvtracker.profiling.modal_syn4d_bedlam import BODY_ROOT as SOURCE_BODY_ROOT

    mapping = _mapping_path()
    plans = dependency_plan(mapping)
    motions = sorted({item["dependencies"].body_motion for item in plans})
    output_root = DATA_ROOT / CONVERTED_BODY_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    missing = [motion for motion in motions if not tuple(output_root.rglob(f"{motion}.npz"))]
    if not missing:
        return {"motion_count": 0, "vertex_cache_count": 0, "skipped": True}
    run = _wandb_run(
        job_type="syn4d-fixed-split-bedlam", name="syn4d-fixed-split-bedlam",
        tags=["bedlam2", "blender", "cpu"], config={"missing_motions": len(missing), **CPU_TAGS},
    )
    addon_parts = sorted((DATA_ROOT / SMPLX_ADDON_PARTS).glob("part-*"))
    if not addon_parts:
        raise FileNotFoundError(f"SMPL-X add-on parts are required: {DATA_ROOT / SMPLX_ADDON_PARTS}")
    addon = Path("/tmp/smplx_blender_addon-1.0.3-20260511.zip")
    with addon.open("wb") as target:
        for part in addon_parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, target, 8 << 20)
    if addon.stat().st_size != SMPLX_ADDON_BYTES:
        raise RuntimeError("SMPL-X add-on parts size mismatch")
    work = Path("/tmp/syn4d-fixed-split-bedlam")
    shutil.rmtree(work, ignore_errors=True)
    motions_root, abc_root, vertices_root = work / "motions", work / "abc", work / "vertices"
    motions_root.mkdir(parents=True)
    for motion in missing:
        shutil.copyfile(DATA_ROOT / SOURCE_BODY_ROOT / f"{motion}.npz", motions_root / f"{motion}.npz")
    install = f"{shlex.quote(str(BLENDER_BIN))} --background --python-expr " + shlex.quote(
        "import bpy; bpy.ops.extensions.package_install_files("
        f"filepath={str(addon)!r}, repo='user_default', enable_on_install=True, overwrite=True); "
        "bpy.ops.wm.save_userpref(); assert hasattr(bpy.context.window_manager, 'smplx_tool')"
    )
    subprocess.run(install, shell=True, check=True)
    subprocess.run([
        "python3", str(BEDLAM_SCRIPTS_ROOT / "smplx_anim_to_alembic_batch.py"),
        str(motions_root), str(abc_root), "8", "--blender", str(BLENDER_BIN),
        "--timeout-seconds", "600", "--retries", "1", "--skip-existing",
    ], check=True)
    subprocess.run([
        "python3", str(BEDLAM_SCRIPTS_ROOT / "smplx_anim_to_objs_batch.py"),
        str(abc_root), str(vertices_root), "8", "--blender", str(BLENDER_BIN),
        "--timeout-seconds", "600", "--retries", "1", "--skip-existing",
    ], check=True)
    files = tuple(vertices_root.rglob("*.npz"))
    if len(files) != len(missing):
        raise RuntimeError(f"expected {len(missing)} body caches, got {len(files)}")
    for source in files:
        target = output_root / source.relative_to(vertices_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    data_volume.commit()
    result = {"motion_count": len(missing), "vertex_cache_count": len(files)}
    run.summary.update(result)
    run.finish()
    return result


def _convert_jobs(environment_names: tuple[str, ...], shard: str) -> dict[str, object]:
    from huggingface_hub import hf_hub_download
    from mvtracker.preprocessing.syn4d import convert_syn4d_sequence

    data_volume.reload()
    run = _wandb_run(
        job_type="syn4d-fixed-split-conversion", name=f"syn4d-fixed-split-{shard}",
        tags=["converter", "t4", f"shard-{shard}"],
        config={"environments": list(environment_names), **T4_TAGS},
    )
    started = time.perf_counter()
    results = []
    for job in jobs_for(environment_names):
        cached = DATA_ROOT / job.output_root / job.output_name / "manifest.json"
        if job.environment == "lab_bald" and job.sequence == "seq_000017" and cached.is_file():
            results.append({
                "scene": job.environment, "sequence": job.sequence,
                "output_path": str(cached.parent), "reused": True,
                "split": job.split,
            })
            run.log({"environment": job.environment, "split": job.split, "reused": 1})
            continue
        work = Path(f"/tmp/syn4d-fixed-split-{job.environment}")
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        archive = work / f"{job.environment}.tar.zst"
        try:
            legacy_relative = LEGACY_ARCHIVES.get(job.environment)
            legacy = DATA_ROOT / legacy_relative if legacy_relative is not None else None
            if legacy is not None and legacy.is_file():
                if legacy.stat().st_size != job.archive_bytes:
                    raise RuntimeError(f"legacy archive size mismatch: {legacy}")
                shutil.copyfile(legacy, archive)
            else:
                hf_work = work / "hf"
                hf_work.mkdir()
                os.environ["HF_HOME"] = str(work / "hf-home")
                os.environ["HF_XET_CACHE"] = str(work / "hf-xet")
                os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "4"
                source = hf_hub_download(
                    repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION,
                    filename=f"{SYN4D_SUBSET}/{job.environment}.tar.zst",
                    token=os.environ["HF_TOKEN"], local_dir=str(hf_work),
                    cache_dir=str(work / "hf-cache"),
                )
                if Path(source).stat().st_size != job.archive_bytes:
                    raise RuntimeError(f"downloaded archive size mismatch: {source}")
                archive = Path(source)
            extracted = work / "extracted"
            extracted.mkdir()
            subprocess.run(["tar", "-I", "zstd -T0", "-xf", str(archive), "-C", str(extracted)], check=True)
            result = convert_syn4d_sequence(
                extracted / job.environment,
                DATA_ROOT / SHARED_METADATA_ROOT,
                DATA_ROOT / job.output_root,
                sequence=job.sequence,
                device="cuda",
                official_visualizer_root=VISUALIZER_ROOT,
            )
            data_volume.commit()
            results.append({**result, "environment": job.environment, "split": job.split})
            run.log({"environment": job.environment, "split": job.split, "completed": 1})
        finally:
            shutil.rmtree(work, ignore_errors=True)
    result = {"shard": shard, "jobs": results, "elapsed_seconds": time.perf_counter() - started}
    run.summary.update({"job_count": len(results), "elapsed_seconds": result["elapsed_seconds"]})
    run.finish()
    return result


@app.function(
    image=t4_image, secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    gpu="T4", cpu=8, memory=32 * 1024, ephemeral_disk=1 * 1024 * 1024,
    timeout=24 * 60 * 60, retries=0, max_containers=1, include_source=False,
)
def convert_shard_a_remote() -> dict[str, object]:
    return _convert_jobs(SHARD_A_ENVIRONMENTS, "a")


@app.function(
    image=t4_image, secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    gpu="T4", cpu=8, memory=32 * 1024, ephemeral_disk=1 * 1024 * 1024,
    timeout=24 * 60 * 60, retries=0, max_containers=1, include_source=False,
)
def convert_shard_b_remote() -> dict[str, object]:
    return _convert_jobs(SHARD_B_ENVIRONMENTS, "b")


def _preflight(tags: dict[str, str], free_slots: int = 1) -> None:
    require_pushed_main_commit(_source_commit())
    preflight_active_containers(required_free_slots=free_slots)
    app.set_tags(tags)


@app.local_entrypoint(name="download")
def download() -> None:
    _preflight(CPU_TAGS)
    print(json.dumps(stage_dependencies_remote.remote(), indent=2, sort_keys=True))


@app.local_entrypoint(name="convert-bedlam")
def convert_bedlam() -> None:
    _preflight(CPU_TAGS)
    print(json.dumps(convert_body_remote.remote(), indent=2, sort_keys=True))


@app.local_entrypoint(name="convert-shard-a")
def convert_shard_a() -> None:
    _preflight(T4_TAGS)
    print(json.dumps(convert_shard_a_remote.remote(), indent=2, sort_keys=True))


@app.local_entrypoint(name="convert-shard-b")
def convert_shard_b() -> None:
    _preflight(T4_TAGS)
    print(json.dumps(convert_shard_b_remote.remote(), indent=2, sort_keys=True))


if __name__ == "__main__":
    print("Use: modal run tools/modal_syn4d_split_setup.py::<download|convert-bedlam|convert-shard-a|convert-shard-b>")
