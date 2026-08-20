"""Index existing MV-Kubric WebDataset TARs on the Modal data Volume.

This job does not rewrite the TARs.  It runs NVIDIA DALI's ``wds2idx`` once
per existing train/validation archive, with bounded CPU parallelism, then
invokes the preprocessing module's record-locator API after all sidecars are
present.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import modal

from modal_training_profile import (
    BASE_TAGS,
    DATA_ROOT,
    _runtime_image,
    _source_commit,
    data_volume,
    wandb_secret,
)


APP_NAME = "jeet-mvtracker-mvkubric-tar-index"
WANDB_PROJECT = "mvtracker-modal-profiling"
WANDB_ENTITY = "jeetucl-ucl"
MODAL_TAGS = {**BASE_TAGS, "experiment": "mvkubric-tar-index"}
TRAIN_TAR_ROOT = DATA_ROOT / "datasets/kubric-multiview-webdataset/train"
VALIDATION_TAR_ROOT = DATA_ROOT / "datasets/kubric-multiview-webdataset/validation"
LOCATOR_MODULE = "mvtracker.preprocessing.mvkubric_webdataset"
LOCATOR_API = "publish_record_locator"


app = modal.App(APP_NAME, tags=MODAL_TAGS)
index_image = _runtime_image()


def _index_path(tar_path: Path) -> Path:
    return tar_path.with_suffix(".idx")


def _wds2idx_command() -> list[str]:
    executable = shutil.which("wds2idx")
    if executable is not None:
        return [executable]

    dali_spec = importlib.util.find_spec("nvidia.dali")
    if dali_spec is not None and dali_spec.origin is not None:
        package_root = Path(dali_spec.origin).parent
        candidates = (
            package_root / "tools/wds2idx.py",
            package_root.parent / "tools/wds2idx.py",
        )
        for candidate in candidates:
            if candidate.is_file():
                return [sys.executable, str(candidate)]
    raise RuntimeError("DALI wds2idx is not installed in the Modal image")


def _index_one(tar_path: Path, *, force: bool, command: list[str]) -> dict[str, object]:
    index_path = _index_path(tar_path)
    if index_path.is_file() and not force:
        return {
            "tar": str(tar_path),
            "index": str(index_path),
            "status": "skipped",
            "bytes": tar_path.stat().st_size,
        }
    partial = index_path.with_suffix(index_path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    started = time.perf_counter()
    subprocess.run([*command, str(tar_path), str(partial)], check=True)
    partial.replace(index_path)
    return {
        "tar": str(tar_path),
        "index": str(index_path),
        "status": "indexed",
        "bytes": tar_path.stat().st_size,
        "seconds": time.perf_counter() - started,
    }


def _run_locator_builder(split_root: Path) -> None:
    module = importlib.import_module(LOCATOR_MODULE)
    builder = getattr(module, LOCATOR_API, None)
    if not callable(builder):
        raise RuntimeError(
            f"{LOCATOR_MODULE} must export callable {LOCATOR_API}"
        )
    builder(split_root)
    print(f"INDEX event=locator_complete root={split_root}", flush=True)


@app.function(
    image=index_image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=32768,
    timeout=4 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def index_remote(
    *,
    train_root: str = str(TRAIN_TAR_ROOT),
    validation_root: str = str(VALIDATION_TAR_ROOT),
    workers: int = 8,
    force: bool = False,
) -> dict[str, object]:
    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    import wandb

    roots = {"train": Path(train_root), "validation": Path(validation_root)}
    archives = {
        split: tuple(sorted(root.glob("*.tar")))
        for split, root in roots.items()
    }
    missing = [str(root) for root in roots.values() if not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"TAR roots are missing: {missing}")
    if any(not paths for paths in archives.values()):
        raise RuntimeError(f"no .tar archives found: {archives}")
    command = _wds2idx_command()
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="mvkubric-tar-index",
        tags=["modal", "cpu", "mv-kubric", "wds2idx"],
        config={
            "source_commit": _source_commit(),
            "train_root": train_root,
            "validation_root": validation_root,
            "workers": workers,
            "force": force,
            "archive_count": sum(len(paths) for paths in archives.values()),
            **MODAL_TAGS,
        },
    )
    results: list[dict[str, object]] = []
    jobs = tuple((split, path) for split, paths in archives.items() for path in paths)
    print(
        f"INDEX event=start archives={len(jobs)} workers={workers} "
        f"train={len(archives['train'])} validation={len(archives['validation'])}",
        flush=True,
    )

    def run_one(job: tuple[str, Path]) -> dict[str, object]:
        split, path = job
        result = _index_one(path, force=force, command=command)
        return {"split": split, **result}

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        pending = {
            executor.submit(run_one, job)
            for job in jobs
        }
        completed = 0
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                print(
                    f"INDEX event=heartbeat progress={completed}/{len(jobs)}",
                    flush=True,
                )
                run.log(
                    {
                        "progress/archives_completed": completed,
                        "progress/archives_total": len(jobs),
                    },
                    commit=True,
                )
                continue
            for future in done:
                completed += 1
                result = future.result()
                results.append(result)
                print(
                    f"INDEX event=archive_complete split={result['split']} "
                    f"tar={result['tar']} status={result['status']} "
                    f"progress={completed}/{len(jobs)}",
                    flush=True,
                )
                run.log(
                    {
                        "progress/archives_completed": completed,
                        "progress/archives_total": len(jobs),
                    },
                    commit=True,
                )

    for split, root in roots.items():
        _run_locator_builder(root)
    results.sort(key=lambda result: str(result["tar"]))
    data_volume.commit()
    summary = {
        "format": "mvtracker-mvkubric-dali-index",
        "source_commit": _source_commit(),
        "archives": results,
        "archive_count": len(results),
        "indexed_count": sum(result["status"] == "indexed" for result in results),
        "skipped_count": sum(result["status"] == "skipped" for result in results),
    }
    run.summary.update(summary)
    run.finish()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.local_entrypoint(name="index")
def index(
    train_root: str = str(TRAIN_TAR_ROOT),
    validation_root: str = str(VALIDATION_TAR_ROOT),
    workers: int = 8,
    force: bool = False,
) -> None:
    print(
        json.dumps(
            index_remote.remote(
                train_root=train_root,
                validation_root=validation_root,
                workers=workers,
                force=force,
            ),
            indent=2,
            sort_keys=True,
        )
    )
