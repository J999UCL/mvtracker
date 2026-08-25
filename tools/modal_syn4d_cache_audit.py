"""Read-only integrity audit for every persisted Syn4D training cache."""

from __future__ import annotations

import json
from pathlib import Path
import time

import modal

from modal_training_profile import BASE_TAGS, DATA_ROOT, _source_commit, data_volume, image, wandb_secret


APP_NAME = "jeet-mvtracker-syn4d-cache-audit"
TAGS = {**BASE_TAGS, "experiment": "syn4d-cache-audit", "gpu": "cpu"}
app = modal.App(APP_NAME, tags=TAGS)


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=8,
    memory=16 * 1024,
    timeout=2 * 60 * 60,
    include_source=False,
)
def audit_remote() -> dict[str, object]:
    import numpy as np
    import wandb

    run = wandb.init(project="mvtracker-modal-profiling", job_type="syn4d-cache-audit", name="syn4d-cache-audit", tags=["modal", "cpu", "syn4d", "integrity"], config={**TAGS, "source_commit": _source_commit()})
    root = DATA_ROOT / "datasets/syn4d-mvtracker"
    failures: list[dict[str, str]] = []
    checked = 0
    started = time.perf_counter()
    try:
        for split in ("train", "validation"):
            for scene in sorted(path for path in (root / split).iterdir() if path.is_dir() and not path.name.startswith(".")):
                try:
                    manifest = json.loads((scene / "manifest.json").read_text())
                    frames, tracks, views = int(manifest["frames"]), int(manifest["tracks"]), int(manifest["views"])
                    if manifest["format"] != "syn4d-tapvid-mv" or views != 8 or tracks <= 0:
                        raise ValueError("invalid manifest")
                    specs = {"tracks_xyz": (np.float32, (frames, tracks, 3)), "track_valid": (np.bool_, (frames, tracks)), "queries_xytv": (np.float32, (tracks, 4)), "motion_path_length": (np.float32, (tracks,))}
                    for name, (dtype, shape) in specs.items():
                        array = np.load(scene / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                        if array.dtype != dtype or array.shape != shape:
                            raise ValueError(f"{name} contract")
                    for view in range(views):
                        view_root = scene / str(view)
                        for name, dtype, shape in (("depth", np.float32, (frames, 384, 683)), ("intrinsics", np.float32, (frames, 3, 3)), ("extrinsics_w2c", np.float32, (frames, 4, 4)), ("visibility", np.bool_, (frames, tracks))):
                            array = np.load(view_root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                            if array.dtype != dtype or array.shape != shape:
                                raise ValueError(f"view {view}/{name} contract")
                        offsets = np.load(scene / f"view_{view}" / "jpeg_offsets.npy", mmap_mode="r", allow_pickle=False)
                        if offsets.dtype != np.int64 or offsets.shape != (frames + 1,) or offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1]):
                            raise ValueError(f"view {view}/jpeg offsets")
                        if offsets[-1] != (scene / f"view_{view}" / "jpeg_bytes.bin").stat().st_size:
                            raise ValueError(f"view {view}/jpeg byte count")
                except Exception as error:
                    failures.append({"split": split, "scene": scene.name, "error": f"{type(error).__name__}: {error}"})
                checked += 1
                if checked % 10 == 0:
                    payload = {"checked": checked, "failures": len(failures), "elapsed_seconds": round(time.perf_counter() - started, 2)}
                    print(json.dumps(payload), flush=True)
                    run.log(payload)
        result = {"checked": checked, "failures": failures, "elapsed_seconds": round(time.perf_counter() - started, 2)}
        run.log({"checked": checked, "failures": len(failures), "elapsed_seconds": result["elapsed_seconds"]})
        print(json.dumps(result), flush=True)
        return result
    finally:
        run.finish()


@app.local_entrypoint()
def audit() -> None:
    print(json.dumps(audit_remote.remote(), indent=2, sort_keys=True))
