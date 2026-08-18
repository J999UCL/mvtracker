"""Materialize the pinned 2,000-scene MV-Kubric inputs on the data Volume."""

from __future__ import annotations

import json
from pathlib import Path

import modal

from modal_training_profile import DATA_ROOT, _runtime_image, data_volume, hf_secret, wandb_secret


app = modal.App(
    "jeet-mvkubric-validation",
    tags={"owner": "jeet", "project": "mvtracker", "purpose": "profiling"},
)
image = _runtime_image()


@app.function(
    image=image,
    secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=65536,
    ephemeral_disk=512 * 1024,
    timeout=12 * 60 * 60,
    max_containers=1,
)
def materialize_validation() -> dict[str, object]:
    import os
    import wandb

    from mvtracker.profiling.modal_mvkubric2000 import materialize_mvkubric2000

    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-modal-profiling",
        job_type="mvkubric-2000-scene-setup",
        tags=["modal", "mv-kubric", "validation", "archive"],
        config={"owner": "jeet", "project": "mvtracker", "purpose": "profiling"},
    )
    try:
        manifest = materialize_mvkubric2000(DATA_ROOT, os.environ["HF_TOKEN"])
        data_volume.commit()
        run.summary.update(
            {
                "train_scene_count": manifest["train_scene_count"],
                "validation_scene_count": manifest["validation_scene_count"],
                "archive_root": str(Path(DATA_ROOT) / "archives/mvkubric/2000-scenes-v1"),
            }
        )
        return manifest
    finally:
        run.finish()


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(materialize_validation.remote(), indent=2, sort_keys=True))
