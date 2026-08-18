"""CPU-only Modal setup for the continual-training dataset image."""

from __future__ import annotations

import modal

from modal_training_profile import (
    DATA_ROOT,
    _runtime_image,
    _source_commit,
    data_volume,
    hf_secret,
    wandb_secret,
)
from mvtracker.profiling.modal_continual_training import (
    EPHEMERAL_DISK_MIB,
    MAX_CONTAINERS,
    PROFILE_TAGS,
    WANDB_ENTITY,
    WANDB_GROUP,
    WANDB_PROJECT,
    require_pushed_main_commit,
)


app = modal.App(
    "jeet-mvtracker-continual-data-setup",
    tags={**PROFILE_TAGS, "experiment": "data-setup", "gpu": "cpu"},
)

@app.function(
    image=_runtime_image(),
    secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=32768,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    timeout=24 * 60 * 60,
    max_containers=MAX_CONTAINERS,
    include_source=False,
)
def setup_training_data_remote() -> dict:
    import wandb

    from mvtracker.profiling.modal_continual_data import (
        materialize_expanded_continual_training_data,
    )

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="data-setup",
        tags=["modal", "data-setup", "gt-depth-replay-v1"],
        config={"source_commit": _source_commit(), **PROFILE_TAGS},
    )
    manifest = materialize_expanded_continual_training_data(DATA_ROOT)
    data_volume.commit()
    run.summary.update(
        {
            "mvkubric_train_scenes": manifest["mvkubric"]["train_scene_count"],
            "mvkubric_validation_scenes": manifest["mvkubric"][
                "validation_scene_count"
            ],
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        }
    )
    run.finish()
    return manifest


@app.local_entrypoint(name="setup-data")
def setup_data() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    call = setup_training_data_remote.spawn()
    print(f"FUNCTION_CALL {call.object_id}")
