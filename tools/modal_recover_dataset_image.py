"""Recover the completed MV-Kubric image layer as a clean filesystem snapshot."""

from __future__ import annotations

import json

import modal


SOURCE_IMAGE_ID = "im-SAhj7qgBbNxXId6CxLb5WG"
EXPECTED_SCENES = tuple(str(scene) for scene in range(1001, 1501))
WANDB_ENTITY = "jeetucl-ucl"
WANDB_PROJECT = "mvtracker-continual-training"

app = modal.App(
    "jeet-mvtracker-dataset-image-recovery",
    tags={
        "owner": "jeet",
        "project": "mvtracker",
        "purpose": "profiling",
        "experiment": "dataset-image-recovery",
        "gpu": "cpu",
    },
)
wandb_secret = modal.Secret.from_name(
    "jeet-mvtracker-wandb", required_keys=["WANDB_API_KEY"]
)


@app.local_entrypoint()
def main() -> None:
    sandbox = modal.Sandbox.create(
        "sleep",
        "infinity",
        app=app,
        image=modal.Image.from_id(SOURCE_IMAGE_ID),
        secrets=[wandb_secret],
        cpu=1,
        memory=2048,
        timeout=30 * 60,
    )
    try:
        verification = sandbox.exec(
            "python",
            "-c",
            (
                "from pathlib import Path; import json; "
                "p=Path('/opt/mvtracker-data/datasets/kubric-multiview/train'); "
                "s=sorted((x.name for x in p.iterdir() if x.is_dir()), key=int); "
                f"assert s == {list(EXPECTED_SCENES)!r}; "
                "print(json.dumps({'scene_count':len(s),'scene_start':s[0],"
                "'scene_end':s[-1]}))"
            ),
        )
        output = verification.stdout.read().strip()
        verification.wait()
        recovered = sandbox.snapshot_filesystem(timeout=20 * 60)
        result = {**json.loads(output), "source_image_id": SOURCE_IMAGE_ID, "image_id": recovered.object_id}
        logging = sandbox.exec(
            "python",
            "-c",
            (
                "import json,os,wandb; r=json.loads(os.environ['RECOVERY_RESULT']); "
                f"w=wandb.init(entity={WANDB_ENTITY!r},project={WANDB_PROJECT!r},"
                "job_type='dataset-image-recovery',tags=['modal','cpu','dataset-image']); "
                "w.summary.update(r); w.finish()"
            ),
            env={"RECOVERY_RESULT": json.dumps(result)},
        )
        logging.wait()
        print(json.dumps(result, indent=2))
    finally:
        sandbox.terminate()
