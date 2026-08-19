"""Single-H100 exactness gate and benchmark for UpdateFormer optimization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

from modal_training_profile import (
    DATA_ROOT,
    RUN_ROOT,
    _dependency_image,
    _source_commit,
    _source_image,
    data_volume,
    run_volume,
    wandb_secret,
)
from mvtracker.profiling.modal_continual_training import (
    preflight_active_containers,
    require_pushed_main_commit,
)


APP_NAME = "jeet-mvtracker-updateformer-research"
CONTRACT_ROOT = RUN_ROOT / "performance-contracts" / "updateformer-v2"
TAGS = {
    "owner": "jeet",
    "project": "mvtracker",
    "purpose": "profiling",
}

app = modal.App(APP_NAME, tags={**TAGS, "experiment": "unclassified", "gpu": "h100"})
image = _source_image(_dependency_image())


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(RUN_ROOT): run_volume},
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def run_contract(action: str, warmup: int = 3, measured: int = 10) -> dict:
    import torch
    import wandb

    from mvtracker.profiling.updateformer_contract import (
        benchmark,
        capture_golden,
        verify_golden,
    )

    if action not in {
        "capture",
        "verify",
        "benchmark",
        "diagnose",
        "checkpoint-study",
        "capacity-study",
        "cuda-graph-study",
        "partial-graph-study",
        "forward-graph-study",
    }:
        raise ValueError(f"unsupported action: {action}")
    torch.set_float32_matmul_precision("high")
    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-modal-profiling",
        group="updateformer-autoresearch-v2",
        job_type=f"updateformer-{action}",
        tags=["modal", "h100", "single-gpu", "updateformer", action],
        config={
            "source_commit": _source_commit(),
            "action": action,
            "warmup": warmup,
            "measured": measured,
            **TAGS,
        },
    )
    if action == "capture":
        result = capture_golden(CONTRACT_ROOT)
        run_volume.commit()
        summary = {
            "contract/baseline_commit": result["baseline_commit"],
            "contract/workloads": len(result["workloads"]),
        }
    elif action == "verify":
        result = verify_golden(CONTRACT_ROOT)
        summary = {"contract/exact": 1}
    elif action == "benchmark":
        result = benchmark(CONTRACT_ROOT, warmup=warmup, measured=measured)
        summary = {"contract/exact": 1}
        for name, measurements in result["results"].items():
            summary[f"benchmark/{name}/median_seconds"] = measurements[
                "median_seconds"
            ]
            summary[f"benchmark/{name}/peak_allocated_gib"] = (
                measurements["peak_allocated_bytes"] / 2**30
            )
    elif action == "diagnose":
        from mvtracker.profiling.updateformer_diagnostics import compare_candidate

        result = compare_candidate(CONTRACT_ROOT)
        summary = {
            "diagnostics/changed_tensors": result["changed_tensors"],
            "diagnostics/low_precision_over_one_ulp": result[
                "low_precision_over_one_ulp"
            ],
        }
    elif action == "checkpoint-study":
        from mvtracker.profiling.updateformer_diagnostics import (
            benchmark_checkpointing,
        )

        verify_golden(CONTRACT_ROOT)
        result = benchmark_checkpointing(CONTRACT_ROOT)
        summary = {"contract/exact": 1}
        for name, measurements in result["results"].items():
            summary[f"checkpoint/{name}/median_seconds"] = measurements[
                "median_seconds"
            ]
            summary[f"checkpoint/{name}/peak_allocated_gib"] = (
                measurements["peak_allocated_bytes"] / 2**30
            )
    elif action == "capacity-study":
        from mvtracker.profiling.updateformer_diagnostics import (
            benchmark_checkpoint_capacity,
        )

        verify_golden(CONTRACT_ROOT)
        result = benchmark_checkpoint_capacity(CONTRACT_ROOT)
        summary = {"contract/exact": 1}
        for name, measurements in result["results"].items():
            summary[f"capacity/{name}/scenes_per_second"] = measurements[
                "scenes_per_second"
            ]
            summary[f"capacity/{name}/peak_allocated_gib"] = (
                measurements["peak_allocated_bytes"] / 2**30
            )
    elif action == "cuda-graph-study":
        from mvtracker.profiling.updateformer_diagnostics import (
            benchmark_cuda_graphs,
        )

        verify_golden(CONTRACT_ROOT)
        result = benchmark_cuda_graphs(CONTRACT_ROOT)
        summary = {"contract/exact": 1}
        for name, measurements in result["results"].items():
            summary[f"cuda_graph/{name}/scenes_per_second"] = measurements[
                "scenes_per_second"
            ]
            summary[f"cuda_graph/{name}/peak_allocated_gib"] = (
                measurements["peak_allocated_bytes"] / 2**30
            )
    elif action == "partial-graph-study":
        from mvtracker.profiling.updateformer_diagnostics import (
            benchmark_graphed_callables,
        )

        verify_golden(CONTRACT_ROOT)
        result = benchmark_graphed_callables(CONTRACT_ROOT)
        summary = {"contract/exact": 1}
        for name, measurements in result["results"].items():
            summary[f"partial_graph/{name}/scenes_per_second"] = measurements[
                "scenes_per_second"
            ]
            summary[f"partial_graph/{name}/peak_allocated_gib"] = (
                measurements["peak_allocated_bytes"] / 2**30
            )
    else:
        from mvtracker.profiling.updateformer_diagnostics import (
            benchmark_forward_graph_checkpoint,
        )

        verify_golden(CONTRACT_ROOT)
        result = benchmark_forward_graph_checkpoint(CONTRACT_ROOT)
        summary = {"contract/exact": 1}
        for name, measurements in result["results"].items():
            summary[f"forward_graph/{name}/scenes_per_second"] = measurements[
                "scenes_per_second"
            ]
            summary[f"forward_graph/{name}/peak_allocated_gib"] = (
                measurements["peak_allocated_bytes"] / 2**30
            )
    output_root = RUN_ROOT / "performance-results" / _source_commit()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{action}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    run_volume.commit()
    run.summary.update({**summary, "result_path": str(output_path)})
    run.finish()
    return {"result_path": str(output_path), **summary}


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu="H100!",
    cpu=32,
    memory=65536,
    timeout=2 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def run_single_gpu_smoke(run_name: str) -> dict:
    commit = _source_commit()
    run_dir = RUN_ROOT / "single-gpu-performance" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_id = __import__("hashlib").sha256(run_name.encode()).hexdigest()[:12]
    environment = os.environ.copy()
    environment.update(
        {
            "MVTRACKER_TRAINING_RUN_DIR": str(run_dir),
            "MVTRACKER_TRAINING_CHECKPOINT": str(
                DATA_ROOT / "checkpoints/mvtracker_200000_june2025.pth"
            ),
            "MVTRACKER_DATA_ROOT": str(DATA_ROOT),
            "MVTRACKER_MVKUBRIC_INDEX_ROOT": str(
                DATA_ROOT / "datasets/kubric-multiview/train/MVTracker_index"
            ),
            "MVTRACKER_TRAINING_SEED": "20260820",
            "MVTRACKER_WANDB_RUN_NAME": run_name,
            "MVTRACKER_WANDB_RUN_ID": wandb_id,
            "WANDB_ENTITY": "jeetucl-ucl",
            "WANDB_PROJECT": "mvtracker-continual-training",
            "WANDB_RUN_GROUP": "updateformer-autoresearch-v2",
            "WANDB_RUN_ID": wandb_id,
            "WANDB_RESUME": "allow",
        }
    )
    log_path = run_dir / "training.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "mvtracker.cli.train",
                "+experiment=diegesis_mvkubric_gt_single_gpu_perf",
            ],
            cwd="/opt/mvtracker",
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    run_volume.commit()
    if completed.returncode != 0:
        raise RuntimeError(f"single-GPU smoke failed; see {log_path}")
    return {
        "source_commit": commit,
        "run_name": run_name,
        "elapsed_seconds": time.perf_counter() - started,
        "log_path": str(log_path),
    }


def _launch(action: str, warmup: int = 3, measured: int = 10) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    app.set_tags(
        {
            **TAGS,
            "experiment": f"updateformer-{action}-{commit[:8]}",
            "gpu": "h100",
        }
    )
    print(json.dumps(run_contract.remote(action, warmup, measured), indent=2))


@app.local_entrypoint(name="capture")
def capture() -> None:
    _launch("capture")


@app.local_entrypoint(name="verify")
def verify() -> None:
    _launch("verify")


@app.local_entrypoint(name="benchmark")
def run_benchmark(warmup: int = 3, measured: int = 10) -> None:
    _launch("benchmark", warmup, measured)


@app.local_entrypoint(name="diagnose")
def diagnose() -> None:
    _launch("diagnose")


@app.local_entrypoint(name="checkpoint-study")
def checkpoint_study() -> None:
    _launch("checkpoint-study")


@app.local_entrypoint(name="capacity-study")
def capacity_study() -> None:
    _launch("capacity-study")


@app.local_entrypoint(name="cuda-graph-study")
def cuda_graph_study() -> None:
    _launch("cuda-graph-study")


@app.local_entrypoint(name="partial-graph-study")
def partial_graph_study() -> None:
    _launch("partial-graph-study")


@app.local_entrypoint(name="forward-graph-study")
def forward_graph_study() -> None:
    _launch("forward-graph-study")


@app.local_entrypoint(name="single-gpu-smoke")
def single_gpu_smoke(run_name: str = "") -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    selected = run_name or f"updateformer-single-h100-{commit[:8]}"
    app.set_tags(
        {
            **TAGS,
            "experiment": selected,
            "gpu": "h100",
        }
    )
    print(json.dumps(run_single_gpu_smoke.remote(selected), indent=2))
