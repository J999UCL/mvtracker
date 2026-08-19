"""Single-H100 exactness gate and benchmark for UpdateFormer optimization."""

from __future__ import annotations

import json
from pathlib import Path

import modal

from modal_training_profile import (
    RUN_ROOT,
    _dependency_image,
    _source_commit,
    _source_image,
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
    else:
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
    output_root = RUN_ROOT / "performance-results" / _source_commit()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{action}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    run_volume.commit()
    run.summary.update({**summary, "result_path": str(output_path)})
    run.finish()
    return {"result_path": str(output_path), **summary}


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
