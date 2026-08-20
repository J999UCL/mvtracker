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
CONTRACT_ROOT = RUN_ROOT / "performance-contracts" / "updateformer-v3"
CANDIDATE_BATCH = (
    RUN_ROOT
    / "continual-training/direct-volume-v2-smoke10-ddp2-h100-20260819T0110Z"
    / "crash_batch_step_000000.pt"
)
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
        "component-diagnose",
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
    elif action == "forward-graph-study":
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
    else:
        from mvtracker.profiling.updateformer_diagnostics import (
            diagnose_fused_components,
        )

        verify_golden(CONTRACT_ROOT)
        result = diagnose_fused_components(CONTRACT_ROOT)
        summary = {}
        for name, measurements in result["variants"].items():
            summary[f"components/{name}/max_abs"] = measurements[
                "maximum_absolute_error"
            ]
            summary[f"components/{name}/mean_abs"] = measurements[
                "mean_absolute_error"
            ]
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
def run_single_gpu_smoke(
    run_name: str,
    backend: str = "eager",
    steps: int = 10,
) -> dict:
    commit = _source_commit()
    inductor_cache = RUN_ROOT / "torchinductor-cache" / "torch2.7.1-cu128-h100"
    inductor_cache.mkdir(parents=True, exist_ok=True)
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
            "TORCHINDUCTOR_CACHE_DIR": str(inductor_cache),
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
                f"model.updateformer_backend={backend}",
                "model.checkpoint_updateformer=false",
                f"trainer.num_steps={steps}",
                f"trainer.save_ckpt_freq={steps}",
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
        "backend": backend,
        "steps": steps,
        "elapsed_seconds": time.perf_counter() - started,
        "log_path": str(log_path),
    }


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu="H100!",
    cpu=8,
    memory=65536,
    timeout=3 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def run_fused_candidate_gate(candidate_backend: str = "fused") -> dict:
    inductor_cache = RUN_ROOT / "torchinductor-cache" / "torch2.7.1-cu128-h100"
    inductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    import torch
    import wandb

    from mvtracker.profiling.full_updateformer_candidate import compare_real_update

    commit = _source_commit()
    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-modal-profiling",
        group="updateformer-fused-backend",
        job_type="full-update-candidate-gate",
        tags=["modal", "h100", "single-gpu", "updateformer", candidate_backend],
        config={
            "source_commit": commit,
            "candidate_backend": candidate_backend,
            **TAGS,
        },
    )
    torch.set_float32_matmul_precision("high")
    output_root = RUN_ROOT / "performance-results" / commit
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{candidate_backend}-candidate-gate.json"
    result = compare_real_update(
        data_root=DATA_ROOT / "datasets",
        checkpoint=DATA_ROOT / "checkpoints/mvtracker_200000_june2025.pth",
        batch_cache=CANDIDATE_BATCH,
        output=output_path,
        candidate_backend=candidate_backend,
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    run_volume.commit()
    run.summary.update({
        "gate/passed": int(result["passed"]),
        "gate/trajectory_rms_m": result["final_trajectory"]["rms"],
        "gate/trajectory_p99_m": result["final_trajectory"]["p99"],
        "gate/visibility_mean_abs": result["final_visibility"]["mean"],
        "gate/gradient_cosine": result["gradients"]["cosine"],
        "gate/update_cosine": result["parameter_updates"]["cosine"],
        "gate/five_update_trajectory_rms_m": result["multi_update"]["final_trajectory"]["rms"],
        "gate/five_update_update_cosine": result["multi_update"]["parameter_updates"]["cosine"],
        "timing/eager_forward_seconds": result["timing"]["eager"]["forward_seconds"],
        "timing/fused_forward_seconds": result["timing"]["fused"].get(
            "forward_seconds",
            result["timing"]["fused"].get("capture_seconds", 0.0),
        ),
        "timing/eager_warm_update_seconds": result["timing"]["eager"]["warm_update_median_seconds"],
        "timing/fused_warm_update_seconds": result["timing"]["fused"]["warm_update_median_seconds"],
        "timing/amortized_speedup_1000": result["amortized_speedup"],
        "result_path": str(output_path),
    })
    if result["eager_repeat"] is not None:
        run.summary.update({
            "baseline/five_update_trajectory_rms_m": result["eager_repeat"][
                "final_trajectory"
            ]["rms"],
            "baseline/five_update_update_cosine": result["eager_repeat"][
                "parameter_updates"
            ]["cosine"],
        })
    run.finish()
    return {"result_path": str(output_path), **result}


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu="H100!",
    cpu=8,
    memory=65536,
    timeout=6 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def run_candidate_sweep() -> dict:
    inductor_cache = RUN_ROOT / "torchinductor-cache" / "torch2.7.1-cu128-h100"
    inductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    import torch
    import wandb

    from mvtracker.profiling.full_updateformer_candidate import compare_real_update

    commit = _source_commit()
    candidates = (
        "bucketed_reduce",
        "graphed",
        "graphed_bucketed",
        "whole_graph",
    )
    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-modal-profiling",
        group="updateformer-autoresearch-v2",
        job_type="whole-update-candidate-sweep",
        tags=["modal", "h100", "single-gpu", "autoresearch", "candidate-sweep"],
        config={
            "source_commit": commit,
            "candidates": candidates,
            **TAGS,
        },
    )
    torch.set_float32_matmul_precision("high")
    output_root = RUN_ROOT / "performance-results" / commit / "candidate-sweep"
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for candidate in candidates:
        print(f"AUTORESEARCH candidate={candidate} status=started", flush=True)
        output_path = output_root / f"{candidate}.json"
        try:
            result = compare_real_update(
                data_root=DATA_ROOT / "datasets",
                checkpoint=DATA_ROOT / "checkpoints/mvtracker_200000_june2025.pth",
                batch_cache=CANDIDATE_BATCH,
                output=output_path,
                candidate_backend=candidate,
            )
        except Exception as error:
            result = {
                "passed": False,
                "candidate_backend": candidate,
                "error": f"{type(error).__name__}: {error}",
            }
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results[candidate] = result
        summary = {
            f"{candidate}/passed": int(result["passed"]),
        }
        if "speedup" in result:
            summary.update({
                f"{candidate}/behavior_passed": int(result["behavior_passed"]),
                f"{candidate}/speedup": result["speedup"],
                f"{candidate}/amortized_speedup": result["amortized_speedup"],
                f"{candidate}/peak_allocated_gib": (
                    result["memory"]["fused"]["peak_allocated_bytes"] / 2**30
                ),
            })
        if "error" in result:
            summary[f"{candidate}/error"] = result["error"]
        run.log(summary)
        print(
            f"AUTORESEARCH candidate={candidate} status=finished "
            f"passed={result['passed']} speedup={result.get('speedup')}",
            flush=True,
        )
    accepted = [
        result for result in results.values()
        if result.get("passed")
    ]
    live_candidates = {
        "bucketed",
        "bucketed_reduce",
        "graphed",
        "graphed_bucketed",
    }
    live_accepted = [
        result for result in accepted
        if result["candidate_backend"] in live_candidates
    ]
    winner = (
        max(accepted, key=lambda result: result["speedup"])["candidate_backend"]
        if accepted else None
    )
    live_winner = (
        max(live_accepted, key=lambda result: result["speedup"])[
            "candidate_backend"
        ]
        if live_accepted else None
    )
    summary_path = output_root / "summary.json"
    summary = {
        "source_commit": commit,
        "winner": winner,
        "live_winner": live_winner,
        "target_2x_reached": any(
            result.get("target_2x_reached", False) for result in accepted
        ),
        "results": {
            name: {
                key: value for key, value in result.items()
                if key in {
                    "passed",
                    "strict_passed",
                    "behavior_passed",
                    "performance_passed",
                    "speedup",
                    "amortized_speedup",
                    "target_2x_reached",
                    "error",
                }
            }
            for name, result in results.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    run_volume.commit()
    run.summary.update({
        "winner": winner or "none",
        "live_winner": live_winner or "none",
        "target_2x_reached": int(summary["target_2x_reached"]),
        "result_path": str(summary_path),
    })
    run.finish()
    return {"result_path": str(summary_path), **summary}


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


@app.local_entrypoint(name="component-diagnose")
def component_diagnose() -> None:
    _launch("component-diagnose")


@app.local_entrypoint(name="single-gpu-smoke")
def single_gpu_smoke(
    run_name: str = "",
    backend: str = "eager",
    steps: int = 10,
) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    selected = run_name or f"updateformer-{backend}-h100-{commit[:8]}"
    app.set_tags(
        {
            **TAGS,
            "experiment": selected,
            "gpu": "h100",
        }
    )
    print(json.dumps(
        run_single_gpu_smoke.remote(selected, backend, steps), indent=2
    ))


@app.local_entrypoint(name="fused-candidate-gate")
def fused_candidate_gate(candidate_backend: str = "fused") -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    app.set_tags(
        {
            **TAGS,
            "experiment": f"updateformer-{candidate_backend}-gate-{commit[:8]}",
            "gpu": "h100",
        }
    )
    print(json.dumps(run_fused_candidate_gate.remote(candidate_backend), indent=2))


@app.local_entrypoint(name="candidate-sweep")
def candidate_sweep() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    app.set_tags({
        **TAGS,
        "experiment": f"updateformer-candidate-sweep-{commit[:8]}",
        "gpu": "h100",
    })
    print(json.dumps(run_candidate_sweep.remote(), indent=2))


@app.local_entrypoint(name="autoresearch")
def autoresearch(steps: int = 10) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    app.set_tags({
        **TAGS,
        "experiment": f"updateformer-autoresearch-{commit[:8]}",
        "gpu": "h100",
    })
    sweep = run_candidate_sweep.remote()
    live_winner = sweep["live_winner"]
    result = {"sweep": sweep, "live_confirmation": None}
    if live_winner is not None:
        run_name = f"updateformer-{live_winner}-live-{commit[:8]}"
        result["live_confirmation"] = run_single_gpu_smoke.remote(
            run_name,
            live_winner,
            steps,
        )
    print(json.dumps(result, indent=2))
