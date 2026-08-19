"""Locked exactness contract and benchmark for UpdateFormer research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch


FORMAT = "mvtracker-updateformer-contract-v1"
MODEL_SEED = 20260819
INPUT_DIM = 581
HIDDEN_SIZE = 256
OUTPUT_DIM = 131


@dataclass(frozen=True)
class Workload:
    name: str
    batch_size: int
    tracks: int
    real_tracks: tuple[int, ...]
    seed: int


WORKLOADS = (
    Workload("single_512", 1, 512, (512,), 1101),
    Workload("single_1536", 1, 1536, (1536,), 1102),
    Workload("paired_ragged_1024", 2, 1024, (1024, 731), 1103),
    Workload("quad_ragged_512", 4, 512, (512, 447, 381, 256), 1104),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verifier_sha256() -> str:
    return _sha256(Path(__file__))


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
    }


def _named_tensor_records(value, prefix="") -> dict[str, dict[str, object]]:
    records = {}
    if isinstance(value, torch.Tensor):
        records[prefix or "tensor"] = tensor_record(value)
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            records.update(
                _named_tensor_records(value[key], f"{prefix}.{key}" if prefix else str(key))
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            records.update(
                _named_tensor_records(item, f"{prefix}.{index}" if prefix else str(index))
            )
    return records


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _build_model(device):
    from mvtracker.models.core.cotracker2.blocks import (
        EfficientUpdateFormer,
        FlashAttention,
    )

    torch.manual_seed(MODEL_SEED)
    model = EfficientUpdateFormer(
        space_depth=6,
        time_depth=6,
        input_dim=INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        num_heads=6,
        output_dim=OUTPUT_DIM,
        mlp_ratio=4.0,
        add_space_attn=True,
        num_virtual_tracks=64,
        attn_class=FlashAttention,
    )
    return model.to(device).train()


def _workload_tensors(workload: Workload, device):
    generator = torch.Generator(device="cpu").manual_seed(workload.seed)
    inputs = torch.randn(
        workload.batch_size,
        workload.tracks,
        12,
        INPUT_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    inputs.requires_grad_(True)
    target = torch.randn(
        workload.batch_size,
        workload.tracks,
        12,
        OUTPUT_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    weights = torch.randn(
        workload.batch_size,
        workload.tracks,
        12,
        OUTPUT_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    mask = torch.zeros(
        workload.batch_size, workload.tracks, dtype=torch.bool, device=device
    )
    for index, count in enumerate(workload.real_tracks):
        mask[index, :count] = True
    return inputs, target, weights, mask


def _run_updateformer_case(workload: Workload, state, device):
    model = _build_model(device)
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    inputs, target, weights, mask = _workload_tensors(workload, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(inputs, point_mask=mask)
        loss = ((output.float() - target) * weights).square().mean()
    loss.backward()
    gradients = {
        name: tensor_record(parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    result = {
        "output": tensor_record(output),
        "loss": tensor_record(loss),
        "input_gradient": tensor_record(inputs.grad),
        "parameter_gradients": gradients,
    }
    optimizer.step()
    result["updated_parameters"] = _named_tensor_records(model.state_dict())
    result["optimizer_state"] = _named_tensor_records(optimizer.state_dict())
    torch.cuda.synchronize()
    return result


def _run_loss_contract(device):
    from mvtracker.models.core.losses import balanced_ce_loss, sequence_loss_3d

    generator = torch.Generator(device="cpu").manual_seed(2201)
    flow_predictions = []
    flow_leaves = []
    flow_ground_truth = []
    visibility = []
    validity = []
    for window, tracks in enumerate((37, 53)):
        predictions = []
        leaves = []
        for _ in range(4):
            leaf = torch.randn(
                2, 12, tracks, 3, generator=generator, dtype=torch.float32
            ).to(device)
            leaf.requires_grad_(True)
            leaves.append(leaf)
            predictions.append(leaf + 0.0)
        flow_predictions.append(predictions)
        flow_leaves.append(leaves)
        flow_ground_truth.append(
            torch.randn(2, 12, tracks, 3, generator=generator).to(device)
        )
        visibility.append(
            (torch.rand(2, 12, tracks, generator=generator) > 0.25).to(device)
        )
        validity.append(
            (torch.rand(2, 12, tracks, generator=generator) > 0.10).to(device)
        )
    flow_loss = sequence_loss_3d(
        flow_predictions, flow_ground_truth, visibility, validity, gamma=0.8
    )
    flow_loss.backward()

    logits = []
    labels = []
    valid_logits = []
    for tracks in (41, 59):
        logit = torch.randn(2, 12, tracks, generator=generator).to(device)
        logit.requires_grad_(True)
        logits.append(logit)
        labels.append(
            (torch.rand(2, 12, tracks, generator=generator) > 0.4).float().to(device)
        )
        valid_logits.append(
            (torch.rand(2, 12, tracks, generator=generator) > 0.1).float().to(device)
        )
    visibility_loss = balanced_ce_loss(logits, labels, valid_logits)
    visibility_loss.backward()
    return {
        "sequence_loss": tensor_record(flow_loss),
        "sequence_gradients": [
            [tensor_record(leaf.grad) for leaf in leaves]
            for leaves in flow_leaves
        ],
        "balanced_ce_loss": tensor_record(visibility_loss),
        "balanced_ce_gradients": [tensor_record(logit.grad) for logit in logits],
    }


def _exact_mismatch(expected, actual, path="root"):
    if type(expected) is not type(actual):
        return f"{path}: type {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return f"{path}: keys differ"
        for key in expected:
            mismatch = _exact_mismatch(expected[key], actual[key], f"{path}.{key}")
            if mismatch:
                return mismatch
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: list lengths differ"
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatch = _exact_mismatch(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
    elif expected != actual:
        return f"{path}: {actual!r} != {expected!r}"
    return None


def capture_golden(root: Path) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("golden capture requires exactly one CUDA device")
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"golden contract already exists: {root}")
    partial_root = root.with_name(f".{root.name}.partial")
    if partial_root.exists():
        raise FileExistsError(f"unfinished golden capture exists: {partial_root}")
    partial_root.mkdir(parents=True)
    device = torch.device("cuda:0")
    model = _build_model(device)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state_path = partial_root / "baseline_state.pt"
    torch.save(state, state_path)
    first = {
        workload.name: _run_updateformer_case(workload, state, device)
        for workload in WORKLOADS
    }
    losses = _run_loss_contract(device)
    second = {
        workload.name: _run_updateformer_case(workload, state, device)
        for workload in WORKLOADS
    }
    second_losses = _run_loss_contract(device)
    mismatch = _exact_mismatch(first, second) or _exact_mismatch(losses, second_losses)
    if mismatch:
        raise RuntimeError(f"baseline is not bit-exact: {mismatch}")
    manifest = {
        "format": FORMAT,
        "baseline_commit": _git_commit(),
        "verifier_sha256": verifier_sha256(),
        "baseline_state": state_path.name,
        "baseline_state_sha256": _sha256(state_path),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "workloads": [asdict(workload) for workload in WORKLOADS],
        "cases": first,
        "loss_contract": losses,
    }
    (partial_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    partial_root.replace(root)
    return manifest


def verify_golden(root: Path) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("verification requires exactly one CUDA device")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != FORMAT:
        raise RuntimeError("unknown verifier contract format")
    if manifest["verifier_sha256"] != verifier_sha256():
        raise RuntimeError("verifier source differs from the locked contract")
    state_path = root / manifest["baseline_state"]
    if _sha256(state_path) != manifest["baseline_state_sha256"]:
        raise RuntimeError("baseline model state hash mismatch")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    device = torch.device("cuda:0")
    actual_cases = {
        workload.name: _run_updateformer_case(workload, state, device)
        for workload in WORKLOADS
    }
    actual_losses = _run_loss_contract(device)
    mismatch = _exact_mismatch(manifest["cases"], actual_cases)
    mismatch = mismatch or _exact_mismatch(manifest["loss_contract"], actual_losses)
    if mismatch:
        raise RuntimeError(f"exactness contract failed: {mismatch}")
    return {
        "status": "exact",
        "baseline_commit": manifest["baseline_commit"],
        "candidate_commit": _git_commit(),
        "workloads": [workload.name for workload in WORKLOADS],
        "loss_contract": "exact",
    }


def benchmark(root: Path, warmup=3, measured=10) -> dict[str, object]:
    verify_golden(root)
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    state = torch.load(
        Path(root) / manifest["baseline_state"], map_location="cpu", weights_only=True
    )
    device = torch.device("cuda:0")
    results = {}
    for workload in WORKLOADS:
        model = _build_model(device)
        model.load_state_dict(state, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
        base_inputs, target, weights, mask = _workload_tensors(workload, device)
        base_inputs = base_inputs.detach()

        def prepare():
            return base_inputs.clone().requires_grad_(True)

        def step(inputs):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(inputs, point_mask=mask)
                loss = ((output.float() - target) * weights).square().mean()
            loss.backward()
            optimizer.step()

        for _ in range(int(warmup)):
            step(prepare())
        durations = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(int(measured)):
            inputs = prepare()
            torch.cuda.synchronize()
            started = time.perf_counter()
            step(inputs)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
        results[workload.name] = {
            "median_seconds": statistics.median(durations),
            "minimum_seconds": min(durations),
            "maximum_seconds": max(durations),
            "durations_seconds": durations,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    return {
        "status": "verified",
        "candidate_commit": _git_commit(),
        "warmup": int(warmup),
        "measured": int(measured),
        "results": results,
    }
