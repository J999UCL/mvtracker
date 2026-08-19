"""Locked exactness contract and benchmark for UpdateFormer research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

import torch


FORMAT = "mvtracker-updateformer-contract-v3"
MODEL_SEED = 20260819
INPUT_DIM = 581
HIDDEN_SIZE = 256
OUTPUT_DIM = 131
FLOAT_RTOL = 1e-4
FLOAT_ATOL = 1e-5
LOW_PRECISION_MAX_ULP = 1
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Workload:
    name: str
    batch_size: int
    tracks: int
    real_tracks: tuple[int, ...]
    seed: int


WORKLOADS = (
    Workload("single_512", 1, 512, (512,), 1101),
    Workload("single_777", 1, 777, (777,), 1105),
    Workload("single_1536", 1, 1536, (1536,), 1102),
    Workload("paired_ragged_900", 2, 900, (900, 623), 1103),
    Workload("quad_ragged_333", 4, 333, (333, 287, 211, 149), 1104),
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


def _cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def _git_commit() -> str:
    commit = os.environ.get("MVTRACKER_MODAL_COMMIT", "")
    if _COMMIT.fullmatch(commit) is None:
        raise RuntimeError("MVTRACKER_MODAL_COMMIT must be one full Git commit")
    return commit


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
        squared_error = ((output.float() - target) * weights).square()
        valid = mask[:, :, None, None]
        loss = (squared_error * valid).sum() / (
            valid.sum() * output.shape[2] * output.shape[3]
        )
    loss.backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    result = {
        "output": [
            output[index, :count]
            for index, count in enumerate(workload.real_tracks)
        ],
        "loss": loss,
        "input_gradient": [
            inputs.grad[index, :count]
            for index, count in enumerate(workload.real_tracks)
        ],
        "parameter_gradients": gradients,
    }
    optimizer.step()
    result["updated_parameters"] = model.state_dict()
    result["optimizer_state"] = optimizer.state_dict()
    torch.cuda.synchronize()
    return _cpu_clone(result)


def _run_loss_contract(device):
    from mvtracker.models.core.losses import balanced_ce_loss, sequence_loss_3d

    generator = torch.Generator(device="cpu").manual_seed(2201)
    flow_predictions = []
    flow_leaves = []
    flow_ground_truth = []
    visibility = []
    validity = []
    for tracks in (37, 53):
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
    return _cpu_clone(
        {
            "sequence_loss": flow_loss,
            "sequence_gradients": [
                [leaf.grad for leaf in leaves] for leaves in flow_leaves
            ],
            "balanced_ce_loss": visibility_loss,
            "balanced_ce_gradients": [logit.grad for logit in logits],
        }
    )


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


def _close_mismatch(expected, actual, path="root"):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            return f"{path}: expected tensor"
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            return (
                f"{path}: tensor metadata differs: "
                f"{tuple(actual.shape)}/{actual.dtype} != "
                f"{tuple(expected.shape)}/{expected.dtype}"
            )
        if expected.is_floating_point():
            low_precision = expected.dtype in {torch.bfloat16, torch.float16}
            if low_precision:
                left_bits = expected.view(torch.int16).to(torch.int32) & 0xFFFF
                right_bits = actual.view(torch.int16).to(torch.int32) & 0xFFFF
                left_ordered = torch.where(
                    left_bits & 0x8000 != 0,
                    0xFFFF - left_bits,
                    0x8000 + left_bits,
                )
                right_ordered = torch.where(
                    right_bits & 0x8000 != 0,
                    0xFFFF - right_bits,
                    0x8000 + right_bits,
                )
                ulp = (right_ordered - left_ordered).abs()
                close = bool((ulp <= LOW_PRECISION_MAX_ULP).all())
            else:
                close = torch.allclose(
                    expected, actual, rtol=FLOAT_RTOL, atol=FLOAT_ATOL
                )
            if not close:
                difference = (actual.float() - expected.float()).abs()
                return f"{path}: max absolute error {difference.max().item():.8g}"
        elif not torch.equal(expected, actual):
            return f"{path}: tensor values differ"
        return None
    if type(expected) is not type(actual):
        return f"{path}: type {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return f"{path}: keys differ"
        for key in expected:
            mismatch = _close_mismatch(expected[key], actual[key], f"{path}.{key}")
            if mismatch:
                return mismatch
    elif isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return f"{path}: sequence lengths differ"
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatch = _close_mismatch(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
    elif expected != actual:
        return f"{path}: {actual!r} != {expected!r}"
    return None


def _difference_summary(expected, actual):
    differences = []

    def collect(left, right):
        if isinstance(left, torch.Tensor) and left.is_floating_point():
            differences.append((right.float() - left.float()).abs().max().item())
        elif isinstance(left, dict):
            for key in left:
                collect(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            for left_item, right_item in zip(left, right):
                collect(left_item, right_item)

    collect(expected, actual)
    return {
        "floating_tensor_count": len(differences),
        "changed_tensor_count": sum(value > 0 for value in differences),
        "maximum_absolute_error": max(differences, default=0.0),
    }


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
    mismatch = _close_mismatch(first, second) or _close_mismatch(losses, second_losses)
    if mismatch:
        raise RuntimeError(f"baseline exceeds the numerical contract: {mismatch}")
    reference = {"cases": first, "loss_contract": losses}
    reference_path = partial_root / "reference.pt"
    torch.save(reference, reference_path)
    manifest = {
        "format": FORMAT,
        "baseline_commit": _git_commit(),
        "verifier_sha256": verifier_sha256(),
        "baseline_state": state_path.name,
        "baseline_state_sha256": _sha256(state_path),
        "reference": reference_path.name,
        "reference_sha256": _sha256(reference_path),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "workloads": [asdict(workload) for workload in WORKLOADS],
        "float_rtol": FLOAT_RTOL,
        "float_atol": FLOAT_ATOL,
        "low_precision_max_ulp": LOW_PRECISION_MAX_ULP,
        "baseline_replay": {
            "cases": _difference_summary(first, second),
            "loss_contract": _difference_summary(losses, second_losses),
        },
        "cases": _named_tensor_records(first),
        "loss_contract": _named_tensor_records(losses),
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
    reference_path = root / manifest["reference"]
    if _sha256(reference_path) != manifest["reference_sha256"]:
        raise RuntimeError("golden reference hash mismatch")
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    device = torch.device("cuda:0")
    actual_cases = {
        workload.name: _run_updateformer_case(workload, state, device)
        for workload in WORKLOADS
    }
    actual_losses = _run_loss_contract(device)
    mismatch = _close_mismatch(reference["cases"], actual_cases)
    mismatch = mismatch or _close_mismatch(
        reference["loss_contract"], actual_losses
    )
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
                squared_error = ((output.float() - target) * weights).square()
                valid = mask[:, :, None, None]
                loss = (squared_error * valid).sum() / (
                    valid.sum() * output.shape[2] * output.shape[3]
                )
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
