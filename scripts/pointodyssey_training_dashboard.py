#!/usr/bin/env python3
"""Serve a live MV-Tracker training dashboard from local files or W&B.

The dashboard is read-only. Local mode consumes TensorBoard, ``train.log``, and
NVML. W&B mode reads the same metrics from one explicitly selected W&B run.
Both modes push fresh snapshots to the browser with server-sent events.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_STREAM_INTERVAL_SECONDS = 1.0
DEFAULT_GPU_SAMPLE_INTERVAL_SECONDS = 2.0
DEFAULT_MAX_GPU_SAMPLES = 10_000
DEFAULT_WANDB_POLL_INTERVAL_SECONDS = 10.0

DATAPOINT_RE = re.compile(
    r"Datapoint:\s+\['([^']+)'\].*?waited\s+([0-9.]+)s"
)
TRACK_COUNT_RE = re.compile(r"FWD pass:.*?num_points=(\d+)")
FAILED_BATCH_RE = re.compile(r"batch is None: failed\s+(\d+)\s+/\s+(\d+)")
OPTIMIZER_CLIP_RE = re.compile(r"\[optimizer:(\d+)\].*?\sclipped=([01])(?:\s|$)")
TIMING_RE = re.compile(
    r"\[timing:(\d+)\]\s+Total:\s*([0-9.]+)s\s*\|\s*"
    r"Data:\s*([0-9.]+)s\s*\|\s*Fwd:\s*([0-9.]+)s\s*\|\s*"
    r"Sync:\s*([0-9.]+)s\s*\|\s*Bwd:\s*([0-9.]+)s"
)
FATAL_MARKERS = (
    "CUDA out of memory",
    "Error executing job with overrides",
    "Forward pass crashed at step",
)


def utc_iso(timestamp: float | None = None) -> str:
    value = time.time() if timestamp is None else timestamp
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def merge_scalar_series(
    preferred: list[dict[str, float | int]],
    supplemental: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    """Fill not-yet-flushed TensorBoard steps from the live training log."""
    by_step = {int(point["step"]): point for point in supplemental}
    by_step.update({int(point["step"]): point for point in preferred})
    return [by_step[step] for step in sorted(by_step)]


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def port_number(value: str) -> int:
    port = positive_int(value)
    if port > 65535:
        raise argparse.ArgumentTypeError("must be at most 65535")
    return port


@dataclass(frozen=True)
class RunConfig:
    total_steps: int
    gradient_accumulation_steps: int
    trajectory_cap: int | None
    num_workers: int | None
    sequence_len: int | None
    eval_frequency: int | None
    eval_datasets: tuple[str, ...]


def run_config_from_mapping(config: dict[str, Any]) -> RunConfig:
    def selected(path: str, default: Any = None) -> Any:
        value: Any = config
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    eval_names = selected("datasets.eval.names", []) or []
    return RunConfig(
        total_steps=int(selected("trainer.num_steps", 0)),
        gradient_accumulation_steps=int(
            selected("trainer.gradient_accumulation_steps", 1)
        ),
        trajectory_cap=(
            int(selected("datasets.train.traj_per_sample"))
            if selected("datasets.train.traj_per_sample") is not None
            else None
        ),
        num_workers=(
            int(selected("datasets.train.num_workers"))
            if selected("datasets.train.num_workers") is not None
            else None
        ),
        sequence_len=(
            int(selected("datasets.train.sequence_len"))
            if selected("datasets.train.sequence_len") is not None
            else None
        ),
        eval_frequency=(
            int(selected("trainer.eval_freq"))
            if selected("trainer.eval_freq") is not None
            else None
        ),
        eval_datasets=tuple(str(name) for name in eval_names),
    )


class HydraConfigReader:
    """Reload the Hydra config only when its file changes."""

    def __init__(self, path: Path):
        self.path = path
        self._signature: tuple[int, int] | None = None
        self._cached: RunConfig | None = None
        self.error: str | None = None

    def read(self) -> RunConfig | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.error = None
            return None
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self._signature:
            return self._cached
        try:
            from omegaconf import OmegaConf

            parsed = run_config_from_mapping(
                OmegaConf.to_container(OmegaConf.load(self.path), resolve=True)
            )
        except Exception as exc:
            self.error = f"cannot read Hydra config {self.path}: {exc}"
            return self._cached
        self._signature = signature
        self._cached = parsed
        self.error = None
        return parsed


class TensorBoardScalarReader:
    """Read all scalar series from live TensorBoard event files."""

    def __init__(self, event_dir: Path):
        self.event_dir = event_dir
        self._signature: tuple[tuple[str, int, int], ...] = ()
        self._event_names: tuple[str, ...] = ()
        self._accumulator: Any = None
        self._cached: dict[str, list[dict[str, float | int]]] = {}
        self.error: str | None = None

    def _event_signature(self) -> tuple[tuple[str, int, int], ...]:
        files = sorted(self.event_dir.glob("events.out.tfevents.*"))
        return tuple(
            (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in files
        )

    def read(self) -> dict[str, list[dict[str, float | int]]]:
        try:
            signature = self._event_signature()
        except OSError as exc:
            self.error = f"cannot inspect TensorBoard directory {self.event_dir}: {exc}"
            return self._cached
        if not signature:
            self.error = None
            return {}
        if signature == self._signature:
            return self._cached
        try:
            from tensorboard.backend.event_processing import event_accumulator

            event_names = tuple(item[0] for item in signature)
            if self._accumulator is None or event_names != self._event_names:
                self._accumulator = event_accumulator.EventAccumulator(
                    str(self.event_dir),
                    size_guidance={event_accumulator.SCALARS: 0},
                )
            self._accumulator.Reload()
            output: dict[str, list[dict[str, float | int]]] = {}
            for tag in sorted(self._accumulator.Tags().get("scalars", [])):
                by_step: dict[int, dict[str, float | int]] = {}
                for event in self._accumulator.Scalars(tag):
                    value = finite_float(event.value)
                    if value is None:
                        continue
                    by_step[int(event.step)] = {
                        "step": int(event.step),
                        "value": value,
                        "wall_time": float(event.wall_time),
                    }
                output[tag] = [by_step[step] for step in sorted(by_step)]
        except Exception as exc:
            self.error = f"cannot read TensorBoard scalars from {self.event_dir}: {exc}"
            return self._cached
        self._signature = signature
        self._event_names = event_names
        self._cached = output
        self.error = None
        return output


class WandbRunReader:
    """Incrementally read scalar history and metadata from one W&B run."""

    def __init__(
        self,
        run_path: str,
        poll_interval_seconds: float,
        api: Any = None,
    ):
        if api is None:
            import wandb

            api = wandb.Api()
        self.run_path = run_path
        self.run = api.run(run_path)
        self.poll_interval_seconds = poll_interval_seconds
        self._last_poll_monotonic: float | None = None
        self._next_history_step = 0
        self._scalars: dict[str, dict[int, dict[str, float | int]]] = {}
        self.error: str | None = None

    def read(self) -> dict[str, list[dict[str, float | int]]]:
        now = time.monotonic()
        if (
            self._last_poll_monotonic is not None
            and now - self._last_poll_monotonic < self.poll_interval_seconds
        ):
            return self._ordered_scalars()
        self._last_poll_monotonic = now
        try:
            self.run.load(force=True)
            rows = self.run.scan_history(
                page_size=1000,
                min_step=self._next_history_step,
            )
            for row in rows:
                history_step = int(row["_step"])
                self._next_history_step = max(
                    self._next_history_step, history_step + 1
                )
                optimizer_step = int(row.get("global_step", history_step))
                wall_time = finite_float(row.get("_timestamp"))
                for tag, raw_value in row.items():
                    if tag.startswith("_") or tag == "global_step":
                        continue
                    value = finite_float(raw_value)
                    if value is None:
                        continue
                    point: dict[str, float | int] = {
                        "step": optimizer_step,
                        "value": value,
                    }
                    if wall_time is not None:
                        point["wall_time"] = wall_time
                    self._scalars.setdefault(tag, {})[optimizer_step] = point
            self.error = None
        except Exception as exc:
            self.error = f"cannot read W&B run {self.run_path}: {exc}"
        return self._ordered_scalars()

    def _ordered_scalars(self) -> dict[str, list[dict[str, float | int]]]:
        return {
            tag: [by_step[step] for step in sorted(by_step)]
            for tag, by_step in self._scalars.items()
        }

    def config(self) -> RunConfig:
        return run_config_from_mapping(dict(self.run.config))

    def status(self) -> str:
        return {
            "finished": "completed",
            "crashed": "failed",
            "killed": "failed",
        }.get(str(self.run.state), str(self.run.state))


class WandbConfigReader:
    def __init__(self, run_reader: WandbRunReader):
        self.run_reader = run_reader
        self.error: str | None = None

    def read(self) -> RunConfig | None:
        try:
            config = self.run_reader.config()
        except Exception as exc:
            self.error = f"cannot read W&B config: {exc}"
            return None
        self.error = None
        return config


def gpu_series_from_scalars(
    scalars: dict[str, list[dict[str, float | int]]],
) -> list[dict[str, Any]]:
    fields = {
        "utilization_percent": "utilization_percent",
        "memory_used_gib": "vram_used_gib",
        "memory_total_gib": "vram_total_gib",
        "memory_utilization_percent": "vram_percent",
    }
    samples: dict[tuple[int, int], dict[str, Any]] = {}
    for tag, points in scalars.items():
        match = re.fullmatch(r"hardware/gpu_(\d+)/(.+)", tag)
        if match is None or match.group(2) not in fields:
            continue
        gpu_index = int(match.group(1))
        field = fields[match.group(2)]
        for point in points:
            step = int(point["step"])
            sample = samples.setdefault(
                (gpu_index, step), {"gpu_index": gpu_index, "step": step}
            )
            sample[field] = float(point["value"])
            if "wall_time" in point:
                sample["wall_time"] = float(point["wall_time"])
    if not samples:
        return []
    start = min(float(sample.get("wall_time", 0)) for sample in samples.values())
    return [
        sample
        | {"elapsed_seconds": float(sample.get("wall_time", start)) - start}
        for _, sample in sorted(samples.items())
    ]


def loss_series_from_scalars(
    scalars: dict[str, list[dict[str, float | int]]],
) -> dict[str, dict[str, list[dict[str, float | int]]]]:
    return {
        "combined": {
            "total": scalars.get("live_total_loss", []),
            "visibility": scalars.get("live_visibility_loss", []),
            "flow": scalars.get("live_flow_loss", []),
        },
        "diegesis": {
            "total": scalars.get("source/diegesis/loss", []),
            "visibility": scalars.get(
                "source/diegesis/component/visibility", []
            ),
            "flow": scalars.get("source/diegesis/component/flow", []),
        },
        "syn4d": {
            "total": scalars.get("source/syn4d/loss", []),
            "visibility": scalars.get(
                "source/syn4d/component/visibility", []
            ),
            "flow": scalars.get("source/syn4d/component/flow", []),
        },
        "mvkubric": {
            "total": scalars.get("source/mvkubric/loss", []),
            "visibility": scalars.get(
                "source/mvkubric/component/visibility", []
            ),
            "flow": scalars.get("source/mvkubric/component/flow", []),
        },
    }


def aggregated_ratio_series(
    numerator: list[dict[str, float | int]],
    denominator: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    """Compute a sparse ratio from two losses already aggregated per step."""
    denominator_by_step = {
        int(point["step"]): point for point in denominator
    }
    output: list[dict[str, float | int]] = []
    for numerator_point in numerator:
        step = int(numerator_point["step"])
        denominator_point = denominator_by_step.get(step)
        if denominator_point is None:
            continue
        numerator_value = finite_float(numerator_point.get("value"))
        denominator_value = finite_float(denominator_point.get("value"))
        if (
            numerator_value is None
            or denominator_value is None
            or denominator_value <= 0.0
        ):
            continue
        point: dict[str, float | int] = {
            "step": step,
            "value": numerator_value / denominator_value,
        }
        wall_time = numerator_point.get("wall_time", denominator_point.get("wall_time"))
        if wall_time is not None:
            point["wall_time"] = float(wall_time)
        output.append(point)
    return output


def validation_series_from_scalars(
    scalars: dict[str, list[dict[str, float | int]]],
) -> dict[str, dict[str, list[dict[str, float | int]]]]:
    """Group validation metrics without mixing full and subset populations."""
    grouped: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    for tag, points in scalars.items():
        if tag.startswith("eval/"):
            _, dataset_name, remainder = tag.split("/", 2)
            metric = remainder.rsplit("/", 1)[-1]
        elif tag.startswith("eval_") and "/" in tag:
            dataset_name, metric = tag.split("/", 1)
            dataset_name = dataset_name.removeprefix("eval_")
        else:
            continue
        if not metric.startswith("model__"):
            continue
        if dataset_name == "tapvid3d-multiview-validation":
            source_name = "diegesis"
        elif dataset_name == "syn4d-multiview-validation":
            source_name = "syn4d"
        elif dataset_name.endswith("validation-full"):
            source_name = "mvkubric_full"
        elif dataset_name.endswith("validation-subset"):
            source_name = "mvkubric_subset"
        else:
            source_name = dataset_name
        grouped.setdefault(source_name, {})[metric] = points

    diegesis = grouped.get("diegesis", {})
    syn4d = grouped.get("syn4d", {})
    for cohort in ("full", "subset"):
        mvkubric = grouped.get(f"mvkubric_{cohort}", {})
        combined = {}
        for metric in diegesis.keys() & syn4d.keys() & mvkubric.keys():
            sources = [
                {int(point["step"]): point for point in metrics[metric]}
                for metrics in (diegesis, syn4d, mvkubric)
            ]
            common_steps = set(sources[0])
            for source in sources[1:]:
                common_steps &= source.keys()
            combined[metric] = [
                {
                    "step": step,
                    "value": sum(
                        float(source[step]["value"])
                        for source in sources
                    )
                    / len(sources),
                }
                for step in sorted(common_steps)
            ]
        if combined:
            grouped[f"combined_{cohort}"] = combined
    return grouped


class TrainingLogReader:
    """Incrementally parse the append-only training log."""

    def __init__(self, path: Path):
        self.path = path
        self._inode: int | None = None
        self._offset = 0
        self._partial = ""
        self.samples: list[dict[str, Any]] = []
        self.failure_events: list[int] = []
        self.timing_rows: list[dict[str, float | int]] = []
        self.optimizer_clipped: dict[int, int] = {}
        self.finished = False
        self.fatal_error: str | None = None
        self.last_message: str | None = None
        self.mtime: float | None = None

    def _reset(self) -> None:
        self._offset = 0
        self._partial = ""
        self.samples.clear()
        self.failure_events.clear()
        self.timing_rows.clear()
        self.optimizer_clipped.clear()
        self.finished = False
        self.fatal_error = None
        self.last_message = None

    def refresh(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.mtime = None
            return
        if self._inode != stat.st_ino or stat.st_size < self._offset:
            self._reset()
            self._inode = stat.st_ino
        self.mtime = stat.st_mtime
        if stat.st_size == self._offset:
            return
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()
        text = self._partial + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._partial = lines.pop()
        for line in lines:
            self._parse_line(line.rstrip("\r"))

    def _parse_line(self, line: str) -> None:
        stripped = line.strip()
        if stripped:
            self.last_message = stripped[-500:]
        datapoint = DATAPOINT_RE.search(line)
        if datapoint:
            self.samples.append(
                {
                    "scene": datapoint.group(1),
                    "wait_seconds": float(datapoint.group(2)),
                    "tracks": None,
                }
            )
        track_count = TRACK_COUNT_RE.search(line)
        if track_count and self.samples and self.samples[-1]["tracks"] is None:
            self.samples[-1]["tracks"] = int(track_count.group(1))
        failure = FAILED_BATCH_RE.search(line)
        if failure:
            self.failure_events.append(len(self.samples))
        optimizer_clip = OPTIMIZER_CLIP_RE.search(line)
        if optimizer_clip:
            self.optimizer_clipped[int(optimizer_clip.group(1))] = int(
                optimizer_clip.group(2)
            )
        timing = TIMING_RE.search(line)
        if timing:
            self.timing_rows.append(
                {
                    "step": int(timing.group(1)),
                    "total": float(timing.group(2)),
                    "data": float(timing.group(3)),
                    "fwd": float(timing.group(4)),
                    "sync": float(timing.group(5)),
                    "bwd": float(timing.group(6)),
                }
            )
        if "FINISHED TRAINING" in line:
            self.finished = True
        for marker in FATAL_MARKERS:
            if marker in line:
                self.fatal_error = stripped[-500:]
                break

    def rolling_clipped_step_rate(
        self,
        window_size: int = 50,
    ) -> list[dict[str, float | int]]:
        """Return the share of recent optimizer steps with any clipped element."""
        window_size = max(1, window_size)
        ordered = sorted(self.optimizer_clipped.items())
        output: list[dict[str, float | int]] = []
        for index, (step, _) in enumerate(ordered):
            window = ordered[max(0, index - window_size + 1) : index + 1]
            output.append(
                {
                    "step": step,
                    "value": statistics.fmean(clipped for _, clipped in window),
                }
            )
        return output

    def pipeline_series(self, accumulation_steps: int) -> list[dict[str, Any]]:
        accumulation_steps = max(1, accumulation_steps)
        accepted = len(self.samples)
        number_of_steps = math.ceil(accepted / accumulation_steps)
        if self.failure_events and number_of_steps == 0:
            number_of_steps = 1
        seen_scenes: set[str] = set()
        previous_failures = 0
        output: list[dict[str, Any]] = []
        for step in range(1, number_of_steps + 1):
            start = (step - 1) * accumulation_steps
            end = min(step * accumulation_steps, accepted)
            batch = self.samples[start:end]
            seen_scenes.update(str(sample["scene"]) for sample in batch)
            cumulative_failures = sum(
                accepted_before < step * accumulation_steps
                for accepted_before in self.failure_events
            )
            accepted_through_step = end
            attempted_through_step = accepted_through_step + cumulative_failures
            tracks = [
                int(sample["tracks"])
                for sample in batch
                if sample["tracks"] is not None
            ]
            waits = [float(sample["wait_seconds"]) for sample in batch]
            output.append(
                {
                    "step": step,
                    "accepted": len(batch),
                    "failed": cumulative_failures - previous_failures,
                    "failed_cumulative": cumulative_failures,
                    "rejection_percent": (
                        100.0 * cumulative_failures / attempted_through_step
                        if attempted_through_step
                        else 0.0
                    ),
                    "tracks_mean": statistics.fmean(tracks) if tracks else None,
                    "tracks_min": min(tracks) if tracks else None,
                    "tracks_max": max(tracks) if tracks else None,
                    "wait_mean_seconds": statistics.fmean(waits) if waits else None,
                    "scenes_cumulative": len(seen_scenes),
                }
            )
            previous_failures = cumulative_failures
        return output


class NVMLGPUReader:
    """Sample one physical GPU using NVIDIA's supported Python binding."""

    def __init__(self, index: int):
        import pynvml

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.index = index
        name = pynvml.nvmlDeviceGetName(self._handle)
        self.name = name.decode("utf-8") if isinstance(name, bytes) else str(name)

    def __call__(self) -> dict[str, float | int | str]:
        pynvml = self._pynvml
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return {
            "gpu_index": self.index,
            "gpu_name": self.name,
            "utilization_percent": int(utilization.gpu),
            "vram_used_gib": float(memory.used / 2**30),
            "vram_total_gib": float(memory.total / 2**30),
            "vram_percent": float(100.0 * memory.used / memory.total),
            "power_watts": float(pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0),
            "temperature_c": int(
                pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
            ),
        }

    def close(self) -> None:
        self._pynvml.nvmlShutdown()


class GPUHistory:
    def __init__(
        self,
        reader: Callable[[], dict[str, Any]],
        interval_seconds: float,
        max_samples: int,
    ):
        self.reader = reader
        self.interval_seconds = interval_seconds
        self.max_samples = max_samples
        self.samples: list[dict[str, Any]] = []
        self.last_sample_monotonic: float | None = None
        self.error: str | None = None

    def sample_if_due(self, enabled: bool) -> None:
        if not enabled:
            return
        now_monotonic = time.monotonic()
        if (
            self.last_sample_monotonic is not None
            and now_monotonic - self.last_sample_monotonic < self.interval_seconds
        ):
            return
        try:
            sample = dict(self.reader())
        except Exception as exc:
            self.error = f"GPU sampling failed: {exc}"
            self.last_sample_monotonic = now_monotonic
            return
        sample["wall_time"] = time.time()
        self.samples.append(sample)
        if len(self.samples) > self.max_samples:
            del self.samples[: len(self.samples) - self.max_samples]
        self.last_sample_monotonic = now_monotonic
        self.error = None

    def series(self) -> list[dict[str, Any]]:
        if not self.samples:
            return []
        start = float(self.samples[0]["wall_time"])
        return [
            sample | {"elapsed_seconds": float(sample["wall_time"]) - start}
            for sample in self.samples
        ]


class TrainingDashboardState:
    """Combine live run artifacts into one stable dashboard snapshot."""

    def __init__(
        self,
        run_dir: Path,
        log_path: Path,
        event_dir: Path,
        gpu_history: GPUHistory,
        stale_after_seconds: float = 60.0,
        scalar_reader: Any = None,
        config_reader: Any = None,
        status_reader: Callable[[], str] | None = None,
        run_label: str | None = None,
        telemetry_from_scalars: bool = False,
    ):
        self.run_dir = run_dir
        self.log_reader = TrainingLogReader(log_path)
        self.scalar_reader = scalar_reader or TensorBoardScalarReader(event_dir)
        self.config_reader = config_reader or HydraConfigReader(
            run_dir / ".hydra" / "config.yaml"
        )
        self.gpu_history = gpu_history
        self.stale_after_seconds = stale_after_seconds
        self.status_reader = status_reader
        self.run_label = run_label or str(run_dir)
        self.telemetry_from_scalars = telemetry_from_scalars
        self._lock = threading.Lock()

    def _status(self, now: float) -> str:
        if self.status_reader is not None:
            return self.status_reader()
        if self.log_reader.finished or (self.run_dir / "model_final.pth").is_file():
            return "completed"
        if self.log_reader.fatal_error:
            return "failed"
        if self.log_reader.mtime is None:
            return "waiting"
        if now - self.log_reader.mtime > self.stale_after_seconds:
            return "stale"
        return "running"

    @staticmethod
    def _summary(
        timing_total: list[dict[str, float | int]],
        pipeline: list[dict[str, Any]],
        accepted: int,
        failed: int,
        accumulation_steps: int,
        gpu: list[dict[str, Any]],
    ) -> dict[str, Any]:
        durations = [float(point["value"]) for point in timing_total]
        attempts = accepted + failed
        peak_gpu = max(
            (float(sample["vram_percent"]) for sample in gpu),
            default=None,
        )
        peak_vram = max(
            (float(sample["vram_used_gib"]) for sample in gpu),
            default=None,
        )
        return {
            "mean_step_seconds": statistics.fmean(durations) if durations else None,
            "median_step_seconds": statistics.median(durations) if durations else None,
            "successful_microbatches_per_second": (
                accumulation_steps / statistics.fmean(durations) if durations else None
            ),
            "accepted_samples": accepted,
            "failed_samples": failed,
            "attempted_samples": attempts,
            "rejection_percent": 100.0 * failed / attempts if attempts else None,
            "unique_scenes": pipeline[-1]["scenes_cumulative"] if pipeline else 0,
            "peak_gpu_vram_percent": peak_gpu,
            "peak_gpu_vram_gib": peak_vram,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            config = self.config_reader.read()
            self.log_reader.refresh()
            scalars = self.scalar_reader.read()
            status = self._status(now)
            if self.telemetry_from_scalars:
                gpu = gpu_series_from_scalars(scalars)
            else:
                self.gpu_history.sample_if_due(status in {"running", "stale"})
                gpu = self.gpu_history.series()
            accumulation_steps = (
                config.gradient_accumulation_steps if config is not None else 1
            )
            pipeline = self.log_reader.pipeline_series(accumulation_steps)
            accepted = len(self.log_reader.samples)
            failed = len(self.log_reader.failure_events)

            losses = loss_series_from_scalars(scalars)
            stationary = scalars.get(
                "baseline/stationary_trajectory_loss",
                [],
            )
            baseline = {
                "stationary": stationary,
                "model_ratio": aggregated_ratio_series(
                    losses["combined"]["flow"],
                    stationary,
                ),
            }
            gradients = {
                "pre_clip": scalars.get("optimization/grad_norm_pre_clip", []),
                "post_clip": scalars.get("optimization/grad_norm_post_clip", []),
                "microbatch_mean": scalars.get(
                    "optimization/microbatch_grad_norm_mean",
                    [],
                ),
                "cosine_mean": scalars.get(
                    "optimization/microbatch_grad_cosine_mean",
                    [],
                ),
                "cosine_min": scalars.get(
                    "optimization/microbatch_grad_cosine_min",
                    [],
                ),
                "max_abs_pre_clip": scalars.get(
                    "optimization/max_abs_grad_pre_clip",
                    [],
                ),
                "norm_retention": scalars.get(
                    "optimization/norm_retention_after_value_clip",
                    [],
                ),
                "clipped_element_fraction": scalars.get(
                    "optimization/gradient_elements_clipped_fraction",
                    [],
                ),
                "clipped_step_rate_50": self.log_reader.rolling_clipped_step_rate(50),
            }
            log_timing = {
                key: [
                    {"step": int(row["step"]), "value": float(row[key])}
                    for row in self.log_reader.timing_rows
                ]
                for key in ("total", "data", "fwd", "sync", "bwd")
            }
            timing = {
                "total": merge_scalar_series(
                    scalars.get("timing/step", []), log_timing["total"]
                ),
                "data": merge_scalar_series(
                    scalars.get("timing/only_dataloader", []), log_timing["data"]
                ),
                "fwd": merge_scalar_series(
                    scalars.get("timing/only_fwd", []), log_timing["fwd"]
                ),
                "sync": merge_scalar_series(
                    scalars.get("timing/only_sync", []), log_timing["sync"]
                ),
                "bwd": merge_scalar_series(
                    scalars.get("timing/only_bwd", []), log_timing["bwd"]
                ),
            }
            motion = {
                name: scalars.get(f"sampling/{tag}", [])
                for name, tag in {
                    "track_count": "motion_track_count",
                    "full_mean": "motion_full_mean_m",
                    "full_median": "motion_full_median_m",
                    "full_p90": "motion_full_p90_m",
                    "window_mean": "motion_window_mean_m",
                    "window_median": "motion_window_median_m",
                    "window_p90": "motion_window_p90_m",
                    "window_static": "motion_window_static_count",
                    "window_dynamic": "motion_window_dynamic_count",
                    "window_very_dynamic": "motion_window_very_dynamic_count",
                    "full_dynamic_window_static": (
                        "motion_full_dynamic_window_static_count"
                    ),
                }.items()
            }
            validation = validation_series_from_scalars(scalars)
            performance = {
                name: scalars.get(f"performance/{name}", [])
                for name in (
                    "samples_per_second",
                    "trajectories_per_second",
                    "global_samples",
                    "global_trajectories",
                )
            }
            hardware = {
                name: scalars.get(f"hardware/container/{name}", [])
                for name in (
                    "cpu_cores_used",
                    "cpu_utilization_percent",
                    "memory_used_gib",
                    "memory_limit_gib",
                    "memory_utilization_percent",
                )
            }
            completed_step = max(
                [int(point["step"]) for point in timing["total"]]
                + [int(row["step"]) for row in self.log_reader.timing_rows]
                + [0]
            )
            errors = {
                key: value
                for key, value in {
                    "config": self.config_reader.error,
                    "scalars": self.scalar_reader.error,
                    "gpu": self.gpu_history.error,
                    "training": self.log_reader.fatal_error,
                }.items()
                if value
            }
            total_steps = config.total_steps if config is not None else None
            return {
                "schema_version": 2,
                "format": "mvtracker_training_dashboard",
                "server_time": utc_iso(now),
                "run_dir": self.run_label,
                "status": status,
                "progress": {
                    "completed_steps": completed_step,
                    "total_steps": total_steps,
                    "percent": (
                        100.0 * completed_step / total_steps
                        if total_steps and total_steps > 0
                        else None
                    ),
                },
                "config": (
                    {
                        "gradient_accumulation_steps": config.gradient_accumulation_steps,
                        "trajectory_cap": config.trajectory_cap,
                        "num_workers": config.num_workers,
                        "sequence_len": config.sequence_len,
                        "eval_frequency": config.eval_frequency,
                        "eval_datasets": list(config.eval_datasets),
                    }
                    if config is not None
                    else None
                ),
                "summary": self._summary(
                    timing["total"],
                    pipeline,
                    accepted,
                    failed,
                    accumulation_steps,
                    gpu,
                ),
                "series": {
                    "losses": losses,
                    "baseline": baseline,
                    "gradients": gradients,
                    "learning_rate": scalars.get("learning_rate", []),
                    "timing": timing,
                    "motion": motion,
                    "validation": validation,
                    "performance": performance,
                    "hardware": hardware,
                    "pipeline": pipeline,
                    "gpu": gpu,
                },
                "latest_log_message": self.log_reader.last_message,
                "errors": errors,
            }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MV-Tracker live training</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f7f8fb; --panel: #ffffff; --panel-2: #eef1f6;
      --text: #172033; --muted: #667085; --border: #d9dee8;
      --primary: #2563eb; --series-1: #2563eb; --series-2: #7c3aed;
      --series-3: #059669; --series-4: #d97706; --series-5: #dc2626;
      --good: #059669; --warn: #d97706; --bad: #dc2626;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #090d14; --panel: #111824; --panel-2: #162131;
        --text: #ebf2fb; --muted: #91a2b8; --border: #26364b;
        --primary: #62a8ff; --series-1: #62a8ff; --series-2: #b597ff;
        --series-3: #4fd6a0; --series-4: #ffc766; --series-5: #ff747f;
        --good: #4fd6a0; --warn: #ffc766; --bad: #ff747f;
      }
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--text); background: var(--bg); font: 14px/1.45 system-ui, sans-serif; }
    main { width: min(1500px, 100%); margin: auto; padding: 22px; }
    header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 18px; }
    h1 { margin: 0; font-size: clamp(21px, 3vw, 30px); font-weight: 500; letter-spacing: -.025em; }
    h2 { margin: 0 0 14px; font-size: 18px; font-weight: 500; }
    h3 { margin: 0 0 8px; font-size: 14px; font-weight: 500; }
    .muted { color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap: anywhere; }
    .status { display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--border); border-radius: 99px; padding: 7px 11px; background: var(--panel); }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--warn); box-shadow: 0 0 10px currentColor; }
    .status.running .dot, .status.completed .dot { background: var(--good); }
    .status.failed .dot, .status.stale .dot, .status.disconnected .dot { background: var(--bad); }
    .progress-track { height: 7px; border-radius: 99px; background: var(--panel-2); overflow: hidden; margin: 12px 0 24px; }
    .progress-fill { height: 100%; width: 0; background: var(--primary); transition: width .25s ease; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 28px; }
    .metric { min-width: 0; border: 1px solid var(--border); border-radius: 12px; background: var(--panel); padding: 14px; }
    .metric .value { font-size: 22px; font-variant-numeric: tabular-nums; margin-top: 3px; overflow: hidden; text-overflow: ellipsis; }
    section { margin-top: 30px; }
    .grid-2, .grid-3 { display: grid; gap: 24px; }
    .grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .loss-stack { display: grid; gap: 28px; }
    .chart-panel { min-width: 0; }
    .chart-wrap { position: relative; height: 250px; width: 100%; }
    .chart-wrap.compact { height: 190px; }
    .chart-wrap.loss-large { height: 360px; }
    .chart-note { color: var(--muted); margin-top: 7px; font-size: 12px; }
    .validation-head { display: flex; align-items: end; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; }
    .chart-controls { display: flex; justify-content: flex-end; margin: 4px 0 18px; }
    .opacity-control { display: flex; align-items: center; gap: 10px; }
    .opacity-control input { width: 170px; accent-color: var(--primary); }
    .opacity-control output { width: 3.5em; color: var(--text); font-variant-numeric: tabular-nums; }
    label { color: var(--muted); display: grid; gap: 4px; }
    select { max-width: min(700px, 100%); border: 1px solid var(--border); border-radius: 8px; padding: 7px 9px; color: var(--text); background: var(--panel); }
    .empty { height: 170px; display: grid; place-items: center; text-align: center; border-left: 1px solid var(--border); border-bottom: 1px solid var(--border); }
    .error { display: none; white-space: pre-wrap; border: 1px solid var(--bad); border-radius: 10px; padding: 12px; color: var(--bad); margin-bottom: 16px; }
    .error.visible { display: block; }
    footer { border-top: 1px solid var(--border); color: var(--muted); padding-top: 12px; margin-top: 28px; font-size: 12px; }
    @media (max-width: 820px) {
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid-2, .grid-3 { grid-template-columns: 1fr; }
      .chart-wrap { height: 225px; }
      .chart-wrap.compact { height: 185px; }
    }
    @media (max-width: 480px) {
      main { padding: 14px; }
      .metrics { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>MV-Tracker live training</h1>
      <div class="muted mono" id="run-path">Waiting for run…</div>
    </div>
    <div class="status waiting" id="status-pill"><span class="dot"></span><strong id="status">connecting</strong><span class="muted" id="step-status"></span></div>
  </header>
  <div class="error" id="errors"></div>
  <div class="progress-track" aria-label="Training progress"><div class="progress-fill" id="progress-fill"></div></div>
  <div class="metrics">
    <div class="metric"><div class="muted">Mean step time</div><div class="value" id="mean-step">—</div><div id="median-step">—</div></div>
    <div class="metric"><div class="muted">Successful throughput</div><div class="value" id="throughput">—</div><div id="microbatch-config">—</div></div>
    <div class="metric"><div class="muted">Rejected attempts</div><div class="value" id="rejection">—</div><div id="attempt-counts">—</div></div>
    <div class="metric"><div class="muted">Peak GPU allocation</div><div class="value" id="peak-gpu">—</div><div id="peak-vram">—</div></div>
  </div>
  <div class="chart-controls">
    <label class="opacity-control" for="raw-opacity">Raw-point visibility
      <input id="raw-opacity" type="range" min="0" max="1" step="0.05" value="0.35">
      <output id="raw-opacity-value" for="raw-opacity">35%</output>
    </label>
    <label class="opacity-control" for="loss-smoothing">Loss smoothing
      <input id="loss-smoothing" type="range" min="0" max="0.99" step="0.01" value="0.9">
      <output id="loss-smoothing-value" for="loss-smoothing">90%</output>
    </label>
  </div>

  <section>
    <h2>Training losses</h2>
    <div class="loss-stack">
      <div class="chart-panel"><h3>Combined</h3><div class="chart-wrap loss-large"><canvas id="loss-combined"></canvas></div></div>
      <div class="chart-panel"><h3>DIEGESIS</h3><div class="chart-wrap loss-large"><canvas id="loss-diegesis"></canvas></div></div>
      <div class="chart-panel"><h3>Syn4D</h3><div class="chart-wrap loss-large"><canvas id="loss-syn4d"></canvas></div></div>
      <div class="chart-panel"><h3>MV-Kubric</h3><div class="chart-wrap loss-large"><canvas id="loss-mvkubric"></canvas></div></div>
    </div>
    <div class="chart-note">Faint points are raw updates; solid lines are debiased exponential moving averages. Each panel shows total, visibility, and 3D trajectory loss.</div>
  </section>

  <section>
    <h2>Trajectory baseline</h2>
    <div class="grid-2">
      <div class="chart-panel"><h3>Model versus stationary prediction</h3><div class="chart-wrap"><canvas id="stationary-baseline"></canvas></div><div class="chart-note">Model loss is shown continuously; stationary loss is shown at diagnostic steps only. Stationary holds every track at its query coordinate.</div></div>
      <div class="chart-panel"><h3>Model / stationary loss</h3><div class="chart-wrap"><canvas id="stationary-ratio"></canvas></div><div class="chart-note">Each point is the ratio of the aggregated model and stationary losses at a diagnostic step; no smoothing is applied. Below 1.0 means the model beats the no-motion baseline.</div></div>
    </div>
  </section>

  <section>
    <div class="validation-head">
      <h2>Validation through training</h2>
      <label>Metric<select id="validation-select" disabled><option>No validation metrics yet</option></select></label>
    </div>
    <div class="grid-2" id="validation-chart-wrap" hidden>
      <div class="chart-panel"><h3>Combined</h3><div class="chart-wrap"><canvas id="validation-combined"></canvas></div><div class="chart-note">Equal-weight mean of DIEGESIS, Syn4D, and MV-Kubric at the same step. MV-Kubric full and subset cohorts remain separate.</div></div>
      <div class="chart-panel"><h3>DIEGESIS</h3><div class="chart-wrap"><canvas id="validation-diegesis"></canvas></div></div>
      <div class="chart-panel"><h3>Syn4D</h3><div class="chart-wrap"><canvas id="validation-syn4d"></canvas></div></div>
      <div class="chart-panel"><h3>MV-Kubric</h3><div class="chart-wrap"><canvas id="validation-mvkubric"></canvas></div><div class="chart-note">Full (27 scenes) and subset (101–102) remain separate so unlike populations are never joined into one trend.</div></div>
    </div>
    <div class="empty" id="validation-empty"><div><strong>No validation series recorded</strong><div class="muted" id="validation-reason">Waiting for evaluation scalars…</div></div></div>
  </section>

  <section>
    <h2>Optimization and throughput</h2>
    <div class="grid-2">
      <div class="chart-panel"><h3>Step timing</h3><div class="chart-wrap"><canvas id="step-timing"></canvas></div><div class="chart-note">Total, blocked dataloader time, forward, and backward/update.</div></div>
      <div class="chart-panel"><h3>Learning-rate schedule</h3><div class="chart-wrap"><canvas id="learning-rate"></canvas></div></div>
    </div>
    <div class="grid-3" style="margin-top:22px">
      <div class="chart-panel"><h3>Gradient norms</h3><div class="chart-wrap compact"><canvas id="gradient-norms"></canvas></div><div class="chart-note">Faint points are diagnostic samples; strong lines are trailing 50-sample means.</div></div>
      <div class="chart-panel"><h3>Microbatch gradient agreement</h3><div class="chart-wrap compact"><canvas id="gradient-cosine"></canvas></div><div class="chart-note">Trailing 50-sample means over diagnostic samples. Negative cosine indicates gradient cancellation.</div></div>
      <div class="chart-panel"><h3>Elementwise gradient clipping</h3><div class="chart-wrap compact"><canvas id="gradient-clipping"></canvas></div><div class="chart-note">Upstream clips each gradient element to ±1; norm retention is post/pre global norm. Clipped-step rate is the share of the latest 50 optimizer steps with any clipped element.</div></div>
    </div>
  </section>

  <section>
    <h2>Data-pipeline behavior</h2>
    <div class="grid-3">
      <div class="chart-panel"><h3>Cumulative rejection rate</h3><div class="chart-wrap compact"><canvas id="rejection-rate"></canvas></div></div>
      <div class="chart-panel"><h3>Tracks per microbatch</h3><div class="chart-wrap compact"><canvas id="track-count"></canvas></div><div class="chart-note">Trailing 50-step means over faint raw minimum, mean and maximum counts.</div></div>
      <div class="chart-panel"><h3>Cumulative scene coverage</h3><div class="chart-wrap compact"><canvas id="scene-coverage"></canvas></div></div>
    </div>
    <div class="grid-3" style="margin-top:22px">
      <div class="chart-panel"><h3>Sampled path length</h3><div class="chart-wrap compact"><canvas id="motion-path-length"></canvas></div><div class="chart-note">Visible 3D path length over the complete scene versus the sampled 24-frame window.</div></div>
      <div class="chart-panel"><h3>Window motion buckets</h3><div class="chart-wrap compact"><canvas id="motion-window-buckets"></canvas></div><div class="chart-note">Static &lt;1 cm, dynamic &gt;10 cm and very dynamic &gt;2 m. Very-dynamic tracks are also included in dynamic.</div></div>
      <div class="chart-panel"><h3>Global/window mismatch</h3><div class="chart-wrap compact"><canvas id="motion-window-mismatch"></canvas></div><div class="chart-note">Tracks moving &gt;10 cm globally but &lt;1 cm inside the sampled window, compared with all sampled tracks.</div></div>
    </div>
  </section>

  <section>
    <h2>GPU telemetry</h2>
    <div class="grid-3">
      <div class="chart-panel"><h3>GPU utilization</h3><div class="chart-wrap compact"><canvas id="gpu-util"></canvas></div></div>
      <div class="chart-panel"><h3>Allocated VRAM</h3><div class="chart-wrap compact"><canvas id="gpu-vram"></canvas></div></div>
      <div class="chart-panel"><h3>Power and temperature</h3><div class="chart-wrap compact"><canvas id="gpu-thermal"></canvas></div></div>
    </div>
    <div class="grid-3" style="margin-top:22px">
      <div class="chart-panel"><h3>Training throughput</h3><div class="chart-wrap compact"><canvas id="training-throughput"></canvas></div></div>
      <div class="chart-panel"><h3>Container CPU</h3><div class="chart-wrap compact"><canvas id="container-cpu"></canvas></div></div>
      <div class="chart-panel"><h3>Container memory</h3><div class="chart-wrap compact"><canvas id="container-memory"></canvas></div></div>
    </div>
    <div class="chart-note">Local mode samples NVML. W&amp;B mode displays the per-rank GPU and container metrics logged by training.</div>
  </section>

  <footer><span id="latest-message">Waiting for training log…</span></footer>
</main>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script>
const css=getComputedStyle(document.documentElement);
const color=name=>css.getPropertyValue(name).trim();
const palette={text:color('--text'),muted:color('--muted'),border:color('--border'),panel:color('--panel'),s1:color('--series-1'),s2:color('--series-2'),s3:color('--series-3'),s4:color('--series-4'),s5:color('--series-5')};
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const fmt=new Intl.NumberFormat(undefined,{maximumFractionDigits:2});
const pct=value=>value==null?'—':`${Number(value).toFixed(2)}%`;
const seconds=value=>value==null?'—':`${Number(value).toFixed(2)} s`;
const line=(label,stroke,extra={})=>({label,data:[],borderColor:stroke,backgroundColor:stroke,pointBackgroundColor:stroke,pointBorderColor:stroke,borderWidth:2,pointRadius:2,pointHoverRadius:4,tension:.18,fill:false,...extra});
const alphaColor=(hex,alpha)=>{
  const match=/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  return match?`rgba(${parseInt(match[1],16)},${parseInt(match[2],16)},${parseInt(match[3],16)},${alpha})`:hex;
};
const meanLine=(label,stroke,extra={})=>line(`${label} · 50-sample mean`,stroke,{borderWidth:2.5,pointRadius:0,pointHoverRadius:3,tension:.18,...extra});
const emaLine=(label,stroke,extra={})=>line(`${label} · EMA`,stroke,{borderWidth:2.5,pointRadius:0,pointHoverRadius:3,tension:.18,...extra});
const rawPoints=(label,stroke,extra={})=>line(`${label} · raw`,alphaColor(stroke,.35),{showInLegend:false,showLine:false,borderWidth:0,pointRadius:1,pointHoverRadius:4,tension:0,isRawPoints:true,rawStroke:stroke,...extra});
const lossLines=()=>[rawPoints('Total',palette.s1,{seriesKey:'total'}),emaLine('Total',palette.s1,{seriesKey:'total'}),rawPoints('Visibility',palette.s2,{seriesKey:'visibility'}),emaLine('Visibility',palette.s2,{borderDash:[6,4],seriesKey:'visibility'}),rawPoints('Trajectory',palette.s3,{seriesKey:'trajectory'}),emaLine('Trajectory',palette.s3,{seriesKey:'trajectory'})];
const compactNumber=value=>{
  const number=Number(value), magnitude=Math.abs(number);
  if(!Number.isFinite(number)) return value;
  if((magnitude>0&&magnitude<.001)||magnitude>=1000) return number.toExponential(1);
  return number.toLocaleString(undefined,{maximumFractionDigits:3});
};
function options(xTitle,yTitle,extra={}){
  return {responsive:true,maintainAspectRatio:false,animation:reduced?false:{duration:180},interaction:{mode:'index',intersect:false},plugins:{legend:{display:extra.legend!==false,position:'bottom',...(extra.legendOnClick?{onClick:extra.legendOnClick}:{}),labels:{color:palette.text,usePointStyle:true,boxWidth:8,filter:(item,data)=>data.datasets[item.datasetIndex].showInLegend!==false}},tooltip:{enabled:true,backgroundColor:palette.panel,titleColor:palette.text,bodyColor:palette.text,borderColor:palette.border,borderWidth:1}},scales:{x:{type:extra.xType||'linear',grid:{color:palette.border},ticks:{color:palette.muted,maxTicksLimit:10},title:{display:true,text:xTitle,color:palette.muted}},y:{min:extra.min,max:extra.max,grid:{color:palette.border},ticks:{color:palette.muted,callback:extra.tickCallback||compactNumber},title:{display:true,text:yTitle,color:palette.muted}},...(extra.scales||{})}};
}
function fitLossScale(chart){const values=chart.data.datasets.flatMap((dataset,index)=>{if(dataset.isRawPoints||!chart.isDatasetVisible(index))return[];const raw=chart.data.datasets[index-1];const input=(raw&&raw.isRawPoints?raw.data:dataset.data).map(point=>Number(point.y)).filter(Number.isFinite);let weighted=0,weightSum=0;return input.map(value=>{weighted=.9*weighted+.1*value;weightSum=.9*weightSum+.1;return weighted/weightSum;});});if(!values.length){delete chart.options.scales.y.min;delete chart.options.scales.y.max;return;}values.sort((a,b)=>a-b);const quantile=p=>values[Math.min(values.length-1,Math.floor((values.length-1)*p))];const minimum=quantile(.01),maximum=quantile(.99),range=maximum-minimum||Math.max(Math.abs(maximum)*.1,1e-6),padding=range*.08;chart.options.scales.y.min=Math.max(0,minimum-padding);chart.options.scales.y.max=maximum+padding;}
function updateLossChart(chart,datasets){datasets.forEach((data,index)=>{chart.data.datasets[index].data=data;});fitLossScale(chart);chart.update(reduced?'none':undefined);}
function toggleLossSeries(_event,legendItem,legend){const chart=legend.chart,key=chart.data.datasets[legendItem.datasetIndex].seriesKey,visible=chart.isDatasetVisible(legendItem.datasetIndex);chart.data.datasets.forEach((dataset,index)=>{if(dataset.seriesKey===key)chart.setDatasetVisibility(index,!visible);});fitLossScale(chart);chart.update(reduced?'none':undefined);}
const charts={
  combined:new Chart(document.getElementById('loss-combined'),{type:'line',data:{datasets:lossLines()},options:options('Optimizer step','Loss',{legendOnClick:toggleLossSeries,xType:'logarithmic'})}),
  diegesis:new Chart(document.getElementById('loss-diegesis'),{type:'line',data:{datasets:lossLines()},options:options('Optimizer step','Loss',{legendOnClick:toggleLossSeries,xType:'logarithmic'})}),
  syn4d:new Chart(document.getElementById('loss-syn4d'),{type:'line',data:{datasets:lossLines()},options:options('Optimizer step','Loss',{legendOnClick:toggleLossSeries,xType:'logarithmic'})}),
  mvkubric:new Chart(document.getElementById('loss-mvkubric'),{type:'line',data:{datasets:lossLines()},options:options('Optimizer step','Loss',{legendOnClick:toggleLossSeries,xType:'logarithmic'})}),
  stationary:new Chart(document.getElementById('stationary-baseline'),{type:'line',data:{datasets:[line('Model trajectory',palette.s1),rawPoints('Stationary baseline',palette.s4)]},options:options('Optimizer step','Trajectory loss',{xType:'logarithmic'})}),
  stationaryRatio:new Chart(document.getElementById('stationary-ratio'),{type:'line',data:{datasets:[line('Model / stationary',palette.s1),line('Parity',palette.muted,{borderDash:[5,4],pointRadius:0})]},options:options('Optimizer step','Loss ratio',{min:0,xType:'logarithmic'})}),
  validationCombined:new Chart(document.getElementById('validation-combined'),{type:'line',data:{datasets:[]},options:options('Optimizer step','Score')}),
  validationDiegesis:new Chart(document.getElementById('validation-diegesis'),{type:'line',data:{datasets:[]},options:options('Optimizer step','Score')}),
  validationSyn4d:new Chart(document.getElementById('validation-syn4d'),{type:'line',data:{datasets:[]},options:options('Optimizer step','Score')}),
  validationMvkubric:new Chart(document.getElementById('validation-mvkubric'),{type:'line',data:{datasets:[]},options:options('Optimizer step','Score')}),
  timing:new Chart(document.getElementById('step-timing'),{type:'line',data:{datasets:[line('Total',palette.s1),line('Data wait',palette.s2),line('Forward',palette.s3),line('Backward',palette.s4)]},options:options('Optimizer step','Seconds')}),
  learningRate:new Chart(document.getElementById('learning-rate'),{type:'line',data:{datasets:[line('Learning rate',palette.s1)]},options:options('Optimizer step','Learning rate',{legend:false,tickCallback:value=>Number(value).toExponential(1)})}),
  gradientNorms:new Chart(document.getElementById('gradient-norms'),{type:'line',data:{datasets:[rawPoints('Pre-clip',palette.s1),meanLine('Pre-clip',palette.s1),rawPoints('Post-clip',palette.s2),meanLine('Post-clip',palette.s2,{borderDash:[6,4]}),rawPoints('Microbatch mean',palette.s3),meanLine('Microbatch mean',palette.s3,{borderDash:[2,3]})]},options:options('Optimizer step','Global L2 norm',{min:0})}),
  gradientCosine:new Chart(document.getElementById('gradient-cosine'),{type:'line',data:{datasets:[rawPoints('Mean cosine',palette.s1),meanLine('Mean cosine',palette.s1),rawPoints('Minimum cosine',palette.s5),meanLine('Minimum cosine',palette.s5,{borderDash:[6,4]})]},options:options('Optimizer step','Cosine similarity',{min:-1,max:1})}),
  gradientClipping:new Chart(document.getElementById('gradient-clipping'),{type:'line',data:{datasets:[line('Norm retention',palette.s1),line('Elements clipped',palette.s5),line('Clipped steps (last 50)',palette.s4)]},options:options('Optimizer step','Fraction',{min:0,max:1})}),
  rejection:new Chart(document.getElementById('rejection-rate'),{type:'line',data:{datasets:[line('Rejected',palette.s1)]},options:options('Optimizer step','Rejected attempts (%)',{min:0,max:100,legend:false})}),
  tracks:new Chart(document.getElementById('track-count'),{type:'line',data:{datasets:[rawPoints('Mean',palette.s1),meanLine('Mean',palette.s1),rawPoints('Maximum',palette.s2),meanLine('Maximum',palette.s2,{borderDash:[6,4]}),rawPoints('Minimum',palette.s3),meanLine('Minimum',palette.s3,{borderDash:[2,3]})]},options:options('Optimizer step','Tracks')}),
  scenes:new Chart(document.getElementById('scene-coverage'),{type:'line',data:{datasets:[line('Seen',palette.s1)]},options:options('Optimizer step','Unique scenes',{min:0})}),
  motionPath:new Chart(document.getElementById('motion-path-length'),{type:'line',data:{datasets:[line('Full mean',palette.s1),line('Window mean',palette.s2),line('Window p90',palette.s4,{borderDash:[6,4]})]},options:options('Optimizer step','Path length (m)',{min:0})}),
  motionBuckets:new Chart(document.getElementById('motion-window-buckets'),{type:'line',data:{datasets:[line('Static',palette.s3),line('Dynamic',palette.s1),line('Very dynamic',palette.s5)]},options:options('Optimizer step','Tracks',{min:0})}),
  motionMismatch:new Chart(document.getElementById('motion-window-mismatch'),{type:'line',data:{datasets:[line('All sampled',palette.s2),line('Global dynamic → window static',palette.s5)]},options:options('Optimizer step','Tracks',{min:0})}),
  gpuUtil:new Chart(document.getElementById('gpu-util'),{type:'line',data:{datasets:[line('GPU 0',palette.s1),line('GPU 1',palette.s2)]},options:options('Elapsed time (min)','Utilization (%)',{min:0,max:100})}),
  gpuVram:new Chart(document.getElementById('gpu-vram'),{type:'line',data:{datasets:[line('GPU 0',palette.s1),line('GPU 1',palette.s2)]},options:options('Elapsed time (min)','VRAM (GiB)',{min:0})}),
  gpuThermal:new Chart(document.getElementById('gpu-thermal'),{type:'line',data:{datasets:[line('Power',palette.s3),line('Temperature',palette.s4,{yAxisID:'temp'})]},options:options('Elapsed time (min)','Power (W)',{scales:{temp:{position:'right',grid:{drawOnChartArea:false},ticks:{color:palette.muted},title:{display:true,text:'Temperature (°C)',color:palette.muted}}}})}),
  throughput:new Chart(document.getElementById('training-throughput'),{type:'line',data:{datasets:[line('Samples/s',palette.s1),line('Trajectories/s',palette.s3,{yAxisID:'trajectories'})]},options:options('Optimizer step','Samples/s',{scales:{trajectories:{position:'right',grid:{drawOnChartArea:false},ticks:{color:palette.muted},title:{display:true,text:'Trajectories/s',color:palette.muted}}}})}),
  containerCpu:new Chart(document.getElementById('container-cpu'),{type:'line',data:{datasets:[line('CPU utilization',palette.s1),line('Cores used',palette.s3,{yAxisID:'cores'})]},options:options('Optimizer step','Utilization (%)',{min:0,scales:{cores:{position:'right',grid:{drawOnChartArea:false},ticks:{color:palette.muted},title:{display:true,text:'CPU cores',color:palette.muted}}}})}),
  containerMemory:new Chart(document.getElementById('container-memory'),{type:'line',data:{datasets:[line('Memory used',palette.s2),line('Memory utilization',palette.s4,{yAxisID:'percent'})]},options:options('Optimizer step','Memory (GiB)',{min:0,scales:{percent:{position:'right',min:0,max:100,grid:{drawOnChartArea:false},ticks:{color:palette.muted},title:{display:true,text:'Utilization (%)',color:palette.muted}}}})})
};
const rawOpacity=document.getElementById('raw-opacity');
const rawOpacityValue=document.getElementById('raw-opacity-value');
const lossSmoothing=document.getElementById('loss-smoothing');
const lossSmoothingValue=document.getElementById('loss-smoothing-value');
let lossSmoothingWeight=Number(lossSmoothing.value);
let currentLosses={};
function setRawOpacity(value){
  const opacity=Math.max(0,Math.min(1,Number(value)));
  rawOpacityValue.value=`${Math.round(opacity*100)}%`;
  Object.values(charts).forEach(chart=>{
    chart.data.datasets.forEach(dataset=>{
      if(!dataset.isRawPoints) return;
      const color=alphaColor(dataset.rawStroke,opacity);
      dataset.borderColor=color;
      dataset.backgroundColor=color;
      dataset.pointBackgroundColor=color;
      dataset.pointBorderColor=color;
    });
    chart.update('none');
  });
}
rawOpacity.addEventListener('input',event=>setRawOpacity(event.target.value));
const points=series=>(series||[]).map(point=>({x:Number(point.step),y:Number(point.value)}));
const movingAverageXY=(input,windowSize=50)=>{
  const window=[]; let sum=0;
  return input.map(point=>{window.push(point.y);sum+=point.y;if(window.length>windowSize)sum-=window.shift();return{x:point.x,y:sum/window.length};});
};
const movingAveragePoints=(series,windowSize=50)=>movingAverageXY(points(series),windowSize);
const debiasedEmaXY=(input,weight=lossSmoothingWeight)=>{
  let weightedSum=0, weightSum=0;
  return input.map(point=>{
    weightedSum=weight*weightedSum+(1-weight)*point.y;
    weightSum=weight*weightSum+(1-weight);
    return{x:point.x,y:weightedSum/weightSum};
  });
};
const debiasedEmaPoints=series=>debiasedEmaXY(points(series));
function renderLossCharts(){
  const lossData=source=>[points(source.total),debiasedEmaPoints(source.total),points(source.visibility),debiasedEmaPoints(source.visibility),points(source.flow),debiasedEmaPoints(source.flow)];
  updateLossChart(charts.combined,lossData(currentLosses.combined||{}));
  updateLossChart(charts.diegesis,lossData(currentLosses.diegesis||{}));
  updateLossChart(charts.syn4d,lossData(currentLosses.syn4d||{}));
  updateLossChart(charts.mvkubric,lossData(currentLosses.mvkubric||{}));
}
lossSmoothing.addEventListener('input',event=>{
  lossSmoothingWeight=Number(event.target.value);
  lossSmoothingValue.value=`${Math.round(lossSmoothingWeight*100)}%`;
  renderLossCharts();
});
const parityPoints=series=>points(series).map(point=>({x:point.x,y:1}));
const pipePoints=(series,key)=>(series||[]).filter(point=>point[key]!=null).map(point=>({x:Number(point.step),y:Number(point[key])}));
const gpuPoints=(series,key,gpuIndex=null)=>(series||[]).filter(point=>point[key]!=null&&(gpuIndex==null||Number(point.gpu_index)===gpuIndex)).map(point=>({x:Number(point.elapsed_seconds)/60,y:Number(point[key])}));
function update(chart,datasets){datasets.forEach((data,index)=>{chart.data.datasets[index].data=data;});chart.update(reduced?'none':undefined);}
const validationLabels={combined_full:'Combined · full validation',combined_subset:'Combined · MV-Kubric 101–102',diegesis:'DIEGESIS',syn4d:'Syn4D',mvkubric_full:'MV-Kubric · full 27',mvkubric_subset:'MV-Kubric · scenes 101–102'};
const validationColors={combined_full:palette.s1,combined_subset:palette.s4,diegesis:palette.s3,syn4d:palette.s2,mvkubric_full:palette.s1,mvkubric_subset:palette.s4};
function updateValidation(chart,entries){chart.data.datasets=entries.map(([name,series],index)=>line(validationLabels[name]||name,validationColors[name]||[palette.s1,palette.s2,palette.s3,palette.s4,palette.s5][index%5],{data:points(series)}));chart.update(reduced?'none':undefined);}
const text=(id,value)=>{document.getElementById(id).textContent=value;};
function render(state){
  const summary=state.summary||{}, config=state.config||{}, progress=state.progress||{};
  const pill=document.getElementById('status-pill'); pill.className=`status ${state.status}`;
  text('status',state.status); text('step-status',`${progress.completed_steps||0} / ${progress.total_steps??'—'} steps`);
  text('run-path',state.run_dir||'Waiting for run…');
  document.getElementById('progress-fill').style.width=`${Math.max(0,Math.min(100,Number(progress.percent)||0))}%`;
  text('mean-step',seconds(summary.mean_step_seconds)); text('median-step',summary.median_step_seconds==null?'—':`${Number(summary.median_step_seconds).toFixed(2)} s median`);
  text('throughput',summary.successful_microbatches_per_second==null?'—':`${Number(summary.successful_microbatches_per_second).toFixed(3)}/s`);
  text('microbatch-config',`${config.gradient_accumulation_steps??'—'} serial microbatches/step`);
  text('rejection',pct(summary.rejection_percent)); text('attempt-counts',`${summary.failed_samples||0} rejected · ${summary.accepted_samples||0} accepted`);
  text('peak-gpu',pct(summary.peak_gpu_vram_percent)); text('peak-vram',summary.peak_gpu_vram_gib==null?'—':`${Number(summary.peak_gpu_vram_gib).toFixed(2)} GiB sampled`);
  const errors=document.getElementById('errors'), entries=Object.entries(state.errors||{}); errors.className=entries.length?'error visible':'error'; errors.textContent=entries.map(([key,value])=>`${key}: ${value}`).join('\n');
  text('latest-message',state.latest_log_message||'Waiting for training log…');

  currentLosses=state.series?.losses||{};
  renderLossCharts();
  const combinedLosses=currentLosses.combined||{};
  const baseline=state.series?.baseline||{};
  update(charts.stationary,[points(combinedLosses.flow),points(baseline.stationary)]);
  update(charts.stationaryRatio,[points(baseline.model_ratio),parityPoints(baseline.model_ratio)]);
  const timing=state.series?.timing||{};
  update(charts.timing,[points(timing.total),points(timing.data),points(timing.fwd),points(timing.bwd)]);
  update(charts.learningRate,[points(state.series?.learning_rate)]);
  const gradients=state.series?.gradients||{};
  update(charts.gradientNorms,[points(gradients.pre_clip),movingAveragePoints(gradients.pre_clip),points(gradients.post_clip),movingAveragePoints(gradients.post_clip),points(gradients.microbatch_mean),movingAveragePoints(gradients.microbatch_mean)]);
  update(charts.gradientCosine,[points(gradients.cosine_mean),movingAveragePoints(gradients.cosine_mean),points(gradients.cosine_min),movingAveragePoints(gradients.cosine_min)]);
  update(charts.gradientClipping,[points(gradients.norm_retention),points(gradients.clipped_element_fraction),points(gradients.clipped_step_rate_50)]);
  const pipeline=state.series?.pipeline||[];
  update(charts.rejection,[pipePoints(pipeline,'rejection_percent')]);
  const trackMean=pipePoints(pipeline,'tracks_mean'), trackMax=pipePoints(pipeline,'tracks_max'), trackMin=pipePoints(pipeline,'tracks_min');
  update(charts.tracks,[trackMean,movingAverageXY(trackMean),trackMax,movingAverageXY(trackMax),trackMin,movingAverageXY(trackMin)]);
  update(charts.scenes,[pipePoints(pipeline,'scenes_cumulative')]);
  const motion=state.series?.motion||{};
  update(charts.motionPath,[points(motion.full_mean),points(motion.window_mean),points(motion.window_p90)]);
  update(charts.motionBuckets,[points(motion.window_static),points(motion.window_dynamic),points(motion.window_very_dynamic)]);
  update(charts.motionMismatch,[points(motion.track_count),points(motion.full_dynamic_window_static)]);
  const gpu=state.series?.gpu||[];
  const gpuIndices=[...new Set(gpu.map(point=>Number(point.gpu_index??0)))].sort((a,b)=>a-b).slice(0,2);
  [charts.gpuUtil,charts.gpuVram].forEach(chart=>chart.data.datasets.forEach((dataset,index)=>dataset.label=gpuIndices[index]==null?`GPU ${index}`:`GPU ${gpuIndices[index]}`));
  update(charts.gpuUtil,gpuIndices.map(index=>gpuPoints(gpu,'utilization_percent',index)).concat([[],[]]).slice(0,2)); update(charts.gpuVram,gpuIndices.map(index=>gpuPoints(gpu,'vram_used_gib',index)).concat([[],[]]).slice(0,2)); update(charts.gpuThermal,[gpuPoints(gpu,'power_watts'),gpuPoints(gpu,'temperature_c')]);
  const performance=state.series?.performance||{}, hardware=state.series?.hardware||{};
  update(charts.throughput,[points(performance.samples_per_second),points(performance.trajectories_per_second)]);
  update(charts.containerCpu,[points(hardware.cpu_utilization_percent),points(hardware.cpu_cores_used)]);
  update(charts.containerMemory,[points(hardware.memory_used_gib),points(hardware.memory_utilization_percent)]);

  const validation=state.series?.validation||{}, tags=[...new Set(Object.values(validation).flatMap(metrics=>Object.keys(metrics)))].sort(), select=document.getElementById('validation-select');
  const previous=select.value;
  select.replaceChildren(...tags.map(tag=>{const option=document.createElement('option');option.value=tag;option.textContent=tag;return option;}));
  const renderValidation=metric=>{const entries=Object.entries(validation).filter(([,metrics])=>metrics[metric]).map(([name,metrics])=>[name,metrics[metric]]);updateValidation(charts.validationCombined,entries.filter(([name])=>name.startsWith('combined_')));updateValidation(charts.validationDiegesis,entries.filter(([name])=>name==='diegesis'));updateValidation(charts.validationSyn4d,entries.filter(([name])=>name==='syn4d'));updateValidation(charts.validationMvkubric,entries.filter(([name])=>name.startsWith('mvkubric_')));};
  if(tags.length){select.disabled=false;select.value=tags.includes(previous)?previous:tags[0];document.getElementById('validation-chart-wrap').hidden=false;document.getElementById('validation-empty').hidden=true;renderValidation(select.value);}
  else{select.disabled=true;const option=document.createElement('option');option.textContent='No validation metrics yet';select.append(option);document.getElementById('validation-chart-wrap').hidden=true;document.getElementById('validation-empty').hidden=false;text('validation-reason',(config.eval_datasets||[]).length?'Waiting for the first scheduled evaluation…':'No evaluation datasets are configured.');}
  select.onchange=()=>renderValidation(select.value);
}
const stream=new EventSource('/api/stream');
stream.onmessage=event=>{try{render(JSON.parse(event.data));}catch(error){const box=document.getElementById('errors');box.className='error visible';box.textContent=`render: ${error.message}`;}};
stream.onerror=()=>{document.getElementById('status-pill').className='status disconnected';text('status','reconnecting');};
</script>
</body>
</html>
"""


def make_handler(
    state: TrainingDashboardState,
    stream_interval_seconds: float,
) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "MVTrackerTrainingDashboard/1"

        def _security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")

        def _send(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._security_headers()
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _snapshot_bytes(self) -> bytes:
            return json.dumps(
                state.snapshot(),
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")

        def _stream(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self._security_headers()
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b"data: " + self._snapshot_bytes() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(stream_interval_seconds)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return
            finally:
                self.close_connection = True

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlsplit(self.path).path
            if path == "/":
                self._send(
                    HTTPStatus.OK,
                    INDEX_HTML.encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return
            if path == "/api/state":
                self._send(
                    HTTPStatus.OK,
                    self._snapshot_bytes(),
                    "application/json; charset=utf-8",
                )
                return
            if path == "/api/stream":
                self._stream()
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain; charset=utf-8")

        def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if urlsplit(self.path).path == "/api/stream":
                self._send(HTTPStatus.METHOD_NOT_ALLOWED, b"", "text/plain; charset=utf-8")
                return
            self.do_GET()

        def log_message(self, format: str, *args: Any) -> None:
            return

    return DashboardHandler


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument(
        "--wandb-run",
        help="W&B run path as entity/project/run_id",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--event-dir", type=Path, default=None)
    parser.add_argument("--gpu-index", type=nonnegative_int)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=port_number, default=DEFAULT_PORT)
    parser.add_argument(
        "--stream-interval-seconds",
        type=positive_float,
        default=DEFAULT_STREAM_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--gpu-sample-interval-seconds",
        type=positive_float,
        default=DEFAULT_GPU_SAMPLE_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--max-gpu-samples",
        type=positive_int,
        default=DEFAULT_MAX_GPU_SAMPLES,
    )
    parser.add_argument(
        "--wandb-poll-interval-seconds",
        type=positive_float,
        default=DEFAULT_WANDB_POLL_INTERVAL_SECONDS,
    )
    args = parser.parse_args(argv)
    if args.run_dir is not None and args.gpu_index is None:
        parser.error("--gpu-index is required with --run-dir")
    if args.wandb_run is not None:
        if len(args.wandb_run.split("/")) != 3:
            parser.error("--wandb-run must be entity/project/run_id")
        if args.log_file is not None or args.event_dir is not None:
            parser.error("--log-file and --event-dir are local-mode options")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.wandb_run is not None:
        run_reader = WandbRunReader(
            args.wandb_run,
            poll_interval_seconds=args.wandb_poll_interval_seconds,
        )
        gpu_reader = None
        state = TrainingDashboardState(
            run_dir=Path("."),
            log_path=Path("/nonexistent/wandb-train.log"),
            event_dir=Path("/nonexistent/wandb-events"),
            gpu_history=GPUHistory(lambda: {}, 1, 1),
            scalar_reader=run_reader,
            config_reader=WandbConfigReader(run_reader),
            status_reader=run_reader.status,
            run_label=f"wandb://{args.wandb_run}",
            telemetry_from_scalars=True,
        )
        source_description = f"wandb={args.wandb_run}"
    else:
        run_dir = args.run_dir.expanduser().resolve()
        log_path = (
            args.log_file.expanduser().resolve()
            if args.log_file is not None
            else run_dir / "train.log"
        )
        event_dir = (
            args.event_dir.expanduser().resolve()
            if args.event_dir is not None
            else run_dir / "runs_0"
        )
        gpu_reader = NVMLGPUReader(args.gpu_index)
        gpu_history = GPUHistory(
            gpu_reader,
            interval_seconds=args.gpu_sample_interval_seconds,
            max_samples=args.max_gpu_samples,
        )
        state = TrainingDashboardState(
            run_dir=run_dir,
            log_path=log_path,
            event_dir=event_dir,
            gpu_history=gpu_history,
        )
        source_description = (
            f"run_dir={run_dir} log={log_path} events={event_dir} "
            f"gpu={args.gpu_index}"
        )
    state.snapshot()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(state, args.stream_interval_seconds),
    )
    server.daemon_threads = True
    print(
        f"MVTRACKER_TRAINING_DASHBOARD http://{args.host}:{server.server_port} "
        f"{source_description}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if gpu_reader is not None:
            gpu_reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
