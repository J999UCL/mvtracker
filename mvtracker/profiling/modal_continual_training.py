"""Launch invariants for the two-H100 continual-training experiment."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Callable


GPU_REQUEST = "H100!:2"
GPU_COUNT = 2
MAX_CONTAINERS = 1
EPHEMERAL_DISK_MIB = 512 * 1024
DATASET_IMAGE_ROOT = "/opt/mvtracker-data"
DATASET_IMAGE_VERSION = "diegesis21-mvkubric100-val101-102-v1"
CONTINUAL_RUN_SUBDIR = "continual-training"
WORKSPACE_CONTAINER_LIMIT = 10
REQUIRED_FREE_SLOTS = 2
MODAL_TAGS = {
    "owner": "jeet",
    "project": "mvtracker",
    "purpose": "training",
}
PROFILE_TAGS = {
    "owner": "jeet",
    "project": "mvtracker",
    "purpose": "profiling",
}
WANDB_ENTITY = "jeetucl-ucl"
WANDB_PROJECT = "mvtracker-continual-training"
WANDB_GROUP = "gt-depth-replay-v1"
MAIN_CONFIRMATION = "RUN_MAIN_1000_STEPS"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class ActiveContainer:
    container_id: str
    app_name: str


def validate_source_commit(commit: str) -> str:
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("source commit must be one full lowercase Git SHA")
    return commit


def validate_run_name(run_name: str) -> str:
    if _RUN_NAME.fullmatch(run_name) is None:
        raise ValueError("run name contains unsupported characters")
    return run_name


def parse_active_containers(payload: str) -> tuple[ActiveContainer, ...]:
    records = json.loads(payload)
    if not isinstance(records, list):
        raise ValueError("Modal container inventory must be a JSON list")
    return tuple(
        ActiveContainer(
            container_id=str(record["container_id"]),
            app_name=str(record["app_name"]),
        )
        for record in records
    )


def require_training_capacity(
    containers: tuple[ActiveContainer, ...],
    *,
    workspace_limit: int = WORKSPACE_CONTAINER_LIMIT,
    required_free_slots: int = REQUIRED_FREE_SLOTS,
) -> None:
    available = workspace_limit - len(containers)
    if available < required_free_slots:
        names = ", ".join(container.app_name for container in containers) or "none"
        raise RuntimeError(
            f"training requires {required_free_slots} free Prism slots but only "
            f"{available} are available; active apps: {names}"
        )


def preflight_active_containers(
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    *,
    required_free_slots: int = REQUIRED_FREE_SLOTS,
) -> tuple[ActiveContainer, ...]:
    completed = runner(
        ["modal", "container", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    containers = parse_active_containers(completed.stdout)
    require_training_capacity(containers, required_free_slots=required_free_slots)
    return containers


def require_pushed_main_commit(
    commit: str,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    validate_source_commit(commit)
    completed = runner(
        ["git", "ls-remote", "origin", "refs/heads/main"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote_commit = completed.stdout.split(maxsplit=1)[0]
    if remote_commit != commit:
        raise RuntimeError(
            f"MVTRACKER_MODAL_COMMIT {commit} is not the pushed origin/main SHA "
            f"({remote_commit})"
        )


def require_main_confirmation(mode: str, confirm_main: bool) -> None:
    if mode not in {"smoke", "smoke10", "main"}:
        raise ValueError("mode must be smoke, smoke10 or main")
    if mode == "main" and not confirm_main:
        raise RuntimeError("main training requires --confirm-main")


def require_remote_main_confirmation(mode: str, confirmation: str) -> None:
    if mode == "main" and confirmation != MAIN_CONFIRMATION:
        raise RuntimeError("main training requires explicit 1000-step confirmation")
