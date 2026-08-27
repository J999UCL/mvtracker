"""Compile an immutable v1 recipe into compact schema-v2 execution sidecars."""

from __future__ import annotations

import argparse
import importlib
import subprocess
from pathlib import Path

from mvtracker.datasets.training_recipe import compile_execution_recipe


def _load_factory(spec: str):
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("--factory must be MODULE:FUNCTION")
    return getattr(importlib.import_module(module_name), function_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--factory",
        required=True,
        help="MODULE:FUNCTION returning (datasets, request_factories)",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    datasets, request_factories = _load_factory(args.factory)()
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    compile_execution_recipe(
        args.source,
        args.output,
        datasets=datasets,
        request_factories=request_factories,
        compiler_commit=commit,
        worker_count=args.workers,
    )


if __name__ == "__main__":
    main()
