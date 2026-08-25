"""Linux page-cache hints for one-pass training data."""

from __future__ import annotations

import os
from pathlib import Path


def discard_file_range(descriptor: int, offset: int = 0, length: int = 0) -> None:
    os.posix_fadvise(
        descriptor,
        int(offset),
        int(length),
        os.POSIX_FADV_DONTNEED,
    )


def flush_and_discard_file(path: str | Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
        discard_file_range(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["discard_file_range", "flush_and_discard_file"]
