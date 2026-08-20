"""Coordinate process-local CUDA decoding with global graph capture."""

from __future__ import annotations

from threading import Lock


CUDA_CAPTURE_LOCK = Lock()
