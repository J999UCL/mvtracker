#!/usr/bin/env python3
"""Build the random-access MVTracker cache for raw multi-view TAPVid-3D."""

from __future__ import annotations

import argparse
from pathlib import Path

from mvtracker.datasets.tapvid3d_multiview_dataset import prepare_tapvid3d_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    counts = prepare_tapvid3d_cache(
        args.raw_root,
        args.cache_root,
        workers=args.workers,
    )
    print(f"prepared={counts['prepared']} reused={counts['reused']}")


if __name__ == "__main__":
    main()
