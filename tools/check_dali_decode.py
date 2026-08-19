"""Check one real MV-Kubric RGB/TIFF pair against the DALI decoder.

This intentionally requires the production CUDA image. It does not silently
fall back to a CPU decoder; the CPU readers are used only to establish the
parity reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def check_pair(rgb_path: Path, depth_path: Path) -> dict[str, object]:
    import torch
    from PIL import Image

    from mvtracker.datasets.tapvid3d_multiview_dataset import DaliEncodedImageDecoder

    if not torch.cuda.is_available():
        raise RuntimeError("DALI parity check requires a CUDA device")
    rgb_reference = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth_reference = np.asarray(Image.open(depth_path))
    decoder = DaliEncodedImageDecoder(torch.device("cuda"), num_threads=1, prefetch_queue_depth=1)
    rgb_decoded, depth_decoded = decoder.decode(
        [rgb_path.read_bytes()], [depth_path.read_bytes()]
    )
    torch.cuda.synchronize()
    rgb_actual = rgb_decoded[0].cpu().numpy()
    depth_actual = depth_decoded[0].cpu().numpy()
    if depth_actual.ndim == 3:
        depth_actual = depth_actual[..., 0]
    if rgb_actual.shape != rgb_reference.shape:
        raise AssertionError(f"RGB shape mismatch: {rgb_actual.shape} != {rgb_reference.shape}")
    if depth_actual.shape != depth_reference.shape:
        raise AssertionError(
            f"depth shape mismatch: {depth_actual.shape} != {depth_reference.shape}"
        )
    rgb_max_abs = int(np.abs(rgb_actual.astype(np.int16) - rgb_reference.astype(np.int16)).max())
    depth_max_abs = float(
        np.abs(depth_actual.astype(np.float64) - depth_reference.astype(np.float64)).max()
    )
    if rgb_max_abs > 1:
        raise AssertionError(f"RGB decoder disagreement exceeds one code value: {rgb_max_abs}")
    if depth_max_abs != 0.0:
        raise AssertionError(f"TIFF depth is not lossless: max absolute error {depth_max_abs}")
    return {
        "rgb": str(rgb_path),
        "depth": str(depth_path),
        "rgb_shape": list(rgb_actual.shape),
        "depth_shape": list(depth_actual.shape),
        "rgb_max_abs_error": rgb_max_abs,
        "depth_max_abs_error": depth_max_abs,
        "rgb_dtype": str(rgb_actual.dtype),
        "depth_dtype": str(depth_actual.dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check_pair(args.rgb, args.depth), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
