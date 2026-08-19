"""Build the indexed-correlation CUDA extension into the source package.

The Modal image runs this after checking out the source tree. The resulting
shared object is imported by the production operator, so CUDA training never
invokes ``torch.utils.cpp_extension.load`` at runtime.
"""

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--build-directory", type=Path)
    args = parser.parse_args()
    package_dir = args.source_dir or (
        Path(__file__).resolve().parents[1]
        / "mvtracker"
        / "models"
        / "core"
        / "mvtracker"
    )
    build_directory = args.build_directory or package_dir
    build_directory.mkdir(parents=True, exist_ok=True)
    load(
        name="mvtracker_indexed_correlation_cuda",
        sources=[
            str(package_dir / "indexed_correlation_cuda.cpp"),
            str(package_dir / "indexed_correlation_cuda.cu"),
        ],
        build_directory=str(build_directory),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
