"""Build the indexed-correlation CUDA extension into the source package.

The Modal image runs this after checking out the source tree. The resulting
shared object is imported by the production operator, so CUDA training never
invokes ``torch.utils.cpp_extension.load`` at runtime.
"""

from pathlib import Path

from torch.utils.cpp_extension import load


def main() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "mvtracker" / "models" / "core" / "mvtracker"
    load(
        name="mvtracker_indexed_correlation_cuda",
        sources=[
            str(package_dir / "indexed_correlation_cuda.cpp"),
            str(package_dir / "indexed_correlation_cuda.cu"),
        ],
        build_directory=str(package_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
