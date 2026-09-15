"""Locate optional NVIDIA wheels before importing the native extension."""

import ctypes
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import sys


# Retain handles for the lifetime of the extension, including NVRTC's builtins.
_libraries = []


def configure_cuda():
    """Use a complete pip CUDA installation, or leave system CUDA discovery alone.

    Preloading by absolute path is necessary: changing LD_LIBRARY_PATH inside a
    running Python process does not update the Linux dynamic loader's search.
    """
    if sys.platform != "linux":
        return []
    components = ("cuda_runtime", "cuda_nvrtc", "cuda_nvcc", "cuda_cccl", "curand")
    roots = {}
    for component in components:
        try:
            package = distribution("nvidia-" + component.replace("_", "-") + "-cu12")
        except PackageNotFoundError:
            return []
        roots[component] = Path(package.locate_file("nvidia/" + component))

    includes = [root / "include" for root in roots.values()]
    # CCCL wheels can put their header trees one level below include/.
    includes.append(roots["cuda_cccl"] / "include" / "cccl")
    for component, pattern in (
        ("cuda_runtime", "libcudart.so.12"),
        ("cuda_nvrtc", "libnvrtc-builtins.so.*"),
        ("cuda_nvrtc", "libnvrtc.so.12"),
    ):
        paths = sorted((roots[component] / "lib").glob(pattern))
        if not paths:
            raise ImportError("Incomplete NVIDIA CUDA wheels: missing " + pattern)
        _libraries.append(ctypes.CDLL(str(paths[0]), mode=ctypes.RTLD_GLOBAL))

    # Return paths to the private binding; do not alter the process environment.
    return [str(path) for path in includes if path.is_dir()]
