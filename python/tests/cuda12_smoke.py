"""Validate pip CUDA headers/compiler without a GPU; optionally run the solver.

Run against an installed wheel with --gpu for a fresh JIT simulation. The default
mode deliberately loads only the Python bootstrap, so CI needs no driver stub.
"""

import argparse
import ctypes
from importlib.metadata import distribution
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile


def main():
    """Compile representative CUDA headers, then optionally check free fall."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", default="deme")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    package = distribution(args.distribution)
    bootstrap = runpy.run_path(str(package.locate_file("deme/_cuda.py")))
    environment_before = dict(os.environ)
    includes = bootstrap["configure_cuda"]()
    assert dict(os.environ) == environment_before, "CUDA bootstrap must not change the environment"
    assert includes, "Install the cuda12 extra before running this test"
    nvrtc = bootstrap["_libraries"][-1]
    program = ctypes.c_void_p()
    source = b"""
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <curand_kernel.h>
    #include <cub/block/block_reduce.cuh>
    extern "C" __global__ void smoke(float* out) {
        curandState state;
        curand_init(123, threadIdx.x, 0, &state);
        out[threadIdx.x] = curand_uniform(&state);
    }
    """
    result = nvrtc.nvrtcCreateProgram(ctypes.byref(program), source, b"smoke.cu", 0, None, None)
    assert result == 0, result
    options = [b"--std=c++17", b"--gpu-architecture=compute_75"]
    options.extend(("-I" + path).encode() for path in includes)
    option_array = (ctypes.c_char_p * len(options))(*options)
    try:
        result = nvrtc.nvrtcCompileProgram(program, len(options), option_array)
        size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        nvrtc.nvrtcGetProgramLog(program, log)
        assert result == 0, log.value.decode()
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    print("PASS: pip NVRTC compiled CUDA runtime, compiler, CURAND, and CCCL headers", flush=True)

    if args.gpu:
        # A new cache ensures initialization really compiles the solver kernels.
        with tempfile.TemporaryDirectory(prefix="deme-cuda12-jit-") as cache:
            os.environ["DEME_JIT_CACHE_DIR"] = cache
            environment_before = dict(os.environ)
            import deme
            import importlib

            importlib.reload(deme)
            assert dict(os.environ) == environment_before, "Import must not export CUDA configuration"
            # An exec'd child must see exactly the original environment, with no Python CUDA override.
            child = subprocess.check_output(
                [sys.executable, "-c", "import os; print(os.environ.get('DEME_CUDA_INCLUDE_PATH', '<unset>'))"],
                text=True,
            ).strip()
            assert child == environment_before.get("DEME_CUDA_INCLUDE_PATH", "<unset>")

            mappings = Path("/proc/self/maps").read_text().splitlines()
            cuda_mappings = [line for line in mappings if "libcudart.so" in line or "libnvrtc" in line]
            assert cuda_mappings and all("/nvidia/" in line for line in cuda_mappings), cuda_mappings
            solver = deme.DEMSolver([0])
            solver.SetVerbosity("ERROR")
            solver.InstructBoxDomainDimension((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))
            solver.SetGravitationalAcceleration([0.0, 0.0, -9.81])
            material = solver.LoadMaterial({"E": 1e7, "nu": 0.3, "CoR": 0.5, "mu": 0.4, "Crr": 0.0})
            sphere = solver.LoadSphereType(1.0, 0.01, material)
            batch = solver.AddClumps(sphere, [[0.0, 0.0, 0.0]])
            tracker = solver.Track(batch)
            solver.SetInitTimeStep(1e-5)
            solver.Initialize()
            solver.DoDynamicsThenSync(0.01)
            fingerprints = list(Path(cache).rglob("fingerprint.txt"))
            assert fingerprints, "Expected fresh JIT cache entries"
            assert all("-I/usr/local/cuda" not in path.read_text() for path in fingerprints)
            position = tracker.Pos()
            assert -0.0006 < position[2] < -0.0004, position
            print("PASS: fresh solver JIT and GPU free fall", flush=True)


if __name__ == "__main__":
    main()
