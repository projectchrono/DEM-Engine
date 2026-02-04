# Building the Python Package

This document describes how to build the Python package for DEM-Engine.

## Prerequisites

- CUDA Toolkit (11.0 or later recommended; tested with 11.0-12.x)
  - Compute capability 6.0+ required (Pascal architecture or newer)
  - Includes CUB library (bundled with CUDA 11.0+)
- CMake (3.18 or later)
- Python (3.8 or later)
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- pip (for Python package installation)

## Building the Package

### Option 1: Using pip (recommended)

**Important for Conda Users:** If you're building in a conda environment, you need to install conda's compilers first to ensure compatibility:

```bash
# Activate your conda environment
conda activate your_env_name

# Install conda's compilers
conda install -c conda-forge gxx_linux-64 gcc_linux-64

# Then build/install
pip install .
```

Without conda's compilers, the wheel will link against your system's libstdc++ which may cause import errors in conda environments.

For non-conda environments:

```bash
pip install .
```

This will use scikit-build-core to automatically configure CMake with the Python build options enabled, compile the C++/CUDA code, and install the package.

### Option 2: Building a wheel

**Important for Conda Users:** If building in a conda environment, install conda's compilers first (see Option 1).

```bash
pip install build
python -m build
```

This creates a `.whl` file in the `dist/` directory that can be distributed and installed on other machines.

### Option 3: Development installation

For development, you can install in editable mode:

```bash
pip install -e .
```

### Option 4: Using conda

For conda users, you can build and install using conda-build. This method automatically handles all dependencies including compilers:

```bash
# Build the conda package (requires conda-build and access to conda-forge channel)
conda install conda-build -c conda-forge
conda build recipe/ -c conda-forge

# Install from local build
conda install --use-local deme
```

**Note:** The `-c conda-forge` flag is required to access build dependencies like cmake, ninja, and the C/C++ compilers.
conda install --use-local deme
```

This is useful for managing the package alongside other conda packages in your environment.

## CMake Options

When building manually with CMake, use the following option to enable Python bindings:

```bash
mkdir build
cd build
cmake .. -DDEME_BUILD_PYTHON=ON
cmake --build .
```

## Usage

After installation, you can import the package in Python:

```python
import DEME

# Create a solver
solver = DEME.DEMSolver()

# Use other classes and functions
# ...
```

## Package Contents

The Python package provides bindings for:
- `DEMSolver` - Main simulation solver class with full API including:
  - Contact detection control (`SetMaxTriTriPenetration`, `SetTriTriPenetration`)
  - Mesh contact settings (`SetMeshUniversalContact`, `SetPersistentContact`)
  - Performance tuning (`DisableJitifyClumpTemplates`)
  - Output control (`EnableOwnerWildcardOutput`, `EnableContactWildcardOutput`)
  - Error thresholds (`SetErrorOutAngularVelocity`, `SetErrorOutVelocity`)
- `DEMMaterial` - Material properties
- `DEMClumpBatch` - Batch operations for clumps
- `DEMExternObj` - External objects
- `DEMMeshConnected` - Triangle mesh support
- Samplers: `PDSampler`, `GridSampler`, `HCPSampler`
- Utility functions for transformations and sampling

## Troubleshooting

### Compiler Issues in Conda Environments

**Problem:** When building with `pip install .` in a conda environment, you get import errors like `GLIBCXX_3.4.30 not found`.

**Solution:** Install conda's compilers before building:
```bash
conda install -c conda-forge gxx_linux-64 gcc_linux-64
pip install .
```

This ensures the wheel is built with libraries compatible with your conda environment.

### Conda Build: Packages Not Found Error

**Problem:** `conda build recipe/` fails with "PackagesNotFoundError: cmake not available from current channels"

**Solution:** Specify the conda-forge channel when building:
```bash
conda build recipe/ -c conda-forge
```

The conda-forge channel provides cmake, ninja, and other build dependencies.

### CUDA Not Found

Ensure that CUDA is installed and the `CUDA_ROOT` or `CUDACXX` environment variable is set:

```bash
export CUDA_ROOT=/usr/local/cuda
```

### CUB Library Not Found

The CUDA Toolkit includes CUB. If CMake cannot find it, you may need to specify its location:

```bash
cmake .. -DDEME_BUILD_PYTHON=ON -DCUB_DIR=/path/to/cuda/lib64/cmake/cub
```

### Python Version Mismatch

Make sure you're using the same Python version for building and running. You can specify the Python executable:

```bash
cmake .. -DDEME_BUILD_PYTHON=ON -DPython_EXECUTABLE=/path/to/python
```

### libstdc++ Version Issues (Conda)

If you get an error like `GLIBCXX_3.4.30 not found` when importing in a conda environment:

1. Make sure you built the package using conda-build (not pip) within the conda environment
2. The conda recipe includes `libstdcxx-ng >=12` as a runtime dependency
3. You can also try updating libstdcxx-ng in your environment:
   ```bash
   conda install -c conda-forge libstdcxx-ng
   ```

Alternatively, build with the system Python (outside conda) if you're using pip installation.
```
