#!/usr/bin/env bash
# Run inside cibuildwheel's manylinux 2.28 (AlmaLinux 8) container. Install only
# compilation/runtime headers and libraries, without GPU drivers or GUI tools.
set -euo pipefail

curl --fail --location --retry 3 \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    --output /etc/yum.repos.d/cuda-rhel8.repo
# The driver development package supplies libcuda.so for linking on GPU-less CI;
# it does not install a GPU driver. Auditwheel excludes this stub from the wheel.
dnf install -y \
    cuda-minimal-build-12-8 cuda-driver-devel-12-8 cuda-nvrtc-devel-12-8 libcurand-devel-12-8 \
    libX11-devel libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel mesa-libGL-devel
dnf clean all

# Fail before a long wheel build if repository resolution selected the wrong compiler.
/usr/local/cuda-12.8/bin/nvcc --version | grep -F 'release 12.8,'

# Check the RPM layout used by the wheel's CMAKE_PREFIX_PATH before building six ABIs.
# Resolve CUB headers and all CUDA targets linked by DEME, including the driver stub.
cuda_probe=$(mktemp -d)
trap 'rm -rf "$cuda_probe"' EXIT
cat > "$cuda_probe/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.18)
project(DEMEWheelCUDACheck LANGUAGES CXX)
find_package(CUDAToolkit REQUIRED)
foreach(component cudart nvrtc cuda_driver)
    if(NOT TARGET CUDA::${component})
        message(FATAL_ERROR "Missing wheel build dependency: CUDA::${component}")
    endif()
endforeach()
find_package(CUB REQUIRED CONFIG)
CMAKE
cmake -S "$cuda_probe" -B "$cuda_probe/build" \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
    -DCMAKE_PREFIX_PATH=/usr/local/cuda-12.8/targets/x86_64-linux
