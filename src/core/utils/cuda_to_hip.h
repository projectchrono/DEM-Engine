//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  Copyright (c) 2026, Advanced Micro Devices, Inc.
//
//  SPDX-License-Identifier: BSD-3-Clause

// CUDA-to-HIP compatibility header for DEM-Engine.
// On ROCm this aliases CUDA runtime API symbols to their HIP equivalents.
// On CUDA this is a passthrough to the standard CUDA runtime.

#ifndef DEME_CUDA_TO_HIP_H
#define DEME_CUDA_TO_HIP_H

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

    #include <hip/hip_runtime.h>

    // Runtime API types
    #define cudaStream_t hipStream_t
    #define cudaEvent_t hipEvent_t
    #define cudaError_t hipError_t
    #define cudaDeviceProp hipDeviceProp_t

    // Error codes
    #define cudaSuccess hipSuccess
    #define cudaErrorInvalidValue hipErrorInvalidValue
    #define cudaErrorMemoryAllocation hipErrorMemoryAllocation
    #define cudaErrorNotSupported hipErrorNotSupported

    // Memory management
    #define cudaMalloc hipMalloc
    #define cudaFree hipFree
    #define cudaMallocManaged hipMallocManaged
    #define cudaMallocHost hipHostMalloc
    #define cudaHostAlloc hipHostMalloc
    #define cudaFreeHost hipHostFree
    #define cudaMemcpy hipMemcpy
    #define cudaMemcpyAsync hipMemcpyAsync
    #define cudaMemset hipMemset
    #define cudaMemsetAsync hipMemsetAsync
    #define cudaMemPrefetchAsync hipMemPrefetchAsync

    // Memory copy kinds
    #define cudaMemcpyHostToHost hipMemcpyHostToHost
    #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
    #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
    #define cudaMemcpyDefault hipMemcpyDefault

    // Memory attach flags
    #define cudaMemAttachGlobal hipMemAttachGlobal

    // Host alloc flags
    #define cudaHostAllocDefault hipHostMallocDefault

    // Stream management
    #define cudaStreamCreate hipStreamCreate
    #define cudaStreamDestroy hipStreamDestroy
    #define cudaStreamSynchronize hipStreamSynchronize

    // Device management
    #define cudaGetDevice hipGetDevice
    #define cudaSetDevice hipSetDevice
    #define cudaGetDeviceCount hipGetDeviceCount
    #define cudaGetDeviceProperties hipGetDeviceProperties
    #define cudaDeviceSynchronize hipDeviceSynchronize
    #define cudaDeviceReset hipDeviceReset

    // Error handling
    #define cudaGetLastError hipGetLastError
    #define cudaGetErrorString hipGetErrorString
    #define cudaPeekAtLastError hipPeekAtLastError

    // Pointer attributes
    #define cudaPointerAttributes hipPointerAttribute_t
    #define cudaPointerGetAttributes hipPointerGetAttributes

    // Memory types - use a namespace alias to preserve scoped access syntax.
    // In hiprtc JIT kernels, hipMemoryType may not be available (device-only runtime).
    // Provide a minimal fallback.
    #ifdef __HIPCC_RTC__
// hiprtc device-only context: define our own enum for compatibility
enum hipMemoryType {
    hipMemoryTypeUnregistered = 0,
    hipMemoryTypeHost = 1,
    hipMemoryTypeDevice = 2,
    hipMemoryTypeManaged = 3
};
    #endif

namespace cudaMemoryType {
constexpr hipMemoryType cudaMemoryTypeUnregistered = hipMemoryTypeUnregistered;
constexpr hipMemoryType cudaMemoryTypeHost = hipMemoryTypeHost;
constexpr hipMemoryType cudaMemoryTypeDevice = hipMemoryTypeDevice;
constexpr hipMemoryType cudaMemoryTypeManaged = hipMemoryTypeManaged;
}  // namespace cudaMemoryType

    // Memory advise
    #define cudaMemAdvise hipMemAdvise
    #define cudaMemoryAdvise hipMemoryAdvise

    // CUB -> hipCUB
    #define cub hipcub

    // cuRAND device headers -> hipRAND (not needed in this shim, handled in API.h)

    // CUDA floating-point intrinsics with directed rounding.
    // HIP/AMD does not provide exact equivalents; use standard operations.
    // The _ru suffix means "round up" (toward +infinity), _rd means "round down",
    // _rn means "round to nearest", _rz means "round toward zero".
    // For most physics simulations, the difference is negligible.
    #if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
__device__ inline float __frcp_ru(float x) {
    return 1.0f / x;
}
__device__ inline double __drcp_ru(double x) {
    return 1.0 / x;
}
__device__ inline float __fmul_ru(float x, float y) {
    return x * y;
}
__device__ inline double __dmul_ru(double x, double y) {
    return x * y;
}
__device__ inline float __fadd_ru(float x, float y) {
    return x + y;
}
__device__ inline double __dadd_ru(double x, double y) {
    return x + y;
}
    #endif

#else  // CUDA

    #include <cuda_runtime.h>

#endif  // USE_HIP

#endif  // DEME_CUDA_TO_HIP_H
