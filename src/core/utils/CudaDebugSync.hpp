// Copyright (c) 2026, SBEL GPU Development Team
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_CUDA_DEBUG_SYNC_HPP
#define DEME_CUDA_DEBUG_SYNC_HPP

#include <atomic>

#include <cuda_runtime_api.h>

namespace deme {

// Debug-only synchronization is process-wide because JIT launch helpers and low-level data migration utilities do not
// belong to a particular DEMSolver. Keep this disabled for normal asynchronous execution; enable it while debugging to
// report asynchronous CUDA errors at the call that enqueued the work.
inline std::atomic<bool>& CudaDebugSyncEnabledFlag() {
    static std::atomic<bool> enabled{false};
    return enabled;
}

inline void SetCudaDebugSyncEnabled(bool enabled) {
    CudaDebugSyncEnabledFlag().store(enabled, std::memory_order_relaxed);
}

inline bool IsCudaDebugSyncEnabled() {
    return CudaDebugSyncEnabledFlag().load(std::memory_order_relaxed);
}

inline cudaError_t DebugSyncStream(cudaStream_t stream) {
    return IsCudaDebugSyncEnabled() ? cudaStreamSynchronize(stream) : cudaSuccess;
}

}  // namespace deme

#endif
