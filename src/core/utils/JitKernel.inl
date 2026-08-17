//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  Copyright (c) 2026, Advanced Micro Devices, Inc.
//
//  SPDX-License-Identifier: BSD-3-Clause

// Template implementation for JitKernel.h
// This file is included at the end of JitKernel.h

#ifndef DEME_JIT_KERNEL_INL
#define DEME_JIT_KERNEL_INL

#include <cstring>
#include <vector>
#include <memory>

namespace deme {
namespace jit {

namespace detail {

// Helper to pack a single argument by copying to a buffer
template <typename T>
inline void packArg(std::vector<void*>& args, std::vector<std::unique_ptr<char[]>>& storage, T arg) {
    auto buf = std::make_unique<char[]>(sizeof(T));
    std::memcpy(buf.get(), &arg, sizeof(T));
    args.push_back(buf.get());
    storage.push_back(std::move(buf));
}

// Recursive pack for variadic args
template <typename T, typename... Rest>
inline void packArgs(std::vector<void*>& args, std::vector<std::unique_ptr<char[]>>& storage, T first, Rest... rest) {
    packArg(args, storage, first);
    if constexpr (sizeof...(rest) > 0) {
        packArgs(args, storage, rest...);
    }
}

// Base case: no args
inline void packArgs(std::vector<void*>&, std::vector<std::unique_ptr<char[]>>&) {}

}  // namespace detail

template <typename... Args>
void KernelLauncher::launch(Args... args) {
    // Pack each argument into a heap buffer and collect pointers to them, then
    // forward to the backend's non-template raw launcher (hipModuleLaunchKernel
    // on ROCm, jitify's cuLaunchKernel wrapper on CUDA). Both runtimes take the
    // same "array of pointers to argument values" convention.
    std::vector<void*> kernel_args;
    std::vector<std::unique_ptr<char[]>> arg_storage;

    if constexpr (sizeof...(args) > 0) {
        detail::packArgs(kernel_args, arg_storage, args...);
    }

    launchRaw(kernel_args.empty() ? nullptr : kernel_args.data(), kernel_args.size());
}

}  // namespace jit
}  // namespace deme

#endif  // DEME_JIT_KERNEL_INL
