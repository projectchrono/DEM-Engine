//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  Copyright (c) 2026, Advanced Micro Devices, Inc.
//
//  SPDX-License-Identifier: BSD-3-Clause

// Unified runtime kernel compilation API for DEM-Engine.
// Abstracts over jitify v1 (CUDA/NVRTC) and hiprtc (ROCm), enabling both
// platforms to use the same fluent kernel launch syntax.

#ifndef DEME_JIT_KERNEL_H
#define DEME_JIT_KERNEL_H

#include <string>
#include <vector>
#include <memory>
#include <cstddef>
#include <utility>

#ifdef USE_HIP
    #include <hip/hip_runtime.h>
using gpuStream_t = hipStream_t;
#else
    #include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#endif

namespace deme {
namespace jit {

class Kernel;
class Program;

// Fluent kernel launch builder - matches existing jitify v1 usage pattern.
// Usage: program->kernel("name")().instantiate().configure(...).launch(args...)
class KernelLauncher {
  public:
    // For kernels without template parameters
    KernelLauncher& instantiate();

    // For kernels with template parameters.
    // type_names: comma-separated type names as they appear in the kernel template.
    // On hiprtc, these get baked into the mangled kernel name.
    KernelLauncher& instantiate(const std::string& type_names);

    // Configure launch parameters
    KernelLauncher& configure(dim3 grid, dim3 block, size_t shared_mem = 0, gpuStream_t stream = 0);

    // Launch with arguments - variadic template forwards to backend
    // The implementation is in the .inl file for HIP, or calls jitify for CUDA
    template <typename... Args>
    void launch(Args... args);

    // Non-templated launch for backend implementations
    void launchRaw(void** kernel_args, size_t num_args);

    // Kernel can create KernelLaunchers
    explicit KernelLauncher(Kernel* k);

  private:
    friend class Kernel;

    Kernel* kernel_;
    dim3 grid_{1, 1, 1};
    dim3 block_{1, 1, 1};
    size_t shared_mem_ = 0;
    gpuStream_t stream_ = 0;
    std::string instantiation_;
    bool instantiated_ = false;
};

// Handle to a kernel function within a compiled program.
// Created by Program::kernel(name).
class Kernel {
  public:
    ~Kernel();
    Kernel(Kernel&&) noexcept;
    Kernel& operator=(Kernel&&) noexcept;

    // Deleted copy operations - kernel handles are move-only
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    // Start a launch chain. Returns a KernelLauncher for fluent chaining.
    KernelLauncher operator()();

    // Direct instantiate() method to match jitify v1 API.
    // Usage: program->kernel("name").instantiate().configure(...).launch(...)
    KernelLauncher& instantiate();
    KernelLauncher& instantiate(const std::string& type_names);

    // Get kernel name (for debugging/error messages)
    const std::string& name() const;

  private:
    friend class Program;
    friend class KernelLauncher;
    Kernel();

    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::unique_ptr<KernelLauncher> launcher_;  // For instantiate() return reference
};

// A compiled GPU program containing one or more kernels.
// Created by ProgramCache::program().
class Program {
  public:
    ~Program();
    Program(Program&&) noexcept;
    Program& operator=(Program&&) noexcept;

    // Deleted copy operations - programs are move-only
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    // Get a kernel by name. The kernel can then be launched via:
    // program->kernel("name")().instantiate().configure(...).launch(...)
    Kernel kernel(const std::string& name);

    // Check if program is valid (compiled successfully)
    explicit operator bool() const;

  private:
    friend class ProgramCache;
    Program();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Caches compiled programs to avoid recompilation.
// Thread-safe: multiple threads can call program() concurrently.
class ProgramCache {
  public:
    ProgramCache();
    ~ProgramCache();

    // Compile a program.
    // - name: identifier for the program (typically the kernel file name)
    // - source: kernel source code (prepended with name for jitify compatibility)
    // - headers: additional header code as (name, source) pairs
    // - flags: compiler flags (-I, -D, -arch, etc.)
    //
    // Returns a new Program by value. Caching is not performed at this level;
    // callers typically cache via shared_ptr<Program> at the application level.
    Program program(const std::string& name,
                    const std::string& source,
                    const std::vector<std::pair<std::string, std::string>>& headers = {},
                    const std::vector<std::string>& flags = {});

    // Clear all cached programs. Invalidates all Program references.
    void clear();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Helper to build compiler flags in a platform-agnostic way.
class CompilerFlags {
  public:
    // Add an include path (-I)
    CompilerFlags& include(const std::string& path);

    // Add a macro definition (-D)
    CompilerFlags& define(const std::string& macro, const std::string& value = "");

    // Set target architecture (compute_75 for CUDA, gfx90a for HIP)
    // The flag format is handled automatically per platform.
    CompilerFlags& arch(const std::string& arch);

    // Add a raw flag (passed through unchanged)
    CompilerFlags& flag(const std::string& raw_flag);

    // Build the final flags vector
    std::vector<std::string> build() const;

  private:
    std::vector<std::string> flags_;
};

}  // namespace jit
}  // namespace deme

// Include the template implementation (must be visible at all call sites)
#include "JitKernel.inl"

#endif  // DEME_JIT_KERNEL_H
