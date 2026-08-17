//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  Copyright (c) 2026, Advanced Micro Devices, Inc.
//
//  SPDX-License-Identifier: BSD-3-Clause

// CUDA backend for JitKernel using jitify v1 (NVRTC)

#ifndef USE_HIP

    #include "JitKernel.h"

    // Jitify is in the build tree at ${CMAKE_BINARY_DIR}/src/jitify/
    // Include path is added by CMake target_include_directories
    #include <jitify/jitify.hpp>
    #include <unordered_map>
    #include <stdexcept>
    #include <mutex>

namespace deme {
namespace jit {

// ============================================================================
// Kernel implementation
// ============================================================================

struct Kernel::Impl {
    std::string name;
    jitify::Program* program = nullptr;  // Non-owning reference to parent program
    jitify::KernelInstantiation instantiation;
    bool has_instantiation = false;
};

Kernel::Kernel() : impl_(std::make_unique<Impl>()) {}
Kernel::~Kernel() = default;
Kernel::Kernel(Kernel&&) noexcept = default;
Kernel& Kernel::operator=(Kernel&&) noexcept = default;

KernelLauncher Kernel::operator()() {
    return KernelLauncher(this);
}

KernelLauncher& Kernel::instantiate() {
    launcher_ = std::make_unique<KernelLauncher>(this);
    return launcher_->instantiate();
}

KernelLauncher& Kernel::instantiate(const std::string& type_names) {
    launcher_ = std::make_unique<KernelLauncher>(this);
    return launcher_->instantiate(type_names);
}

const std::string& Kernel::name() const {
    return impl_->name;
}

// ============================================================================
// KernelLauncher implementation
// ============================================================================

KernelLauncher::KernelLauncher(Kernel* k) : kernel_(k) {}

KernelLauncher& KernelLauncher::instantiate() {
    kernel_->impl_->instantiation = kernel_->impl_->program->kernel(kernel_->impl_->name).instantiate();
    kernel_->impl_->has_instantiation = true;
    instantiated_ = true;
    return *this;
}

KernelLauncher& KernelLauncher::instantiate(const std::string& type_names) {
    // Parse comma-separated type names and instantiate with them.
    // Jitify uses variadic templates for instantiation, but we can pass
    // the type string directly to the kernel() call via name mangling.
    // For CUDA/jitify, we use the instantiate(type_name) overload.
    kernel_->impl_->instantiation = kernel_->impl_->program->kernel(kernel_->impl_->name).instantiate({type_names});
    kernel_->impl_->has_instantiation = true;
    instantiated_ = true;
    instantiation_ = type_names;
    return *this;
}

KernelLauncher& KernelLauncher::configure(dim3 grid, dim3 block, size_t shared_mem, gpuStream_t stream) {
    grid_ = grid;
    block_ = block;
    shared_mem_ = shared_mem;
    stream_ = stream;
    return *this;
}

// ============================================================================
// Program implementation
// ============================================================================

struct Program::Impl {
    jitify::Program jit_program;
};

Program::Program() : impl_(std::make_unique<Impl>()) {}
Program::~Program() = default;
Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) noexcept = default;

Program::operator bool() const {
    return impl_ != nullptr;
}

Kernel Program::kernel(const std::string& name) {
    Kernel k;
    k.impl_->name = name;
    k.impl_->program = &impl_->jit_program;
    return k;
}

// ============================================================================
// ProgramCache implementation
// ============================================================================

struct ProgramCache::Impl {
    jitify::JitCache cache;
};

ProgramCache::ProgramCache() : impl_(std::make_unique<Impl>()) {}
ProgramCache::~ProgramCache() = default;

Program ProgramCache::program(const std::string& name,
                              const std::string& source,
                              const std::vector<std::pair<std::string, std::string>>& headers,
                              const std::vector<std::string>& flags) {
    // jitify::JitCache handles caching internally, so we just wrap the result
    (void)name;  // Name is embedded in source for jitify

    // Extract header sources (jitify v1 takes just the source strings)
    std::vector<std::string> header_sources;
    for (const auto& h : headers) {
        header_sources.push_back(h.second);
    }

    Program prog;
    prog.impl_->jit_program = impl_->cache.program(source, header_sources, flags);
    return prog;
}

void ProgramCache::clear() {
    // No-op: jitify::JitCache doesn't expose a clear method
}

// ============================================================================
// CompilerFlags implementation
// ============================================================================

CompilerFlags& CompilerFlags::include(const std::string& path) {
    flags_.push_back("-I" + path);
    return *this;
}

CompilerFlags& CompilerFlags::define(const std::string& macro, const std::string& value) {
    if (value.empty()) {
        flags_.push_back("-D" + macro);
    } else {
        flags_.push_back("-D" + macro + "=" + value);
    }
    return *this;
}

CompilerFlags& CompilerFlags::arch(const std::string& arch) {
    flags_.push_back("-arch=" + arch);
    return *this;
}

CompilerFlags& CompilerFlags::flag(const std::string& raw_flag) {
    flags_.push_back(raw_flag);
    return *this;
}

std::vector<std::string> CompilerFlags::build() const {
    return flags_;
}

// ============================================================================
// KernelLauncher::launchRaw() - launch via jitify's cuLaunchKernel wrapper
// ============================================================================

// The templated launch() (in JitKernel.inl) packs each argument into a buffer
// and hands the resulting array of pointers here, identically to the HIP
// backend. jitify's launch takes the same "vector of pointers to argument
// values" convention, so we forward straight to it.
void KernelLauncher::launchRaw(void** kernel_args, size_t num_args) {
    if (!kernel_->impl_->has_instantiation) {
        throw std::runtime_error("Kernel '" + kernel_->impl_->name + "' not instantiated before launch");
    }

    // Skip launch if any grid dimension is zero (no work); keeps parity with the
    // HIP backend, where a zero grid is an error rather than a silent no-op.
    if (grid_.x == 0 || grid_.y == 0 || grid_.z == 0) {
        return;
    }

    std::vector<void*> arg_ptrs(kernel_args, kernel_args + num_args);
    kernel_->impl_->instantiation.configure(grid_, block_, shared_mem_, stream_).safe_launch(arg_ptrs);
}

}  // namespace jit
}  // namespace deme

#endif  // !USE_HIP
