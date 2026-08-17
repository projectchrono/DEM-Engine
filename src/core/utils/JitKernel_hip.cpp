//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  Copyright (c) 2026, Advanced Micro Devices, Inc.
//
//  SPDX-License-Identifier: BSD-3-Clause

// HIP backend for JitKernel using hiprtc.
//
// Key design: hiprtc requires all name expressions (kernel names to resolve)
// be registered BEFORE compilation via hiprtcAddNameExpression(). Since the
// DEM-Engine API calls buildProgram() first, then kernel().instantiate() later,
// we use lazy compilation: defer hiprtcCompileProgram until the first kernel
// launch, by which point all kernel+instantiation requests have been collected.

#ifdef USE_HIP

    #include "JitKernel.h"
    #include <hip/hiprtc.h>
    #include <hip/hip_runtime.h>
    #include <unordered_map>
    #include <unordered_set>
    #include <stdexcept>
    #include <mutex>
    #include <vector>
    #include <cstring>
    #include <algorithm>

namespace deme {
namespace jit {

namespace {

void checkHiprtc(hiprtcResult result, const char* msg) {
    if (result != HIPRTC_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + hiprtcGetErrorString(result));
    }
}

void checkHip(hipError_t result, const char* msg) {
    if (result != hipSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(result));
    }
}

// Build the name expression string for hiprtcAddNameExpression.
// For a templated kernel like "modifyComponents" instantiated with "deme::DEMDataKT",
// the expression is "modifyComponents<deme::DEMDataKT>".
std::string buildNameExpression(const std::string& name, const std::string& instantiation) {
    if (instantiation.empty()) {
        return name;
    }
    return name + "<" + instantiation + ">";
}

}  // namespace

// Forward declaration of ProgramImpl so Kernel::Impl can reference it
struct ProgramImpl {
    // Source and compilation options (stored for lazy/deferred compilation)
    std::string name;
    std::string source;
    std::vector<std::pair<std::string, std::string>> headers;
    std::vector<std::string> flags;

    // Kernel name expressions that have been requested
    // Key: name expression (e.g., "modifyComponents<deme::DEMDataKT>")
    // Value: lowered/mangled name after compilation
    std::unordered_map<std::string, std::string> loweredNames;

    // Name expressions pending addition (requested but not yet compiled)
    std::unordered_set<std::string> pendingNameExpressions;

    // Compiled module (nullptr until first compilation)
    hipModule_t module = nullptr;

    // Compiled code buffer (kept for potential recompilation)
    std::vector<char> codeBuffer;

    // Whether we need to recompile (new name expressions added since last compile)
    bool needsRecompile = true;

    std::mutex mutex;

    ~ProgramImpl() {
        if (module) {
            (void)hipModuleUnload(module);
        }
    }

    // Compile or recompile with all registered name expressions
    void ensureCompiled() {
        std::lock_guard<std::mutex> lock(mutex);

        if (!needsRecompile && module != nullptr) {
            return;  // Already compiled with current name expressions
        }

        // Unload previous module if any
        if (module) {
            (void)hipModuleUnload(module);
            module = nullptr;
        }

        // Create hiprtc program
        hiprtcProgram prog;
        std::vector<const char*> header_names;
        std::vector<const char*> header_sources;
        for (const auto& h : headers) {
            header_names.push_back(h.first.c_str());
            header_sources.push_back(h.second.c_str());
        }

        checkHiprtc(hiprtcCreateProgram(&prog, source.c_str(), name.c_str(), static_cast<int>(header_sources.size()),
                                        header_sources.data(), header_names.data()),
                    "hiprtcCreateProgram");

        // Register all name expressions (pending ones plus any previously resolved)
        std::vector<std::string> allNameExpressions;
        for (const auto& pending : pendingNameExpressions) {
            allNameExpressions.push_back(pending);
        }
        for (const auto& resolved : loweredNames) {
            // Re-register previously resolved names (needed for recompilation)
            if (pendingNameExpressions.find(resolved.first) == pendingNameExpressions.end()) {
                allNameExpressions.push_back(resolved.first);
            }
        }

        for (const auto& expr : allNameExpressions) {
            hiprtcResult res = hiprtcAddNameExpression(prog, expr.c_str());
            if (res != HIPRTC_SUCCESS) {
                hiprtcDestroyProgram(&prog);
                throw std::runtime_error("hiprtcAddNameExpression failed for '" + expr +
                                         "': " + hiprtcGetErrorString(res));
            }
        }

        // Convert flags to const char* array
        std::vector<const char*> options;
        for (const auto& f : flags) {
            options.push_back(f.c_str());
        }

        // Compile
        hiprtcResult compileResult = hiprtcCompileProgram(prog, static_cast<int>(options.size()), options.data());

        if (compileResult != HIPRTC_SUCCESS) {
            size_t logSize;
            hiprtcGetProgramLogSize(prog, &logSize);
            std::string log(logSize, '\0');
            hiprtcGetProgramLog(prog, log.data());
            hiprtcDestroyProgram(&prog);
            throw std::runtime_error("hiprtc compilation failed for '" + name + "':\n" + log);
        }

        // Get lowered names for all expressions
        for (const auto& expr : allNameExpressions) {
            const char* lowered = nullptr;
            hiprtcResult res = hiprtcGetLoweredName(prog, expr.c_str(), &lowered);
            if (res != HIPRTC_SUCCESS) {
                hiprtcDestroyProgram(&prog);
                throw std::runtime_error("hiprtcGetLoweredName failed for '" + expr +
                                         "': " + hiprtcGetErrorString(res));
            }
            loweredNames[expr] = std::string(lowered);
        }

        // Clear pending set - all are now resolved
        pendingNameExpressions.clear();

        // Get compiled code
        size_t codeSize;
        checkHiprtc(hiprtcGetCodeSize(prog, &codeSize), "hiprtcGetCodeSize");
        codeBuffer.resize(codeSize);
        checkHiprtc(hiprtcGetCode(prog, codeBuffer.data()), "hiprtcGetCode");

        hiprtcDestroyProgram(&prog);

        // Load module
        checkHip(hipModuleLoadData(&module, codeBuffer.data()), "hipModuleLoadData");

        needsRecompile = false;
    }

    // Register a name expression (kernel + optional instantiation)
    // Returns the lowered name, triggering compilation if needed
    std::string getLoweredName(const std::string& nameExpr) {
        {
            std::lock_guard<std::mutex> lock(mutex);

            // Check if already resolved
            auto it = loweredNames.find(nameExpr);
            if (it != loweredNames.end() && !needsRecompile) {
                return it->second;
            }

            // Add to pending if not already there
            if (loweredNames.find(nameExpr) == loweredNames.end()) {
                pendingNameExpressions.insert(nameExpr);
                needsRecompile = true;
            }
        }

        // Compile (releases and re-acquires lock internally)
        ensureCompiled();

        // Now look up the lowered name
        std::lock_guard<std::mutex> lock(mutex);
        auto it = loweredNames.find(nameExpr);
        if (it == loweredNames.end()) {
            throw std::runtime_error("Internal error: lowered name not found for '" + nameExpr + "'");
        }
        return it->second;
    }
};

// ============================================================================
// Kernel implementation
// ============================================================================

struct Kernel::Impl {
    std::string name;
    std::string instantiation;
    hipFunction_t func = nullptr;
    ProgramImpl* program = nullptr;  // Non-owning reference to parent program
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
    return instantiate("");
}

KernelLauncher& KernelLauncher::instantiate(const std::string& type_names) {
    kernel_->impl_->instantiation = type_names;
    instantiation_ = type_names;

    if (kernel_->impl_->program) {
        // Build the name expression and register it
        std::string nameExpr = buildNameExpression(kernel_->impl_->name, type_names);

        // Get the lowered name (this triggers compilation if needed)
        std::string loweredName = kernel_->impl_->program->getLoweredName(nameExpr);

        // Ensure module is compiled
        kernel_->impl_->program->ensureCompiled();

        // Now get the function handle
        hipError_t err =
            hipModuleGetFunction(&kernel_->impl_->func, kernel_->impl_->program->module, loweredName.c_str());
        if (err != hipSuccess) {
            throw std::runtime_error("hipModuleGetFunction failed for '" + nameExpr + "' (lowered: " + loweredName +
                                     "): " + hipGetErrorString(err));
        }
    }
    instantiated_ = true;
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
// Program implementation - wraps ProgramImpl
// ============================================================================

struct Program::Impl : public ProgramImpl {};

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
    k.impl_->program = impl_.get();
    return k;
}

// ============================================================================
// ProgramCache implementation
// ============================================================================

struct ProgramCache::Impl {
    // No caching at this level; callers cache via shared_ptr<Program>
};

ProgramCache::ProgramCache() : impl_(std::make_unique<Impl>()) {}
ProgramCache::~ProgramCache() = default;

Program ProgramCache::program(const std::string& name,
                              const std::string& source,
                              const std::vector<std::pair<std::string, std::string>>& headers,
                              const std::vector<std::string>& flags) {
    // Create a new Program with deferred compilation (lazy compilation on first kernel use)
    Program result;
    result.impl_->name = name;
    result.impl_->source = source;
    result.impl_->headers = headers;
    result.impl_->flags = flags;
    result.impl_->needsRecompile = true;
    return result;
}

void ProgramCache::clear() {
    // No-op: no caching at this level
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
    flags_.push_back("--gpu-architecture=" + arch);
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
// KernelLauncher::launchRaw() - non-templated launch for HIP
// ============================================================================

void KernelLauncher::launchRaw(void** kernel_args, size_t num_args) {
    if (!kernel_->impl_->func) {
        throw std::runtime_error("Kernel function '" + kernel_->impl_->name + "' not resolved");
    }

    // Skip launch if grid has zero dimension (no work to do).
    // CUDA may silently accept grid=0, but HIP returns hipErrorInvalidValue.
    if (grid_.x == 0 || grid_.y == 0 || grid_.z == 0) {
        return;
    }

    hipError_t err = hipModuleLaunchKernel(kernel_->impl_->func, grid_.x, grid_.y, grid_.z, block_.x, block_.y,
                                           block_.z, shared_mem_, stream_, kernel_args, nullptr);

    checkHip(err, ("hipModuleLaunchKernel failed for '" + kernel_->impl_->name + "'").c_str());
    (void)num_args;
}

}  // namespace jit
}  // namespace deme

#endif  // USE_HIP
