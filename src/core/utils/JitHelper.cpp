//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <filesystem>
#include <string>
#include <regex>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <cstring>

#include "cuda_to_hip.h"

// Compile-time default architecture fallback.
// Can be overridden at build time via -DDEME_DEFAULT_CUDA_ARCH_STR="compute_XY" (CUDA)
// or -DDEME_DEFAULT_HIP_ARCH_STR="gfx90a" (HIP).
// At runtime, the environment variable DEME_DEFAULT_CUDA_ARCH (CUDA) or
// DEME_DEFAULT_HIP_ARCH (HIP) takes precedence.
#if defined(USE_HIP)
    #ifndef DEME_DEFAULT_HIP_ARCH_STR
        #define DEME_DEFAULT_HIP_ARCH_STR "gfx90a"
    #endif
#else
    #ifndef DEME_DEFAULT_CUDA_ARCH_STR
        #define DEME_DEFAULT_CUDA_ARCH_STR "compute_75"
    #endif
#endif

#include <core/ApiVersion.h>
#include "RuntimeData.h"
#include "JitHelper.h"

deme::jit::ProgramCache* JitHelper::kcache = nullptr;

std::filesystem::path JitHelper::KERNEL_DIR = DEMERuntimeDataHelper::data_path / "kernel";
std::filesystem::path JitHelper::KERNEL_INCLUDE_DIR = DEMERuntimeDataHelper::include_path;

JitHelper::Header::Header(const std::filesystem::path& sourcefile) {
    this->_source = JitHelper::loadSourceFile(sourcefile);
}

const std::string& JitHelper::Header::getSource() {
    return _source;
}

void JitHelper::Header::substitute(const std::string& symbol, const std::string& value) {
    // find occurrences of `symbol` until there are none left
    for (size_t p = this->_source.find(symbol); p != std::string::npos; p = this->_source.find(symbol)) {
        // Replace this occurrence with the new value
        this->_source.replace(p, symbol.length(), value);
    }
}

deme::jit::Program JitHelper::buildProgram(const std::string& name,
                                           const std::filesystem::path& source,
                                           std::unordered_map<std::string, std::string> substitutions,
                                           std::vector<std::string> flags) {
    // Double ensure include paths for runtime headers + CUDA/CCCL (cuda::std) or ROCm
    auto add_inc = [&](const std::filesystem::path& p) {
        if (p.empty())
            return;
        std::error_code ec;
        if (!std::filesystem::exists(p, ec))
            return;
        const std::string inc_flag = "-I" + p.string();
        if (std::find(flags.begin(), flags.end(), inc_flag) == flags.end())
            flags.push_back(inc_flag);
    };

    // Project/runtime includes
    add_inc(KERNEL_INCLUDE_DIR);

    // Also add the source tree for header includes (cuda_to_hip.h, etc.)
    // KERNEL_INCLUDE_DIR is the build tree, but some headers live in src/
    // The source tree is one level up from build, then into src/
    std::error_code ec;
    std::filesystem::path src_include = std::filesystem::canonical(KERNEL_INCLUDE_DIR / "..", ec);
    if (!ec) {
        add_inc(src_include / "src");
    }

    // Common fallbacks
#if defined(USE_HIP)
    // Helper: find clang builtin headers by scanning <base>/<version>/include.
    // Linux ROCm keeps them under lib/llvm/lib/clang; the Windows HIP SDK under lib/clang.
    auto add_clang_builtins = [&](const std::filesystem::path& rocm_root) {
        for (const auto& clang_base : {rocm_root / "lib" / "llvm" / "lib" / "clang", rocm_root / "lib" / "clang"}) {
            std::error_code scan_ec;
            for (const auto& entry : std::filesystem::directory_iterator(clang_base, scan_ec)) {
                if (entry.is_directory()) {
                    add_inc(entry.path() / "include");
                    break;  // use the first (and typically only) version dir
                }
            }
        }
    };

    // ROCm include paths for hipRTC. ROCM_PATH is the Linux convention; the Windows HIP SDK
    // sets HIP_PATH instead, so accept either (hiprtc on Windows has no implicit include path,
    // which makes this lookup load-bearing there, not just a fallback).
    const char* rocm_path = std::getenv("ROCM_PATH");
    if (rocm_path == nullptr || rocm_path[0] == '\0')
        rocm_path = std::getenv("HIP_PATH");
    if (rocm_path != nullptr && rocm_path[0] != '\0') {
        add_inc(std::filesystem::path(rocm_path) / "include");
        add_inc(std::filesystem::path(rocm_path) / "include" / "hipcub");
        add_inc(std::filesystem::path(rocm_path) / "include" / "rocprim");
        // Clang builtin headers (stddef.h, etc.) for hiprtc -- scan for actual version dir
        add_clang_builtins(std::filesystem::path(rocm_path));
    }
    add_inc("/opt/rocm/include");
    add_inc("/opt/rocm/include/hipcub");
    add_inc("/opt/rocm/include/rocprim");
    // Clang builtin headers for hiprtc (needed for stddef.h, stdint.h, etc.)
    add_clang_builtins(std::filesystem::path("/opt/rocm"));
#else
    if (const char* cuda_home = std::getenv("CUDA_HOME")) {
        add_inc(std::filesystem::path(cuda_home) / "include");
        add_inc(std::filesystem::path(cuda_home) / "include" / "cccl");
    }
    add_inc("/usr/local/cuda/include");
    add_inc("/usr/local/cuda/include/cccl");
#endif

#if defined(USE_HIP)
    // For hiprtc: start with hip_runtime.h to provide HIP types/functions in device code.
    // The program name is passed separately to hiprtcCreateProgram.
    std::string code = "#include <hip/hip_runtime.h>\n";
#else
    // For CUDA/jitify: the program name on the first line is a jitify requirement.
    std::string code = name + "\n";
#endif

    code.append(JitHelper::loadSourceFile(source));
    // Apply the substitutions
    for (auto& subst : substitutions) {
        code = std::regex_replace(code, std::regex(subst.first), subst.second);
    }

    if (kcache == nullptr)
        kcache = new deme::jit::ProgramCache();

    // Detect GPU architecture and add appropriate flags
    std::string arch_flag;
    {
        int dev = 0;
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
#if defined(USE_HIP)
            // HIP: use gcnArchName (e.g., "gfx90a")
            std::string detected_arch = prop.gcnArchName;
            // Strip any suffix after the base arch name (e.g., "gfx90a:sramecc+:xnack-" -> "gfx90a")
            size_t colon_pos = detected_arch.find(':');
            if (colon_pos != std::string::npos) {
                detected_arch = detected_arch.substr(0, colon_pos);
            }
            arch_flag = "--gpu-architecture=" + detected_arch;
#else
            if (prop.major > 0 && prop.minor >= 0) {
                std::string detected_arch = "compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
                arch_flag = "-arch=" + detected_arch;
            }
#endif
        }
    }

    // If device detection failed, use fallback
    if (arch_flag.empty()) {
#if defined(USE_HIP)
        const char* env_arch = std::getenv("DEME_DEFAULT_HIP_ARCH");
        const std::string fallback_arch =
            (env_arch != nullptr && env_arch[0] != '\0') ? std::string(env_arch) : DEME_DEFAULT_HIP_ARCH_STR;
        arch_flag = "--gpu-architecture=" + fallback_arch;
#else
        const char* env_arch = std::getenv("DEME_DEFAULT_CUDA_ARCH");
        const std::string fallback_arch =
            (env_arch != nullptr && env_arch[0] != '\0') ? std::string(env_arch) : DEME_DEFAULT_CUDA_ARCH_STR;
        arch_flag = "-arch=" + fallback_arch;
#endif
    }

    std::vector<std::string> final_flags = flags;
    final_flags.push_back(arch_flag);

    // Use the unified JitKernel abstraction - no headers needed for our usage
    std::vector<std::pair<std::string, std::string>> headers;
    return kcache->program(name, code, headers, final_flags);
}
