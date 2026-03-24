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

#include <cuda_runtime_api.h>

// Compile-time default CUDA architecture fallback.
// Can be overridden at build time via -DDEME_DEFAULT_CUDA_ARCH_STR="compute_XY".
// At runtime, the environment variable DEME_DEFAULT_CUDA_ARCH takes precedence.
#ifndef DEME_DEFAULT_CUDA_ARCH_STR
    #define DEME_DEFAULT_CUDA_ARCH_STR "compute_75"
#endif

#include <core/ApiVersion.h>
#include "RuntimeData.h"
#include "JitHelper.h"

jitify::JitCache* JitHelper::kcache = nullptr;

const std::filesystem::path JitHelper::KERNEL_DIR = DEMERuntimeDataHelper::data_path / "kernel";
const std::filesystem::path JitHelper::KERNEL_INCLUDE_DIR = DEMERuntimeDataHelper::include_path;

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

jitify::Program JitHelper::buildProgram(
    const std::string& name,
    const std::filesystem::path& source,
    std::unordered_map<std::string, std::string> substitutions,
    // std::vector<JitHelper::Header> headers, // THIS PARAMETER PROBABLY WON'T EVER BE USED
    std::vector<std::string> flags) {
    // Double ensure include paths for runtime headers + CUDA/CCCL (cuda::std)
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

    // Common fallbacks
    if (const char* cuda_home = std::getenv("CUDA_HOME")) {
        add_inc(std::filesystem::path(cuda_home) / "include");
        add_inc(std::filesystem::path(cuda_home) / "include" / "cccl");
    }
    add_inc("/usr/local/cuda/include");
    add_inc("/usr/local/cuda/include/cccl");

    std::string code = name + "\n";

    code.append(JitHelper::loadSourceFile(source));
    // Apply the substitutions
    for (auto& subst : substitutions) {
        code = std::regex_replace(code, std::regex(subst.first), subst.second);
    }

    std::vector<std::string> header_code;
    // THIS BLOCK IS ONLY NEEDED IF THE headers PARAMETER IS USED
    /*
    for (auto it = headers.begin(); it != headers.end(); it++) {
        header_code.push_back(it->getSource());
    }
    */

    if (kcache == nullptr)
        kcache = new jitify::JitCache();

    // Stage 1: Explicit architecture detection.
    // Query the active CUDA device compute capability and build an -arch flag.
    {
        int dev = 0;
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
            prop.major > 0 && prop.minor >= 0) {
            std::string detected_arch = "compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
            std::vector<std::string> arch_flags = flags;
            arch_flags.push_back("-arch=" + detected_arch);
            return kcache->program(code, header_code, arch_flags);
        }
    }

    // Stage 2: Jitify auto-detect.
    // Device detection failed or returned an invalid compute capability; compile
    // without an explicit -arch flag and let Jitify detect the architecture.
    try {
        return kcache->program(code, header_code, flags);
    } catch (const std::exception& e) {
        const std::string err_msg = e.what();
        // Only fall through to Stage 3 when the failure is architecture-related.
        const bool is_arch_error =
            (err_msg.find("arch") != std::string::npos || err_msg.find("compute_") != std::string::npos ||
             err_msg.find("sm_") != std::string::npos);
        if (!is_arch_error) {
            throw;
        }

        // Stage 3: Hardcoded default fallback.
        // Use the DEME_DEFAULT_CUDA_ARCH environment variable if set; otherwise fall
        // back to the compile-time constant DEME_DEFAULT_CUDA_ARCH_STR.
        const char* env_arch = std::getenv("DEME_DEFAULT_CUDA_ARCH");
        const std::string fallback_arch =
            (env_arch != nullptr && env_arch[0] != '\0') ? std::string(env_arch) : DEME_DEFAULT_CUDA_ARCH_STR;
        std::vector<std::string> fallback_flags = flags;
        fallback_flags.push_back("-arch=" + fallback_arch);
        try {
            return kcache->program(code, header_code, fallback_flags);
        } catch (const std::exception& e3) {
            std::string ctx = "Jitify compilation failed with fallback arch '";
            ctx += fallback_arch;
            ctx += (env_arch != nullptr && env_arch[0] != '\0') ? "' (from DEME_DEFAULT_CUDA_ARCH env var): "
                                                                : "' (compile-time default): ";
            ctx += e3.what();
            throw std::runtime_error(ctx);
        }
    }
}
