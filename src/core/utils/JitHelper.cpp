//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//  SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <regex>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <utility>
#include <mutex>
#include <memory>
#include <vector>
#include <stdexcept>

#include <cuda_runtime_api.h>
#include <nvrtc.h>

// Compile-time default CUDA architecture fallback.
// Can be overridden at build time via -DDEME_DEFAULT_CUDA_ARCH_STR="compute_XY".
// At runtime, the environment variable DEME_DEFAULT_CUDA_ARCH takes precedence.
#ifndef DEME_DEFAULT_CUDA_ARCH_STR
    #define DEME_DEFAULT_CUDA_ARCH_STR "compute_75"
#endif

#include <core/ApiVersion.h>
#include "RuntimeData.h"
#include "JitHelper.h"
#include "Logger.hpp"

#ifdef DEME_PYTHON_JIT
    #include "DEM/python/PythonCudaIncludes.hpp"
#endif

namespace {

constexpr uint64_t kFNVOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFNVPrime = 0x100000001b3ULL;

constexpr int getCudaVersion() {
#ifdef CUDA_VERSION
    return CUDA_VERSION;
#else
    return CUDART_VERSION;
#endif
}

std::pair<int, int> getNvrtcVersion() {
    int major = 0;
    int minor = 0;
    if (nvrtcVersion(&major, &minor) != NVRTC_SUCCESS) {
        return {0, 0};
    }
    return {major, minor};
}

bool cudaHeadersMatch(const std::filesystem::path& include_dir, int nvrtc_major, int nvrtc_minor) {
    std::ifstream cuda_header(include_dir / "cuda.h");
    if (!cuda_header) {
        return false;
    }

    std::string line;
    const std::regex version_pattern(R"(^\s*#\s*define\s+CUDA_VERSION\s+([0-9]+))");
    std::smatch match;
    while (std::getline(cuda_header, line)) {
        if (std::regex_search(line, match, version_pattern)) {
            const int encoded_version = std::stoi(match[1].str());
            const int header_major = encoded_version / 1000;
            const int header_minor = (encoded_version % 1000) / 10;
            return header_major == nvrtc_major && header_minor == nvrtc_minor;
        }
    }
    return false;
}

std::string sanitizeFilename(const std::string& name) {
    std::string sanitized = name;
    for (auto& c : sanitized) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) {
            c = '_';
        }
    }
    return sanitized;
}

bool isArchitectureError(const std::exception& error) {
    const std::string message = error.what();
    return message.find("arch") != std::string::npos || message.find("compute_") != std::string::npos ||
           message.find("sm_") != std::string::npos;
}

}  // namespace

std::filesystem::path JitHelper::KERNEL_DIR = DEMERuntimeDataHelper::data_path / "kernel";
std::filesystem::path JitHelper::KERNEL_INCLUDE_DIR = DEMERuntimeDataHelper::include_path;
std::filesystem::path JitHelper::CACHE_DIR;

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

JitHelper::CachedProgram JitHelper::buildProgram(const std::string& name,
                                                 const std::filesystem::path& source,
                                                 std::unordered_map<std::string, std::string> substitutions,
                                                 std::vector<std::string> flags) {
    std::string code = name + "\n";

    code.append(JitHelper::loadSourceFile(source));
    // Apply the substitutions deterministically (unordered_map iteration is non-deterministic)
    std::vector<std::pair<std::string, std::string>> ordered_subs(substitutions.begin(), substitutions.end());
    std::sort(ordered_subs.begin(), ordered_subs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (auto& subst : ordered_subs) {
        code = std::regex_replace(code, std::regex(subst.first), subst.second);
    }
    {
        // Select CUDA headers matching the NVRTC library loaded on the deployment host. A wheel's build-time CUDA
        // path can exist on the host while pointing at another toolkit version, which makes CUDA's internal headers
        // fail with misleading undefined-intrinsic errors during JIT compilation.
        const auto [nvrtc_major, nvrtc_minor] = getNvrtcVersion();
        std::vector<std::filesystem::path> include_paths;
        {
            std::string dirs = DEME_CUDA_TOOLKIT_INCLUDE_DIRS;  // "dir1;dir2;dir3"
            std::stringstream ss(dirs);
            std::string dir;
            while (std::getline(ss, dir, ';')) {
                if (!dir.empty()) {
                    include_paths.emplace_back(dir);
                }
            }
        }
        auto add_inc = [&](const std::filesystem::path& p) {
            if (p.empty()) {
                return;
            }
            std::error_code ec;
            if (!std::filesystem::exists(p, ec)) {
                return;
            }
            std::string inc_flag = "-I" + p.string();
            if (std::find(flags.begin(), flags.end(), inc_flag) == flags.end()) {
                flags.push_back(inc_flag);
            }
        };
        bool found_matching_cuda_headers = false;
        auto add_cuda_inc = [&](const std::filesystem::path& p) {
            if (nvrtc_major == 0 || !cudaHeadersMatch(p, nvrtc_major, nvrtc_minor)) {
                return;
            }
            add_inc(p);
            add_inc(p / "cccl");
            found_matching_cuda_headers = true;
        };
#ifdef DEME_PYTHON_JIT
        // Import passes the split NVIDIA wheel directories directly into this extension's core.
        // Native C++ does not compile this branch or inherit configuration from a Python parent.
        const auto& python_include_paths = deme::python::CudaIncludePaths();
        if (!python_include_paths.empty()) {
            for (const auto& path : python_include_paths) {
                add_cuda_inc(path);
            }
            if (!found_matching_cuda_headers) {
                DEME_ERROR("The pip CUDA headers do not match loaded NVRTC %d.%d. Use a consistent CUDA environment.",
                           nvrtc_major, nvrtc_minor);
            }
            for (const auto& path : python_include_paths) {
                add_inc(path);
            }
            add_inc(KERNEL_INCLUDE_DIR);
        } else {
#endif
            for (auto& p : include_paths) {
                add_cuda_inc(p);
            }
            add_inc(KERNEL_INCLUDE_DIR);
            if (const char* cuda_home = std::getenv("CUDA_HOME")) {
                add_cuda_inc(std::filesystem::path(cuda_home) / "include");
            }
            if (nvrtc_major > 0) {
                add_cuda_inc("/usr/local/cuda-" + std::to_string(nvrtc_major) + "." + std::to_string(nvrtc_minor) +
                             "/include");
            }
            add_cuda_inc("/usr/local/cuda/include");
#ifdef DEME_PYTHON_JIT
        }
#endif
        if (!found_matching_cuda_headers) {
            DEME_ERROR(
                "NVRTC %d.%d is loaded, but matching CUDA headers were not found. Install that CUDA Toolkit or set "
                "CUDA_HOME to its root directory.",
                nvrtc_major, nvrtc_minor);
        }
    }

    int device = 0;
    cudaDeviceProp prop{};
    bool architecture_detected = cudaGetDevice(&device) == cudaSuccess &&
                                 cudaGetDeviceProperties(&prop, device) == cudaSuccess && prop.major > 0;
    if (!architecture_detected) {
        device = 0;
        prop.major = 0;
        prop.minor = 0;
    }
    const std::string arch_tag =
        architecture_detected ? "compute_" + std::to_string(prop.major) + std::to_string(prop.minor) : "auto";
    // Prefer the active device's exact compute capability. If CUDA device discovery is unavailable, leave the flag
    // unset so Jitify can perform its own architecture detection during compilation.
    if (architecture_detected) {
        flags.push_back("-arch=" + arch_tag);
    }

    std::vector<std::string> flags_sorted = flags;
    std::sort(flags_sorted.begin(), flags_sorted.end());
    const std::string flags_sig = jitify::reflection::reflect_list(flags_sorted);
    const auto [nvrtc_major, nvrtc_minor] = getNvrtcVersion();
    const std::string nvrtc_tag = std::to_string(nvrtc_major) + "." + std::to_string(nvrtc_minor);
    const std::string fingerprint = code + "|flags:" + flags_sig + "|api:" + std::to_string(DEME_API_VERSION) +
                                    "|cuda:" + std::to_string(getCudaVersion()) + "|nvrtc:" + nvrtc_tag +
                                    "|arch:" + arch_tag;
    std::string program_hash = hashString(fingerprint);

    static std::once_flag cache_dir_init_flag;
    std::call_once(cache_dir_init_flag, []() { CACHE_DIR = resolveCacheDir(); });
    const auto program_dir = CACHE_DIR / program_hash;
    auto storage = std::make_shared<CachedProgram::ProgramStorage>(code, flags);
    storage->programHash = program_hash;
    storage->cacheDir = program_dir;
    storage->device = device;
    storage->archTag = arch_tag;

    std::error_code ec;
    std::filesystem::create_directories(storage->cacheDir, ec);
    std::ofstream fp_out(storage->cacheDir / "fingerprint.txt", std::ios::trunc);
    if (fp_out) {
        fp_out << program_hash << "\n" << fingerprint;
    }

    return CachedProgram(storage);
}

std::string JitHelper::hashString(const std::string& in) {
    uint64_t hash = kFNVOffset;
    for (unsigned char c : in) {
        hash ^= c;
        hash *= kFNVPrime;
    }
    return toHex(hash);
}

std::string JitHelper::toHex(uint64_t value) {
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << value;
    return oss.str();
}

std::filesystem::path JitHelper::resolveCacheDir() {
    if (const char* env = std::getenv("DEME_JIT_CACHE_DIR")) {
        return std::filesystem::path(env);
    }
    // Prefer to keep cache alongside other runtime data in the build tree
    std::filesystem::path default_path = DEMERuntimeDataHelper::data_path / "jit_cache";
    if (std::filesystem::exists(DEMERuntimeDataHelper::data_path)) {
        return default_path;
    }
    return std::filesystem::temp_directory_path() / "dem-jit";
}
JitHelper::CachedProgram::CachedProgram(std::shared_ptr<ProgramStorage> storage) : m_storage(std::move(storage)) {}
JitHelper::CachedProgram::Kernel JitHelper::CachedProgram::kernel(const std::string& name,
                                                                  std::vector<std::string> options) const {
    return Kernel(m_storage, name, std::move(options));
}
JitHelper::CachedProgram::ProgramStorage::ProgramStorage(std::string code_in, std::vector<std::string> flags_in)
    : code(std::move(code_in)), flags(std::move(flags_in)) {}
JitHelper::CachedProgram::Kernel::Kernel(std::shared_ptr<ProgramStorage> storage,
                                         std::string name,
                                         std::vector<std::string> options)
    : m_storage(std::move(storage)), m_name(std::move(name)), m_options(std::move(options)) {}

std::shared_ptr<jitify::experimental::KernelInstantiation> JitHelper::CachedProgram::Kernel::getKernelInstantiation(
    const std::vector<std::string>& template_args) const {
    const std::string template_suffix =
        template_args.empty() ? std::string() : jitify::reflection::reflect_template(template_args);

    std::vector<std::string> options_sorted = m_options;
    std::sort(options_sorted.begin(), options_sorted.end());
    const std::string options_sig = jitify::reflection::reflect_list(options_sorted);

    auto make_key = [&]() {
        const auto [nvrtc_major, nvrtc_minor] = getNvrtcVersion();
        const std::string key_material = m_storage->programHash + "|" + m_name + "|" + template_suffix + "|" +
                                         options_sig + "|cuda:" + std::to_string(getCudaVersion()) +
                                         "|nvrtc:" + std::to_string(nvrtc_major) + "." + std::to_string(nvrtc_minor) +
                                         "|api:" + std::to_string(DEME_API_VERSION) + "|" + m_storage->archTag;
        return JitHelper::hashString(key_material);
    };
    std::string key = make_key();
    std::filesystem::path cache_file = m_storage->cacheDir / (sanitizeFilename(m_name) + "_" + key + ".jit");
    std::lock_guard<std::mutex> storage_lock(m_storage->mutex);
    if (auto it = m_storage->kernelCache.find(key); it != m_storage->kernelCache.end()) {
        return it->second;
    }

    std::shared_ptr<jitify::experimental::KernelInstantiation> inst;
    if (std::filesystem::exists(cache_file)) {
        std::ifstream input(cache_file, std::ios::binary);
        if (input) {
            std::stringstream buffer;
            buffer << input.rdbuf();
            try {
                inst = std::make_shared<jitify::experimental::KernelInstantiation>(
                    jitify::experimental::KernelInstantiation::deserialize(buffer.str()));
            } catch (const std::exception&) {
                inst.reset();
            }
        }
    }
    if (!inst) {  // Compile if changed and not already there and make user aware
        DEME_INFO("jit-compiling for %s ...", m_name.c_str());
        if (!m_storage->program) {
            try {
                // With an unresolved architecture, first allow Jitify to auto-detect it.
                m_storage->program = std::make_unique<jitify::experimental::Program>(
                    m_storage->code, std::vector<std::string>(), m_storage->flags);
            } catch (const std::exception& error) {
                if (m_storage->archTag != "auto" || !isArchitectureError(error)) {
                    throw;
                }

                const char* configured_arch = std::getenv("DEME_DEFAULT_CUDA_ARCH");
                const std::string fallback_arch = configured_arch && configured_arch[0] != '\0'
                                                      ? std::string(configured_arch)
                                                      : DEME_DEFAULT_CUDA_ARCH_STR;
                m_storage->flags.push_back("-arch=" + fallback_arch);
                m_storage->archTag = fallback_arch;
                // The resolved fallback architecture must participate in both the program directory and kernel key;
                // otherwise binaries for different GPU targets could collide on disk.
                m_storage->programHash = JitHelper::hashString(m_storage->programHash + "|arch:" + fallback_arch);
                m_storage->cacheDir = JitHelper::CACHE_DIR / m_storage->programHash;
                key = make_key();
                cache_file = m_storage->cacheDir / (sanitizeFilename(m_name) + "_" + key + ".jit");
                std::error_code cache_error;
                std::filesystem::create_directories(m_storage->cacheDir, cache_error);
                std::ofstream fingerprint_out(m_storage->cacheDir / "fingerprint.txt", std::ios::trunc);
                if (fingerprint_out) {
                    fingerprint_out << m_storage->programHash << "\narch:" << fallback_arch;
                }
                try {
                    m_storage->program = std::make_unique<jitify::experimental::Program>(
                        m_storage->code, std::vector<std::string>(), m_storage->flags);
                } catch (const std::exception& fallback_error) {
                    throw std::runtime_error("Jitify compilation failed with fallback architecture '" + fallback_arch +
                                             "': " + fallback_error.what());
                }
            }
        }
        auto kernel = m_storage->program->kernel(m_name, m_options);
        inst = std::make_shared<jitify::experimental::KernelInstantiation>(kernel, template_args);
        std::error_code ec;
        std::filesystem::create_directories(cache_file.parent_path(), ec);
        std::ofstream output(cache_file, std::ios::binary | std::ios::trunc);
        if (output) {
            output << inst->serialize();
        }
    }
    m_storage->kernelCache[key] = inst;
    return inst;
}

JitHelper::CachedProgram::Kernel::KernelInstantiation JitHelper::CachedProgram::Kernel::instantiate(
    std::vector<std::string> template_args) const {
    return KernelInstantiation(getKernelInstantiation(template_args));
}

JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelInstantiation(
    std::shared_ptr<jitify::experimental::KernelInstantiation> impl)
    : m_impl(std::move(impl)) {}
