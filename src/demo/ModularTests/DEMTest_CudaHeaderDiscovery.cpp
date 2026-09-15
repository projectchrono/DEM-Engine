// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "core/utils/JitHelper.h"

// A native process must ignore the old Python environment override. Building a
// program selects and validates headers without compiling the full solver kernels.
int main() {
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("deme-native-headers-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    const auto source = directory / "probe.cu";
    std::ofstream(source) << "extern \"C\" __global__ void probe() {}\n";
#ifdef _WIN32
    _putenv_s("DEME_CUDA_INCLUDE_PATH", "Z:/missing-python-cuda-headers");
    _putenv_s("DEME_JIT_CACHE_DIR", directory.string().c_str());
#else
    setenv("DEME_CUDA_INCLUDE_PATH", "/missing-python-cuda-headers", 1);
    setenv("DEME_JIT_CACHE_DIR", directory.string().c_str(), 1);
#endif
    try {
        const auto program = JitHelper::buildProgram("native_header_probe", source);
        if (!std::filesystem::exists(directory / program.key() / "fingerprint.txt")) {
            throw std::runtime_error("Native header selection did not create a program fingerprint");
        }
        std::filesystem::remove_all(directory);
        std::cout << "PASS: native CUDA discovery ignores Python configuration\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(directory);
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
