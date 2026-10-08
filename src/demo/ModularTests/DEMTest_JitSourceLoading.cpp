// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "core/utils/JitHelper.h"
#include "core/utils/Logger.hpp"

namespace {
// Both header and kernel entry points must reject failed I/O before any CUDA/NVRTC work, with a usable path.
void expectSourceError(const std::filesystem::path& path, bool kernel = false) {
    try {
        if (kernel)
            JitHelper::buildProgram("missing_source_probe", path);
        else
            JitHelper::Header{path};
    } catch (const deme::SolverException& error) {
        if (std::string(error.what()).find(std::filesystem::absolute(path).string()) == std::string::npos)
            throw std::runtime_error("Source diagnostic omitted the absolute path");
        return;
    }
    throw std::runtime_error("Invalid source path was silently accepted: " + path.string());
}
}  // namespace

int main() {
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("deme-source-loading-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directories(directory);
        const auto source = directory / "header.cuh";
        // Exercise multiple read buffers and preserve bytes that are not a character representation of EOF.
        const std::string contents = std::string(20000, 'x') + char(0xff) + "\n// trailing source\n";
        std::ofstream(source) << contents;
        JitHelper::Header header(source);
        if (header.getSource() != contents)
            throw std::runtime_error("Source text was truncated or changed");
        const auto empty = directory / "empty.cuh";
        std::ofstream(empty).close();
        if (!JitHelper::Header(empty).getSource().empty())
            throw std::runtime_error("Empty source fragment was not preserved");

        std::cout << "Checking expected source-loading errors (missing, directory and unreadable paths)." << std::endl;
        expectSourceError(directory / "missing.cuh");
        expectSourceError(directory / "missing.cu", true);
        // Opening a directory either fails immediately or fails during reading, depending on the platform.
        expectSourceError(directory);

        // Permission enforcement varies (notably Windows permissions and privileged POSIX users).
        std::error_code permission_error;
        const auto saved_permissions = std::filesystem::status(source).permissions();
        std::filesystem::permissions(source, std::filesystem::perms::none, permission_error);
        if (!permission_error && !std::ifstream(source).is_open()) {
            expectSourceError(source);
        } else {
            std::cout << "SKIP: unreadable-file check (read permissions are not enforced here).\n";
        }
        std::filesystem::permissions(source, saved_permissions, permission_error);
        std::filesystem::remove_all(directory);
        std::cout << "PASS: JIT source loading preserves content and empty fragments, and reports failed I/O.\n";
        return 0;
    } catch (const std::exception& error) {
        std::error_code cleanup_error;
        std::filesystem::remove_all(directory, cleanup_error);
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
