// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "jitify/jitify.hpp"

namespace {
// Read the serialized records directly so successful saves cannot be mistaken for in-memory cache reuse.
void checkCache(const std::filesystem::path& path, uint64_t expected_count) {
    std::ifstream input(path, std::ios::binary);
    std::string magic, signature;
    jitify::detail::read_string_record(input, &magic);
    jitify::detail::read_string_record(input, &signature);
    uint64_t count = 0;
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!input || magic != "JITIFY_HEADER_CACHE_V2" || signature != "cache-test" || count != expected_count)
        throw std::runtime_error("Persistent header cache did not contain the expected saved records");
    for (uint64_t i = 0; i < count; ++i) {
        std::string name, source, fullpath;
        jitify::detail::read_string_record(input, &name);
        jitify::detail::read_string_record(input, &source);
        jitify::detail::read_string_record(input, &fullpath);
        if (!input || source != "// " + name)
            throw std::runtime_error("Persistent header cache contained an incomplete record");
    }
}

// Select a private cache file without touching a user's existing persistent cache.
void setCachePath(const std::filesystem::path& path) {
#ifdef _WIN32
    _putenv_s("DEME_PERSISTENT_JITIFY_CACHE", path.string().c_str());
#else
    setenv("DEME_PERSISTENT_JITIFY_CACHE", path.string().c_str(), 1);
#endif
}
}  // namespace

// Exercise creation, replacement, failed publication and retry without constructing a CUDA context.
int main() {
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("deme-header-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directories(directory);
        const auto cache_path = directory / "headers.bin";
        setCachePath(cache_path);
        auto& sources = jitify::detail::global_header_source_cache();
        sources.clear();
        jitify::detail::global_header_fullpath_cache().clear();
        jitify::detail::persistent_header_cache_saved_count() = 0;
        sources["first.h"] = "// first.h";
        jitify::detail::save_persistent_header_cache_if_needed("cache-test");
        checkCache(cache_path, 1);
        sources["second.h"] = "// second.h";
        jitify::detail::save_persistent_header_cache_if_needed("cache-test");
        checkCache(cache_path, 2);

        // Windows input streams deny delete sharing. Hold the old cache open to force replacement to fail;
        // POSIX permits replacing open files, so use an existing directory as the blocked destination there.
#ifdef _WIN32
        std::ifstream blocked_file(cache_path, std::ios::binary);
        if (!blocked_file)
            throw std::runtime_error("Could not hold the old cache open for the failure test");
#else
        const auto blocked_path = directory / "blocked";
        std::filesystem::create_directory(blocked_path);
        setCachePath(blocked_path);
#endif
        sources["third.h"] = "// third.h";
        jitify::detail::save_persistent_header_cache_if_needed("cache-test");
#ifdef _WIN32
        blocked_file.close();
#else
        if (!std::filesystem::is_directory(blocked_path))
            throw std::runtime_error("Failed publication changed the blocked destination");
#endif
        if (jitify::detail::persistent_header_cache_saved_count() != 2)
            throw std::runtime_error("Failed publication changed the destination or marked the cache saved");
        checkCache(cache_path, 2);
        setCachePath(cache_path);
        jitify::detail::save_persistent_header_cache_if_needed("cache-test");
        checkCache(cache_path, 3);
        if (std::filesystem::exists(cache_path.string() + ".tmp"))
            throw std::runtime_error("Successful publication left its temporary file behind");

        // Loading after clearing process state verifies that a fresh process can recover every expanded record.
        sources.clear();
        jitify::detail::persistent_header_cache_saved_count() = 0;
        jitify::detail::persistent_header_cache_loaded() = false;
        jitify::detail::load_persistent_header_cache_once("cache-test");
        if (sources.size() != 3 || sources.at("third.h") != "// third.h")
            throw std::runtime_error("Expanded header cache did not reload correctly");
        std::filesystem::remove_all(directory);
        std::cout << "PASS: header-cache creation, replacement, failed publication, retry and reload.\n";
        return 0;
    } catch (const std::exception& error) {
        std::error_code cleanup_error;
        std::filesystem::remove_all(directory, cleanup_error);
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
