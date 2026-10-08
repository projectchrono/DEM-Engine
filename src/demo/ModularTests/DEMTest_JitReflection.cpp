// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "jitify/jitify.hpp"

namespace {
// Include different argument shapes so workers exercise the common demangler across multiple reflected types.
std::array<std::string, 6> reflectTypes() {
    return {jitify::reflection::reflect<int>(),      jitify::reflection::reflect<unsigned int>(),
            jitify::reflection::reflect<float>(),    jitify::reflection::reflect<const float*>(),
            jitify::reflection::reflect<double**>(), jitify::reflection::reflect<const int(&)[3]>()};
}
}  // namespace

// No CUDA context is needed: under MSVC this stresses DbgHelp directly, as concurrent dT/kT launches do.
int main() try {
    const auto expected = reflectTypes();
    for (const auto& name : expected) {
        if (name.empty())
            throw std::runtime_error("Empty reflected type name");
    }
    std::atomic<unsigned int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> failed{false};
    constexpr unsigned int worker_count = 8;
    std::vector<std::thread> workers;
    for (unsigned int i = 0; i < worker_count; ++i) {
        workers.emplace_back([&]() {
            ++ready;
            while (!start.load())
                std::this_thread::yield();
            try {
                for (unsigned int iteration = 0; iteration < 2000; ++iteration) {
                    if (reflectTypes() != expected)
                        failed = true;
                }
            } catch (...) {
                failed = true;
            }
        });
    }
    while (ready.load() != worker_count)
        std::this_thread::yield();
    start = true;
    for (auto& worker : workers)
        worker.join();
    if (failed.load())
        throw std::runtime_error("Concurrent type reflection changed names or threw an exception");
#ifdef _MSC_VER
    std::cout << "PASS: concurrent MSVC/DbgHelp type reflection.\n";
#else
    std::cout << "PASS: concurrent native type reflection (DbgHelp coverage requires MSVC).\n";
#endif
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
