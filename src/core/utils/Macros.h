//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <exception>

extern size_t m_approx_bytes_used;

#define SGPS_ERROR(...)                                          \
    {                                                            \
        char error_message[256];                                 \
        sprintf(error_message, __VA_ARGS__);                     \
        printf("ERROR!");                                        \
        printf("\n%s", error_message);                           \
        printf("\n%s", __func__);                                \
        throw std::runtime_error("\nEXITING SGPS SIMULATION\n"); \
    }

#define TRACKED_QUICK_VECTOR_RESIZE(vec, newsize)              \
    {                                                          \
        size_t item_size = sizeof(decltype(vec)::value_type);  \
        size_t old_size = vec.size();                          \
        vec.resize(newsize);                                   \
        size_t new_size = vec.size();                          \
        size_t byte_delta = item_size * (new_size - old_size); \
        m_approx_bytes_used += byte_delta;                     \
    }

#define TRACKED_VECTOR_RESIZE(vec, newsize, name, val)         \
    {                                                          \
        size_t item_size = sizeof(decltype(vec)::value_type);  \
        size_t old_size = vec.size();                          \
        vec.resize(newsize, val);                              \
        size_t new_size = vec.size();                          \
        size_t byte_delta = item_size * (new_size - old_size); \
        m_approx_bytes_used += byte_delta;                     \
    }
// INFO_PRINTF("Resizing vector %s, old size %zu, new size %zu, byte delta %s\n", name, old_size, new_size,
// pretty_format_bytes(byte_delta).c_str());

inline std::string pretty_format_bytes(size_t bytes) {
    // set up byte prefixes
    constexpr size_t KIBI = 1024;
    constexpr size_t MEBI = KIBI * KIBI;
    constexpr size_t GIBI = KIBI * KIBI * KIBI;
    float gibival = float(bytes) / GIBI;
    float mebival = float(bytes) / MEBI;
    float kibival = float(bytes) / KIBI;
    std::stringstream ret;
    if (gibival > 1) {
        ret << gibival << " GiB";
    } else if (mebival > 1) {
        ret << mebival << " MiB";
    } else if (kibival > 1) {
        ret << kibival << " KiB";
    } else {
        ret << bytes << " B";
    }
    return ret.str();
}