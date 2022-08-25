//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef SGPS_GPU_ERROR_H
#define SGPS_GPU_ERROR_H

#include <iostream>
#include <exception>
#include <cuda_runtime_api.h>

#define GPU_CALL(res) \
    { gpu_assert((res), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char* filename, int line, bool except = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPU Assertion: " << cudaGetErrorString(code) << " in " << filename << ":" << line << "\n";
        if (except) {
            throw std::runtime_error("GPU Assertion Failed!");
        }
    }
}

#endif