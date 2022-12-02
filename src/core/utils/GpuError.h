//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_GPU_ERROR_H
#define DEME_GPU_ERROR_H

#include <iostream>
#include <sstream>
#include <exception>
#include <cuda_runtime_api.h>

#define DEME_GPU_CALL(res) \
    { gpu_assert((res), __FILE__, __LINE__); }

#define DEME_GPU_CALL_NOTHROW(res) \
    { gpu_assert((res), __FILE__, __LINE__, false); }

#define DEME_GPU_CALL_WATCH_BETA(res) \
    { gpu_assert_watch_beta((res), __FILE__, __LINE__, granData->maxVel, simParams->beta, true); }

inline bool gpu_assert(cudaError_t code, const char* filename, int line, bool except = true) {
    if (code != cudaSuccess) {
        if (except) {
            std::stringstream out;
            out << "GPU Assertion: " << cudaGetErrorString(code) << ". This happened in " << filename << ":" << line
                << "\n";
            throw std::runtime_error(out.str());
        }
        return false;
    }
    return true;
}

inline bool gpu_assert_watch_beta(cudaError_t code,
                                  const char* filename,
                                  int line,
                                  float max_vel,
                                  float beta,
                                  bool except = true) {
    if (code != cudaSuccess) {
        if (except) {
            std::stringstream out;
            out << "GPU Assertion: " << cudaGetErrorString(code) << ". This happened in " << filename << ":" << line
                << "\n";
            out << "Right now, the dT reported (by user-specification or by calculation) max velocity is ";
            out << max_vel << "\n";
            out << "The contact margin thickness is ";
            out << beta << "\n";
            out << "If the velocity is extremely large, then the simulation probably diverged due to encountering "
                   "large particle velocities, and decreasing the step size could help.\n";
            out << "If the velocity is fair but the margin is large compared to particle sizes, then perhaps too many "
                   "contact geometries are in one bin, and decreasing the step size, update frequency or the bin size "
                   "could help.\n";
            throw std::runtime_error(out.str());
        }
        return false;
    }
    return true;
}

#endif