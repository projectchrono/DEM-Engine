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
            std::stringstream out_msg;
            out_msg << "\n\n-------- Simulation crashed potentially due to too many geometries in a bin --------\n";
            out_msg << "Right now, the dT reported (by user-specification or by calculation) max velocity is ";
            out_msg << max_vel << "\n";
            out_msg << "The contact margin thickness is ";
            out_msg << beta << "\n";
            out_msg << "If the velocity is extremely large, then the simulation probably diverged due to encountering "
                       "large particle velocities, and decreasing the step size could help.\n";
            out_msg << "If the velocity is fair but the margin is large compared to particle sizes, then perhaps too "
                       "many contact geometries are in one bin, and decreasing the step size, update frequency or the "
                       "bin size could help.\n";
            out_msg
                << "If they are both fair and you do not see \"exceeding maximum allowance\" reports before the crash, "
                   "then it is probably not too many geometries in a bin and it crashed for other reasons.\n\n";
            std::cerr << out_msg.str();
            std::stringstream out;
            out << "GPU Assertion: " << cudaGetErrorString(code) << ". This happened in " << filename << ":" << line
                << "\n";
            throw std::runtime_error(out.str());
        }
        return false;
    }
    return true;
}

#endif