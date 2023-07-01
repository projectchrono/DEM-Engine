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
    { gpu_assert_watch_beta((res), __FILE__, __LINE__, granData->maxVel, true); }

inline bool gpu_assert(cudaError_t code, const char* filename, int line, bool except = true) {
    if (code != cudaSuccess) {
        if (except) {
            std::stringstream out;
            out << "GPU Assertion: " << cudaGetErrorString(code) << ". This happened in " << filename << ":" << line
                << "\n";
            out << "You can check out the troubleshoot section of DEME to see if it helps, or post this error on "
                   "Chrono's forum https://groups.google.com/g/projectchrono \n";
            throw std::runtime_error(out.str());
        }
        return false;
    }
    return true;
}

inline bool gpu_assert_watch_beta(cudaError_t code, const char* filename, int line, float max_vel, bool except = true) {
    if (code != cudaSuccess) {
        if (except) {
            std::stringstream out_msg;
            out_msg << "\n\n-------- Simulation crashed \"potentially\" due to too many geometries in a bin --------\n";
            out_msg << "Right now, the dT reported (by user-specification or by calculation) max velocity is ";
            out_msg << max_vel << "\n";
            out_msg << "------------------------------------\n";
            out_msg << "If the velocity is extremely large, then the simulation probably diverged due to encountering "
                       "large particle velocities.\nDecreasing the step size could help, and remember to check if your "
                       "simulation objects are initially within the domain you specified.\n";
            out_msg << "------------------------------------\n";
            out_msg
                << "If the velocity is fair, then perhaps for some reason too "
                   "many contact geometries appeared in one bin.\nAre you using a contact model that promotes large "
                   "penetrations, so it leads to this?\nAnd if you are using manual bin size, then decreasing the "
                   "bin size could help.\n\n";
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