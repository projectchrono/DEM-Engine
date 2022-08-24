//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  
//	SPDX-License-Identifier: BSD-3-Clause

#include <cub/cub.cuh>
#include <core/utils/JitHelper.h>
#include <algorithms/DEMCubBasedSubroutines.h>

#include <core/utils/GpuError.h>
#include <algorithms/DEMCubWrappers.cu>

namespace sgps {

// These functions interconnecting the cub-part and cpp-part of the code cannot be templated... because of cmake
// restrictions. Not much that I can do, other than writing them all out.

void sumReduce(double* d_in, double* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateData& scratchPad) {
    cubDEMSum<double, DEMSolverStateData>(d_in, d_out, n, this_stream, scratchPad);
}

void boolMaxReduce(notStupidBool_t* d_in,
                   notStupidBool_t* d_out,
                   size_t n,
                   cudaStream_t& this_stream,
                   DEMSolverStateData& scratchPad) {
    cubDEMMax<notStupidBool_t, DEMSolverStateData>(d_in, d_out, n, this_stream, scratchPad);
}

void floatMaxReduce(float* d_in, float* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateData& scratchPad) {
    cubDEMMax<float, DEMSolverStateData>(d_in, d_out, n, this_stream, scratchPad);
}

}  // namespace sgps
