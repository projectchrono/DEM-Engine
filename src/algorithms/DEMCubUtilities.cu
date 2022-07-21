//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cub/cub.cuh>
#include <core/utils/JitHelper.h>
#include <algorithms/DEMCubBasedSubroutines.h>

#include <core/utils/GpuError.h>
#include <algorithms/DEMCubWrappers.cu>

namespace sgps {

void sumReduce(float* d_in, float* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateData& scratchPad) {
    cubDEMSum<float, DEMSolverStateData>(d_in, d_out, n, this_stream, scratchPad);
}

void flagMaxReduce(notStupidBool_t* d_in,
                   notStupidBool_t* d_out,
                   size_t n,
                   cudaStream_t& this_stream,
                   DEMSolverStateData& scratchPad) {
    cubDEMMax<notStupidBool_t, DEMSolverStateData>(d_in, d_out, n, this_stream, scratchPad);
}

}  // namespace sgps
