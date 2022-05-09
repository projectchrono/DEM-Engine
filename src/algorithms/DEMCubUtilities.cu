//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cub/cub.cuh>
#include <core/utils/JitHelper.h>
#include <algorithms/DEMCubHelperFunctions.h>

#include <core/utils/GpuError.h>

namespace sgps {

void cubDEMSum(float* d_in, float* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateDataDT& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceReduce::Sum(NULL, cub_scratch_bytes, d_in, d_out, n, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceReduce::Sum(d_scratch_space, cub_scratch_bytes, d_in, d_out, n, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
}

}  // namespace sgps
