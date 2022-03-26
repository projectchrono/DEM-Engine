//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cub/cub.cuh>
#include <core/utils/JitHelper.h>

#include <algorithms/DEMCubHelperFunctions.h>

#include <core/utils/GpuError.h>

namespace sgps {

void cubPrefixScan(sgps::binsSphereTouches_t* d_in,
                   sgps::binsSphereTouchesScan_t* d_out,
                   size_t n,
                   GpuManager::StreamInfo& streamInfo,
                   sgps::DEMSolverStateData& scratchPad) {
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, d_in, d_out, n);
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_scratch_space, temp_storage_bytes, d_in, d_out, n);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
}

void cubSortByKeys(sgps::binID_t* d_keys,
                   sgps::bodyID_t* d_vals,
                   size_t n,
                   GpuManager::StreamInfo& streamInfo,
                   sgps::DEMSolverStateData& scratchPad) {
    /*
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_keys, d_vals, n, 0,
    sizeof(sgps::binID_t)*BITS_PER_BYTE,streamInfo.stream,false); void* d_scratch_space =
    (void*)scratchPad.allocateScratchSpace(temp_storage_bytes); cub::DeviceRadixSort::SortPairs(d_scratch_space,
    temp_storage_bytes, d_keys, d_vals, n, 0, sizeof(sgps::binID_t)*BITS_PER_BYTE,streamInfo.stream,false);
    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
    */
}

}  // namespace sgps
