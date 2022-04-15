//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cub/cub.cuh>
#include <core/utils/JitHelper.h>

#include <algorithms/DEMCubHelperFunctions.h>

#include <core/utils/GpuError.h>

namespace sgps {

// template <typename T1, typename T2>
// void cubPrefixScan(T1* d_in,
//                    T2* d_out,
//                    size_t n,
//                    cudaStream_t& this_stream,
//                    DEMSolverStateData& scratchPad) {

void cubPrefixScan_binSphere(binsSphereTouches_t* d_in,
                             binSphereTouchPairs_t* d_out,
                             size_t n,
                             cudaStream_t& this_stream,
                             DEMSolverStateDataKT& scratchPad) {
    // NOTE!!! Why did I not use ExclusiveSum? I found that when for a cub scan operation, if the d_in and d_out are of
    // different types, then cub defaults to use d_in type to store the scan result, but will switch to d_out type if
    // there are too many items to scan. There is however, a region where cub does not choose to switch to d_out type,
    // but d_in type is not big enough to store the scan result. This is very bad. I made a trick: use ExclusiveScan and
    // (d_out type)0 as the initial value, and this forces cub to store results as d_out type.
    size_t cub_scratch_bytes = 0;
    cub::DeviceScan::ExclusiveScan(NULL, cub_scratch_bytes, d_in, d_out, cub::Sum(), (binSphereTouchPairs_t)0, n,
                                   this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceScan::ExclusiveScan(d_scratch_space, cub_scratch_bytes, d_in, d_out, cub::Sum(),
                                   (binSphereTouchPairs_t)0, n, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
}

void cubPrefixScan_contacts(spheresBinTouches_t* d_in,
                            contactPairs_t* d_out,
                            size_t n,
                            cudaStream_t& this_stream,
                            DEMSolverStateDataKT& scratchPad) {
    // NOTE!!! Why did I not use ExclusiveSum? I found that when for a cub scan operation, if the d_in and d_out are of
    // different types, then cub defaults to use d_in type to store the scan result, but will switch to d_out type if
    // there are too many items to scan. There is however, a region where cub does not choose to switch to d_out type,
    // but d_in type is not big enough to store the scan result. This is very bad. I made a trick: use ExclusiveScan and
    // (d_out type)0 as the initial value, and this forces cub to store results as d_out type.
    size_t cub_scratch_bytes = 0;
    cub::DeviceScan::ExclusiveScan(NULL, cub_scratch_bytes, d_in, d_out, cub::Sum(), (contactPairs_t)0, n, this_stream,
                                   false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceScan::ExclusiveScan(d_scratch_space, cub_scratch_bytes, d_in, d_out, cub::Sum(), (contactPairs_t)0, n,
                                   this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
}

void cubSortByKeys(binID_t* d_keys_in,
                   binID_t* d_keys_out,
                   bodyID_t* d_vals_in,
                   bodyID_t* d_vals_out,
                   size_t n,
                   cudaStream_t& this_stream,
                   DEMSolverStateDataKT& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, cub_scratch_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, n, 0,
                                    sizeof(sgps::binID_t) * SGPS_BITS_PER_BYTE, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceRadixSort::SortPairs(d_scratch_space, cub_scratch_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, n,
                                    0, sizeof(sgps::binID_t) * SGPS_BITS_PER_BYTE, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
}

void cubUnique(binID_t* d_in,
               binID_t* d_out,
               size_t* d_num_out,
               size_t n,
               cudaStream_t& this_stream,
               DEMSolverStateDataKT& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceSelect::Unique(NULL, cub_scratch_bytes, d_in, d_out, d_num_out, n, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceSelect::Unique(d_scratch_space, cub_scratch_bytes, d_in, d_out, d_num_out, n, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
}

void cubRunLengthEncode(binID_t* d_in,
                        binID_t* d_unique_out,
                        spheresBinTouches_t* d_counts_out,
                        size_t* d_num_out,
                        size_t n,
                        cudaStream_t& this_stream,
                        DEMSolverStateDataKT& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(NULL, cub_scratch_bytes, d_in, d_unique_out, d_counts_out, d_num_out, n,
                                       this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceRunLengthEncode::Encode(d_scratch_space, cub_scratch_bytes, d_in, d_unique_out, d_counts_out, d_num_out,
                                       n, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
}

void cubSum(float* d_in, float* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateDataDT& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceReduce::Sum(NULL, cub_scratch_bytes, d_in, d_out, n, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceReduce::Sum(d_scratch_space, cub_scratch_bytes, d_in, d_out, n, this_stream, false);
    GPU_CALL(cudaStreamSynchronize(this_stream));
}

}  // namespace sgps
