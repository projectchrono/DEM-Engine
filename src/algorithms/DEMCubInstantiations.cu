//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <cub/cub.cuh>
#include <algorithms/DEMStaticDeviceSubroutines.h>

#include <core/utils/GpuError.h>
#include <algorithms/DEMCubWrappers.cu>

namespace deme {

// ========================================================================
// Instantiations of CUB-based subroutines
// These functions interconnecting the cub-part and cpp-part of the code cannot be fully templated...
// Instead, all possible uses must be explicitly instantiated.
// ========================================================================

////////////////////////////////////////////////////////////////////////////////
// Reduce::Sum
////////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2>
void cubSumReduce(T1* d_in, T2* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad) {
    cubDEMSum<T1, T2>(d_in, d_out, n, this_stream, scratchPad);
}
template void cubSumReduce<float, float>(float* d_in,
                                         float* d_out,
                                         size_t n,
                                         cudaStream_t& this_stream,
                                         DEMSolverScratchData& scratchPad);
template void cubSumReduce<double, double>(double* d_in,
                                           double* d_out,
                                           size_t n,
                                           cudaStream_t& this_stream,
                                           DEMSolverScratchData& scratchPad);
template void cubSumReduce<notStupidBool_t, size_t>(notStupidBool_t* d_in,
                                                    size_t* d_out,
                                                    size_t n,
                                                    cudaStream_t& this_stream,
                                                    DEMSolverScratchData& scratchPad);

template <typename T1, typename T2>
void cubSumReduceByKey(T1* d_keys_in,
                       T1* d_unique_out,
                       T2* d_vals_in,
                       T2* d_aggregates_out,
                       size_t* d_num_out,
                       size_t n,
                       cudaStream_t& this_stream,
                       DEMSolverScratchData& scratchPad) {
    // I'm not sure how to pass cuda cub::Sum() as a template argument here, so I used a custom add...
    CubOpAdd<T2> add_op;
    cubDEMReduceByKeys<T1, T2, CubOpAdd<T2>>(d_keys_in, d_unique_out, d_vals_in, d_aggregates_out, d_num_out, add_op, n,
                                             this_stream, scratchPad);
}
template void cubSumReduceByKey<notStupidBool_t, float>(notStupidBool_t* d_keys_in,
                                                        notStupidBool_t* d_unique_out,
                                                        float* d_vals_in,
                                                        float* d_aggregates_out,
                                                        size_t* d_num_out,
                                                        size_t n,
                                                        cudaStream_t& this_stream,
                                                        DEMSolverScratchData& scratchPad);
template void cubSumReduceByKey<notStupidBool_t, double>(notStupidBool_t* d_keys_in,
                                                         notStupidBool_t* d_unique_out,
                                                         double* d_vals_in,
                                                         double* d_aggregates_out,
                                                         size_t* d_num_out,
                                                         size_t n,
                                                         cudaStream_t& this_stream,
                                                         DEMSolverScratchData& scratchPad);

////////////////////////////////////////////////////////////////////////////////
// Reduce::Max
////////////////////////////////////////////////////////////////////////////////

template <typename T1>
void cubMaxReduce(T1* d_in, T1* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad) {
    cubDEMMax<T1>(d_in, d_out, n, this_stream, scratchPad);
}
template void cubMaxReduce<notStupidBool_t>(notStupidBool_t* d_in,
                                            notStupidBool_t* d_out,
                                            size_t n,
                                            cudaStream_t& this_stream,
                                            DEMSolverScratchData& scratchPad);
template void cubMaxReduce<float>(float* d_in,
                                  float* d_out,
                                  size_t n,
                                  cudaStream_t& this_stream,
                                  DEMSolverScratchData& scratchPad);
template void cubMaxReduce<double>(double* d_in,
                                   double* d_out,
                                   size_t n,
                                   cudaStream_t& this_stream,
                                   DEMSolverScratchData& scratchPad);

template <typename T1, typename T2>
void cubMaxReduceByKey(T1* d_keys_in,
                       T1* d_unique_out,
                       T2* d_vals_in,
                       T2* d_aggregates_out,
                       size_t* d_num_out,
                       size_t n,
                       cudaStream_t& this_stream,
                       DEMSolverScratchData& scratchPad) {
    // I'm not sure how to pass cuda cub::Max() as a template argument here, so I used a custom max...
    CubOpMax<T2> max_op;
    cubDEMReduceByKeys<T1, T2, CubOpMax<T2>>(d_keys_in, d_unique_out, d_vals_in, d_aggregates_out, d_num_out, max_op, n,
                                             this_stream, scratchPad);
}
template void cubMaxReduceByKey<notStupidBool_t, float>(notStupidBool_t* d_keys_in,
                                                        notStupidBool_t* d_unique_out,
                                                        float* d_vals_in,
                                                        float* d_aggregates_out,
                                                        size_t* d_num_out,
                                                        size_t n,
                                                        cudaStream_t& this_stream,
                                                        DEMSolverScratchData& scratchPad);
template void cubMaxReduceByKey<notStupidBool_t, double>(notStupidBool_t* d_keys_in,
                                                         notStupidBool_t* d_unique_out,
                                                         double* d_vals_in,
                                                         double* d_aggregates_out,
                                                         size_t* d_num_out,
                                                         size_t n,
                                                         cudaStream_t& this_stream,
                                                         DEMSolverScratchData& scratchPad);

////////////////////////////////////////////////////////////////////////////////
// Reduce::Min
////////////////////////////////////////////////////////////////////////////////

template <typename T1>
void cubMinReduce(T1* d_in, T1* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad) {
    cubDEMMin<T1>(d_in, d_out, n, this_stream, scratchPad);
}
template void cubMinReduce<float>(float* d_in,
                                  float* d_out,
                                  size_t n,
                                  cudaStream_t& this_stream,
                                  DEMSolverScratchData& scratchPad);
template void cubMinReduce<double>(double* d_in,
                                   double* d_out,
                                   size_t n,
                                   cudaStream_t& this_stream,
                                   DEMSolverScratchData& scratchPad);

template <typename T1, typename T2>
void cubMinReduceByKey(T1* d_keys_in,
                       T1* d_unique_out,
                       T2* d_vals_in,
                       T2* d_aggregates_out,
                       size_t* d_num_out,
                       size_t n,
                       cudaStream_t& this_stream,
                       DEMSolverScratchData& scratchPad) {
    // I'm not sure how to pass cuda cub::Min() as a template argument here, so I used a custom min...
    CubOpMin<T2> min_op;
    cubDEMReduceByKeys<T1, T2, CubOpMin<T2>>(d_keys_in, d_unique_out, d_vals_in, d_aggregates_out, d_num_out, min_op, n,
                                             this_stream, scratchPad);
}
template void cubMinReduceByKey<notStupidBool_t, float>(notStupidBool_t* d_keys_in,
                                                        notStupidBool_t* d_unique_out,
                                                        float* d_vals_in,
                                                        float* d_aggregates_out,
                                                        size_t* d_num_out,
                                                        size_t n,
                                                        cudaStream_t& this_stream,
                                                        DEMSolverScratchData& scratchPad);
template void cubMinReduceByKey<notStupidBool_t, double>(notStupidBool_t* d_keys_in,
                                                         notStupidBool_t* d_unique_out,
                                                         double* d_vals_in,
                                                         double* d_aggregates_out,
                                                         size_t* d_num_out,
                                                         size_t n,
                                                         cudaStream_t& this_stream,
                                                         DEMSolverScratchData& scratchPad);

////////////////////////////////////////////////////////////////////////////////
// Sort
////////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2>
void cubSortByKey(T1* d_keys_in,
                  T1* d_keys_out,
                  T2* d_vals_in,
                  T2* d_vals_out,
                  size_t n,
                  cudaStream_t& this_stream,
                  DEMSolverScratchData& scratchPad) {
    cubDEMSortByKeys<T1, T2>(d_keys_in, d_keys_out, d_vals_in, d_vals_out, n, this_stream, scratchPad);
}
template void cubSortByKey<notStupidBool_t, float>(notStupidBool_t* d_keys_in,
                                                   notStupidBool_t* d_keys_out,
                                                   float* d_vals_in,
                                                   float* d_vals_out,
                                                   size_t n,
                                                   cudaStream_t& this_stream,
                                                   DEMSolverScratchData& scratchPad);
template void cubSortByKey<notStupidBool_t, double>(notStupidBool_t* d_keys_in,
                                                    notStupidBool_t* d_keys_out,
                                                    double* d_vals_in,
                                                    double* d_vals_out,
                                                    size_t n,
                                                    cudaStream_t& this_stream,
                                                    DEMSolverScratchData& scratchPad);
}  // namespace deme
