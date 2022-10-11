//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <cub/cub.cuh>
#include <core/utils/JitHelper.h>
#include <algorithms/DEMCubBasedSubroutines.h>

#include <core/utils/GpuError.h>
#include <algorithms/DEMCubWrappers.cu>

namespace deme {

// These functions interconnecting the cub-part and cpp-part of the code cannot be templated... because of cmake
// restrictions. Not much that I can do, other than writing them all out.

void doubleSumReduce(double* d_in, double* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateData& scratchPad) {
    cubDEMSum<double, double, DEMSolverStateData>(d_in, d_out, n, this_stream, scratchPad);
}
void floatSumReduce(float* d_in, float* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateData& scratchPad) {
    cubDEMSum<float, float, DEMSolverStateData>(d_in, d_out, n, this_stream, scratchPad);
}

void boolSumReduce(notStupidBool_t* d_in,
                   size_t* d_out,
                   size_t n,
                   cudaStream_t& this_stream,
                   DEMSolverStateData& scratchPad) {
    cubDEMSum<notStupidBool_t, size_t, DEMSolverStateData>(d_in, d_out, n, this_stream, scratchPad);
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
void doubleMaxReduce(double* d_in, double* d_out, size_t n, cudaStream_t& this_stream, DEMSolverStateData& scratchPad) {
    cubDEMMax<double, DEMSolverStateData>(d_in, d_out, n, this_stream, scratchPad);
}

void floatSumReduceByKey(notStupidBool_t* d_keys_in,
                         notStupidBool_t* d_unique_out,
                         float* d_vals_in,
                         float* d_aggregates_out,
                         size_t* d_num_out,
                         size_t n,
                         cudaStream_t& this_stream,
                         DEMSolverStateData& scratchPad) {
    // I'm not sure how to pass cuda cub::Sum() as a template argument here, so I used a custom add...
    CubFloatAdd add_op;
    cubDEMReduceByKeys<notStupidBool_t, float, CubFloatAdd, DEMSolverStateData>(
        d_keys_in, d_unique_out, d_vals_in, d_aggregates_out, d_num_out, add_op, n, this_stream, scratchPad);
}
void doubleSumReduceByKey(notStupidBool_t* d_keys_in,
                          notStupidBool_t* d_unique_out,
                          double* d_vals_in,
                          double* d_aggregates_out,
                          size_t* d_num_out,
                          size_t n,
                          cudaStream_t& this_stream,
                          DEMSolverStateData& scratchPad) {
    // I'm not sure how to pass cuda cub::Sum() as a template argument here, so I used a custom add...
    CubFloatAdd add_op;
    cubDEMReduceByKeys<notStupidBool_t, double, CubFloatAdd, DEMSolverStateData>(
        d_keys_in, d_unique_out, d_vals_in, d_aggregates_out, d_num_out, add_op, n, this_stream, scratchPad);
}

void floatSortByKey(notStupidBool_t* d_keys_in,
                    notStupidBool_t* d_keys_out,
                    float* d_vals_in,
                    float* d_vals_out,
                    size_t n,
                    cudaStream_t& this_stream,
                    DEMSolverStateData& scratchPad) {
    cubDEMSortByKeys<notStupidBool_t, float, DEMSolverStateData>(d_keys_in, d_keys_out, d_vals_in, d_vals_out, n,
                                                                 this_stream, scratchPad);
}
void doubleSortByKey(notStupidBool_t* d_keys_in,
                     notStupidBool_t* d_keys_out,
                     double* d_vals_in,
                     double* d_vals_out,
                     size_t n,
                     cudaStream_t& this_stream,
                     DEMSolverStateData& scratchPad) {
    cubDEMSortByKeys<notStupidBool_t, double, DEMSolverStateData>(d_keys_in, d_keys_out, d_vals_in, d_vals_out, n,
                                                                  this_stream, scratchPad);
}

}  // namespace deme
