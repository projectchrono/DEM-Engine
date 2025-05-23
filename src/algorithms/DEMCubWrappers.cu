//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <cub/cub.cuh>
#include <DEM/Defines.h>
#include <DEM/Structs.h>
#include <core/utils/GpuError.h>

namespace deme {

// Functor type for selecting values less than some criteria
template <typename T>
struct CubEqualTo {
    T compare;
    CUB_RUNTIME_FUNCTION __forceinline__ CubEqualTo(T compare) : compare(compare) {}
    CUB_RUNTIME_FUNCTION __forceinline__ bool operator()(const T& a) const { return (a == (T)compare); }
};

struct CubFloat3Add {
    CUB_RUNTIME_FUNCTION __forceinline__ __device__ __host__ float3 operator()(const float3& a, const float3& b) const {
        return ::make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
};

template <typename T>
struct CubOpAdd {
    CUB_RUNTIME_FUNCTION __forceinline__ __device__ __host__ T operator()(const T& a, const T& b) const {
        return a + b;
    }
};

// Custom min and max functor
template <typename T>
struct CubOpMin {
    CUB_RUNTIME_FUNCTION __forceinline__ __device__ __host__ T operator()(const T& a, const T& b) const {
        return (b < a) ? b : a;
    }
};

template <typename T>
struct CubOpMax {
    CUB_RUNTIME_FUNCTION __forceinline__ __device__ __host__ T operator()(const T& a, const T& b) const {
        return (b > a) ? b : a;
    }
};

template <typename T1, typename T2>
inline void cubDEMSelectFlagged(T1* d_in,
                                T1* d_out,
                                T2* d_flags,
                                size_t* d_num_out,
                                size_t n,
                                cudaStream_t& this_stream,
                                DEMSolverScratchData& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceSelect::Flagged(NULL, cub_scratch_bytes, d_in, d_flags, d_out, d_num_out, n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceSelect::Flagged(d_scratch_space, cub_scratch_bytes, d_in, d_flags, d_out, d_num_out, n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
}

template <typename T1, typename T2>
inline void cubDEMPrefixScan(T1* d_in,
                             T2* d_out,
                             size_t n,
                             cudaStream_t& this_stream,
                             DEMSolverScratchData& scratchPad) {
    // NOTE!!! Why did I not use ExclusiveSum? I found that when for a cub scan operation, if the d_in and d_out are of
    // different types, then cub defaults to use d_in type to store the scan result, but will switch to d_out type if
    // there are too many items to scan. There is however, a region where cub does not choose to switch to d_out type,
    // but d_in type is not big enough to store the scan result. This causes overflow and cub certainly does not care to
    // let you know when it happens. I made a trick: use ExclusiveScan and (T2)0 as the initial value, and this forces
    // cub to store results as T2 type.
    size_t cub_scratch_bytes = 0;
    cub::DeviceScan::ExclusiveScan(NULL, cub_scratch_bytes, d_in, d_out, cub::Sum(), (T2)0, n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceScan::ExclusiveScan(d_scratch_space, cub_scratch_bytes, d_in, d_out, cub::Sum(), (T2)0, n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
}

template <typename T1, typename T2>
inline void cubDEMSortByKeys(T1* d_keys_in,
                             T1* d_keys_out,
                             T2* d_vals_in,
                             T2* d_vals_out,
                             size_t n,
                             cudaStream_t& this_stream,
                             DEMSolverScratchData& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, cub_scratch_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, n, 0,
                                    sizeof(T1) * DEME_BITS_PER_BYTE, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceRadixSort::SortPairs(d_scratch_space, cub_scratch_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, n,
                                    0, sizeof(T1) * DEME_BITS_PER_BYTE, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
}

template <typename T1>
inline void cubDEMUnique(T1* d_in,
                         T1* d_out,
                         size_t* d_num_out,
                         size_t n,
                         cudaStream_t& this_stream,
                         DEMSolverScratchData& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceSelect::Unique(NULL, cub_scratch_bytes, d_in, d_out, d_num_out, n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceSelect::Unique(d_scratch_space, cub_scratch_bytes, d_in, d_out, d_num_out, n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
}

template <typename T1, typename T2>
inline void cubDEMRunLengthEncode(T1* d_in,
                                  T1* d_unique_out,
                                  T2* d_counts_out,
                                  size_t* d_num_out,
                                  size_t n,
                                  cudaStream_t& this_stream,
                                  DEMSolverScratchData& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(NULL, cub_scratch_bytes, d_in, d_unique_out, d_counts_out, d_num_out, n,
                                       this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceRunLengthEncode::Encode(d_scratch_space, cub_scratch_bytes, d_in, d_unique_out, d_counts_out, d_num_out,
                                       n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
}

template <typename T1, typename T2, typename T3>
inline void cubDEMReduceByKeys(T1* d_keys_in,
                               T1* d_unique_out,
                               T2* d_vals_in,
                               T2* d_aggregates_out,
                               size_t* d_num_out,
                               T3& reduce_op,
                               size_t n,
                               cudaStream_t& this_stream,
                               DEMSolverScratchData& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceReduce::ReduceByKey(NULL, cub_scratch_bytes, d_keys_in, d_unique_out, d_vals_in, d_aggregates_out,
                                   d_num_out, reduce_op, n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceReduce::ReduceByKey(d_scratch_space, cub_scratch_bytes, d_keys_in, d_unique_out, d_vals_in,
                                   d_aggregates_out, d_num_out, reduce_op, n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
}

template <typename T1, typename T2>
void cubDEMSum(T1* d_in, T2* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceReduce::Reduce(NULL, cub_scratch_bytes, d_in, d_out, n, cub::Sum(), (T2)0, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceReduce::Reduce(d_scratch_space, cub_scratch_bytes, d_in, d_out, n, cub::Sum(), (T2)0, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
}

// template <typename T1>
// void cubDEMSum(T1* d_in, T1* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad) {
//     size_t cub_scratch_bytes = 0;
//     cub::DeviceReduce::Sum(NULL, cub_scratch_bytes, d_in, d_out, n, this_stream);
//     DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
//     void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
//     cub::DeviceReduce::Sum(d_scratch_space, cub_scratch_bytes, d_in, d_out, n, this_stream);
//     DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
// }

template <typename T1>
void cubDEMMax(T1* d_in, T1* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceReduce::Max(NULL, cub_scratch_bytes, d_in, d_out, n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceReduce::Max(d_scratch_space, cub_scratch_bytes, d_in, d_out, n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
}

template <typename T1>
void cubDEMMin(T1* d_in, T1* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad) {
    size_t cub_scratch_bytes = 0;
    cub::DeviceReduce::Min(NULL, cub_scratch_bytes, d_in, d_out, n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    cub::DeviceReduce::Min(d_scratch_space, cub_scratch_bytes, d_in, d_out, n, this_stream);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
}

}  // namespace deme
