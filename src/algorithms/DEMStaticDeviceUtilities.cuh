//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_STATIC_DEVICE_UTIL_CUH
#define DEME_STATIC_DEVICE_UTIL_CUH

#include <DEM/Defines.h>

namespace deme {

// ========================================================================
// Some simple, static device-side utilities are here, and they need a place
// to live with cuda compilation environment
// ========================================================================

template <typename T1, typename T2, typename T3>
__global__ void oneNumAdd(T1* res, const T2* a, const T3* b) {
    T1 T1a = (T1)(*a);
    T1 T1b = (T1)(*b);
    *res = T1a + T1b;
}
template <typename T1, typename T2, typename T3>
void deviceAdd(T1* res, const T2* a, const T3* b, cudaStream_t& this_stream) {
    oneNumAdd<T1, T2, T3><<<1, 1, 0, this_stream>>>(res, a, b);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
}

template <typename T1, typename T2>
__global__ void oneNumAssign(T1* res, const T2* a) {
    T1 T1a = (T1)(*a);
    *res = T1a;
}
template <typename T1, typename T2>
void deviceAssign(T1* res, const T2* a, cudaStream_t& this_stream) {
    oneNumAssign<T1, T2><<<1, 1, 0, this_stream>>>(res, a);
    DEME_GPU_CALL(cudaStreamSynchronize(this_stream));
}

}  // namespace deme

#endif
