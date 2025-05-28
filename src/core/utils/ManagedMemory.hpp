//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_MANAGED_MEM_H
#define DEME_MANAGED_MEM_H

#include <vector>

#include "CudaAllocator.hpp"

// Convenience functions to help with Managed Memory (allocated using ManagedAllocator, typically)
namespace deme {

// Underlying implementation
template <class T>
inline void __migrate_impl(T* data, std::size_t size, int device, cudaStream_t stream = 0) {
    cudaMemPrefetchAsync(static_cast<void*>(data), size * sizeof(T), device, stream);
}

// Pointer syntax, basically the same as cudaMemPrefetchAsync(...)
template <class T>
void migrate(T* data, std::size_t size, int device, cudaStream_t stream = 0) {
    __migrate_impl<T>(data, size, device, stream);
}

// Inadviseable, but necessary to force transfer of const arrays to device
// template <class T>
// void migrate(const T* data, std::size_t size, int device, cudaStream_t stream = 0) {
//     __migrate_impl<T>(std::const_cast<T*>(data), size, device, stream);
// }

// Migrate a single class or value
template <class T>
void migrate(T& datum, int device, cudaStream_t stream = 0) {
    __migrate_impl<T>(&datum, 1, device, stream);
}

// Migrate a single const class or value
template <class T>
void migrate(const T& datum, int device, cudaStream_t stream = 0) {
    __migrate_impl<T>(&datum, 1, device, stream);
}

// Migrate the data contained in a vector
template <class T>
void migrate(std::vector<T>& data, int device, cudaStream_t stream = 0) {
    __migrate_impl<T>(data.data(), data.size(), device, stream);
}

// Migrate the data contained in a const vector
template <class T>
void migrate(const std::vector<T>& data, int device, cudaStream_t stream = 0) {
    __migrate_impl<T>(data.data(), data.size(), device, stream);
}

// Aliases for cudaMemoryAdvise constants
enum class ManagedAdvice {
    READ_MOSTLY = cudaMemAdviseSetReadMostly,
    PREFERRED_LOC = cudaMemAdviseSetPreferredLocation,
    ACCESSED_BY = cudaMemAdviseSetAccessedBy,
    UNSET_READ_MOSTLY = cudaMemAdviseUnsetReadMostly,
    UNSET_PREFERRED_LOC = cudaMemAdviseUnsetPreferredLocation,
    UNSET_ACCESSED_BY = cudaMemAdviseUnsetAccessedBy
};

// Underlying implementation
template <class T>
void __advise_impl(const T* data, std::size_t size, ManagedAdvice advice, int device) {
    cudaMemAdvise(data, size * sizeof(T), (cudaMemoryAdvise)advice, device);
}

// Advice for raw pointer
template <class T>
void advise(const T* data, std::size_t size, ManagedAdvice advice, int device) {
    __advise_impl(data, size, advice, device);
}

// Advice for single class or value
template <class T>
void advise(const T& datum, ManagedAdvice advice, int device) {
    __advise_impl(&datum, 1, advice, device);
}

// Advice for underlying vector storage
template <class T>
void advise(const std::vector<T>& data, ManagedAdvice advice, int device) {
    __advise_impl(data.data(), data.size(), advice, device);
}

}  // END namespace deme

#endif
