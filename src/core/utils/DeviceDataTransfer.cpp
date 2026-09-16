// Copyright (c) 2021, SBEL GPU Development Team
// Copyright (c) 2021, University of Wisconsin - Madison
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DeviceDataTransfer.hpp"

namespace deme {
namespace device_data {

cudaError_t ValidateOutputPointer(const void* pointer, size_t bytes, int device) {
    if (bytes == 0) {
        return cudaSuccess;
    }
    if (!pointer || device < 0) {
        return cudaErrorInvalidValue;
    }

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        return status;
    }
    if (device >= device_count) {
        return cudaErrorInvalidDevice;
    }

    cudaPointerAttributes attributes{};
    status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) {
        return status;
    }
    if (attributes.type == cudaMemoryTypeDevice && attributes.device != device) {
        return cudaErrorInvalidDevicePointer;
    }
    if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) {
        return cudaErrorInvalidDevicePointer;
    }
    return cudaSuccess;
}

cudaError_t TransferBuffer::Copy(void* destination,
                                 int /* destination_device */,
                                 const void* source,
                                 int /* source_device */,
                                 size_t bytes) {
    if (bytes == 0) {
        return cudaSuccess;
    }
    if (!destination || !source) {
        return cudaErrorInvalidValue;
    }
    // DEME's regular dT/kT exchange uses this CUDA path when direct peer access is unavailable. With unified virtual
    // addressing, CUDA identifies both owning devices and selects either a direct peer route or driver-managed
    // staging. If the topology cannot support the copy, return that CUDA error to the caller.
    return cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice);
}

}  // namespace device_data
}  // namespace deme
