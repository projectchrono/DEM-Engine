// Copyright (c) 2021, SBEL GPU Development Team
// Copyright (c) 2021, University of Wisconsin - Madison
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_OWNER_DATA_RETRIEVAL_H
#define DEME_OWNER_DATA_RETRIEVAL_H

#include <cuda_runtime_api.h>

#include <cstddef>

#include "DEM/Structs.h"

namespace deme {

/// Fixed-size owner properties supported by the GPU retrieval path.
enum class OwnerDataField {
    POSITION,
    VELOCITY,
    ANGULAR_VELOCITY_LOCAL,
    ANGULAR_VELOCITY_GLOBAL,
    ORIENTATION,
    CONTACT_ACCELERATION,
    CONTACT_ANGULAR_ACCELERATION_LOCAL,
    CONTACT_ANGULAR_ACCELERATION_GLOBAL,
    FAMILY,
    MASS,
    MOI,
    WILDCARD
};

/// Owner state fields supported by the GPU input path.
enum class OwnerStateField {
    POSITION,
    VELOCITY,
    ANGULAR_VELOCITY_LOCAL,
    ANGULAR_VELOCITY_GLOBAL,
    ORIENTATION,
    NEXT_STEP_ACCELERATION,
    NEXT_STEP_ANGULAR_ACCELERATION_LOCAL
};

/// Pack one consecutive owner range from DEME's internal SoA representation into the public AoS/scalar representation.
void PackOwnerData(void* output,
                   OwnerDataField field,
                   bodyID_t owner_begin,
                   size_t count,
                   const DEMSimParams* sim_params,
                   const DEMDataDT* data,
                   bool mass_properties_are_jitified,
                   unsigned int wildcard_index,
                   cudaStream_t stream);

/// Return the size of one output element for a retrieval field.
size_t OwnerDataElementSize(OwnerDataField field);

/// Check that every public-order quaternion is finite and has nonzero length before an owner-state update.
void ValidateOwnerOrientations(const float4* input, size_t count, unsigned int* invalid, cudaStream_t stream);

/// Unpack one public AoS owner-state range directly into DEME's internal SoA representation.
void UnpackOwnerState(const void* input,
                      OwnerStateField field,
                      bodyID_t owner_begin,
                      size_t count,
                      const DEMSimParams* sim_params,
                      DEMDataDT* data,
                      cudaStream_t stream);

/// Reduce the current patch-contact records into one global-frame resultant wrench per consecutive owner.
void ReduceOwnerContactWrenches(float3* forces,
                                float3* torques,
                                bodyID_t owner_begin,
                                size_t count,
                                const DEMDataDT* data,
                                size_t num_contacts,
                                cudaStream_t stream);

}  // namespace deme

#endif
