// Copyright (c) 2021, SBEL GPU Development Team
// Copyright (c) 2021, University of Wisconsin - Madison
//
// SPDX-License-Identifier: BSD-3-Clause

#include "DEMOwnerDataRetrieval.h"

#include "DEM/Defines.h"
#include "kernel/DEMHelperKernels.cuh"

namespace deme {

namespace {

constexpr unsigned int OWNER_DATA_RETRIEVAL_BLOCK_SIZE = 256;

__global__ void ValidateOwnerOrientationsKernel(const float4* input, size_t count, unsigned int* invalid) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float4 q = input[index];
    const double norm_squared = static_cast<double>(q.x) * q.x + static_cast<double>(q.y) * q.y +
                                static_cast<double>(q.z) * q.z + static_cast<double>(q.w) * q.w;
    if (!isfinite(norm_squared) || norm_squared == 0.0) {
        atomicExch(invalid, 1u);
    }
}

__global__ void PackOwnerDataKernel(void* output,
                                    OwnerDataField field,
                                    bodyID_t owner_begin,
                                    size_t count,
                                    const DEMSimParams* sim_params,
                                    const DEMDataDT* data,
                                    bool mass_properties_are_jitified,
                                    unsigned int wildcard_index) {
    const size_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= count) {
        return;
    }
    const bodyID_t owner = owner_begin + output_index;

    const float4 orientation =
        make_float4(data->oriQx[owner], data->oriQy[owner], data->oriQz[owner], data->oriQw[owner]);
    switch (field) {
        case OwnerDataField::POSITION: {
            double x, y, z;
            voxelIDToPosition<double, voxelID_t, subVoxelPos_t>(
                x, y, z, data->voxelID[owner], data->locX[owner], data->locY[owner], data->locZ[owner],
                sim_params->nvXp2, sim_params->nvYp2, sim_params->voxelSize, sim_params->l);
            static_cast<float3*>(output)[output_index] =
                make_float3(x + sim_params->LBFX, y + sim_params->LBFY, z + sim_params->LBFZ);
            break;
        }
        case OwnerDataField::VELOCITY:
            static_cast<float3*>(output)[output_index] = make_float3(data->vX[owner], data->vY[owner], data->vZ[owner]);
            break;
        case OwnerDataField::ANGULAR_VELOCITY_LOCAL:
            static_cast<float3*>(output)[output_index] =
                make_float3(data->omgBarX[owner], data->omgBarY[owner], data->omgBarZ[owner]);
            break;
        case OwnerDataField::ANGULAR_VELOCITY_GLOBAL: {
            float3 value = make_float3(data->omgBarX[owner], data->omgBarY[owner], data->omgBarZ[owner]);
            applyOriQToVector3(value, orientation);
            static_cast<float3*>(output)[output_index] = value;
            break;
        }
        case OwnerDataField::ORIENTATION:
            static_cast<float4*>(output)[output_index] = orientation;
            break;
        case OwnerDataField::CONTACT_ACCELERATION:
            static_cast<float3*>(output)[output_index] = make_float3(data->aX[owner], data->aY[owner], data->aZ[owner]);
            break;
        case OwnerDataField::CONTACT_ANGULAR_ACCELERATION_LOCAL:
            static_cast<float3*>(output)[output_index] =
                make_float3(data->alphaX[owner], data->alphaY[owner], data->alphaZ[owner]);
            break;
        case OwnerDataField::CONTACT_ANGULAR_ACCELERATION_GLOBAL: {
            float3 value = make_float3(data->alphaX[owner], data->alphaY[owner], data->alphaZ[owner]);
            applyOriQToVector3(value, orientation);
            static_cast<float3*>(output)[output_index] = value;
            break;
        }
        case OwnerDataField::FAMILY:
            static_cast<unsigned int*>(output)[output_index] = static_cast<unsigned int>(+(data->familyID[owner]));
            break;
        case OwnerDataField::MASS: {
            const inertiaOffset_t offset = mass_properties_are_jitified ? data->inertiaPropOffsets[owner] : owner;
            static_cast<float*>(output)[output_index] = data->massOwnerBody[offset];
            break;
        }
        case OwnerDataField::MOI: {
            const inertiaOffset_t offset = mass_properties_are_jitified ? data->inertiaPropOffsets[owner] : owner;
            static_cast<float3*>(output)[output_index] =
                make_float3(data->mmiXX[offset], data->mmiYY[offset], data->mmiZZ[offset]);
            break;
        }
        case OwnerDataField::WILDCARD:
            static_cast<float*>(output)[output_index] = data->ownerWildcards[wildcard_index][owner];
            break;
    }
}

__global__ void UnpackOwnerStateKernel(const void* input,
                                       OwnerStateField field,
                                       bodyID_t owner_begin,
                                       size_t count,
                                       const DEMSimParams* sim_params,
                                       DEMDataDT* data) {
    const size_t input_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (input_index >= count) {
        return;
    }
    const bodyID_t owner = owner_begin + input_index;

    switch (field) {
        case OwnerStateField::POSITION: {
            const float3 position = static_cast<const float3*>(input)[input_index];
            const double x = static_cast<double>(position.x) - sim_params->LBFX;
            const double y = static_cast<double>(position.y) - sim_params->LBFY;
            const double z = static_cast<double>(position.z) - sim_params->LBFZ;
            positionToVoxelID<voxelID_t, subVoxelPos_t, double>(
                data->voxelID[owner], data->locX[owner], data->locY[owner], data->locZ[owner], x, y, z,
                sim_params->nvXp2, sim_params->nvYp2, sim_params->voxelSize, sim_params->l);
            break;
        }
        case OwnerStateField::VELOCITY: {
            const float3 velocity = static_cast<const float3*>(input)[input_index];
            data->vX[owner] = velocity.x;
            data->vY[owner] = velocity.y;
            data->vZ[owner] = velocity.z;
            break;
        }
        case OwnerStateField::ANGULAR_VELOCITY_LOCAL: {
            const float3 angular_velocity = static_cast<const float3*>(input)[input_index];
            data->omgBarX[owner] = angular_velocity.x;
            data->omgBarY[owner] = angular_velocity.y;
            data->omgBarZ[owner] = angular_velocity.z;
            break;
        }
        case OwnerStateField::ANGULAR_VELOCITY_GLOBAL: {
            float3 angular_velocity = static_cast<const float3*>(input)[input_index];
            const float4 inverse_orientation =
                make_float4(-data->oriQx[owner], -data->oriQy[owner], -data->oriQz[owner], data->oriQw[owner]);
            applyOriQToVector3(angular_velocity, inverse_orientation);
            data->omgBarX[owner] = angular_velocity.x;
            data->omgBarY[owner] = angular_velocity.y;
            data->omgBarZ[owner] = angular_velocity.z;
            break;
        }
        case OwnerStateField::ORIENTATION: {
            const float4 orientation = static_cast<const float4*>(input)[input_index];
            const double norm_squared = static_cast<double>(orientation.x) * orientation.x +
                                        static_cast<double>(orientation.y) * orientation.y +
                                        static_cast<double>(orientation.z) * orientation.z +
                                        static_cast<double>(orientation.w) * orientation.w;
            const float inverse_norm = static_cast<float>(1.0 / sqrt(norm_squared));
            data->oriQx[owner] = orientation.x * inverse_norm;
            data->oriQy[owner] = orientation.y * inverse_norm;
            data->oriQz[owner] = orientation.z * inverse_norm;
            data->oriQw[owner] = orientation.w * inverse_norm;
            break;
        }
        case OwnerStateField::NEXT_STEP_ACCELERATION: {
            const float3 acceleration = static_cast<const float3*>(input)[input_index];
            data->aX[owner] = acceleration.x;
            data->aY[owner] = acceleration.y;
            data->aZ[owner] = acceleration.z;
            data->accSpecified[owner] = 1;
            break;
        }
        case OwnerStateField::NEXT_STEP_ANGULAR_ACCELERATION_LOCAL: {
            const float3 angular_acceleration = static_cast<const float3*>(input)[input_index];
            data->alphaX[owner] = angular_acceleration.x;
            data->alphaY[owner] = angular_acceleration.y;
            data->alphaZ[owner] = angular_acceleration.z;
            data->angAccSpecified[owner] = 1;
            break;
        }
    }
}

__device__ void AccumulateOwnerContactWrench(float3* forces,
                                             float3* torques,
                                             bodyID_t owner_begin,
                                             size_t count,
                                             const DEMDataDT* data,
                                             bodyID_t owner,
                                             const float3& force_a,
                                             const float3& extra_torque_force_a,
                                             float sign,
                                             const float3& local_contact_point) {
    if (owner < owner_begin || static_cast<size_t>(owner - owner_begin) >= count) {
        return;
    }
    const size_t output = static_cast<size_t>(owner - owner_begin);
    const float3 signed_force = force_a * sign;
    float3 total_force_form = (force_a + extra_torque_force_a) * sign;
    const float4 orientation =
        make_float4(data->oriQx[owner], data->oriQy[owner], data->oriQz[owner], data->oriQw[owner]);
    applyOriQToVector3(total_force_form, make_float4(-orientation.x, -orientation.y, -orientation.z, orientation.w));
    float3 global_torque = cross(local_contact_point, total_force_form);
    applyOriQToVector3(global_torque, orientation);

    atomicAdd(&forces[output].x, signed_force.x);
    atomicAdd(&forces[output].y, signed_force.y);
    atomicAdd(&forces[output].z, signed_force.z);
    atomicAdd(&torques[output].x, global_torque.x);
    atomicAdd(&torques[output].y, global_torque.y);
    atomicAdd(&torques[output].z, global_torque.z);
}

__global__ void ReduceOwnerContactWrenchesKernel(float3* forces,
                                                 float3* torques,
                                                 bodyID_t owner_begin,
                                                 size_t count,
                                                 const DEMDataDT* granData,
                                                 size_t num_contacts) {
    const size_t contact = blockIdx.x * blockDim.x + threadIdx.x;
    if (contact >= num_contacts || !granData->contactTypePatch || !granData->idPatchA || !granData->idPatchB) {
        return;
    }
    const contact_t contact_type = granData->contactTypePatch[contact];
    if (contact_type == NOT_A_CONTACT) {
        return;
    }

    const bodyID_t owner_a = DEME_GET_PATCH_OWNER_ID(granData->idPatchA[contact], decodeTypeA(contact_type));
    const bodyID_t owner_b = DEME_GET_PATCH_OWNER_ID(granData->idPatchB[contact], decodeTypeB(contact_type));
    const float3 force_a = granData->contactForces[contact];
    const float3 extra_torque_force_a = granData->contactTorque_convToForce[contact];
    if (length(force_a) + length(extra_torque_force_a) < DEME_TINY_FLOAT) {
        return;
    }

    // contactTorque_convToForce is DEME's force-like representation of force-model-only torque. Using the same
    // contact arm as forceToAcc yields r x force plus rolling-resistance (or other model-provided) torque.
    AccumulateOwnerContactWrench(forces, torques, owner_begin, count, granData, owner_a, force_a, extra_torque_force_a,
                                 1.f, granData->contactPointGeometryA[contact]);
    AccumulateOwnerContactWrench(forces, torques, owner_begin, count, granData, owner_b, force_a, extra_torque_force_a,
                                 -1.f, granData->contactPointGeometryB[contact]);
}

}  // namespace

void PackOwnerData(void* output,
                   OwnerDataField field,
                   bodyID_t owner_begin,
                   size_t count,
                   const DEMSimParams* sim_params,
                   const DEMDataDT* data,
                   bool mass_properties_are_jitified,
                   unsigned int wildcard_index,
                   cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    const size_t blocks = (count + OWNER_DATA_RETRIEVAL_BLOCK_SIZE - 1) / OWNER_DATA_RETRIEVAL_BLOCK_SIZE;
    PackOwnerDataKernel<<<blocks, OWNER_DATA_RETRIEVAL_BLOCK_SIZE, 0, stream>>>(
        output, field, owner_begin, count, sim_params, data, mass_properties_are_jitified, wildcard_index);
    DEME_GPU_DEBUG_SYNC(stream);
}

size_t OwnerDataElementSize(OwnerDataField field) {
    switch (field) {
        case OwnerDataField::ORIENTATION:
            return sizeof(float4);
        case OwnerDataField::FAMILY:
            return sizeof(unsigned int);
        case OwnerDataField::MASS:
        case OwnerDataField::WILDCARD:
            return sizeof(float);
        default:
            return sizeof(float3);
    }
}

void ValidateOwnerOrientations(const float4* input, size_t count, unsigned int* invalid, cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    const size_t blocks = (count + OWNER_DATA_RETRIEVAL_BLOCK_SIZE - 1) / OWNER_DATA_RETRIEVAL_BLOCK_SIZE;
    ValidateOwnerOrientationsKernel<<<blocks, OWNER_DATA_RETRIEVAL_BLOCK_SIZE, 0, stream>>>(input, count, invalid);
    DEME_GPU_DEBUG_SYNC(stream);
}

void UnpackOwnerState(const void* input,
                      OwnerStateField field,
                      bodyID_t owner_begin,
                      size_t count,
                      const DEMSimParams* sim_params,
                      DEMDataDT* data,
                      cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    const size_t blocks = (count + OWNER_DATA_RETRIEVAL_BLOCK_SIZE - 1) / OWNER_DATA_RETRIEVAL_BLOCK_SIZE;
    UnpackOwnerStateKernel<<<blocks, OWNER_DATA_RETRIEVAL_BLOCK_SIZE, 0, stream>>>(input, field, owner_begin, count,
                                                                                   sim_params, data);
    DEME_GPU_DEBUG_SYNC(stream);
}

void ReduceOwnerContactWrenches(float3* forces,
                                float3* torques,
                                bodyID_t owner_begin,
                                size_t count,
                                const DEMDataDT* data,
                                size_t num_contacts,
                                cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    DEME_GPU_CALL(cudaMemsetAsync(forces, 0, count * sizeof(float3), stream));
    DEME_GPU_CALL(cudaMemsetAsync(torques, 0, count * sizeof(float3), stream));
    if (num_contacts == 0) {
        return;
    }
    const size_t blocks = (num_contacts + OWNER_DATA_RETRIEVAL_BLOCK_SIZE - 1) / OWNER_DATA_RETRIEVAL_BLOCK_SIZE;
    ReduceOwnerContactWrenchesKernel<<<blocks, OWNER_DATA_RETRIEVAL_BLOCK_SIZE, 0, stream>>>(
        forces, torques, owner_begin, count, data, num_contacts);
    DEME_GPU_DEBUG_SYNC(stream);
}

}  // namespace deme
