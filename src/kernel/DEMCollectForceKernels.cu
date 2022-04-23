// DEM force computation related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

__global__ void cashInOwnerIndex(sgps::bodyID_t* idOwner,
                                 sgps::bodyID_t* id,
                                 sgps::bodyID_t* ownerClumpBody,
                                 sgps::contact_t* contactType,
                                 size_t nContactPairs) {
    const sgps::bodyID_t objOwner[_nAnalGMSafe_] = {_objOwner_};
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        sgps::bodyID_t thisBodyID = id[myID];
        sgps::contact_t thisCntType = contactType[myID];
        if (thisCntType == sgps::DEM_SPHERE_SPHERE_CONTACT) {
            idOwner[myID] = ownerClumpBody[thisBodyID];
        } else {
            // This is a sphere--analytical geometry contactm its owner is jitified
            idOwner[myID] = objOwner[thisBodyID];
        }
    }
}

__global__ void cashInMassMoiIndex(float* massOwner,
                                   float3* moiOwner,
                                   sgps::clumpBodyInertiaOffset_t* inertiaPropOffsets,
                                   sgps::bodyID_t* idOwner,
                                   size_t nContactPairs) {
    // CUDA does not support initializing shared arrays, so we have to manually load them
    __shared__ float moiX[_nDistinctClumpBodyTopologies_];
    __shared__ float moiY[_nDistinctClumpBodyTopologies_];
    __shared__ float moiZ[_nDistinctClumpBodyTopologies_];
    __shared__ float ClumpMasses[_nDistinctClumpBodyTopologies_];
    if (threadIdx.x < _nActiveLoadingThreads_) {
        const float jitifiedMoiX[_nDistinctClumpBodyTopologies_] = {_moiX_};
        const float jitifiedMoiY[_nDistinctClumpBodyTopologies_] = {_moiY_};
        const float jitifiedMoiZ[_nDistinctClumpBodyTopologies_] = {_moiZ_};
        const float jitifiedMass[_nDistinctClumpBodyTopologies_] = {_ClumpMasses_};
        for (sgps::clumpBodyInertiaOffset_t i = threadIdx.x; i < _nDistinctClumpBodyTopologies_;
             i += _nActiveLoadingThreads_) {
            ClumpMasses[i] = jitifiedMass[i];
            moiX[i] = jitifiedMoiX[i];
            moiY[i] = jitifiedMoiY[i];
            moiZ[i] = jitifiedMoiZ[i];
        }
    }
    __syncthreads();

    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        sgps::bodyID_t thisOwnerID = idOwner[myID];
        sgps::clumpBodyInertiaOffset_t myMassOffset = inertiaPropOffsets[thisOwnerID];
        float3 moi;
        moi.x = moiX[myMassOffset];
        moi.y = moiY[myMassOffset];
        moi.z = moiZ[myMassOffset];
        massOwner[myID] = ClumpMasses[myMassOffset];
        moiOwner[myID] = moi;
    }
}

// computes a ./ b
__global__ void forceToAcc(float3* acc,
                           float3* F,
                           sgps::bodyID_t* owner,
                           float modifier,
                           size_t n,
                           sgps::clumpBodyInertiaOffset_t* inertiaPropOffsets) {
    // CUDA does not support initializing shared arrays, so we have to manually load them
    __shared__ float ClumpMasses[_nDistinctClumpBodyTopologies_];
    if (threadIdx.x < _nActiveLoadingThreads_) {
        const float jitifiedMass[_nDistinctClumpBodyTopologies_] = {_ClumpMasses_};
        for (sgps::clumpBodyInertiaOffset_t i = threadIdx.x; i < _nDistinctClumpBodyTopologies_;
             i += _nActiveLoadingThreads_) {
            ClumpMasses[i] = jitifiedMass[i];
        }
    }
    __syncthreads();
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        sgps::bodyID_t thisOwnerID = owner[myID];
        sgps::clumpBodyInertiaOffset_t myMassOffset = inertiaPropOffsets[thisOwnerID];
        float myMass = ClumpMasses[myMassOffset];
        acc[myID] = F[myID] * modifier / myMass;
    }
}

// computes cross(a, b) ./ c
__global__ void forceToAngAcc(float3* angAcc,
                              float3* cntPnt,
                              float3* F,
                              sgps::bodyID_t* owner,
                              float modifier,
                              size_t n,
                              sgps::clumpBodyInertiaOffset_t* inertiaPropOffsets) {
    // CUDA does not support initializing shared arrays, so we have to manually load them
    __shared__ float moiX[_nDistinctClumpBodyTopologies_];
    __shared__ float moiY[_nDistinctClumpBodyTopologies_];
    __shared__ float moiZ[_nDistinctClumpBodyTopologies_];
    if (threadIdx.x < _nActiveLoadingThreads_) {
        const float jitifiedMoiX[_nDistinctClumpBodyTopologies_] = {_moiX_};
        const float jitifiedMoiY[_nDistinctClumpBodyTopologies_] = {_moiY_};
        const float jitifiedMoiZ[_nDistinctClumpBodyTopologies_] = {_moiZ_};
        for (sgps::clumpBodyInertiaOffset_t i = threadIdx.x; i < _nDistinctClumpBodyTopologies_;
             i += _nActiveLoadingThreads_) {
            moiX[i] = jitifiedMoiX[i];
            moiY[i] = jitifiedMoiY[i];
            moiZ[i] = jitifiedMoiZ[i];
        }
    }
    __syncthreads();
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        sgps::bodyID_t thisOwnerID = owner[myID];
        sgps::clumpBodyInertiaOffset_t myMassOffset = inertiaPropOffsets[thisOwnerID];
        float3 moi;
        moi.x = moiX[myMassOffset];
        moi.y = moiY[myMassOffset];
        moi.z = moiZ[myMassOffset];
        auto myCntPnt = cntPnt[myID];
        auto myF = F[myID] * modifier;
        angAcc[myID] = cross(myCntPnt, myF) / moi;
    }
}

// Place information to an array based on an index array and a value array
__global__ void stashElem(float* out1, float* out2, float* out3, sgps::bodyID_t* index, float3* value, size_t n) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        // my_index is unique, no race condition
        sgps::bodyID_t my_index = index[myID];
        float3 my_value = value[myID];
        out1[my_index] += my_value.x;
        out2[my_index] += my_value.y;
        out3[my_index] += my_value.z;
    }
}
