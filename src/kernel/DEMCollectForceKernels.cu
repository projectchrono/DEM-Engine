// DEM force computation related custom kernels
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>
_kernelIncludes_;

// For analytical entities' owners
__constant__ __device__ deme::bodyID_t objOwner[] = {_objOwner_};
// Mass properties are below, if jitified mass properties are in use
_massDefs_;
_moiDefs_;

__global__ void cashInOwnerIndexA(deme::bodyID_t* idOwner,
                                  deme::bodyID_t* id,
                                  deme::bodyID_t* ownerClumpBody,
                                  deme::contact_t* contactType,
                                  size_t nContactPairs) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        deme::bodyID_t thisBodyID = id[myID];
        idOwner[myID] = ownerClumpBody[thisBodyID];
    }
}

__global__ void cashInOwnerIndexB(deme::bodyID_t* idOwner,
                                  deme::bodyID_t* id,
                                  deme::bodyID_t* ownerClumpBody,
                                  deme::bodyID_t* ownerMesh,
                                  deme::contact_t* contactType,
                                  size_t nContactPairs) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        deme::bodyID_t thisBodyID = id[myID];
        deme::contact_t thisCntType = contactType[myID];
        if (thisCntType == deme::SPHERE_SPHERE_CONTACT) {
            idOwner[myID] = ownerClumpBody[thisBodyID];
        } else if (thisCntType == deme::SPHERE_MESH_CONTACT) {
            idOwner[myID] = ownerMesh[thisBodyID];
        } else {
            // This is a sphere--analytical geometry contact, its owner is jitified
            idOwner[myID] = objOwner[thisBodyID];
        }
    }
}

/*
__global__ void cashInMassMoiIndex(float* massOwner,
                                   float3* moiOwner,
                                   deme::inertiaOffset_t* inertiaPropOffsets,
                                   deme::bodyID_t* idOwner,
                                   size_t nContactPairs) {
    // _nDistinctMassProperties_  elements are in these arrays
    const float moiX[] = {_moiX_};
    const float moiY[] = {_moiY_};
    const float moiZ[] = {_moiZ_};
    const float MassProperties[] = {_MassProperties_};

    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        deme::bodyID_t thisOwnerID = idOwner[myID];
        deme::inertiaOffset_t myMassOffset = inertiaPropOffsets[thisOwnerID];
        float3 moi;
        moi.x = moiX[myMassOffset];
        moi.y = moiY[myMassOffset];
        moi.z = moiZ[myMassOffset];
        massOwner[myID] = MassProperties[myMassOffset];
        moiOwner[myID] = moi;
    }
}
*/

// computes a ./ b
__global__ void forceToAcc(float3* acc,
                           float3* F,
                           deme::bodyID_t* owner,
                           float modifier,
                           size_t n,
                           deme::DEMDataDT* granData) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        float myMass;
        const deme::bodyID_t myOwner = owner[myID];
        // Get my mass info from either jitified arrays or global memory
        // Outputs myMass
        // Use an input named exactly `myOwner' which is the id of this owner
        { _massAcqStrat_; }
        acc[myID] = F[myID] * modifier / myMass;
    }
}

// computes cross(a, b) ./ c
__global__ void forceToAngAcc(float3* angAcc,
                              float3* cntPnt,
                              deme::oriQ_t* oriQw,
                              deme::oriQ_t* oriQx,
                              deme::oriQ_t* oriQy,
                              deme::oriQ_t* oriQz,
                              float3* F,
                              float3* torque_inForceForm,
                              deme::bodyID_t* owner,
                              float modifier,
                              size_t n,
                              deme::DEMDataDT* granData) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const deme::bodyID_t myOwner = owner[myID];
        float3 myMOI;
        // Get my mass info from either jitified arrays or global memory
        // Outputs myMOI
        // Use an input named exactly `myOwner' which is the id of this owner
        { _moiAcqStrat_; }
        const deme::oriQ_t myOriQw = oriQw[myOwner];
        const deme::oriQ_t myOriQx = oriQx[myOwner];
        const deme::oriQ_t myOriQy = oriQy[myOwner];
        const deme::oriQ_t myOriQz = oriQz[myOwner];

        float3 myCntPnt = cntPnt[myID];
        // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque only, not
        // linear velocity
        float3 myF = (F[myID] + torque_inForceForm[myID]) * modifier;
        // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
        applyOriQToVector3<float, deme::oriQ_t>(myF.x, myF.y, myF.z, myOriQw, -myOriQx, -myOriQy, -myOriQz);
        angAcc[myID] = cross(myCntPnt, myF) / myMOI;
    }
}

// Place information to an array based on an index array and a value array
__global__ void stashElem(float* out1, float* out2, float* out3, deme::bodyID_t* index, float3* value, size_t n) {
    deme::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        // my_index is unique, no race condition
        deme::bodyID_t my_index = index[myID];
        float3 my_value = value[myID];
        out1[my_index] += my_value.x;
        out2[my_index] += my_value.y;
        out3[my_index] += my_value.z;
    }
}
