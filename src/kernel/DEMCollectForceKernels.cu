// DEM force computation related custom kernels
#include <DEM/DEMDefines.h>
#include <kernel/DEMHelperKernels.cu>

__global__ void cashInOwnerIndexA(sgps::bodyID_t* idOwner,
                                  sgps::bodyID_t* id,
                                  sgps::bodyID_t* ownerClumpBody,
                                  sgps::contact_t* contactType,
                                  size_t nContactPairs) {
    sgps::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        sgps::bodyID_t thisBodyID = id[myID];
        idOwner[myID] = ownerClumpBody[thisBodyID];
    }
}

__global__ void cashInOwnerIndexB(sgps::bodyID_t* idOwner,
                                  sgps::bodyID_t* id,
                                  sgps::bodyID_t* ownerClumpBody,
                                  sgps::contact_t* contactType,
                                  size_t nContactPairs) {
    // May have analytical entities in it
    const sgps::bodyID_t objOwner[_nAnalGMSafe_] = {_objOwner_};

    sgps::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        sgps::bodyID_t thisBodyID = id[myID];
        sgps::contact_t thisCntType = contactType[myID];
        if (thisCntType == sgps::DEM_SPHERE_SPHERE_CONTACT) {
            idOwner[myID] = ownerClumpBody[thisBodyID];
        } else {
            // This is a sphere--analytical geometry contact, its owner is jitified
            idOwner[myID] = objOwner[thisBodyID];
        }
    }
}

__global__ void cashInMassMoiIndex(float* massOwner,
                                   float3* moiOwner,
                                   sgps::inertiaOffset_t* inertiaPropOffsets,
                                   sgps::bodyID_t* idOwner,
                                   size_t nContactPairs) {
    // _nDistinctMassProperties_  elements are in these arrays
    const float moiX[] = {_moiX_};
    const float moiY[] = {_moiY_};
    const float moiZ[] = {_moiZ_};
    const float MassProperties[] = {_MassProperties_};

    sgps::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        sgps::bodyID_t thisOwnerID = idOwner[myID];
        sgps::inertiaOffset_t myMassOffset = inertiaPropOffsets[thisOwnerID];
        float3 moi;
        moi.x = moiX[myMassOffset];
        moi.y = moiY[myMassOffset];
        moi.z = moiZ[myMassOffset];
        massOwner[myID] = MassProperties[myMassOffset];
        moiOwner[myID] = moi;
    }
}

// computes a ./ b
__global__ void forceToAcc(float3* acc,
                           float3* F,
                           sgps::bodyID_t* owner,
                           float modifier,
                           size_t n,
                           sgps::inertiaOffset_t* inertiaPropOffsets) {
    // _nDistinctMassProperties_  elements are in these arrays
    const float MassProperties[] = {_MassProperties_};

    sgps::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const sgps::bodyID_t thisOwnerID = owner[myID];
        const sgps::inertiaOffset_t myMassOffset = inertiaPropOffsets[thisOwnerID];
        const float myMass = MassProperties[myMassOffset];
        acc[myID] = F[myID] * modifier / myMass;
    }
}

// computes cross(a, b) ./ c
__global__ void forceToAngAcc(float3* angAcc,
                              float3* cntPnt,
                              sgps::oriQ_t* oriQ0,
                              sgps::oriQ_t* oriQ1,
                              sgps::oriQ_t* oriQ2,
                              sgps::oriQ_t* oriQ3,
                              float3* F,
                              float3* torque_inForceForm,
                              sgps::bodyID_t* owner,
                              float modifier,
                              size_t n,
                              sgps::inertiaOffset_t* inertiaPropOffsets) {
    // _nDistinctMassProperties_  elements are in these arrays
    const float moiX[] = {_moiX_};
    const float moiY[] = {_moiY_};
    const float moiZ[] = {_moiZ_};

    sgps::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const sgps::bodyID_t thisOwnerID = owner[myID];
        const sgps::inertiaOffset_t myMassOffset = inertiaPropOffsets[thisOwnerID];
        const sgps::oriQ_t myOriQ0 = oriQ0[thisOwnerID];
        const sgps::oriQ_t myOriQ1 = oriQ1[thisOwnerID];
        const sgps::oriQ_t myOriQ2 = oriQ2[thisOwnerID];
        const sgps::oriQ_t myOriQ3 = oriQ3[thisOwnerID];
        float3 moi;
        moi.x = moiX[myMassOffset];
        moi.y = moiY[myMassOffset];
        moi.z = moiZ[myMassOffset];
        float3 myCntPnt = cntPnt[myID];
        // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque only, not
        // linear velocity
        float3 myF = (F[myID] + torque_inForceForm[myID]) * modifier;
        // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
        applyOriQToVector3<float, sgps::oriQ_t>(myF.x, myF.y, myF.z, myOriQ0, -myOriQ1, -myOriQ2, -myOriQ3);
        angAcc[myID] = cross(myCntPnt, myF) / moi;
    }
}

// Place information to an array based on an index array and a value array
__global__ void stashElem(float* out1, float* out2, float* out3, sgps::bodyID_t* index, float3* value, size_t n) {
    sgps::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        // my_index is unique, no race condition
        sgps::bodyID_t my_index = index[myID];
        float3 my_value = value[myID];
        out1[my_index] += my_value.x;
        out2[my_index] += my_value.y;
        out3[my_index] += my_value.z;
    }
}
