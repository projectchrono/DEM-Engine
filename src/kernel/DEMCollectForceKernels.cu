// DEM force computation related custom kernels
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>
_kernelIncludes_;

// For analytical entities' owners
__constant__ __device__ deme::bodyID_t objOwner[] = {_objOwner_};
// Mass properties are below, if jitified mass properties are in use
_massDefs_;
_moiDefs_;

__global__ void cashInOwnerIndex(deme::bodyID_t* idAOwner,
                                 deme::bodyID_t* idBOwner,
                                 deme::DEMDataDT* granData,
                                 size_t nContactPairs) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        deme::bodyID_t idGeoA = granData->idGeometryA[myID];
        deme::bodyID_t idGeoB = granData->idGeometryB[myID];
        deme::contact_t thisCntType = granData->contactType[myID];
        idAOwner[myID] = DEME_GET_GEO_OWNER_ID(idGeoA, deme::decodeTypeA(thisCntType));
        idBOwner[myID] = DEME_GET_GEO_OWNER_ID(idGeoB, deme::decodeTypeB(thisCntType));
    }
}

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
