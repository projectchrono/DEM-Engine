// DEM kernels that does things such as modifying the system as per user instruction
#include <DEMHelperKernels.cuh>
#include <DEMCollisionKernels_SphSph.cuh>
#include <DEM/Defines.h>
_kernelIncludes_;

// Definitions of analytical entites and clump components are below
_clumpTemplateDefs_;
_analyticalEntityDefs_;

__device__ __forceinline__ float getApproxAbsVel(deme::DEMSimParams* simParams,
                                                 const deme::bodyID_t& ownerID,
                                                 const float* absVel_owner,
                                                 const float* absAngVel_owner,
                                                 const float3& myRelPos) {
    float abs_v = absVel_owner[ownerID];
    float abs_angv = absAngVel_owner[ownerID];
    if (!isfinite(abs_v) || !isfinite(abs_angv)) {
        // May produce messy error messages, but it's still good to know what entities went wrong
        DEME_ABORT_KERNEL(
            "Absolute velocity or angular velocity for ownerID %llu is infinite (it's worse than "
            "max-velocity-exceeded-allowance).\n",
            static_cast<unsigned long long>(ownerID));
    }
    // Compute this primitive sphere's velocity, optionally including the owner's angular velocity contribution.
    float vel = abs_v;
    if (simParams->useAngVelMargin) {
        // This is an estimation as no direction is considered.
        vel += length(myRelPos) * abs_angv;
    }

    vel = fminf(vel, simParams->dyn.approxMaxVel);
    return vel;
}

__global__ void computeMarginFromAbsv_implSph(deme::DEMSimParams* simParams,
                                              deme::DEMDataKT* granData,
                                              const float* absVel_owner,
                                              const float* absAngVel_owner,
                                              float* ts,
                                              unsigned int* maxDrift,
                                              size_t n) {
    size_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < n) {
        float3 myRelPos;
        float myRadius;  // Not actually used
        // Get my component offset info from either jitified arrays or global memory
        // Outputs myRelPos, myRadius
        // Use an input named exactly `sphereID' which is the id of this sphere component
        { _componentAcqStrat_; }

        deme::bodyID_t ownerID = granData->ownerClumpBody[sphereID];
        float vel = getApproxAbsVel(simParams, ownerID, absVel_owner, absAngVel_owner, myRelPos);
        unsigned int my_family = granData->familyID[ownerID];
        granData->marginSizeSphere[sphereID] =
            (double)(vel * simParams->dyn.expSafetyMulti + simParams->dyn.expSafetyAdder) * (*ts) * (*maxDrift) +
            granData->familyExtraMarginSize[my_family];
    }
}

__global__ void computeMarginFromAbsv_implTri(deme::DEMSimParams* simParams,
                                              deme::DEMDataKT* granData,
                                              const float* absVel_owner,
                                              const float* absAngVel_owner,
                                              float* ts,
                                              unsigned int* maxDrift,
                                              double* maxTriTriPenetration,
                                              bool meshUniversalContact,
                                              size_t n) {
    size_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID < n) {
        float3 triBNode1 = granData->relPosNode1[triID];
        float3 triBNode2 = granData->relPosNode2[triID];
        float3 triBNode3 = granData->relPosNode3[triID];
        float3 myRelPos = triangleCentroid<float3>(triBNode1, triBNode2, triBNode3);

        deme::bodyID_t ownerID = granData->ownerTriMesh[triID];
        float vel = getApproxAbsVel(simParams, ownerID, absVel_owner, absAngVel_owner, myRelPos);
        unsigned int my_family = granData->familyID[ownerID];

        // Compute additional margin based on max tri-tri penetration if meshUniversalContact is enabled. This is needed
        // as our meshed particle representation is surface only, so we need to account for existing penetration length
        // in our future-proof contact detection, always.
        double penetrationMargin = *maxTriTriPenetration;
        penetrationMargin = (meshUniversalContact && penetrationMargin > 0.0) ? penetrationMargin : 0.0;
        // Clamp penetration margin to the maximum allowed value to prevent super large margins
        if (penetrationMargin > simParams->capTriTriPenetration) {
            penetrationMargin = simParams->capTriTriPenetration;
        }

        granData->marginSizeTriangle[triID] =
            (double)(vel * simParams->dyn.expSafetyMulti + simParams->dyn.expSafetyAdder) * (*ts) * (*maxDrift) +
            penetrationMargin + granData->familyExtraMarginSize[my_family];
    }
}

__global__ void computeMarginFromAbsv_implAnal(deme::DEMSimParams* simParams,
                                               deme::DEMDataKT* granData,
                                               const float* absVel_owner,
                                               const float* absAngVel_owner,
                                               float* ts,
                                               unsigned int* maxDrift,
                                               size_t n) {
    size_t objID = blockIdx.x * blockDim.x + threadIdx.x;
    if (objID < n) {
        deme::bodyID_t ownerID = granData->ownerAnalBody[objID];
        float3 myRelPos = make_float3(objRelPosX[objID], objRelPosY[objID], objRelPosZ[objID]);

        float vel = getApproxAbsVel(simParams, ownerID, absVel_owner, absAngVel_owner, myRelPos);
        unsigned int my_family = granData->familyID[ownerID];
        granData->marginSizeAnalytical[objID] =
            (double)(vel * simParams->dyn.expSafetyMulti + simParams->dyn.expSafetyAdder) * (*ts) * (*maxDrift) +
            granData->familyExtraMarginSize[my_family];
    }
}
