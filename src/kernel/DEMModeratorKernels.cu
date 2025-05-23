// DEM kernels that does some wildcard stuff, such as modifying the system as per user instruction
#include <DEMHelperKernels.cuh>
#include <DEM/Defines.h>
_kernelIncludes_;

// Mass properties are below, if jitified mass properties are in use
_massDefs_;
_moiDefs_;

__global__ void applyFamilyChanges(deme::DEMSimParams* simParams, deme::DEMDataDT* granData, size_t nOwnerBodies) {
    deme::bodyID_t myOwner = blockIdx.x * blockDim.x + threadIdx.x;
    if (myOwner < nOwnerBodies) {
        // The user may make references to owner positions, velocities, accelerations and simulation time
        double3 pos;
        float3 vel, acc;
        float mass;
        deme::family_t family_code = granData->familyID[myOwner];
        // Get my mass info from either jitified arrays or global memory
        // Outputs myMass
        // Use an input named exactly `myOwner' which is the id of this owner
        {
            float myMass;
            _massAcqStrat_;
            mass = myMass;
        }
        voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
            pos.x, pos.y, pos.z, granData->voxelID[myOwner], granData->locX[myOwner], granData->locY[myOwner],
            granData->locZ[myOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        pos.x += simParams->LBFX;
        pos.y += simParams->LBFY;
        pos.z += simParams->LBFZ;

        vel.x = granData->vX[myOwner];
        vel.y = granData->vY[myOwner];
        vel.z = granData->vZ[myOwner];
        acc.x = granData->aX[myOwner];
        acc.y = granData->aY[myOwner];
        acc.z = granData->aZ[myOwner];

        // Standardize names...
        double X = pos.x;
        double Y = pos.y;
        double Z = pos.z;
        float vX = vel.x;
        float vY = vel.y;
        float vZ = vel.z;
        float accX = acc.x;
        float accY = acc.y;
        float accZ = acc.z;

        float ts = simParams->h;
        float time = simParams->timeElapsed;

        // Carry out user's instructions
        { _familyChangeRules_; }
    }
}
