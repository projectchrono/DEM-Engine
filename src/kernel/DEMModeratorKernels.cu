// DEM kernels that does some wildcard stuff, such as modifying the system as per user instruction
#include <kernel/DEMHelperKernels.cu>
#include <DEM/DEMDefines.h>

// Mass properties are below, if jitified mass properties are in use
_massDefs_;
_moiDefs_;

__global__ void applyFamilyChanges(smug::DEMDataDT* granData, size_t nOwnerBodies, float h, float t) {
    smug::bodyID_t myOwner = blockIdx.x * blockDim.x + threadIdx.x;
    if (myOwner < nOwnerBodies) {
        // The user may make references to owner positions, velocities, accelerations and simulation time
        double3 pos;
        float3 vel, acc;
        float mass;
        smug::family_t family_code = granData->familyID[myOwner];
        // Get my mass info from either jitified arrays or global memory
        // Outputs myMass
        // Use an input named exactly `myOwner' which is the id of this owner
        {
            float myMass;
            _massAcqStrat_;
            mass = myMass;
        }
        voxelID2Position<double, smug::voxelID_t, smug::subVoxelPos_t>(
            pos.x, pos.y, pos.z, granData->voxelID[myOwner], granData->locX[myOwner], granData->locY[myOwner],
            granData->locZ[myOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        vel.x = granData->vX[myOwner];
        vel.y = granData->vY[myOwner];
        vel.z = granData->vZ[myOwner];
        acc.x = granData->aX[myOwner];
        acc.y = granData->aY[myOwner];
        acc.z = granData->aZ[myOwner];

        // Carry out user's instructions
        { _familyChangeRules_; }
    }
}
