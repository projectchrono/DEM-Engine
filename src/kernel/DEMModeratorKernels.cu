// DEM kernels that does some wildcard stuff, such as modifying the system as per user instruction
#include <kernel/DEMHelperKernels.cu>
#include <DEM/DEMDefines.h>

__global__ void applyFamilyChanges(sgps::DEMDataDT* granData, float h, float t) {
    // _nDistinctMassProperties_  elements are in these arrays
    const float moiX[] = {_moiX_};
    const float moiY[] = {_moiY_};
    const float moiZ[] = {_moiZ_};
    const float MassProperties[] = {_MassProperties_};

    sgps::bodyID_t thisClump = blockIdx.x * blockDim.x + threadIdx.x;
    if (thisClump < _nOwnerBodies_) {
        // The user may make references to owner positions, velocities, accelerations and simulation time
        double3 pos;
        float3 vel, acc;
        float mass;
        sgps::family_t family_code = granData->familyID[thisClump];
        sgps::inertiaOffset_t myMassOffset = granData->inertiaPropOffsets[thisClump];
        mass = MassProperties[myMassOffset];
        voxelID2Position<double, sgps::voxelID_t, sgps::subVoxelPos_t>(
            pos.x, pos.y, pos.z, granData->voxelID[thisClump], granData->locX[thisClump], granData->locY[thisClump],
            granData->locZ[thisClump], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        vel.x = granData->vX[thisClump];
        vel.y = granData->vY[thisClump];
        vel.z = granData->vZ[thisClump];
        acc.x = granData->aX[thisClump];
        acc.y = granData->aY[thisClump];
        acc.z = granData->aZ[thisClump];

        // Carry out user's instructions
        { _familyChangeRules_; }
    }
}
