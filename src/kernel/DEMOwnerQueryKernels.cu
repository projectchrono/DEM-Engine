// DEM kernels used for quarrying (statistical) information from the current simulation system
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>
_kernelIncludes_;

// Mass properties are below, if jitified mass properties are in use
_massDefs_;
_moiDefs_;
_volumeDefs_;

__global__ void inspectOwnerProperty(deme::DEMDataDT* granData,
                                     deme::DEMSimParams* simParams,
                                     float* quantity,
                                     deme::notStupidBool_t* not_in_region,
                                     size_t nOwnerBodies,
                                     deme::ownerType_t owner_type) {
    deme::bodyID_t myOwner = blockIdx.x * blockDim.x + threadIdx.x;
    if (myOwner < nOwnerBodies) {
        deme::ownerType_t myType = granData->ownerTypes[myOwner];
        if (myType & owner_type) {
            float oriQw, oriQx, oriQy, oriQz;
            double ownerX, ownerY, ownerZ;
            float myMass;
            float3 myMOI;
            // Get my mass info from either jitified arrays or global memory
            // Outputs myMass
            // Use an input named exactly `myOwner' which is the id of this owner
            { _massAcqStrat_; }

            // Get my mass info from either jitified arrays or global memory
            // Outputs myMOI
            // Use an input named exactly `myOwner' which is the id of this owner
            { _moiAcqStrat_; }

            voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[myOwner], granData->locX[myOwner], granData->locY[myOwner],
                granData->locZ[myOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            oriQw = granData->oriQw[myOwner];
            oriQx = granData->oriQx[myOwner];
            oriQy = granData->oriQy[myOwner];
            oriQz = granData->oriQz[myOwner];

            // Use sphereXYZ to determine if this sphere is in the region that should be counted
            // And don't forget adding LBF as an offset
            float X = ownerX + simParams->LBFX;
            float Y = ownerY + simParams->LBFY;
            float Z = ownerZ + simParams->LBFZ;
            { _inRegionPolicy_; }

            // Now it's a problem of what quantity to query
            { _quantityQueryProcess_; }
        } else {
            not_in_region[myOwner] = 1;
        }
    }
}
