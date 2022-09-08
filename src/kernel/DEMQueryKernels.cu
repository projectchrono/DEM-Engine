// DEM kernels used for quarrying (statistical) information from the current simulation system
#include <DEM/DEMDefines.h>
#include <kernel/DEMHelperKernels.cu>

// Mass properties are below, if jitified mass properties are in use
_massDefs_;
_moiDefs_;

__global__ void computeKE(smug::DEMDataDT* granData, size_t nOwnerBodies, double* KE) {
    smug::bodyID_t myOwner = blockIdx.x * blockDim.x + threadIdx.x;
    if (myOwner < nOwnerBodies) {
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

        // First lin energy
        double myVX = granData->vX[myOwner];
        double myVY = granData->vY[myOwner];
        double myVZ = granData->vZ[myOwner];
        double myKE = 0.5 * myMass * (myVX * myVX + myVY * myVY + myVZ * myVZ);
        // Then rot energy
        myVX = granData->omgBarX[myOwner];
        myVY = granData->omgBarY[myOwner];
        myVZ = granData->omgBarZ[myOwner];
        myKE += 0.5 * ((double)myMOI.x * myVX * myVX + (double)myMOI.y * myVY * myVY + (double)myMOI.z * myVZ * myVZ);
        KE[myOwner] = myKE;
    }
}

__global__ void inspectSphereProperty(smug::DEMDataDT* granData,
                                      smug::DEMSimParams* simParams,
                                      float* quantity,
                                      smug::notStupidBool_t* in_region,
                                      size_t nSpheres) {
    size_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < nSpheres) {
        // Get my owner ID
        smug::bodyID_t myOwner = granData->ownerClumpBody[sphereID];
        float myRelPosX, myRelPosY, myRelPosZ;
        float myRadius;
        float oriQw, oriQx, oriQy, oriQz;
        double ownerX, ownerY, ownerZ;
        // Get my component offset info from either jitified arrays or global memory
        // Outputs myRelPosXYZ, myRadius
        // Use an input named exactly `sphereID' which is the id of this sphere component
        { _componentAcqStrat_; }

        voxelID2Position<double, smug::voxelID_t, smug::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[myOwner], granData->locX[myOwner], granData->locY[myOwner],
            granData->locZ[myOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        oriQw = granData->oriQw[myOwner];
        oriQx = granData->oriQx[myOwner];
        oriQy = granData->oriQy[myOwner];
        oriQz = granData->oriQz[myOwner];
        applyOriQToVector3<float, smug::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, oriQw, oriQx, oriQy, oriQz);

        // Use sphereXYZ to determine if this sphere is in the region that should be counted
        // And don't forget adding LBF as an offset
        float sphereX = ownerX + myRelPosX + simParams->LBFX;
        float sphereY = ownerY + myRelPosY + simParams->LBFY;
        float sphereZ = ownerZ + myRelPosZ + simParams->LBFZ;
        { _inRegionPolicy_; }

        // Now it's a problem of what quantity to query
        { _quantityQueryProcess_; }
    }
}
