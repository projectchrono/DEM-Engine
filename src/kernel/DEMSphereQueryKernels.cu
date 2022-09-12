// DEM kernels used for quarrying (statistical) information from the current simulation system
#include <DEM/DEMDefines.h>
#include <kernel/DEMHelperKernels.cu>

// Mass properties are below... but inspecting spheres doesn't seem to require mass or MOI
// _massDefs_;
// _moiDefs_;

__global__ void inspectSphereProperty(smug::DEMDataDT* granData,
                                      smug::DEMSimParams* simParams,
                                      float* quantity,
                                      smug::notStupidBool_t* not_in_region,
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

        voxelIDToPosition<double, smug::voxelID_t, smug::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[myOwner], granData->locX[myOwner], granData->locY[myOwner],
            granData->locZ[myOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        oriQw = granData->oriQw[myOwner];
        oriQx = granData->oriQx[myOwner];
        oriQy = granData->oriQy[myOwner];
        oriQz = granData->oriQz[myOwner];
        applyOriQToVector3<float, smug::oriQ_t>(myRelPosX, myRelPosY, myRelPosZ, oriQw, oriQx, oriQy, oriQz);

        // Use sphereXYZ to determine if this sphere is in the region that should be counted
        // And don't forget adding LBF as an offset
        float X = ownerX + myRelPosX + simParams->LBFX;
        float Y = ownerY + myRelPosY + simParams->LBFY;
        float Z = ownerZ + myRelPosZ + simParams->LBFZ;
        { _inRegionPolicy_; }

        // Now it's a problem of what quantity to query
        { _quantityQueryProcess_; }
    }
}
