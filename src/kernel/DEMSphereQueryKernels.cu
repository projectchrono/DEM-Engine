// DEM kernels used for quarrying (statistical) information from the current simulation system
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>
_kernelIncludes_;

// If clump templates are jitified, they will be below
_clumpTemplateDefs_;

// Mass properties are below... but inspecting spheres doesn't seem to require mass or MOI
// _massDefs_;
// _moiDefs_;

DEME_KERNEL void inspectSphereProperty(deme::DEMDataDT* granData,
                                       deme::DEMSimParams* simParams,
                                       float* quantity,
                                       deme::notStupidBool_t* not_in_region,
                                       size_t nSpheres,
                                       deme::ownerType_t owner_type) {
    size_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < nSpheres) {
        // Get my owner ID
        deme::bodyID_t myOwner = granData->ownerClumpBody[sphereID];
        float3 myRelPos;
        float myRadius;
        float4 oriQ;
        double ownerX, ownerY, ownerZ;
        // Get my component offset info from either jitified arrays or global memory
        // Outputs myRelPos, myRadius
        // Use an input named exactly `sphereID' which is the id of this sphere component
        { _componentAcqStrat_; }

        voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[myOwner], granData->locX[myOwner], granData->locY[myOwner],
            granData->locZ[myOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        oriQ.w = granData->oriQw[myOwner];
        oriQ.x = granData->oriQx[myOwner];
        oriQ.y = granData->oriQy[myOwner];
        oriQ.z = granData->oriQz[myOwner];
        applyOriQToVector3(myRelPos, oriQ);

        // Use sphereXYZ to determine if this sphere is in the region that should be counted
        // And don't forget adding LBF as an offset
        float X = ownerX + myRelPos.x + simParams->LBFX;
        float Y = ownerY + myRelPos.y + simParams->LBFY;
        float Z = ownerZ + myRelPos.z + simParams->LBFZ;
        { _inRegionPolicy_; }

        // Now it's a problem of what quantity to query
        { _quantityQueryProcess_; }
    }
}
