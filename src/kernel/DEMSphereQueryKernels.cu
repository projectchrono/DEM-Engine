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
        double ownerX, ownerY, ownerZ;
        // Get my component offset info from either jitified arrays or global memory
        // Outputs myRelPos, myRadius
        // Use an input named exactly `sphereID' which is the id of this sphere component
        { _componentAcqStrat_; }

        voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
            ownerX, ownerY, ownerZ, granData->voxelID[myOwner], granData->locX[myOwner], granData->locY[myOwner],
            granData->locZ[myOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        // Keep these scalar names available for injected inspector snippets. Built-in and user-provided sphere
        // query code has historically referenced oriQw/oriQx/oriQy/oriQz directly.
        const deme::oriQ_t oriQw = granData->oriQw[myOwner];
        const deme::oriQ_t oriQx = granData->oriQx[myOwner];
        const deme::oriQ_t oriQy = granData->oriQy[myOwner];
        const deme::oriQ_t oriQz = granData->oriQz[myOwner];
        const float4 oriQ = make_float4(oriQx, oriQy, oriQz, oriQw);
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
