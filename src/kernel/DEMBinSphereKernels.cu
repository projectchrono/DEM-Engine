// DEM bin--sphere relations-related custom kernels
#include <DEM/Defines.h>
#include <DEMCollisionKernels_SphSph.cuh>
_kernelIncludes_;

// If clump templates are jitified, they will be below
_clumpTemplateDefs_;
// Definitions of analytical entites are below
_analyticalEntityDefs_;


__global__ void getNumberOfBinsEachSphereTouches(deme::DEMSimParams* simParams,
                                                 deme::DEMDataKT* granData,
                                                 deme::binsSphereTouches_t* numBinsSphereTouches,
                                                 deme::objID_t* numAnalGeoSphereTouches) {
    deme::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < simParams->nSpheresGM) {
        // Register sphere--analytical geometry contacts
        deme::objID_t contact_count = 0;
        // Sphere's family ID
        unsigned int sphFamilyNum;
        double3 myPosXYZ;
        double myRadius;
        {
            // My sphere voxel ID and my relPos
            deme::bodyID_t myOwnerID = granData->ownerClumpBody[sphereID];
            sphFamilyNum = granData->familyID[myOwnerID];
            float3 myRelPos;
            double3 ownerXYZ;

            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPos, myRadius (in CD kernels, radius needs to be expanded)
            // Use an input named exactly `sphereID' which is the id of this sphere component
            {
                _componentAcqStrat_;
                myRadius += granData->marginSizeSphere[sphereID];
            }

            {
                voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                    ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[myOwnerID], granData->locX[myOwnerID],
                    granData->locY[myOwnerID], granData->locZ[myOwnerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
                const float myOriQw = granData->oriQw[myOwnerID];
                const float myOriQx = granData->oriQx[myOwnerID];
                const float myOriQy = granData->oriQy[myOwnerID];
                const float myOriQz = granData->oriQz[myOwnerID];
                applyOriQToVector3<float, deme::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, myOriQw, myOriQx, myOriQy,
                                                        myOriQz);
                myPosXYZ = ownerXYZ + to_double3(myRelPos);
            }

            deme::binsSphereTouches_t numX, numY, numZ;
            {
                // The bin number that I live in (with fractions)?
                const double invBinSize = simParams->dyn.inv_binSize;
                const int    nbX     = (int)simParams->nbX;
                const int    nbY     = (int)simParams->nbY;
                const int    nbZ     = (int)simParams->nbZ;
                // Mixed precision counting with early exits
                const deme::AxisBounds bx = axis_bounds(myPosXYZ.x, myRadius, nbX, invBinSize);
                if (bx.imax < bx.imin) { 
                    numBinsSphereTouches[sphereID] = 0;
                    numAnalGeoSphereTouches[sphereID] = 0;   // ← dazu
                    return; 
                }
                const deme::AxisBounds by = axis_bounds(myPosXYZ.y, myRadius, nbY, invBinSize);
                if (by.imax < by.imin) { 
                    numBinsSphereTouches[sphereID] = 0;
                    numAnalGeoSphereTouches[sphereID] = 0;   // ← dazu
                    return; 
                }
                const deme::AxisBounds bz = axis_bounds(myPosXYZ.z, myRadius, nbZ, invBinSize);
                if (bz.imax < bz.imin) { 
                    numBinsSphereTouches[sphereID] = 0;
                    numAnalGeoSphereTouches[sphereID] = 0;   // ← dazu
                    return; 
                }
                numX = bx.imax - bx.imin + 1;
                numY = by.imax - by.imin + 1;
                numZ = bz.imax - bz.imin + 1;
                //// TODO: Add an error message if numX * numY * numZ > MAX(binsSphereTouches_t)
            }

            // Write the number of bins this sphere touches back to the global array
            numBinsSphereTouches[sphereID] = numX * numY * numZ;
            // printf("This sp takes num of bins: %u\n", numX * numY * numZ);
        }

        // Each sphere entity should also check if it overlaps with an analytical boundary-type geometry
        for (deme::objID_t objB = 0; objB < simParams->nAnalGM; objB++) {
            deme::bodyID_t objBOwner = objOwner[objB];
            // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
            unsigned int objFamilyNum = granData->familyID[objBOwner];
            unsigned int maskMatID =
                locateMaskPair<unsigned int>((unsigned int)sphFamilyNum, (unsigned int)objFamilyNum);
            // If marked no contact, skip ths iteration
            if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                continue;
            }
            double3 ownerXYZ;
            voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[objBOwner], granData->locX[objBOwner],
                granData->locY[objBOwner], granData->locZ[objBOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            const float ownerOriQw = granData->oriQw[objBOwner];
            const float ownerOriQx = granData->oriQx[objBOwner];
            const float ownerOriQy = granData->oriQy[objBOwner];
            const float ownerOriQz = granData->oriQz[objBOwner];
            float objBRelPosX = objRelPosX[objB];
            float objBRelPosY = objRelPosY[objB];
            float objBRelPosZ = objRelPosZ[objB];
            float objBRotX = objRotX[objB];
            float objBRotY = objRotY[objB];
            float objBRotZ = objRotZ[objB];
            applyOriQToVector3<float, deme::oriQ_t>(objBRelPosX, objBRelPosY, objBRelPosZ, ownerOriQw, ownerOriQx,
                                                    ownerOriQy, ownerOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(objBRotX, objBRotY, objBRotZ, ownerOriQw, ownerOriQx, ownerOriQy,
                                                    ownerOriQz);
            double3 objBPosXYZ = ownerXYZ + make_double3(objBRelPosX, objBRelPosY, objBRelPosZ);

            double overlapDepth, overlapArea;  // overlapArea matters not here
            deme::contact_t contact_type;
            {
                double3 cntPnt;  // cntPnt here is a placeholder
                float3 cntNorm;  // cntNorm is placeholder too
                contact_type = checkSphereEntityOverlap<double3, float, double>(
                    myPosXYZ, myRadius, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ),
                    objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB],
                    granData->marginSizeAnalytical[objB], cntPnt, cntNorm, overlapDepth, overlapArea);
            }
            // overlapDepth (which has both entities' full margins) needs to be larger than the smaller one of the two
            // added margin to be considered in-contact.
            double marginThres =
                (granData->familyExtraMarginSize[sphFamilyNum] < granData->familyExtraMarginSize[objFamilyNum])
                    ? granData->familyExtraMarginSize[sphFamilyNum]
                    : granData->familyExtraMarginSize[objFamilyNum];
            if (contact_type && overlapDepth > marginThres) {
                contact_count++;
            }
        }
        numAnalGeoSphereTouches[sphereID] = contact_count;
    }
}

__global__ void populateBinSphereTouchingPairs(deme::DEMSimParams* simParams,
                                               deme::DEMDataKT* granData,
                                               deme::binSphereTouchPairs_t* numBinsSphereTouchesScan,
                                               deme::binSphereTouchPairs_t* numAnalGeoSphereTouchesScan,
                                               deme::binID_t* binIDsEachSphereTouches,
                                               deme::bodyID_t* sphereIDsEachBinTouches,
                                               deme::bodyID_t* idGeoA,
                                               deme::bodyID_t* idGeoB,
                                               deme::contact_t* contactTypePrimitive) {
    deme::bodyID_t sphereID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphereID < simParams->nSpheresGM) {
        double3 myPosXYZ;
        double myRadius;
        unsigned int sphFamilyNum;

        {
            // My sphere voxel ID and my relPos
            deme::bodyID_t myOwnerID = granData->ownerClumpBody[sphereID];
            sphFamilyNum = granData->familyID[myOwnerID];
            float3 myRelPos;
            double3 ownerXYZ;

            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPos, myRadius (in CD kernels, radius needs to be expanded)
            // Use an input named exactly `sphereID' which is the id of this sphere component
            {
                _componentAcqStrat_;
                myRadius += granData->marginSizeSphere[sphereID];
            }

            // Get the offset of my spot where I should start writing back to the global bin--sphere pair registration
            // array
            deme::binSphereTouchPairs_t myReportOffset = numBinsSphereTouchesScan[sphereID];
            const deme::binSphereTouchPairs_t myReportOffset_end = numBinsSphereTouchesScan[sphereID + 1];

            {
                voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                    ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[myOwnerID], granData->locX[myOwnerID],
                    granData->locY[myOwnerID], granData->locZ[myOwnerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
                const float myOriQw = granData->oriQw[myOwnerID];
                const float myOriQx = granData->oriQx[myOwnerID];
                const float myOriQy = granData->oriQy[myOwnerID];
                const float myOriQz = granData->oriQz[myOwnerID];
                applyOriQToVector3<float, deme::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, myOriQw, myOriQx, myOriQy,
                                                        myOriQz);
                myPosXYZ = ownerXYZ + to_double3(myRelPos);
            }
            // The bin number that I live in (with fractions)?
            const double invBinSize = simParams->dyn.inv_binSize;
            const int nbX = (int)simParams->nbX;
            const int nbY = (int)simParams->nbY;
            const int nbZ = (int)simParams->nbZ;
            const int nbXY = nbX * nbY;
            // Bounds and avoid unnecessary work
            const deme::AxisBounds bx = axis_bounds(myPosXYZ.x, myRadius, nbX, invBinSize);
            if (bx.imax < bx.imin) { return; }
            const deme::AxisBounds by = axis_bounds(myPosXYZ.y, myRadius, nbY, invBinSize);
            if (by.imax < by.imin) { return; }
            const deme::AxisBounds bz = axis_bounds(myPosXYZ.z, myRadius, nbZ, invBinSize);
            if (bz.imax < bz.imin) { return; }
            // Alias names for the loop
            const int ix0 = bx.imin, ix1 = bx.imax;
            const int iy0 = by.imin, iy1 = by.imax;
            const int iz0 = bz.imin, iz1 = bz.imax;
            // Now, write the IDs of those bins that I touch, back to the global memory
            for (int k = iz0; k <= iz1; ++k) {
                const int baseZ = k * nbXY;
                for (int j = iy0; j <= iy1; ++j) {
                    const int baseYZ = baseZ + j * nbX;
                    for (int i = ix0; i <= ix1; ++i) {
                        if (myReportOffset >= myReportOffset_end) {
                            return; // No stepping on the next one's domain
                        }
                        const deme::binID_t binLin = (deme::binID_t)(baseYZ + i);  // = i + nbX*(j + nbY*k)
                        binIDsEachSphereTouches[myReportOffset] = binLin;
                        sphereIDsEachBinTouches[myReportOffset] = sphereID;
                        ++myReportOffset;
                    }
                }
            }
            // First found that this `not filled' problem can happen in the triangle bin--tri intersection detections
            // part... Quite peculiar.
            for (; myReportOffset < myReportOffset_end; ++myReportOffset) {
                binIDsEachSphereTouches[myReportOffset] = deme::NULL_BINID;
                sphereIDsEachBinTouches[myReportOffset] = sphereID;
            }
        }

        // ----- Analytical geometry contacts -----
        deme::binSphereTouchPairs_t mySphereGeoReportOffset     = numAnalGeoSphereTouchesScan[sphereID];
        deme::binSphereTouchPairs_t mySphereGeoReportOffset_end = numAnalGeoSphereTouchesScan[sphereID + 1];
        // Only check analytical geometry if capacity was reserved
        if (mySphereGeoReportOffset < mySphereGeoReportOffset_end) {
            // Each sphere entity should also check if it overlaps with an analytical boundary-type geometry
            for (deme::objID_t objB = 0; objB < simParams->nAnalGM; ++objB) {
                deme::bodyID_t objBOwner = objOwner[objB];
                // Grab family number from memory (not jitified: because family number can change frequently in a sim)
                unsigned int objFamilyNum = granData->familyID[objBOwner];
                unsigned int maskMatID = locateMaskPair<unsigned int>(sphFamilyNum, objFamilyNum);
                // If marked no contact, skip this iteration
                if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                    continue;
                }
                double3 ownerXYZ;
                voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                    ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[objBOwner], granData->locX[objBOwner],
                    granData->locY[objBOwner], granData->locZ[objBOwner], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
                const float ownerOriQw = granData->oriQw[objBOwner];
                const float ownerOriQx = granData->oriQx[objBOwner];
                const float ownerOriQy = granData->oriQy[objBOwner];
                const float ownerOriQz = granData->oriQz[objBOwner];
                float objBRelPosX = objRelPosX[objB];
                float objBRelPosY = objRelPosY[objB];
                float objBRelPosZ = objRelPosZ[objB];
                float objBRotX = objRotX[objB];
                float objBRotY = objRotY[objB];
                float objBRotZ = objRotZ[objB];
                applyOriQToVector3<float, deme::oriQ_t>(objBRelPosX, objBRelPosY, objBRelPosZ, ownerOriQw, ownerOriQx,
                                                        ownerOriQy, ownerOriQz);
                applyOriQToVector3<float, deme::oriQ_t>(objBRotX, objBRotY, objBRotZ, ownerOriQw, ownerOriQx, ownerOriQy,
                                                        ownerOriQz);
                double3 objBPosXYZ = ownerXYZ + make_double3(objBRelPosX, objBRelPosY, objBRelPosZ);

                double overlapDepth, overlapArea;  // overlapArea matters not here
                deme::contact_t contact_type;
                {
                    double3 cntPnt;  // cntPnt here is a placeholder
                    float3 cntNorm;  // cntNorm is placeholder too
                    contact_type = checkSphereEntityOverlap<double3, float, double>(
                        myPosXYZ, myRadius, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ),
                        objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB],
                        granData->marginSizeAnalytical[objB], cntPnt, cntNorm, overlapDepth, overlapArea);
                }
                // overlapDepth (which has both entities' full margins) needs to be larger than the smaller one of the two
                // added margin to be considered in-contact.
                double marginThres =
                    (granData->familyExtraMarginSize[sphFamilyNum] < granData->familyExtraMarginSize[objFamilyNum])
                        ? granData->familyExtraMarginSize[sphFamilyNum]
                        : granData->familyExtraMarginSize[objFamilyNum];
                if (contact_type && overlapDepth > marginThres) {
                    // Check capacity BEFORE writing (prevents OOB when capacity is exactly exhausted)
                    if (mySphereGeoReportOffset >= mySphereGeoReportOffset_end) {
                        return;  // Don't step on the next sphere's domain
                    }
                    idGeoA[mySphereGeoReportOffset] = sphereID;
                    idGeoB[mySphereGeoReportOffset] = (deme::bodyID_t)objB;
                    contactTypePrimitive[mySphereGeoReportOffset] = contact_type;
                    ++mySphereGeoReportOffset;
                }
            }
            // In practice, I've never seen non-illed contact slots that need to be resolved this way. It's purely for
            // ultra safety.
            for (; mySphereGeoReportOffset < mySphereGeoReportOffset_end; ++mySphereGeoReportOffset) {
                contactTypePrimitive[mySphereGeoReportOffset] = deme::NOT_A_CONTACT;
            }
        }  // end capacity guard for analytical geometry
    }
}
