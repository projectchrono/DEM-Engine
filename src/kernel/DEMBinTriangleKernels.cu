// DEM bin--sphere relations-related custom kernels
#include <DEM/Defines.h>
#include <DEMCollisionKernels_SphTri_TriTri.cuh>
#include <DEMTriangleBoxIntersect.cu>
_kernelIncludes_;

// Definitions of analytical entites are below
_analyticalEntityDefs_;

inline __device__ float3
sandwichVertex(float3 vertex, const float3& incenter, const float3& side, const float3& normal, float beta) {
    // The vector along which we enlarge the triangle
    float3 expandVec = normalize(vertex - incenter);

    // Use a side starting from the vertex and the vector from the vertex to the incenter to figure out the half angle
    const float cos_halfangle = dot(-expandVec, side) / length(side);
    // Then the distance to advance the vertex along the expand vector...
    const float enlarge_dist = beta / sqrt(1. - cos_halfangle * cos_halfangle);

    vertex += expandVec * enlarge_dist;
    vertex += normal * beta;
    return vertex;
}

__global__ void makeTriangleSandwich(deme::DEMSimParams* simParams,
                                     deme::DEMDataKT* granData,
                                     float3* sandwichANode1,
                                     float3* sandwichANode2,
                                     float3* sandwichANode3,
                                     float3* sandwichBNode1,
                                     float3* sandwichBNode2,
                                     float3* sandwichBNode3) {
    deme::bodyID_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID < simParams->nTriGM) {
        // Get my component offset info from global array
        const float3 p1 = granData->relPosNode1[triID];
        const float3 p2 = granData->relPosNode2[triID];
        const float3 p3 = granData->relPosNode3[triID];
        const deme::bodyID_t myOwnerID = granData->ownerTriMesh[triID];
        const float margin = granData->marginSizeTriangle[triID];

        // Get the incenter of this triangle.
        // This is because we use the incenter to enalrge a triangle. See for example, this
        // https://stackoverflow.com/questions/36554898/algorithm-for-putting-double-border-around-isosceles-triangle.
        const float3 incenter = triangleIncenter<float3>(p1, p2, p3);
        // Generate normal using RHR from nodes 1, 2, and 3
        float3 triNormal = face_normal<float3>(p1, p2, p3);

        sandwichANode1[triID] = sandwichVertex(p1, incenter, p2 - p1, triNormal, margin);
        sandwichANode2[triID] = sandwichVertex(p2, incenter, p3 - p2, triNormal, margin);
        sandwichANode3[triID] = sandwichVertex(p3, incenter, p1 - p3, triNormal, margin);
        // The other sandwich triangle needs to have an opposite normal direction
        sandwichBNode1[triID] = sandwichVertex(p1, incenter, p2 - p1, -triNormal, margin);
        sandwichBNode2[triID] = sandwichVertex(p3, incenter, p1 - p3, -triNormal, margin);
        sandwichBNode3[triID] = sandwichVertex(p2, incenter, p3 - p2, -triNormal, margin);
    }
}


// Compute triangle AABB -> bin index bounds using mixed precision (FP32 fast path + FP64 fallback).
// This mirrors the sphere binning approach (axis_bounds) to avoid precision issues when a bound lies
// close to a bin boundary.
inline __device__ bool boundingBoxIntersectBinAxisBounds(deme::binID_t* L,
                                                         deme::binID_t* U,
                                                         const float3& vA,
                                                         const float3& vB,
                                                         const float3& vC,
                                                         deme::DEMSimParams* simParams) {
    float3 min_pt;
    min_pt.x = DEME_MIN(vA.x, DEME_MIN(vB.x, vC.x));
    min_pt.y = DEME_MIN(vA.y, DEME_MIN(vB.y, vC.y));
    min_pt.z = DEME_MIN(vA.z, DEME_MIN(vB.z, vC.z));

    float3 max_pt;
    max_pt.x = DEME_MAX(vA.x, DEME_MAX(vB.x, vC.x));
    max_pt.y = DEME_MAX(vA.y, DEME_MAX(vB.y, vC.y));
    max_pt.z = DEME_MAX(vA.z, DEME_MAX(vB.z, vC.z));

    // Enlarge bounding box, so that no triangle lies right between 2 layers of bins
    const float enlarge = (float)DEME_BIN_ENLARGE_RATIO_FOR_FACETS * (float)simParams->dyn.binSize;
    min_pt -= enlarge;
    max_pt += enlarge;

    const double invBinSize = simParams->dyn.inv_binSize;
    const int nbX = (int)simParams->nbX;
    const int nbY = (int)simParams->nbY;
    const int nbZ = (int)simParams->nbZ;

    // Convert [min,max] to (center, half-range) and use axis_bounds (FP32 fast path with FP64 fallback).
    const double cx = 0.5 * ((double)min_pt.x + (double)max_pt.x);
    const double rx = 0.5 * ((double)max_pt.x - (double)min_pt.x);
    const deme::AxisBounds bx = axis_bounds(cx, rx, nbX, invBinSize);
    if (bx.imax < bx.imin) return false;

    const double cy = 0.5 * ((double)min_pt.y + (double)max_pt.y);
    const double ry = 0.5 * ((double)max_pt.y - (double)min_pt.y);
    const deme::AxisBounds by = axis_bounds(cy, ry, nbY, invBinSize);
    if (by.imax < by.imin) return false;

    const double cz = 0.5 * ((double)min_pt.z + (double)max_pt.z);
    const double rz = 0.5 * ((double)max_pt.z - (double)min_pt.z);
    const deme::AxisBounds bz = axis_bounds(cz, rz, nbZ, invBinSize);
    if (bz.imax < bz.imin) return false;

    L[0] = (deme::binID_t)bx.imin;
    U[0] = (deme::binID_t)bx.imax;
    L[1] = (deme::binID_t)by.imin;
    U[1] = (deme::binID_t)by.imax;
    L[2] = (deme::binID_t)bz.imin;
    U[2] = (deme::binID_t)bz.imax;
    return true;
}

inline __device__ bool figureOutNodeAndBoundingBox(deme::DEMSimParams* simParams,
                                                   deme::DEMDataKT* granData,
                                                   const deme::bodyID_t& triID,
                                                   float3& vA,
                                                   float3& vB,
                                                   float3& vC,
                                                   deme::binID_t L[],
                                                   deme::binID_t U[],
                                                   float3 loc_vA,
                                                   float3 loc_vB,
                                                   float3 loc_vC) {
    // My sphere voxel ID and my relPos
    deme::bodyID_t myOwnerID = granData->ownerTriMesh[triID];

    float3 ownerXYZ;
    voxelIDToPosition<float, deme::voxelID_t, deme::subVoxelPos_t>(
        ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[myOwnerID], granData->locX[myOwnerID],
        granData->locY[myOwnerID], granData->locZ[myOwnerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
    const float myOriQw = granData->oriQw[myOwnerID];
    const float myOriQx = granData->oriQx[myOwnerID];
    const float myOriQy = granData->oriQy[myOwnerID];
    const float myOriQz = granData->oriQz[myOwnerID];
    applyOriQToVector3<float, deme::oriQ_t>(loc_vA.x, loc_vA.y, loc_vA.z, myOriQw, myOriQx, myOriQy, myOriQz);
    applyOriQToVector3<float, deme::oriQ_t>(loc_vB.x, loc_vB.y, loc_vB.z, myOriQw, myOriQx, myOriQy, myOriQz);
    applyOriQToVector3<float, deme::oriQ_t>(loc_vC.x, loc_vC.y, loc_vC.z, myOriQw, myOriQx, myOriQy, myOriQz);
    vA = ownerXYZ + loc_vA;
    vB = ownerXYZ + loc_vB;
    vC = ownerXYZ + loc_vC;

    return boundingBoxIntersectBinAxisBounds(L, U, vA, vB, vC, simParams);
}

__global__ void getNumberOfBinsEachTriangleTouches(deme::DEMSimParams* simParams,
                                                   deme::DEMDataKT* granData,
                                                   deme::binsTriangleTouches_t* numBinsTriTouches,
                                                   deme::objID_t* numAnalGeoTriTouches,
                                                   float3* nodeA1,
                                                   float3* nodeB1,
                                                   float3* nodeC1,
                                                   float3* nodeA2,
                                                   float3* nodeB2,
                                                   float3* nodeC2,
                                                   bool meshUniversalContact) {
    deme::bodyID_t triID = blockIdx.x * blockDim.x + threadIdx.x;

    if (triID < simParams->nTriGM) {
        // 3 vertices of the triangle, in true space location but without adding LBF point (since purely voxel- and
        // bin-based locations don't need that)
        float3 vA1, vB1, vC1, vA2, vB2, vC2;
        deme::binID_t L1[3], L2[3], U1[3], U2[3];
        const bool ok1 = figureOutNodeAndBoundingBox(simParams, granData, triID, vA1, vB1, vC1, L1, U1, nodeA1[triID],
                                                     nodeB1[triID], nodeC1[triID]);
        const bool ok2 = figureOutNodeAndBoundingBox(simParams, granData, triID, vA2, vB2, vC2, L2, U2, nodeA2[triID],
                                                     nodeB2[triID], nodeC2[triID]);

        // If neither triangle sandwich intersects the bin grid, it cannot touch any bin.
        if (!ok1 && !ok2) {
            numBinsTriTouches[triID] = 0;
            if (meshUniversalContact) {
                numAnalGeoTriTouches[triID] = 0;
            }
            return;
        }

        // Merge bounds (or take the valid one, if only one is valid).
        if (ok1 && ok2) {
            L1[0] = DEME_MIN(L1[0], L2[0]);
            L1[1] = DEME_MIN(L1[1], L2[1]);
            L1[2] = DEME_MIN(L1[2], L2[2]);
            U1[0] = DEME_MAX(U1[0], U2[0]);
            U1[1] = DEME_MAX(U1[1], U2[1]);
            U1[2] = DEME_MAX(U1[2], U2[2]);
        } else if (!ok1) {
            L1[0] = L2[0];
            L1[1] = L2[1];
            L1[2] = L2[2];
            U1[0] = U2[0];
            U1[1] = U2[1];
            U1[2] = U2[2];
        }

        unsigned int numSDsTouched = 0;
        // Triangle may span a collection of bins...
        // BTW, I don't know why Chrono::GPU had to check the so-called 3 cases, and create thread divergence like that.
        // Just sweep through all potential bins and you are fine.
        float BinCenter[3];
        const float binSizeF = (float)simParams->dyn.binSize;
        const float binHalfSpan = binSizeF * (0.5f + (float)DEME_BIN_ENLARGE_RATIO_FOR_FACETS);
        float BinHalfSizes[3] = {binHalfSpan, binHalfSpan, binHalfSpan};
        const float startX = binSizeF * (float)L1[0] + 0.5f * binSizeF;
        const float startY = binSizeF * (float)L1[1] + 0.5f * binSizeF;
        const float startZ = binSizeF * (float)L1[2] + 0.5f * binSizeF;
        for (deme::binID_t i = L1[0], ix = 0; i <= U1[0]; i++, ix++) {
            float cy0 = startY;
            BinCenter[0] = startX + ix * binSizeF;
            for (deme::binID_t j = L1[1]; j <= U1[1]; j++) {
                float cz = startZ;
                BinCenter[1] = cy0;
                for (deme::binID_t k = L1[2]; k <= U1[2]; k++) {
                    BinCenter[2] = cz;

                    if (check_TriangleBoxOverlap(BinCenter, BinHalfSizes, vA1, vB1, vC1) ||
                        check_TriangleBoxOverlap(BinCenter, BinHalfSizes, vA2, vB2, vC2)) {
                        numSDsTouched++;
                    }
                    cz += binSizeF;
                }
                cy0 += binSizeF;
            }
        }
        numBinsTriTouches[triID] = numSDsTouched;

        // No need to do the following if meshUniversalContact is false
        if (meshUniversalContact) {
            // Register sphere--analytical geometry contacts
            deme::objID_t contact_count = 0;
            // Each triangle should also check if it overlaps with an analytical boundary-type geometry
            for (deme::objID_t objB = 0; objB < simParams->nAnalGM; objB++) {
                deme::bodyID_t objBOwner = objOwner[objB];
                // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                unsigned int objFamilyNum = granData->familyID[objBOwner];
                deme::bodyID_t triOwnerID = granData->ownerTriMesh[triID];
                unsigned int triFamilyNum = granData->familyID[triOwnerID];
                unsigned int maskMatID = locateMaskPair<unsigned int>(triFamilyNum, objFamilyNum);
                // If marked no contact, skip ths iteration
                if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                    continue;
                }
                float3 ownerXYZ;
                voxelIDToPosition<float, deme::voxelID_t, deme::subVoxelPos_t>(
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
                applyOriQToVector3<float, deme::oriQ_t>(objBRotX, objBRotY, objBRotZ, ownerOriQw, ownerOriQx,
                                                        ownerOriQy, ownerOriQz);
                float3 objBPosXYZ = ownerXYZ + make_float3(objBRelPosX, objBRelPosY, objBRelPosZ);

                deme::contact_t contact_type = checkTriEntityOverlapFP32(
                    vA1, vB1, vC1, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ),
                    objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB],
                    granData->marginSizeAnalytical[objB]);
                if (contact_type == deme::NOT_A_CONTACT) {
                    contact_type = checkTriEntityOverlapFP32(
                        vA2, vB2, vC2, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ),
                        objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB],
                        granData->marginSizeAnalytical[objB]);
                }
                // Unlike the sphere-X contact case, we do not test against family extra margin here. This may result in
                // more fake contact pairs, but the efficiency in the mesh-based particle case is not our top priority
                // yet.
                if (contact_type == deme::TRIANGLE_ANALYTICAL_CONTACT) {
                    contact_count++;
                }
            }
            numAnalGeoTriTouches[triID] = contact_count;
        }
    }
}

__global__ void populateBinTriangleTouchingPairs(deme::DEMSimParams* simParams,
                                                 deme::DEMDataKT* granData,
                                                 deme::binsTriangleTouchPairs_t* numBinsTriTouchesScan,
                                                 deme::binsTriangleTouchPairs_t* numAnalGeoTriTouchesScan,
                                                 deme::binID_t* binIDsEachTriTouches,
                                                 deme::bodyID_t* triIDsEachBinTouches,
                                                 float3* nodeA1,
                                                 float3* nodeB1,
                                                 float3* nodeC1,
                                                 float3* nodeA2,
                                                 float3* nodeB2,
                                                 float3* nodeC2,
                                                 deme::bodyID_t* idGeoA,
                                                 deme::bodyID_t* idGeoB,
                                                 deme::contact_t* contactTypePrimitive,
                                                 bool meshUniversalContact) {
    deme::bodyID_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID < simParams->nTriGM) {
        // 3 vertices of the triangle
        float3 vA1, vB1, vC1, vA2, vB2, vC2;
        deme::binID_t L1[3], L2[3], U1[3], U2[3];
        const bool ok1 = figureOutNodeAndBoundingBox(simParams, granData, triID, vA1, vB1, vC1, L1, U1, nodeA1[triID],
                                                     nodeB1[triID], nodeC1[triID]);
        const bool ok2 = figureOutNodeAndBoundingBox(simParams, granData, triID, vA2, vB2, vC2, L2, U2, nodeA2[triID],
                                                     nodeB2[triID], nodeC2[triID]);

        // If neither triangle sandwich intersects the bin grid, it cannot touch any bin.
        if (!ok1 && !ok2) {
            return;
        }

        // Merge bounds (or take the valid one, if only one is valid).
        if (ok1 && ok2) {
            L1[0] = DEME_MIN(L1[0], L2[0]);
            L1[1] = DEME_MIN(L1[1], L2[1]);
            L1[2] = DEME_MIN(L1[2], L2[2]);
            U1[0] = DEME_MAX(U1[0], U2[0]);
            U1[1] = DEME_MAX(U1[1], U2[1]);
            U1[2] = DEME_MAX(U1[2], U2[2]);
        } else if (!ok1) {
            L1[0] = L2[0];
            L1[1] = L2[1];
            L1[2] = L2[2];
            U1[0] = U2[0];
            U1[1] = U2[1];
            U1[2] = U2[2];
        }

        deme::binsTriangleTouchPairs_t myReportOffset = numBinsTriTouchesScan[triID];
        // In case this sweep does not agree with the previous one, we need to intercept such potential segfaults
        const deme::binsTriangleTouchPairs_t myReportOffset_end = numBinsTriTouchesScan[triID + 1];

        // Triangle may span a collection of bins...
        float BinCenter[3];
        const float binSizeF = (float)simParams->dyn.binSize;
        const float binHalfSpan = binSizeF * (0.5f + (float)DEME_BIN_ENLARGE_RATIO_FOR_FACETS);
        float BinHalfSizes[3] = {binHalfSpan, binHalfSpan, binHalfSpan};
        const float startX = binSizeF * (float)L1[0] + 0.5f * binSizeF;
        const float startY = binSizeF * (float)L1[1] + 0.5f * binSizeF;
        const float startZ = binSizeF * (float)L1[2] + 0.5f * binSizeF;
        for (deme::binID_t i = L1[0], ix = 0; i <= U1[0]; i++, ix++) {
            BinCenter[0] = startX + ix * binSizeF;
            float cy0 = startY;
            for (deme::binID_t j = L1[1]; j <= U1[1]; j++) {
                BinCenter[1] = cy0;
                float cz = startZ;
                for (deme::binID_t k = L1[2]; k <= U1[2]; k++) {
                    if (myReportOffset >= myReportOffset_end) {
                        continue;  // Don't step on the next triangle's domain
                    }
                    BinCenter[2] = cz;

                    if (check_TriangleBoxOverlap(BinCenter, BinHalfSizes, vA1, vB1, vC1) ||
                        check_TriangleBoxOverlap(BinCenter, BinHalfSizes, vA2, vB2, vC2)) {
                        binIDsEachTriTouches[myReportOffset] =
                            binIDFrom3Indices<deme::binID_t>(i, j, k, simParams->nbX, simParams->nbY, simParams->nbZ);
                        triIDsEachBinTouches[myReportOffset] = triID;
                        myReportOffset++;
                    }
                    cz += binSizeF;
                }
                cy0 += binSizeF;
            }
        }
        // This can happen for like 1 in 10^9 chance, for the tri--bin contact algorithm has stochasticity on GPU
        for (; myReportOffset < myReportOffset_end; myReportOffset++) {
            binIDsEachTriTouches[myReportOffset] = deme::NULL_BINID;
            triIDsEachBinTouches[myReportOffset] = triID;
        }

        // No need to do the following if meshUniversalContact is false
        if (meshUniversalContact) {
            deme::binsTriangleTouchPairs_t myTriGeoReportOffset = numAnalGeoTriTouchesScan[triID];
            deme::binsTriangleTouchPairs_t myTriGeoReportOffset_end = numAnalGeoTriTouchesScan[triID + 1];
            for (deme::objID_t objB = 0; objB < simParams->nAnalGM; objB++) {
                deme::bodyID_t objBOwner = objOwner[objB];
                // Grab family number from memory (not jitified: b/c family number can change frequently in a sim)
                unsigned int objFamilyNum = granData->familyID[objBOwner];
                deme::bodyID_t triOwnerID = granData->ownerTriMesh[triID];
                unsigned int triFamilyNum = granData->familyID[triOwnerID];
                unsigned int maskMatID = locateMaskPair<unsigned int>(triFamilyNum, objFamilyNum);
                // If marked no contact, skip ths iteration
                if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                    continue;
                }
                float3 ownerXYZ;
                voxelIDToPosition<float, deme::voxelID_t, deme::subVoxelPos_t>(
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
                applyOriQToVector3<float, deme::oriQ_t>(objBRotX, objBRotY, objBRotZ, ownerOriQw, ownerOriQx,
                                                        ownerOriQy, ownerOriQz);
                float3 objBPosXYZ = ownerXYZ + make_float3(objBRelPosX, objBRelPosY, objBRelPosZ);

                deme::contact_t contact_type = checkTriEntityOverlapFP32(
                    vA1, vB1, vC1, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ),
                    objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB],
                    granData->marginSizeAnalytical[objB]);
                if (contact_type == deme::NOT_A_CONTACT) {
                    contact_type = checkTriEntityOverlapFP32(
                        vA2, vB2, vC2, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ),
                        objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB],
                        granData->marginSizeAnalytical[objB]);
                }
                // Unlike the sphere-X contact case, we do not test against family extra margin here, which is more
                // lenient and perhaps makes more fake contacts.
                if (contact_type == deme::TRIANGLE_ANALYTICAL_CONTACT) {
                    idGeoA[myTriGeoReportOffset] = triID;
                    idGeoB[myTriGeoReportOffset] = (deme::bodyID_t)objB;
                    contactTypePrimitive[myTriGeoReportOffset] = contact_type;
                    myTriGeoReportOffset++;
                    if (myTriGeoReportOffset >= myTriGeoReportOffset_end) {
                        return;  // Don't step on the next triangle's domain
                    }
                }
            }
            // Take care of potentially unfilled slots in the report
            for (; myTriGeoReportOffset < myTriGeoReportOffset_end; myTriGeoReportOffset++) {
                contactTypePrimitive[myTriGeoReportOffset] = deme::NOT_A_CONTACT;
            }
        }
    }
}

__global__ void mapTriActiveBinsToSphActiveBins(deme::binID_t* activeBinIDsForTri,
                                                deme::binID_t* activeBinIDs,
                                                deme::binID_t* mapTriActBinToSphActBin,
                                                size_t numActiveBinsForTri,
                                                size_t numActiveBinsForSph) {
    size_t threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID < numActiveBinsForTri) {
        deme::binID_t binID = activeBinIDsForTri[threadID];
        deme::binID_t indexInOther;
        bool found = cuda_binary_search<deme::binID_t, deme::binID_t>(activeBinIDs, binID, 0, numActiveBinsForSph - 1,
                                                                      indexInOther);
        if (found) {
            mapTriActBinToSphActBin[threadID] = indexInOther;
        } else {
            mapTriActBinToSphActBin[threadID] = deme::NULL_BINID;
        }
    }
}
