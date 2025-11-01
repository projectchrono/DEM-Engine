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
        const deme::bodyID_t myOwnerID = granData->ownerMesh[triID];

        // Get the incenter of this triangle.
        // This is because we use the incenter to enalrge a triangle. See for example, this
        // https://stackoverflow.com/questions/36554898/algorithm-for-putting-double-border-around-isosceles-triangle.
        const float3 incenter = triangleIncenter<float3>(p1, p2, p3);
        // Generate normal using RHR from nodes 1, 2, and 3
        float3 triNormal = face_normal<float3>(p1, p2, p3);

        sandwichANode1[triID] = sandwichVertex(p1, incenter, p2 - p1, triNormal, granData->marginSize[myOwnerID]);
        sandwichANode2[triID] = sandwichVertex(p2, incenter, p3 - p2, triNormal, granData->marginSize[myOwnerID]);
        sandwichANode3[triID] = sandwichVertex(p3, incenter, p1 - p3, triNormal, granData->marginSize[myOwnerID]);
        // The other sandwich triangle needs to have an opposite normal direction
        sandwichBNode1[triID] = sandwichVertex(p1, incenter, p2 - p1, -triNormal, granData->marginSize[myOwnerID]);
        sandwichBNode2[triID] = sandwichVertex(p3, incenter, p1 - p3, -triNormal, granData->marginSize[myOwnerID]);
        sandwichBNode3[triID] = sandwichVertex(p2, incenter, p3 - p2, -triNormal, granData->marginSize[myOwnerID]);
    }
}

inline __device__ void figureOutNodeAndBoundingBox(deme::DEMSimParams* simParams,
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
    deme::bodyID_t myOwnerID = granData->ownerMesh[triID];

    double3 ownerXYZ;
    voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
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

    boundingBoxIntersectBin(L, U, vA, vB, vC, simParams);
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
        figureOutNodeAndBoundingBox(simParams, granData, triID, vA1, vB1, vC1, L1, U1, nodeA1[triID], nodeB1[triID],
                                    nodeC1[triID]);
        figureOutNodeAndBoundingBox(simParams, granData, triID, vA2, vB2, vC2, L2, U2, nodeA2[triID], nodeB2[triID],
                                    nodeC2[triID]);
        L1[0] = DEME_MIN(L1[0], L2[0]);
        L1[1] = DEME_MIN(L1[1], L2[1]);
        L1[2] = DEME_MIN(L1[2], L2[2]);
        U1[0] = DEME_MAX(U1[0], U2[0]);
        U1[1] = DEME_MAX(U1[1], U2[1]);
        U1[2] = DEME_MAX(U1[2], U2[2]);

        unsigned int numSDsTouched = 0;
        // Triangle may span a collection of bins...
        // BTW, I don't know why Chrono::GPU had to check the so-called 3 cases, and create thread divergence like that.
        // Just sweep through all potential bins and you are fine.
        float BinCenter[3];
        float BinHalfSizes[3];
        BinHalfSizes[0] = simParams->binSize / 2. + DEME_BIN_ENLARGE_RATIO_FOR_FACETS * simParams->binSize;
        BinHalfSizes[1] = simParams->binSize / 2. + DEME_BIN_ENLARGE_RATIO_FOR_FACETS * simParams->binSize;
        BinHalfSizes[2] = simParams->binSize / 2. + DEME_BIN_ENLARGE_RATIO_FOR_FACETS * simParams->binSize;
        for (deme::binID_t i = L1[0]; i <= U1[0]; i++) {
            for (deme::binID_t j = L1[1]; j <= U1[1]; j++) {
                for (deme::binID_t k = L1[2]; k <= U1[2]; k++) {
                    BinCenter[0] = simParams->binSize * i + simParams->binSize / 2.;
                    BinCenter[1] = simParams->binSize * j + simParams->binSize / 2.;
                    BinCenter[2] = simParams->binSize * k + simParams->binSize / 2.;

                    if (check_TriangleBoxOverlap(BinCenter, BinHalfSizes, vA1, vB1, vC1) ||
                        check_TriangleBoxOverlap(BinCenter, BinHalfSizes, vA2, vB2, vC2)) {
                        numSDsTouched++;
                    }
                }
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
                deme::bodyID_t triOwnerID = granData->ownerMesh[triID];
                unsigned int triFamilyNum = granData->familyID[triOwnerID];
                unsigned int maskMatID = locateMaskPair<unsigned int>(triFamilyNum, objFamilyNum);
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
                applyOriQToVector3<float, deme::oriQ_t>(objBRotX, objBRotY, objBRotZ, ownerOriQw, ownerOriQx,
                                                        ownerOriQy, ownerOriQz);
                double3 objBPosXYZ = ownerXYZ + make_double3(objBRelPosX, objBRelPosY, objBRelPosZ);

                double3 nodeA, nodeB, nodeC;
                nodeA = to_real3<float3, double3>(vA1);
                nodeB = to_real3<float3, double3>(vB1);
                nodeC = to_real3<float3, double3>(vC1);
                deme::contact_t contact_type = checkTriEntityOverlap<double3>(
                    nodeA, nodeB, nodeC, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ),
                    objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB], granData->marginSize[objBOwner]);
                if (contact_type == deme::NOT_A_CONTACT) {
                    nodeA = to_real3<float3, double3>(vA2);
                    nodeB = to_real3<float3, double3>(vB2);
                    nodeC = to_real3<float3, double3>(vC2);
                    contact_type = checkTriEntityOverlap<double3>(nodeA, nodeB, nodeC, objType[objB], objBPosXYZ,
                                                                  make_float3(objBRotX, objBRotY, objBRotZ),
                                                                  objSize1[objB], objSize2[objB], objSize3[objB],
                                                                  objNormal[objB], granData->marginSize[objBOwner]);
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
                                                 deme::contact_t* contactType,
                                                 bool meshUniversalContact) {
    deme::bodyID_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID < simParams->nTriGM) {
        // 3 vertices of the triangle
        float3 vA1, vB1, vC1, vA2, vB2, vC2;
        deme::binID_t L1[3], L2[3], U1[3], U2[3];
        figureOutNodeAndBoundingBox(simParams, granData, triID, vA1, vB1, vC1, L1, U1, nodeA1[triID], nodeB1[triID],
                                    nodeC1[triID]);
        figureOutNodeAndBoundingBox(simParams, granData, triID, vA2, vB2, vC2, L2, U2, nodeA2[triID], nodeB2[triID],
                                    nodeC2[triID]);
        L1[0] = DEME_MIN(L1[0], L2[0]);
        L1[1] = DEME_MIN(L1[1], L2[1]);
        L1[2] = DEME_MIN(L1[2], L2[2]);
        U1[0] = DEME_MAX(U1[0], U2[0]);
        U1[1] = DEME_MAX(U1[1], U2[1]);
        U1[2] = DEME_MAX(U1[2], U2[2]);

        deme::binsTriangleTouchPairs_t myReportOffset = numBinsTriTouchesScan[triID];
        // In case this sweep does not agree with the previous one, we need to intercept such potential segfaults
        const deme::binsTriangleTouchPairs_t myReportOffset_end = numBinsTriTouchesScan[triID + 1];

        // Triangle may span a collection of bins...
        float BinCenter[3];
        float BinHalfSizes[3];
        BinHalfSizes[0] = simParams->binSize / 2. + DEME_BIN_ENLARGE_RATIO_FOR_FACETS * simParams->binSize;
        BinHalfSizes[1] = simParams->binSize / 2. + DEME_BIN_ENLARGE_RATIO_FOR_FACETS * simParams->binSize;
        BinHalfSizes[2] = simParams->binSize / 2. + DEME_BIN_ENLARGE_RATIO_FOR_FACETS * simParams->binSize;
        for (deme::binID_t i = L1[0]; i <= U1[0]; i++) {
            for (deme::binID_t j = L1[1]; j <= U1[1]; j++) {
                for (deme::binID_t k = L1[2]; k <= U1[2]; k++) {
                    if (myReportOffset >= myReportOffset_end) {
                        continue;  // Don't step on the next triangle's domain
                    }
                    BinCenter[0] = simParams->binSize * i + simParams->binSize / 2.;
                    BinCenter[1] = simParams->binSize * j + simParams->binSize / 2.;
                    BinCenter[2] = simParams->binSize * k + simParams->binSize / 2.;

                    if (check_TriangleBoxOverlap(BinCenter, BinHalfSizes, vA1, vB1, vC1) ||
                        check_TriangleBoxOverlap(BinCenter, BinHalfSizes, vA2, vB2, vC2)) {
                        binIDsEachTriTouches[myReportOffset] =
                            binIDFrom3Indices<deme::binID_t>(i, j, k, simParams->nbX, simParams->nbY, simParams->nbZ);
                        triIDsEachBinTouches[myReportOffset] = triID;
                        myReportOffset++;
                    }
                }
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
                deme::bodyID_t triOwnerID = granData->ownerMesh[triID];
                unsigned int triFamilyNum = granData->familyID[triOwnerID];
                unsigned int maskMatID = locateMaskPair<unsigned int>(triFamilyNum, objFamilyNum);
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
                applyOriQToVector3<float, deme::oriQ_t>(objBRotX, objBRotY, objBRotZ, ownerOriQw, ownerOriQx,
                                                        ownerOriQy, ownerOriQz);
                double3 objBPosXYZ = ownerXYZ + make_double3(objBRelPosX, objBRelPosY, objBRelPosZ);

                double3 nodeA, nodeB, nodeC;
                nodeA = to_real3<float3, double3>(vA1);
                nodeB = to_real3<float3, double3>(vB1);
                nodeC = to_real3<float3, double3>(vC1);
                deme::contact_t contact_type = checkTriEntityOverlap<double3>(
                    nodeA, nodeB, nodeC, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ),
                    objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB], granData->marginSize[objBOwner]);
                if (contact_type == deme::NOT_A_CONTACT) {
                    nodeA = to_real3<float3, double3>(vA2);
                    nodeB = to_real3<float3, double3>(vB2);
                    nodeC = to_real3<float3, double3>(vC2);
                    contact_type = checkTriEntityOverlap<double3>(nodeA, nodeB, nodeC, objType[objB], objBPosXYZ,
                                                                  make_float3(objBRotX, objBRotY, objBRotZ),
                                                                  objSize1[objB], objSize2[objB], objSize3[objB],
                                                                  objNormal[objB], granData->marginSize[objBOwner]);
                }
                // Unlike the sphere-X contact case, we do not test against family extra margin here.
                if (contact_type == deme::TRIANGLE_ANALYTICAL_CONTACT) {
                    idGeoA[myTriGeoReportOffset] = triID;
                    idGeoB[myTriGeoReportOffset] = (deme::bodyID_t)objB;
                    contactType[myTriGeoReportOffset] = contact_type;
                    myTriGeoReportOffset++;
                    if (myTriGeoReportOffset >= myTriGeoReportOffset_end) {
                        return;  // Don't step on the next triangle's domain
                    }
                }
            }
            // Take care of potentially unfilled slots in the report
            for (; myTriGeoReportOffset < myTriGeoReportOffset_end; myTriGeoReportOffset++) {
                contactType[myTriGeoReportOffset] = deme::NOT_A_CONTACT;
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
