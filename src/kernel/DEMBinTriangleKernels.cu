// DEM bin--sphere relations-related custom kernels
#include <DEM/Defines.h>
#include <kernel/DEMHelperKernels.cu>
#include <kernel/DEMTriangleBoxIntersect.cu>

inline __device__ float3 sandwichVertex(float3 vertex, const float3& centroid, const float3& normal, float beta) {
    // The vector along which we enlarge the triangle
    const float3 expandVec = normalize(vertex - centroid);
    vertex += expandVec * beta;
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

        // Get the controid of this triangle
        const float3 centroid = triangleCentroid<float3>(p1, p2, p3);
        // Generate normal using RHR from nodes 1, 2, and 3
        float3 triNormal = face_normal<float3>(p1, p2, p3);

        sandwichANode1[triID] = sandwichVertex(p1, centroid, triNormal, simParams->beta);
        sandwichANode2[triID] = sandwichVertex(p2, centroid, triNormal, simParams->beta);
        sandwichANode3[triID] = sandwichVertex(p3, centroid, triNormal, simParams->beta);
        sandwichBNode1[triID] = sandwichVertex(p1, centroid, -triNormal, simParams->beta);
        sandwichBNode2[triID] = sandwichVertex(p2, centroid, -triNormal, simParams->beta);
        sandwichBNode3[triID] = sandwichVertex(p3, centroid, -triNormal, simParams->beta);
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
                                                   float3* nodeA1,
                                                   float3* nodeB1,
                                                   float3* nodeC1,
                                                   float3* nodeA2,
                                                   float3* nodeB2,
                                                   float3* nodeC2) {
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
    }
}

__global__ void populateBinTriangleTouchingPairs(deme::DEMSimParams* simParams,
                                                 deme::DEMDataKT* granData,
                                                 deme::binsTriangleTouchPairs_t* numBinsTriTouchesScan,
                                                 deme::binID_t* binIDsEachTriTouches,
                                                 deme::bodyID_t* triIDsEachBinTouches,
                                                 float3* nodeA1,
                                                 float3* nodeB1,
                                                 float3* nodeC1,
                                                 float3* nodeA2,
                                                 float3* nodeB2,
                                                 float3* nodeC2) {
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
        // Triangle may span a collection of bins...
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
                        binIDsEachTriTouches[myReportOffset] =
                            binIDFrom3Indices<deme::binID_t>(i, j, k, simParams->nbX, simParams->nbY);
                        triIDsEachBinTouches[myReportOffset] = triID;
                        myReportOffset++;
                    }
                }
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
