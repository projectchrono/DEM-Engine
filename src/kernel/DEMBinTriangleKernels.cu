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
    deme::triID_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID < simParams->nTriGM) {
        // Get my component offset info from global array
        const float3 p1 = granData->relPosNode1[triID];
        const float3 p2 = granData->relPosNode2[triID];
        const float3 p3 = granData->relPosNode3[triID];

        // Get the controid of this triangle
        const float3 centroid = triangleCentroid(p1, p2, p3);
        // Generate normal using RHR from nodes 1, 2, and 3
        float3 triNormal = normalize(cross(p2 - p1, p3 - p1));

        sandwichANode1[triID] = sandwichVertex(p1, centroid, triNormal, simParams->beta);
        sandwichANode2[triID] = sandwichVertex(p2, centroid, triNormal, simParams->beta);
        sandwichANode3[triID] = sandwichVertex(p3, centroid, triNormal, simParams->beta);
        sandwichBNode1[triID] = sandwichVertex(p1, centroid, -triNormal, simParams->beta);
        sandwichBNode2[triID] = sandwichVertex(p2, centroid, -triNormal, simParams->beta);
        sandwichBNode3[triID] = sandwichVertex(p3, centroid, -triNormal, simParams->beta);
    }
}

__global__ void getNumberOfBinsEachTriangleTouches(deme::DEMSimParams* simParams,
                                                   deme::DEMDataKT* granData,
                                                   deme::binsTriangleTouches_t* numBinsTriTouches,
                                                   float3* node1,
                                                   float3* node2,
                                                   float3* node3) {
    deme::triID_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID < simParams->nTriGM) {
        // 3 vertices of the triangle
        float3 vA, vB, vC;
        {
            // My sphere voxel ID and my relPos
            deme::bodyID_t myOwnerID = granData->ownerMesh[triID];
            unsigned int triFamilyNum = granData->familyID[myOwnerID];
            // Get my component offset info from input array
            float3 loc_vA = node1[triID];
            float3 loc_vB = node2[triID];
            float3 loc_vC = node3[triID];

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
        }
        unsigned int L[3];
        unsigned int U[3];
        boundingBoxIntersectBin(L, U, vA, vB, vC, simParams);

        // Case 1: All vetices are in the same SD
        if (L[0] == U[0] && L[1] == U[1] && L[2] == U[2]) {
            numBinsTriTouches[triID] = 1;
            return;
        }

        unsigned int n_axes_diff = 0;  // Count axes that have different SD bounds
        unsigned int axes_diff;        // axis of variation (if only one)

        for (int i = 0; i < 3; i++) {
            if (L[i] != U[i]) {
                axes_diff = i;  // If there are more than one, this won't be used anyway
                n_axes_diff++;
            }
        }

        // Case 2: Triangle lies in a Nx1x1, 1xNx1, or 1x1xN block of SDs
        if (n_axes_diff == 1) {
            numBinsTriTouches[triID] = U[axes_diff] - L[axes_diff] + 1;
            return;
        }

        unsigned int numSDsTouched = 0;
        // Case 3: Triangle spans more than one dimension of spheresTouchingThisSD
        float BinCenter[3];
        float BinHalfSizes[3];
        BinHalfSizes[0] = simParams->binSize / 2. + DEME_BIN_ENLARGE_RATIO_FOR_FACETS * simParams->binSize;
        BinHalfSizes[1] = simParams->binSize / 2. + DEME_BIN_ENLARGE_RATIO_FOR_FACETS * simParams->binSize;
        BinHalfSizes[2] = simParams->binSize / 2. + DEME_BIN_ENLARGE_RATIO_FOR_FACETS * simParams->binSize;
        for (unsigned i = L[0]; i <= U[0]; i++) {
            for (unsigned j = L[1]; j <= U[1]; j++) {
                for (unsigned k = L[2]; k <= U[2]; k++) {
                    BinCenter[0] = simParams->binSize * i + simParams->binSize / 2.;
                    BinCenter[1] = simParams->binSize * j + simParams->binSize / 2.;
                    BinCenter[2] = simParams->binSize * k + simParams->binSize / 2.;

                    if (check_TriangleBoxOverlap(BinCenter, BinHalfSizes, vA, vB, vC)) {
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
                                                 deme::binsTriangleTouchPairs_t* numBinTriTouchesScan,
                                                 deme::binID_t* binIDsEachTriTouches,
                                                 deme::triID_t* triIDsEachBinTouches) {
    deme::bodyID_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID < simParams->nTriGM) {
    }
}
