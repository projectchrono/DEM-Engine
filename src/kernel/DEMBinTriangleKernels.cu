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
    if (bx.imax < bx.imin)
        return false;

    const double cy = 0.5 * ((double)min_pt.y + (double)max_pt.y);
    const double ry = 0.5 * ((double)max_pt.y - (double)min_pt.y);
    const deme::AxisBounds by = axis_bounds(cy, ry, nbY, invBinSize);
    if (by.imax < by.imin)
        return false;

    const double cz = 0.5 * ((double)min_pt.z + (double)max_pt.z);
    const double rz = 0.5 * ((double)max_pt.z - (double)min_pt.z);
    const deme::AxisBounds bz = axis_bounds(cz, rz, nbZ, invBinSize);
    if (bz.imax < bz.imin)
        return false;

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


__global__ void precomputeTriangleSandwichData(deme::DEMSimParams* simParams,
                                              deme::DEMDataKT* granData,
                                              // World-space vertices for both sandwich triangles
                                              float3* vA1_all,
                                              float3* vB1_all,
                                              float3* vC1_all,
                                              float3* vA2_all,
                                              float3* vB2_all,
                                              float3* vC2_all,
                                              // Per-triangle translation B = A + shift_world
                                              float3* shift_world_all,
                                              // Per-triangle bounds for A and B (only valid if ok flag true)
                                              int3* LA_all,
                                              int3* UA_all,
                                              int3* LB_all,
                                              int3* UB_all,
                                              // ok flags
                                              unsigned char* ok1_all,
                                              unsigned char* ok2_all,
                                              // sandwich nodes (local, as produced by makeTriangleSandwich)
                                              float3* nodeA1,
                                              float3* nodeB1,
                                              float3* nodeC1,
                                              float3* nodeA2,
                                              float3* nodeB2,
                                              float3* nodeC2) {
    deme::bodyID_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID >= simParams->nTriGM) {
        return;
    }

    float3 vA1, vB1, vC1, vA2, vB2, vC2;
    deme::binID_t L1[3], L2[3], U1[3], U2[3];

    const bool ok1 = figureOutNodeAndBoundingBox(simParams, granData, triID, vA1, vB1, vC1, L1, U1,
                                                 nodeA1[triID], nodeB1[triID], nodeC1[triID]);
    const bool ok2 = figureOutNodeAndBoundingBox(simParams, granData, triID, vA2, vB2, vC2, L2, U2,
                                                 nodeA2[triID], nodeB2[triID], nodeC2[triID]);

    vA1_all[triID] = vA1;
    vB1_all[triID] = vB1;
    vC1_all[triID] = vC1;
    vA2_all[triID] = vA2;
    vB2_all[triID] = vB2;
    vC2_all[triID] = vC2;

    ok1_all[triID] = (unsigned char)(ok1 ? 1 : 0);
    ok2_all[triID] = (unsigned char)(ok2 ? 1 : 0);

    if (ok1) {
        LA_all[triID] = make_int3(L1[0], L1[1], L1[2]);
        UA_all[triID] = make_int3(U1[0], U1[1], U1[2]);
    }
    if (ok2) {
        LB_all[triID] = make_int3(L2[0], L2[1], L2[2]);
        UB_all[triID] = make_int3(U2[0], U2[1], U2[2]);
    }

    // Precompute sandwich translation (B = A + shift_world) in numerically safe way.
    float3 shift_world = make_float3(0.f, 0.f, 0.f);
    if (ok2) {
        const deme::bodyID_t owner = granData->ownerTriMesh[triID];
        const float qw = granData->oriQw[owner];
        const float qx = granData->oriQx[owner];
        const float qy = granData->oriQy[owner];
        const float qz = granData->oriQz[owner];
        float3 shift_local = make_float3(nodeA2[triID].x - nodeA1[triID].x,
                                         nodeA2[triID].y - nodeA1[triID].y,
                                         nodeA2[triID].z - nodeA1[triID].z);
        applyOriQToVector3<float, deme::oriQ_t>(shift_local.x, shift_local.y, shift_local.z, qw, qx, qy, qz);
        shift_world = shift_local;
    }
    shift_world_all[triID] = shift_world;
}

// Prepass versions of the existing kernels (signature includes precomputed arrays).
__global__ void getNumberOfBinsEachTriangleTouches(deme::DEMSimParams* simParams,
                                                           deme::DEMDataKT* granData,
                                                           deme::binsTriangleTouches_t* numBinsTriTouches,
                                                           deme::objID_t* numAnalGeoTriTouches,
                                                           // precomputed
                                                           const float3* vA1_all,
                                                           const float3* vB1_all,
                                                           const float3* vC1_all,
                                                           const float3* vA2_all,
                                                           const float3* vB2_all,
                                                           const float3* vC2_all,
                                                           const float3* shift_world_all,
                                                           const int3* LA_all,
                                                           const int3* UA_all,
                                                           const int3* LB_all,
                                                           const int3* UB_all,
                                                           const unsigned char* ok1_all,
                                                           const unsigned char* ok2_all,
                                                           bool meshUniversalContact) {
    deme::bodyID_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID >= simParams->nTriGM) {
        return;
    }

    const bool ok1 = (ok1_all[triID] != 0);
    const bool ok2 = (ok2_all[triID] != 0);

    if (!ok1 && !ok2) {
        numBinsTriTouches[triID] = 0;
        if (meshUniversalContact) {
            numAnalGeoTriTouches[triID] = 0;
        }
        return;
    }

    const float3 vA1 = vA1_all[triID];
    const float3 vB1 = vB1_all[triID];
    const float3 vC1 = vC1_all[triID];
    const float3 vA2 = vA2_all[triID];
    const float3 vB2 = vB2_all[triID];
    const float3 vC2 = vC2_all[triID];
    const float3 shift_world = shift_world_all[triID];

    int3 LA = make_int3(0, 0, 0), UA = make_int3(-1, -1, -1);
    int3 LB = make_int3(0, 0, 0), UB = make_int3(-1, -1, -1);
    if (ok1) {
        LA = LA_all[triID];
        UA = UA_all[triID];
    }
    if (ok2) {
        LB = LB_all[triID];
        UB = UB_all[triID];
    }

    // Union bounds
    deme::binID_t Lx, Ly, Lz, Ux, Uy, Uz;
    if (ok1 && ok2) {
        Lx = (deme::binID_t)DEME_MIN(LA.x, LB.x);
        Ly = (deme::binID_t)DEME_MIN(LA.y, LB.y);
        Lz = (deme::binID_t)DEME_MIN(LA.z, LB.z);
        Ux = (deme::binID_t)DEME_MAX(UA.x, UB.x);
        Uy = (deme::binID_t)DEME_MAX(UA.y, UB.y);
        Uz = (deme::binID_t)DEME_MAX(UA.z, UB.z);
    } else if (ok1) {
        Lx = (deme::binID_t)LA.x;
        Ly = (deme::binID_t)LA.y;
        Lz = (deme::binID_t)LA.z;
        Ux = (deme::binID_t)UA.x;
        Uy = (deme::binID_t)UA.y;
        Uz = (deme::binID_t)UA.z;
    } else {
        Lx = (deme::binID_t)LB.x;
        Ly = (deme::binID_t)LB.y;
        Lz = (deme::binID_t)LB.z;
        Ux = (deme::binID_t)UB.x;
        Uy = (deme::binID_t)UB.y;
        Uz = (deme::binID_t)UB.z;
    }

    unsigned int numSDsTouched = 0;
    const float binSizeF = (float)simParams->dyn.binSize;
    const float binHalfSpan = binSizeF * (0.5f + (float)DEME_BIN_ENLARGE_RATIO_FOR_FACETS);
    const float startX = binSizeF * (float)Lx + 0.5f * binSizeF;
    const float startY = binSizeF * (float)Ly + 0.5f * binSizeF;
    const float startZ = binSizeF * (float)Lz + 0.5f * binSizeF;

    float BinCenter[3];
    for (deme::binID_t i = Lx, ix = 0; i <= Ux; i++, ix++) {
        float cy0 = startY;
        BinCenter[0] = startX + ix * binSizeF;
        for (deme::binID_t j = Ly; j <= Uy; j++) {
            float cz = startZ;
            BinCenter[1] = cy0;
            for (deme::binID_t k = Lz; k <= Uz; k++) {
                BinCenter[2] = cz;
                const float3 c = make_float3(BinCenter[0], BinCenter[1], BinCenter[2]);

                const bool inA = ok1 && (i >= (deme::binID_t)LA.x && i <= (deme::binID_t)UA.x &&
                                        j >= (deme::binID_t)LA.y && j <= (deme::binID_t)UA.y &&
                                        k >= (deme::binID_t)LA.z && k <= (deme::binID_t)UA.z);
                const bool inB = ok2 && (i >= (deme::binID_t)LB.x && i <= (deme::binID_t)UB.x &&
                                        j >= (deme::binID_t)LB.y && j <= (deme::binID_t)UB.y &&
                                        k >= (deme::binID_t)LB.z && k <= (deme::binID_t)UB.z);
                if (!inA && !inB) {
                    cz += binSizeF;
                    continue;
                }

                const float3 a0 = make_float3(vA1.x - c.x, vA1.y - c.y, vA1.z - c.z);
                const float3 a1 = make_float3(vB1.x - c.x, vB1.y - c.y, vB1.z - c.z);
                const float3 a2 = make_float3(vC1.x - c.x, vC1.y - c.y, vC1.z - c.z);
                const bool hit = triBoxOverlapBinLocalEdgesUnionShiftFP32(a0, a1, a2, shift_world, binHalfSpan, inA, inB);
                if (hit) {
                    numSDsTouched++;
                }
                cz += binSizeF;
            }
            cy0 += binSizeF;
        }
    }

    numBinsTriTouches[triID] = numSDsTouched;

    if (meshUniversalContact) {
        deme::objID_t contact_count = 0;
        for (deme::objID_t objB = 0; objB < simParams->nAnalGM; objB++) {
            deme::bodyID_t objBOwner = objOwner[objB];
            unsigned int objFamilyNum = granData->familyID[objBOwner];
            deme::bodyID_t triOwnerID = granData->ownerTriMesh[triID];
            unsigned int triFamilyNum = granData->familyID[triOwnerID];
            unsigned int maskMatID = locateMaskPair<unsigned int>(triFamilyNum, objFamilyNum);
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

            applyOriQToVector3<float, deme::oriQ_t>(objBRelPosX, objBRelPosY, objBRelPosZ,
                                                   ownerOriQw, ownerOriQx, ownerOriQy, ownerOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(objBRotX, objBRotY, objBRotZ,
                                                   ownerOriQw, ownerOriQx, ownerOriQy, ownerOriQz);

            float3 objBPosXYZ = ownerXYZ + make_float3(objBRelPosX, objBRelPosY, objBRelPosZ);

            deme::contact_t contact_type = checkTriEntityOverlapFP32(
                vA1, vB1, vC1, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ), objSize1[objB],
                objSize2[objB], objSize3[objB], objNormal[objB], granData->marginSizeAnalytical[objB]);

            if (contact_type == deme::NOT_A_CONTACT) {
                contact_type = checkTriEntityOverlapFP32(
                    vA2, vB2, vC2, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ),
                    objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB], granData->marginSizeAnalytical[objB]);
            }

            if (contact_type == deme::TRIANGLE_ANALYTICAL_CONTACT) {
                contact_count++;
            }
        }
        numAnalGeoTriTouches[triID] = contact_count;
    }
}

__global__ void populateBinTriangleTouchingPairs(deme::DEMSimParams* simParams,
                                                         deme::DEMDataKT* granData,
                                                         deme::binsTriangleTouchPairs_t* numBinsTriTouchesScan,
                                                         deme::binsTriangleTouchPairs_t* numAnalGeoTriTouchesScan,
                                                         deme::binID_t* binIDsEachTriTouches,
                                                         deme::bodyID_t* triIDsEachBinTouches,
                                                         // precomputed
                                                         const float3* vA1_all,
                                                         const float3* vB1_all,
                                                         const float3* vC1_all,
                                                         const float3* vA2_all,
                                                         const float3* vB2_all,
                                                         const float3* vC2_all,
                                                         const float3* shift_world_all,
                                                         const int3* LA_all,
                                                         const int3* UA_all,
                                                         const int3* LB_all,
                                                         const int3* UB_all,
                                                         const unsigned char* ok1_all,
                                                         const unsigned char* ok2_all,
                                                         // tri-anal output
                                                         deme::bodyID_t* idGeoA,
                                                         deme::bodyID_t* idGeoB,
                                                         deme::contact_t* contactTypePrimitive,
                                                         bool meshUniversalContact) {
    deme::bodyID_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID >= simParams->nTriGM) {
        return;
    }

    const bool ok1 = (ok1_all[triID] != 0);
    const bool ok2 = (ok2_all[triID] != 0);

    if (!ok1 && !ok2) {
        return;
    }

    const float3 vA1 = vA1_all[triID];
    const float3 vB1 = vB1_all[triID];
    const float3 vC1 = vC1_all[triID];
    const float3 vA2 = vA2_all[triID];
    const float3 vB2 = vB2_all[triID];
    const float3 vC2 = vC2_all[triID];
    const float3 shift_world = shift_world_all[triID];

    int3 LA = make_int3(0, 0, 0), UA = make_int3(-1, -1, -1);
    int3 LB = make_int3(0, 0, 0), UB = make_int3(-1, -1, -1);
    if (ok1) {
        LA = LA_all[triID];
        UA = UA_all[triID];
    }
    if (ok2) {
        LB = LB_all[triID];
        UB = UB_all[triID];
    }

    // Union bounds
    deme::binID_t Lx, Ly, Lz, Ux, Uy, Uz;
    if (ok1 && ok2) {
        Lx = (deme::binID_t)DEME_MIN(LA.x, LB.x);
        Ly = (deme::binID_t)DEME_MIN(LA.y, LB.y);
        Lz = (deme::binID_t)DEME_MIN(LA.z, LB.z);
        Ux = (deme::binID_t)DEME_MAX(UA.x, UB.x);
        Uy = (deme::binID_t)DEME_MAX(UA.y, UB.y);
        Uz = (deme::binID_t)DEME_MAX(UA.z, UB.z);
    } else if (ok1) {
        Lx = (deme::binID_t)LA.x;
        Ly = (deme::binID_t)LA.y;
        Lz = (deme::binID_t)LA.z;
        Ux = (deme::binID_t)UA.x;
        Uy = (deme::binID_t)UA.y;
        Uz = (deme::binID_t)UA.z;
    } else {
        Lx = (deme::binID_t)LB.x;
        Ly = (deme::binID_t)LB.y;
        Lz = (deme::binID_t)LB.z;
        Ux = (deme::binID_t)UB.x;
        Uy = (deme::binID_t)UB.y;
        Uz = (deme::binID_t)UB.z;
    }

    // Write tri-bin pairs
    const deme::binsTriangleTouchPairs_t myReportOffset = numBinsTriTouchesScan[triID];
    const deme::binsTriangleTouchPairs_t myUpperBound = numBinsTriTouchesScan[triID + 1];

    deme::binsTriangleTouchPairs_t count = 0;
    const float binSizeF = (float)simParams->dyn.binSize;
    const float binHalfSpan = binSizeF * (0.5f + (float)DEME_BIN_ENLARGE_RATIO_FOR_FACETS);
    const float startX = binSizeF * (float)Lx + 0.5f * binSizeF;
    const float startY = binSizeF * (float)Ly + 0.5f * binSizeF;
    const float startZ = binSizeF * (float)Lz + 0.5f * binSizeF;

    float BinCenter[3];
    for (deme::binID_t i = Lx, ix = 0; i <= Ux; i++, ix++) {
        float cy0 = startY;
        BinCenter[0] = startX + ix * binSizeF;
        for (deme::binID_t j = Ly; j <= Uy; j++) {
            float cz = startZ;
            BinCenter[1] = cy0;
            for (deme::binID_t k = Lz; k <= Uz; k++) {
                BinCenter[2] = cz;
                const float3 c = make_float3(BinCenter[0], BinCenter[1], BinCenter[2]);

                const bool inA = ok1 && (i >= (deme::binID_t)LA.x && i <= (deme::binID_t)UA.x &&
                                        j >= (deme::binID_t)LA.y && j <= (deme::binID_t)UA.y &&
                                        k >= (deme::binID_t)LA.z && k <= (deme::binID_t)UA.z);
                const bool inB = ok2 && (i >= (deme::binID_t)LB.x && i <= (deme::binID_t)UB.x &&
                                        j >= (deme::binID_t)LB.y && j <= (deme::binID_t)UB.y &&
                                        k >= (deme::binID_t)LB.z && k <= (deme::binID_t)UB.z);
                if (!inA && !inB) {
                    cz += binSizeF;
                    continue;
                }

                const float3 a0 = make_float3(vA1.x - c.x, vA1.y - c.y, vA1.z - c.z);
                const float3 a1 = make_float3(vB1.x - c.x, vB1.y - c.y, vB1.z - c.z);
                const float3 a2 = make_float3(vC1.x - c.x, vC1.y - c.y, vC1.z - c.z);
                const bool hit = triBoxOverlapBinLocalEdgesUnionShiftFP32(a0, a1, a2, shift_world, binHalfSpan, inA, inB);
                if (hit) {
                    const deme::binsTriangleTouchPairs_t outIdx = myReportOffset + count;
                    if (outIdx < myUpperBound) {
                        binIDsEachTriTouches[outIdx] = binIDFrom3Indices<deme::binID_t>(i, j, k, simParams->nbX, simParams->nbY,
                                                                                       simParams->nbZ);
                        triIDsEachBinTouches[outIdx] = triID;
                    }
                    count++;
                }

                cz += binSizeF;
            }
            cy0 += binSizeF;
        }
    }

    // Tri-anal contacts: keep identical to original populate kernel
    if (meshUniversalContact) {
        const deme::binsTriangleTouchPairs_t myAnalOffset = numAnalGeoTriTouchesScan[triID];
        deme::binsTriangleTouchPairs_t analCount = 0;
        for (deme::objID_t objB = 0; objB < simParams->nAnalGM; objB++) {
            deme::bodyID_t objBOwner = objOwner[objB];
            unsigned int objFamilyNum = granData->familyID[objBOwner];
            deme::bodyID_t triOwnerID = granData->ownerTriMesh[triID];
            unsigned int triFamilyNum = granData->familyID[triOwnerID];
            unsigned int maskMatID = locateMaskPair<unsigned int>(triFamilyNum, objFamilyNum);
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

            applyOriQToVector3<float, deme::oriQ_t>(objBRelPosX, objBRelPosY, objBRelPosZ,
                                                   ownerOriQw, ownerOriQx, ownerOriQy, ownerOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(objBRotX, objBRotY, objBRotZ,
                                                   ownerOriQw, ownerOriQx, ownerOriQy, ownerOriQz);

            float3 objBPosXYZ = ownerXYZ + make_float3(objBRelPosX, objBRelPosY, objBRelPosZ);

            deme::contact_t contact_type = checkTriEntityOverlapFP32(
                vA1, vB1, vC1, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ), objSize1[objB],
                objSize2[objB], objSize3[objB], objNormal[objB], granData->marginSizeAnalytical[objB]);
            if (contact_type == deme::NOT_A_CONTACT) {
                contact_type = checkTriEntityOverlapFP32(
                    vA2, vB2, vC2, objType[objB], objBPosXYZ, make_float3(objBRotX, objBRotY, objBRotZ),
                    objSize1[objB], objSize2[objB], objSize3[objB], objNormal[objB], granData->marginSizeAnalytical[objB]);
            }

            if (contact_type == deme::TRIANGLE_ANALYTICAL_CONTACT) {
                const deme::binsTriangleTouchPairs_t outIdx = myAnalOffset + analCount;
                idGeoA[outIdx] = triID;
                idGeoB[outIdx] = (deme::bodyID_t)objB;
                contactTypePrimitive[outIdx] = contact_type;
                analCount++;
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
