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

inline __device__ float distSquaredPoint(const float3& p, const float3& c) {
    const float dx = p.x - c.x;
    const float dy = p.y - c.y;
    const float dz = p.z - c.z;
    return dx * dx + dy * dy + dz * dz;
}

inline __device__ void updatePlaneMinMax(const float3& p,
                                         const float3& origin,
                                         const float3& n,
                                         float& min_d,
                                         float& max_d) {
    const float3 pg = p - origin;
    const float d = dot(pg, n);
    min_d = DEME_MIN(min_d, d);
    max_d = DEME_MAX(max_d, d);
}

DEME_KERNEL void makeTriangleSandwich(deme::DEMSimParams* simParams,
                                      deme::DEMDataKT* granData,
                                      float3* sandwichANode1,
                                      float3* sandwichANode2,
                                      float3* sandwichANode3,
                                      float3* sandwichBNode1) {
    deme::bodyID_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID < simParams->nTriGM) {
        // Get my component offset info from global array
        const float3 p1 = granData->relPosNode1[triID];
        const float3 p2 = granData->relPosNode2[triID];
        const float3 p3 = granData->relPosNode3[triID];
        float margin = granData->marginSizeTriangle[triID];
        if (granData->ownerMeshShellHalfThickness) {
            const deme::bodyID_t ownerID = granData->ownerTriMesh[triID];
            margin += fmaxf(granData->ownerMeshShellHalfThickness[ownerID], 0.f);
        }

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
    }
}

// Precompute mesh-owner pose (position + rotation matrix rows) once per time step.
// Mesh owners are assumed to occupy the last `nTriMeshes` owner slots:
// [0 .. nOwnerClumps-1] clumps, [.. + nExtObj -1] external objects, [.. + nTriMeshes -1] meshes.
// This matches kT initialization (owner_offset_for_mesh_obj = nExistOwners + nClumps + nExtObj).
DEME_KERNEL void precomputeMeshOwnerPose(deme::DEMSimParams* simParams,
                                         deme::DEMDataKT* granData,
                                         float3* meshOwnerPos,  // length nTriMeshes
                                         float3* meshR1,        // length nTriMeshes (row 0)
                                         float3* meshR2,        // length nTriMeshes (row 1)
                                         float3* meshR3) {      // length nTriMeshes (row 2)
    const deme::bodyID_t mesh_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (mesh_i >= simParams->nTriMeshes) {
        return;
    }
    const deme::bodyID_t mesh_owner_start = simParams->nOwnerBodies - simParams->nTriMeshes;
    const deme::bodyID_t ownerID = mesh_owner_start + mesh_i;

    float3 ownerXYZ;
    voxelIDToPosition<float, deme::voxelID_t, deme::subVoxelPos_t>(
        ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[ownerID], granData->locX[ownerID],
        granData->locY[ownerID], granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
    meshOwnerPos[mesh_i] = ownerXYZ;

    const float qw = granData->oriQw[ownerID];
    const float qx = granData->oriQx[ownerID];
    const float qy = granData->oriQy[ownerID];
    const float qz = granData->oriQz[ownerID];

    // Convert quaternion to rotation matrix rows (right-handed, active rotation).
    const float xx = qx * qx;
    const float yy = qy * qy;
    const float zz = qz * qz;
    const float xy = qx * qy;
    const float xz = qx * qz;
    const float yz = qy * qz;
    const float wx = qw * qx;
    const float wy = qw * qy;
    const float wz = qw * qz;

    meshR1[mesh_i] = make_float3(1.f - 2.f * (yy + zz), 2.f * (xy - wz), 2.f * (xz + wy));
    meshR2[mesh_i] = make_float3(2.f * (xy + wz), 1.f - 2.f * (xx + zz), 2.f * (yz - wx));
    meshR3[mesh_i] = make_float3(2.f * (xz - wy), 2.f * (yz + wx), 1.f - 2.f * (xx + yy));
}

inline __device__ float3 applyRotRows(const float3& v, const float3& r1, const float3& r2, const float3& r3) {
    return make_float3(dot(r1, v), dot(r2, v), dot(r3, v));
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
    const float4 myOriQ = make_float4(myOriQx, myOriQy, myOriQz, myOriQw);
    applyOriQToVector3(loc_vA, myOriQ);
    applyOriQToVector3(loc_vB, myOriQ);
    applyOriQToVector3(loc_vC, myOriQ);
    vA = ownerXYZ + loc_vA;
    vB = ownerXYZ + loc_vB;
    vC = ownerXYZ + loc_vC;

    return boundingBoxIntersectBinAxisBounds(L, U, vA, vB, vC, simParams);
}

DEME_KERNEL void precomputeTriangleSandwichData(
    deme::DEMSimParams* simParams,
    deme::DEMDataKT* granData,
    // World-space vertices for A-face triangle
    float3* vA1_all,
    float3* vB1_all,
    float3* vC1_all,
    // Per-triangle translation B = A + shift_world (world-space)
    float3* shift_world_all,
    // Per-triangle bounds for A and B (only valid if ok flag true)
    int3* LA_all,
    int3* UA_all,
    int3* LB_all,
    int3* UB_all,
    // ok flags for A and B
    unsigned char* ok1_all,
    unsigned char* ok2_all,
    // Precomputed mesh-owner pose (length nTriMeshes); may be nullptr if no meshes
    const float3* meshOwnerPos,
    const float3* meshR1,
    const float3* meshR2,
    const float3* meshR3,
    // sandwich nodes (local, as produced by makeTriangleSandwich)
    const float3* nodeA1,
    const float3* nodeA2,
    const float3* nodeA3,
    const float3* nodeB1_only) {
    const deme::bodyID_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID >= simParams->nTriGM) {
        return;
    }

    const deme::bodyID_t ownerID = granData->ownerTriMesh[triID];
    const deme::bodyID_t mesh_owner_start = simParams->nOwnerBodies - simParams->nTriMeshes;

    float3 ownerXYZ;
    float3 r1, r2, r3;

    // Fast path: mesh owners live in [mesh_owner_start, mesh_owner_start + nTriMeshes)
    if (simParams->nTriMeshes > 0 && meshOwnerPos && ownerID >= mesh_owner_start &&
        ownerID < (mesh_owner_start + (deme::bodyID_t)simParams->nTriMeshes)) {
        const deme::bodyID_t mi = ownerID - mesh_owner_start;
        ownerXYZ = meshOwnerPos[mi];
        r1 = meshR1[mi];
        r2 = meshR2[mi];
        r3 = meshR3[mi];
    } else {
        // Fallback: compute pose directly
        voxelIDToPosition<float, deme::voxelID_t, deme::subVoxelPos_t>(
            ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[ownerID], granData->locX[ownerID],
            granData->locY[ownerID], granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);

        const float qw = granData->oriQw[ownerID];
        const float qx = granData->oriQx[ownerID];
        const float qy = granData->oriQy[ownerID];
        const float qz = granData->oriQz[ownerID];

        const float xx = qx * qx;
        const float yy = qy * qy;
        const float zz = qz * qz;
        const float xy = qx * qy;
        const float xz = qx * qz;
        const float yz = qy * qz;
        const float wx = qw * qx;
        const float wy = qw * qy;
        const float wz = qw * qz;

        r1 = make_float3(1.f - 2.f * (yy + zz), 2.f * (xy - wz), 2.f * (xz + wy));
        r2 = make_float3(2.f * (xy + wz), 1.f - 2.f * (xx + zz), 2.f * (yz - wx));
        r3 = make_float3(2.f * (xz - wy), 2.f * (yz + wx), 1.f - 2.f * (xx + yy));
    }

    // Transform A-face nodes to world.
    const float3 lA1 = nodeA1[triID];
    const float3 lA2 = nodeA2[triID];
    const float3 lA3 = nodeA3[triID];

    const float3 vA1 = ownerXYZ + applyRotRows(lA1, r1, r2, r3);
    const float3 vB1 = ownerXYZ + applyRotRows(lA2, r1, r2, r3);
    const float3 vC1 = ownerXYZ + applyRotRows(lA3, r1, r2, r3);

    vA1_all[triID] = vA1;
    vB1_all[triID] = vB1;
    vC1_all[triID] = vC1;

    // Compute shift_world from B1 - A1 in local, then rotate (no translation).
    const float3 lB1 = nodeB1_only[triID];
    float3 shift_local = make_float3(lB1.x - lA1.x, lB1.y - lA1.y, lB1.z - lA1.z);
    const float3 shift_world = applyRotRows(shift_local, r1, r2, r3);
    shift_world_all[triID] = shift_world;

    // Compute bounds for A and B (B is reconstructed from A + shift; note permutation for opposite normal).
    deme::binID_t L1[3], U1[3], L2[3], U2[3];
    const bool ok1 = boundingBoxIntersectBinAxisBounds(L1, U1, vA1, vB1, vC1, simParams);

    const float3 vA2 = vA1 + shift_world;
    const float3 vB2 = vC1 + shift_world;  // swap 2<->3 for inverted normal
    const float3 vC2 = vB1 + shift_world;
    const bool ok2 = boundingBoxIntersectBinAxisBounds(L2, U2, vA2, vB2, vC2, simParams);

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
}

DEME_KERNEL void getNumberOfBinsEachTriangleTouches(deme::DEMSimParams* simParams,
                                                    deme::DEMDataKT* granData,
                                                    deme::binsTriangleTouches_t* numBinsTriTouches,
                                                    deme::objID_t* numAnalGeoTriTouches,
                                                    // precomputed
                                                    const float3* vA1_all,
                                                    const float3* vB1_all,
                                                    const float3* vC1_all,
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
    const float3 shift_world = shift_world_all[triID];
    const float3 vA2 = vA1 + shift_world;
    const float3 vB2 = vC1 + shift_world;  // swapped
    const float3 vC2 = vB1 + shift_world;

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

    // Incremental bin-local coordinates: avoid recomputing (v - c) for every bin.
    for (deme::binID_t i = Lx, ix = 0; i <= Ux; i++, ix++) {
        const float cx = startX + (float)ix * binSizeF;

        const float a0x = vA1.x - cx;
        const float a1x = vB1.x - cx;
        const float a2x = vC1.x - cx;

        float cy = startY;
        for (deme::binID_t j = Ly; j <= Uy; j++) {
            const float a0y = vA1.y - cy;
            const float a1y = vB1.y - cy;
            const float a2y = vC1.y - cy;

            float3 a0 = make_float3(a0x, a0y, vA1.z - startZ);
            float3 a1 = make_float3(a1x, a1y, vB1.z - startZ);
            float3 a2 = make_float3(a2x, a2y, vC1.z - startZ);

            for (deme::binID_t k = Lz; k <= Uz; k++) {
                const bool inA =
                    ok1 && (i >= (deme::binID_t)LA.x && i <= (deme::binID_t)UA.x && j >= (deme::binID_t)LA.y &&
                            j <= (deme::binID_t)UA.y && k >= (deme::binID_t)LA.z && k <= (deme::binID_t)UA.z);
                const bool inB =
                    ok2 && (i >= (deme::binID_t)LB.x && i <= (deme::binID_t)UB.x && j >= (deme::binID_t)LB.y &&
                            j <= (deme::binID_t)UB.y && k >= (deme::binID_t)LB.z && k <= (deme::binID_t)UB.z);

                if (inA || inB) {
                    const bool hit =
                        triBoxOverlapBinLocalEdgesUnionShiftFP32(a0, a1, a2, shift_world, binHalfSpan, inA, inB);
                    if (hit) {
                        numSDsTouched++;
                    }
                }

                // Advance bin center in +Z => (v.z - cz) decreases.
                a0.z -= binSizeF;
                a1.z -= binSizeF;
                a2.z -= binSizeF;
            }
            cy += binSizeF;
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

            const float4 ownerOriQ = make_float4(ownerOriQx, ownerOriQy, ownerOriQz, ownerOriQw);
            float3 objBRelPos = make_float3(objRelPosX[objB], objRelPosY[objB], objRelPosZ[objB]);
            float3 objBRot = make_float3(objRotX[objB], objRotY[objB], objRotZ[objB]);

            applyOriQToVector3(objBRelPos, ownerOriQ);
            applyOriQToVector3(objBRot, ownerOriQ);

            float3 objBPosXYZ = ownerXYZ + objBRelPos;

            deme::contact_t contact_type = checkTriEntityOverlapFP32(
                vA1, vB1, vC1, objType[objB], objBPosXYZ, objBRot, objSize1[objB], objSize2[objB], objSize3[objB],
                objNormal[objB], granData->marginSizeAnalytical[objB]);

            if (contact_type == deme::NOT_A_CONTACT) {
                contact_type = checkTriEntityOverlapFP32(vA2, vB2, vC2, objType[objB], objBPosXYZ, objBRot,
                                                         objSize1[objB], objSize2[objB], objSize3[objB],
                                                         objNormal[objB], granData->marginSizeAnalytical[objB]);
            }

            if (contact_type == deme::TRIANGLE_ANALYTICAL_CONTACT) {
                contact_count++;
            }
        }
        numAnalGeoTriTouches[triID] = contact_count;
    }
}

DEME_KERNEL void populateBinTriangleTouchingPairs(deme::DEMSimParams* simParams,
                                                  deme::DEMDataKT* granData,
                                                  deme::binsTriangleTouchPairs_t* numBinsTriTouchesScan,
                                                  deme::binsTriangleTouchPairs_t* numAnalGeoTriTouchesScan,
                                                  deme::binID_t* binIDsEachTriTouches,
                                                  deme::bodyID_t* triIDsEachBinTouches,
                                                  // precomputed
                                                  const float3* vA1_all,
                                                  const float3* vB1_all,
                                                  const float3* vC1_all,
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
    const float3 shift_world = shift_world_all[triID];
    const float3 vA2 = vA1 + shift_world;
    const float3 vB2 = vC1 + shift_world;  // swapped
    const float3 vC2 = vB1 + shift_world;

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

    // Incremental bin-local coordinates: avoid recomputing (v - c) for every bin.
    for (deme::binID_t i = Lx, ix = 0; i <= Ux; i++, ix++) {
        const float cx = startX + (float)ix * binSizeF;

        const float a0x = vA1.x - cx;
        const float a1x = vB1.x - cx;
        const float a2x = vC1.x - cx;

        float cy = startY;
        for (deme::binID_t j = Ly; j <= Uy; j++) {
            const float a0y = vA1.y - cy;
            const float a1y = vB1.y - cy;
            const float a2y = vC1.y - cy;

            float3 a0 = make_float3(a0x, a0y, vA1.z - startZ);
            float3 a1 = make_float3(a1x, a1y, vB1.z - startZ);
            float3 a2 = make_float3(a2x, a2y, vC1.z - startZ);

            for (deme::binID_t k = Lz; k <= Uz; k++) {
                const bool inA =
                    ok1 && (i >= (deme::binID_t)LA.x && i <= (deme::binID_t)UA.x && j >= (deme::binID_t)LA.y &&
                            j <= (deme::binID_t)UA.y && k >= (deme::binID_t)LA.z && k <= (deme::binID_t)UA.z);
                const bool inB =
                    ok2 && (i >= (deme::binID_t)LB.x && i <= (deme::binID_t)UB.x && j >= (deme::binID_t)LB.y &&
                            j <= (deme::binID_t)UB.y && k >= (deme::binID_t)LB.z && k <= (deme::binID_t)UB.z);

                if (inA || inB) {
                    const bool hit =
                        triBoxOverlapBinLocalEdgesUnionShiftFP32(a0, a1, a2, shift_world, binHalfSpan, inA, inB);
                    if (hit) {
                        const deme::binsTriangleTouchPairs_t outIdx = myReportOffset + count;
                        if (outIdx < myUpperBound) {
                            binIDsEachTriTouches[outIdx] = binIDFrom3Indices<deme::binID_t>(
                                i, j, k, simParams->nbX, simParams->nbY, simParams->nbZ);
                            triIDsEachBinTouches[outIdx] = triID;
                        }
                        count++;
                    }
                }

                a0.z -= binSizeF;
                a1.z -= binSizeF;
                a2.z -= binSizeF;
            }
            cy += binSizeF;
        }
    }
    // As an ultra-safety net, neutralize any reserved-but-unwritten slots.
    // Small count/populate mismatches (e.g., due floating-point branch jitter) should not leak stale bin IDs.
    for (deme::binsTriangleTouchPairs_t outIdx = myReportOffset + count; outIdx < myUpperBound; ++outIdx) {
        binIDsEachTriTouches[outIdx] = deme::NULL_BINID;
        triIDsEachBinTouches[outIdx] = triID;
    }

    // Tri-anal contacts: keep identical to original populate kernel
    if (meshUniversalContact) {
        const deme::binsTriangleTouchPairs_t myAnalOffset = numAnalGeoTriTouchesScan[triID];
        const deme::binsTriangleTouchPairs_t myAnalUpperBound = numAnalGeoTriTouchesScan[triID + 1];
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

            const float4 ownerOriQ = make_float4(ownerOriQx, ownerOriQy, ownerOriQz, ownerOriQw);
            float3 objBRelPos = make_float3(objRelPosX[objB], objRelPosY[objB], objRelPosZ[objB]);
            float3 objBRot = make_float3(objRotX[objB], objRotY[objB], objRotZ[objB]);

            applyOriQToVector3(objBRelPos, ownerOriQ);
            applyOriQToVector3(objBRot, ownerOriQ);

            float3 objBPosXYZ = ownerXYZ + objBRelPos;

            deme::contact_t contact_type = checkTriEntityOverlapFP32(
                vA1, vB1, vC1, objType[objB], objBPosXYZ, objBRot, objSize1[objB], objSize2[objB], objSize3[objB],
                objNormal[objB], granData->marginSizeAnalytical[objB]);
            if (contact_type == deme::NOT_A_CONTACT) {
                contact_type = checkTriEntityOverlapFP32(vA2, vB2, vC2, objType[objB], objBPosXYZ, objBRot,
                                                         objSize1[objB], objSize2[objB], objSize3[objB],
                                                         objNormal[objB], granData->marginSizeAnalytical[objB]);
            }

            if (contact_type == deme::TRIANGLE_ANALYTICAL_CONTACT) {
                const deme::binsTriangleTouchPairs_t outIdx = myAnalOffset + analCount;
                if (outIdx < myAnalUpperBound) {
                    idGeoA[outIdx] = triID;
                    idGeoB[outIdx] = (deme::bodyID_t)objB;
                    contactTypePrimitive[outIdx] = contact_type;
                }
                analCount++;
            }
        }
        // Keep unwritten reserved slots deterministic to avoid stale/invalid contact types.
        for (deme::binsTriangleTouchPairs_t outIdx = myAnalOffset + analCount; outIdx < myAnalUpperBound; ++outIdx) {
            contactTypePrimitive[outIdx] = deme::NOT_A_CONTACT;
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
