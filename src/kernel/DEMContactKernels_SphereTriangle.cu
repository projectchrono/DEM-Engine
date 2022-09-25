// DEM contact detection-related custom kernels
#include <DEM/Defines.h>
#include <kernel/DEMHelperKernels.cu>
#include <kernel/DEMCollisionKernels.cu>

// #include <cub/block/block_load.cuh>
// #include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

// If clump templates are jitified, they will be below
_clumpTemplateDefs_;
// Family mask, _nFamilyMaskEntries_ elements are in this array
// __constant__ __device__ bool familyMasks[] = {_familyMasks_};

__global__ void getNumberOfSphTriContactsEachBin(deme::DEMSimParams* simParams,
                                                 deme::DEMDataKT* granData,
                                                 deme::bodyID_t* sphereIDsEachBinTouches_sorted,
                                                 deme::binID_t* activeBinIDs,
                                                 deme::spheresBinTouches_t* numSpheresBinTouches,
                                                 deme::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                                 deme::binID_t* mapTriActBinToSphActBin,
                                                 deme::bodyID_t* triIDsEachBinTouches_sorted,
                                                 deme::binID_t* activeBinIDsForTri,
                                                 deme::trianglesBinTouches_t* numTrianglesBinTouches,
                                                 deme::binsTriangleTouchPairs_t* triIDsLookUpTable,
                                                 deme::spheresBinTouches_t* numTriSphContactsInEachBin,
                                                 float3* sandwichANode1,
                                                 float3* sandwichANode2,
                                                 float3* sandwichANode3,
                                                 float3* sandwichBNode1,
                                                 float3* sandwichBNode2,
                                                 float3* sandwichBNode3,
                                                 size_t nActiveBinsForTri) {
    // Shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ deme::bodyID_t triOwnerIDs[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ float3 triANode1[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ float3 triANode2[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ float3 triANode3[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ float3 triBNode1[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ float3 triBNode2[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ float3 triBNode3[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ deme::family_t triOwnerFamilies[DEME_MAX_TRIANGLES_PER_BIN];

    typedef cub::BlockReduce<deme::spheresBinTouches_t, DEME_MAX_SPHERES_PER_BIN> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    const deme::trianglesBinTouches_t nTriInBin = numTrianglesBinTouches[blockIdx.x];
    // No need to check if spheres exceed max now... already did in another kernel
    if (threadIdx.x == 0 && nTriInBin > DEME_MAX_TRIANGLES_PER_BIN) {
        DEME_ABORT_KERNEL("Bin %u contains %u triangular mesh facets, exceeding maximum allowance (%u)\n", blockIdx.x,
                          nTriInBin, DEME_MAX_TRIANGLES_PER_BIN);
    }
    deme::spheresBinTouches_t myThreadID = threadIdx.x;
    const deme::binID_t binID = activeBinIDsForTri[blockIdx.x];
    // But what is the index of the same binID in array activeBinIDs? Well, mapTriActBinToSphActBin comes to rescure.
    const deme::binID_t indForAcqSphInfo = mapTriActBinToSphActBin[blockIdx.x];
    // If it is not an active bin from the perspective of the spheres, then we can move on
    if (indForAcqSphInfo == deme::NULL_BINID) {
        if (threadIdx.x == 0) {
            numTriSphContactsInEachBin[blockIdx.x] = 0;
        }
        return;
    }

    // If got here, this bin is active. I need to work on shared memory allocation... for triangles.
    if (myThreadID < nTriInBin) {
        deme::binsTriangleTouchPairs_t thisTriTableEntry = triIDsLookUpTable[blockIdx.x];
        deme::bodyID_t triID = triIDsEachBinTouches_sorted[thisTriTableEntry + myThreadID];
        deme::bodyID_t ownerID = granData->ownerMesh[triID];
        triOwnerIDs[myThreadID] = ownerID;
        triOwnerFamilies[myThreadID] = granData->familyID[ownerID];
        double3 ownerXYZ;
        float3 node1, node2, node3;

        voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
            ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[ownerID], granData->locX[ownerID],
            granData->locY[ownerID], granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        float myOriQw = granData->oriQw[ownerID];
        float myOriQx = granData->oriQx[ownerID];
        float myOriQy = granData->oriQy[ownerID];
        float myOriQz = granData->oriQz[ownerID];

        {
            node1 = sandwichANode1[triID];
            node2 = sandwichANode2[triID];
            node3 = sandwichANode3[triID];
            applyOriQToVector3<float, deme::oriQ_t>(node1.x, node1.y, node1.z, myOriQw, myOriQx, myOriQy, myOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(node2.x, node2.y, node2.z, myOriQw, myOriQx, myOriQy, myOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(node3.x, node3.y, node3.z, myOriQw, myOriQx, myOriQy, myOriQz);
            triANode1[myThreadID] = ownerXYZ + node1;
            triANode2[myThreadID] = ownerXYZ + node2;
            triANode3[myThreadID] = ownerXYZ + node3;
        }
        {
            node1 = sandwichBNode1[triID];
            node2 = sandwichBNode2[triID];
            node3 = sandwichBNode3[triID];
            applyOriQToVector3<float, deme::oriQ_t>(node1.x, node1.y, node1.z, myOriQw, myOriQx, myOriQy, myOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(node2.x, node2.y, node2.z, myOriQw, myOriQx, myOriQy, myOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(node3.x, node3.y, node3.z, myOriQw, myOriQx, myOriQy, myOriQz);
            triBNode1[myThreadID] = ownerXYZ + node1;
            triBNode2[myThreadID] = ownerXYZ + node2;
            triBNode3[myThreadID] = ownerXYZ + node3;
        }
    }
    __syncthreads();

    deme::spheresBinTouches_t contact_count = 0;
    const deme::spheresBinTouches_t nSphInBin = numSpheresBinTouches[indForAcqSphInfo];
    if (myThreadID < nSphInBin) {
        // Have to allocate my sphere info first (each thread handles a sphere)
        const deme::binSphereTouchPairs_t thisSphereTableEntry = sphereIDsLookUpTable[indForAcqSphInfo];

        deme::bodyID_t sphereID = sphereIDsEachBinTouches_sorted[thisSphereTableEntry + myThreadID];
        deme::bodyID_t ownerID = granData->ownerClumpBody[sphereID];
        deme::family_t ownerFamily = granData->familyID[ownerID];

        float myRadius;
        float3 sphXYZ;
        {
            float3 myRelPos;
            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPos, myRadius (in CD kernels, radius needs to be expanded)
            // Use an input named exactly `sphereID' which is the id of this sphere component
            {
                _componentAcqStrat_;
                myRadius += simParams->beta;
            }

            double ownerX, ownerY, ownerZ;
            voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[ownerID], granData->locX[ownerID], granData->locY[ownerID],
                granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            float myOriQw = granData->oriQw[ownerID];
            float myOriQx = granData->oriQx[ownerID];
            float myOriQy = granData->oriQy[ownerID];
            float myOriQz = granData->oriQz[ownerID];
            applyOriQToVector3<float, deme::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, myOriQw, myOriQx, myOriQy,
                                                    myOriQz);
            sphXYZ.x = ownerX + (double)myRelPos.x;
            sphXYZ.y = ownerY + (double)myRelPos.y;
            sphXYZ.z = ownerZ + (double)myRelPos.z;
        }

        // We can stop if this thread reaches the end of all potential pairs, nPairsNeedHandling
        for (unsigned int ind = 0; ind < nTriInBin; ind++) {
            // A mesh facet and a sphere may have the same owner... although it is not possible with the current
            // implementation...
            if (ownerID == triOwnerIDs[ind])
                continue;

            // Grab family number from memory
            unsigned int maskMatID = locateMaskPair<unsigned int>(ownerFamily, triOwnerFamilies[ind]);
            // If marked no contact, skip ths iteration
            if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                continue;
            }

            float3 cntPnt, normal;
            float depth;
            bool in_contact;
            // NOTE: triangle_sphere_CD_directional, instead of triangle_sphere_CD, is in use here. This is because if
            // the later is in use, then if a sphere is between 2 sandwiching triangles, then its potential contact with
            // the original triangle will not be registered. At the same time, we don't want to use
            // triangle_sphere_CD_directional for the real force calculation, and the concern is mainly "sphere near
            // needle tip" scenario. Think about it.
            in_contact = triangle_sphere_CD_directional<float3, float>(triANode1[ind], triANode2[ind], triANode3[ind],
                                                                       sphXYZ, myRadius, normal, depth, cntPnt);
            deme::binID_t contactPntBin = getPointBinID<deme::binID_t>(cntPnt.x, cntPnt.y, cntPnt.z, simParams->binSize,
                                                                       simParams->nbX, simParams->nbY);

            // If already in contact with A, no need to do B
            if (in_contact && (contactPntBin == binID)) {
                contact_count++;
                continue;
            }
            // And triangle B...
            in_contact = triangle_sphere_CD_directional<float3, float>(triBNode1[ind], triBNode2[ind], triBNode3[ind],
                                                                       sphXYZ, myRadius, normal, depth, cntPnt);
            contactPntBin = getPointBinID<deme::binID_t>(cntPnt.x, cntPnt.y, cntPnt.z, simParams->binSize,
                                                         simParams->nbX, simParams->nbY);
            if (in_contact && (contactPntBin == binID)) {
                contact_count++;
            }
        }
    }
    __syncthreads();
    deme::spheresBinTouches_t total_count = BlockReduceT(temp_storage).Sum(contact_count);
    if (myThreadID == 0) {
        numTriSphContactsInEachBin[blockIdx.x] = total_count;
    }
}

__global__ void populateTriSphContactsEachBin(deme::DEMSimParams* simParams,
                                              deme::DEMDataKT* granData,
                                              deme::bodyID_t* sphereIDsEachBinTouches_sorted,
                                              deme::binID_t* activeBinIDs,
                                              deme::spheresBinTouches_t* numSpheresBinTouches,
                                              deme::binSphereTouchPairs_t* sphereIDsLookUpTable,
                                              deme::binID_t* mapTriActBinToSphActBin,
                                              deme::bodyID_t* triIDsEachBinTouches_sorted,
                                              deme::binID_t* activeBinIDsForTri,
                                              deme::trianglesBinTouches_t* numTrianglesBinTouches,
                                              deme::binsTriangleTouchPairs_t* triIDsLookUpTable,
                                              deme::contactPairs_t* triSphContactReportOffsets,
                                              deme::bodyID_t* idSphA,
                                              deme::bodyID_t* idTriB,
                                              deme::contact_t* dType,
                                              float3* sandwichANode1,
                                              float3* sandwichANode2,
                                              float3* sandwichANode3,
                                              float3* sandwichBNode1,
                                              float3* sandwichBNode2,
                                              float3* sandwichBNode3,
                                              size_t nActiveBinsForTri) {
    // Shared storage for bodies involved in this bin. Pre-allocated so that each threads can easily use.
    __shared__ deme::bodyID_t triOwnerIDs[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ deme::bodyID_t triIDs[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ float3 triANode1[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ float3 triANode2[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ float3 triANode3[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ float3 triBNode1[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ float3 triBNode2[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ float3 triBNode3[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ deme::family_t triOwnerFamilies[DEME_MAX_TRIANGLES_PER_BIN];
    __shared__ unsigned int blockPairCnt;

    const deme::trianglesBinTouches_t nTriInBin = numTrianglesBinTouches[blockIdx.x];
    // No need to check if spheres exceed max now... already did in another kernel
    if (threadIdx.x == 0 && nTriInBin > DEME_MAX_TRIANGLES_PER_BIN) {
        DEME_ABORT_KERNEL("Bin %u contains %u triangular mesh facets, exceeding maximum allowance (%u)\n", blockIdx.x,
                          nTriInBin, DEME_MAX_TRIANGLES_PER_BIN);
    }
    deme::spheresBinTouches_t myThreadID = threadIdx.x;
    const deme::binID_t binID = activeBinIDsForTri[blockIdx.x];
    // But what is the index of the same binID in array activeBinIDs? Well, mapTriActBinToSphActBin comes to rescure.
    const deme::binID_t indForAcqSphInfo = mapTriActBinToSphActBin[blockIdx.x];
    // If it is not an active bin from the perspective of the spheres, then we can move on
    if (indForAcqSphInfo == deme::NULL_BINID) {
        return;
    }

    // If got here, this bin is active. I need to work on shared memory allocation... for triangles.
    if (myThreadID < nTriInBin) {
        if (myThreadID == 0)
            blockPairCnt = 0;
        deme::binsTriangleTouchPairs_t thisTriTableEntry = triIDsLookUpTable[blockIdx.x];
        deme::bodyID_t triID = triIDsEachBinTouches_sorted[thisTriTableEntry + myThreadID];
        deme::bodyID_t ownerID = granData->ownerMesh[triID];
        triOwnerIDs[myThreadID] = ownerID;
        triIDs[myThreadID] = triID;
        triOwnerFamilies[myThreadID] = granData->familyID[ownerID];
        double3 ownerXYZ;
        float3 node1, node2, node3;

        voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
            ownerXYZ.x, ownerXYZ.y, ownerXYZ.z, granData->voxelID[ownerID], granData->locX[ownerID],
            granData->locY[ownerID], granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
        float myOriQw = granData->oriQw[ownerID];
        float myOriQx = granData->oriQx[ownerID];
        float myOriQy = granData->oriQy[ownerID];
        float myOriQz = granData->oriQz[ownerID];

        {
            node1 = sandwichANode1[triID];
            node2 = sandwichANode2[triID];
            node3 = sandwichANode3[triID];
            applyOriQToVector3<float, deme::oriQ_t>(node1.x, node1.y, node1.z, myOriQw, myOriQx, myOriQy, myOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(node2.x, node2.y, node2.z, myOriQw, myOriQx, myOriQy, myOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(node3.x, node3.y, node3.z, myOriQw, myOriQx, myOriQy, myOriQz);
            triANode1[myThreadID] = ownerXYZ + node1;
            triANode2[myThreadID] = ownerXYZ + node2;
            triANode3[myThreadID] = ownerXYZ + node3;
        }
        {
            node1 = sandwichBNode1[triID];
            node2 = sandwichBNode2[triID];
            node3 = sandwichBNode3[triID];
            applyOriQToVector3<float, deme::oriQ_t>(node1.x, node1.y, node1.z, myOriQw, myOriQx, myOriQy, myOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(node2.x, node2.y, node2.z, myOriQw, myOriQx, myOriQy, myOriQz);
            applyOriQToVector3<float, deme::oriQ_t>(node3.x, node3.y, node3.z, myOriQw, myOriQx, myOriQy, myOriQz);
            triBNode1[myThreadID] = ownerXYZ + node1;
            triBNode2[myThreadID] = ownerXYZ + node2;
            triBNode3[myThreadID] = ownerXYZ + node3;
        }
    }
    __syncthreads();

    // Get my offset for writing back to the global arrays that contain contact pair info
    const deme::contactPairs_t myReportOffset = triSphContactReportOffsets[blockIdx.x];

    const deme::spheresBinTouches_t nSphInBin = numSpheresBinTouches[indForAcqSphInfo];
    if (myThreadID < nSphInBin) {
        // Have to allocate my sphere info first (each thread handles a sphere)
        const deme::binSphereTouchPairs_t thisSphereTableEntry = sphereIDsLookUpTable[indForAcqSphInfo];

        deme::bodyID_t sphereID = sphereIDsEachBinTouches_sorted[thisSphereTableEntry + myThreadID];
        deme::bodyID_t ownerID = granData->ownerClumpBody[sphereID];
        deme::family_t ownerFamily = granData->familyID[ownerID];

        float myRadius;
        float3 sphXYZ;
        {
            float3 myRelPos;
            // Get my component offset info from either jitified arrays or global memory
            // Outputs myRelPos, myRadius (in CD kernels, radius needs to be expanded)
            // Use an input named exactly `sphereID' which is the id of this sphere component
            {
                _componentAcqStrat_;
                myRadius += simParams->beta;
            }

            double ownerX, ownerY, ownerZ;
            voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                ownerX, ownerY, ownerZ, granData->voxelID[ownerID], granData->locX[ownerID], granData->locY[ownerID],
                granData->locZ[ownerID], _nvXp2_, _nvYp2_, _voxelSize_, _l_);
            float myOriQw = granData->oriQw[ownerID];
            float myOriQx = granData->oriQx[ownerID];
            float myOriQy = granData->oriQy[ownerID];
            float myOriQz = granData->oriQz[ownerID];
            applyOriQToVector3<float, deme::oriQ_t>(myRelPos.x, myRelPos.y, myRelPos.z, myOriQw, myOriQx, myOriQy,
                                                    myOriQz);
            sphXYZ.x = ownerX + (double)myRelPos.x;
            sphXYZ.y = ownerY + (double)myRelPos.y;
            sphXYZ.z = ownerZ + (double)myRelPos.z;
        }

        // We can stop if this thread reaches the end of all potential pairs, nPairsNeedHandling
        for (unsigned int ind = 0; ind < nTriInBin; ind++) {
            // A mesh facet and a sphere may have the same owner... although it is not possible with the current
            // implementation...
            if (ownerID == triOwnerIDs[ind])
                continue;

            // Grab family number from memory
            unsigned int maskMatID = locateMaskPair<unsigned int>(ownerFamily, triOwnerFamilies[ind]);
            // If marked no contact, skip ths iteration
            if (granData->familyMasks[maskMatID] != deme::DONT_PREVENT_CONTACT) {
                continue;
            }

            float3 cntPnt, normal;
            float depth;
            bool in_contact;
            // NOTE: triangle_sphere_CD_directional, instead of triangle_sphere_CD, is in use here. This is because if
            // the later is in use, then if a sphere is between 2 sandwiching triangles, then its potential contact with
            // the original triangle will not be registered. At the same time, we don't want to use
            // triangle_sphere_CD_directional for the real force calculation, and the concern is mainly "sphere near
            // needle tip" scenario. Think about it.
            in_contact = triangle_sphere_CD_directional<float3, float>(triANode1[ind], triANode2[ind], triANode3[ind],
                                                                       sphXYZ, myRadius, normal, depth, cntPnt);
            deme::binID_t contactPntBin = getPointBinID<deme::binID_t>(cntPnt.x, cntPnt.y, cntPnt.z, simParams->binSize,
                                                                       simParams->nbX, simParams->nbY);

            // If already in contact with A, no need to do B
            if (in_contact && (contactPntBin == binID)) {
                unsigned int inBlockOffset = atomicAdd_block(&blockPairCnt, 1);
                idSphA[myReportOffset + inBlockOffset] = sphereID;
                idTriB[myReportOffset + inBlockOffset] = triIDs[ind];
                dType[myReportOffset + inBlockOffset] = deme::SPHERE_MESH_CONTACT;
                continue;
            }
            // And triangle B...
            in_contact = triangle_sphere_CD_directional<float3, float>(triBNode1[ind], triBNode2[ind], triBNode3[ind],
                                                                       sphXYZ, myRadius, normal, depth, cntPnt);
            contactPntBin = getPointBinID<deme::binID_t>(cntPnt.x, cntPnt.y, cntPnt.z, simParams->binSize,
                                                         simParams->nbX, simParams->nbY);
            if (in_contact && (contactPntBin == binID)) {
                unsigned int inBlockOffset = atomicAdd_block(&blockPairCnt, 1);
                idSphA[myReportOffset + inBlockOffset] = sphereID;
                idTriB[myReportOffset + inBlockOffset] = triIDs[ind];
                dType[myReportOffset + inBlockOffset] = deme::SPHERE_MESH_CONTACT;
            }
        }
    }
}
