// DEM history mapping related custom kernels

#ifndef DEME_CD_KERNELS_CUH
#define DEME_CD_KERNELS_CUH

#include <DEM/Defines.h>
#include <kernel/DEMHelperKernels.cuh>

#include <stdio.h>

namespace deme {

// Kernel to add a constant offset to each element of an array
__global__ void addOffsetToArray(contactPairs_t* arr, contactPairs_t offset, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] += offset;
    }
}

// Kernel to fill a contact_t array with a constant value
__global__ void fillContactTypeArray(contact_t* arr, contact_t val, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] = val;
    }
}

__global__ void markBoolIf(notStupidBool_t* bool_arr, notStupidBool_t* value_arr, notStupidBool_t val, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        notStupidBool_t my_val = value_arr[myID];
        if (my_val == val) {
            bool_arr[myID] = 1;
        } else {
            bool_arr[myID] = 0;
        }
    }
}

// Mark primitive contacts that should survive into patch-contact derivation. NOT_A_CONTACT entries are temporary
// sentinels emitted by CD kernels; they do not carry valid patch identity or force/history state.
__global__ void markPhysicalPrimitiveContacts(notStupidBool_t* keep_flags, const contact_t* contact_types, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        keep_flags[myID] = (contact_types[myID] == NOT_A_CONTACT) ? 0 : 1;
    }
}

__global__ void setArr(notStupidBool_t* arr, size_t n, notStupidBool_t val) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] = val;
    }
}

__global__ void markDuplicateContacts(primitivesPrimTouches_t* idA_runlength,
                                      contactPairs_t* idA_scanned_runlength,
                                      bodyID_t* idB,
                                      contact_t* contactTypePrimitive,
                                      notStupidBool_t* persistency,
                                      notStupidBool_t* retain_list,
                                      size_t n,
                                      bool persistency_affect) {
    bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        primitivesPrimTouches_t cnt_count = idA_runlength[myID];
        // If this idA has non-zero runlength in new: a potential removal needed
        if (cnt_count > 0) {
            // Where should I start looking? Grab the offset.
            contactPairs_t cnt_offset = idA_scanned_runlength[myID];
            for (primitivesPrimTouches_t i = 0; i < cnt_count - 1; i++) {
                for (primitivesPrimTouches_t j = i + 1; j < cnt_count; j++) {
                    // Current contact numbers we are inspecting
                    contactPairs_t contactA = cnt_offset + i;
                    contactPairs_t contactB = cnt_offset + j;
                    bodyID_t contactA_idB = idB[contactA];
                    contact_t contactA_cntType = contactTypePrimitive[contactA];
                    bodyID_t contactB_idB = idB[contactB];
                    contact_t contactB_cntType = contactTypePrimitive[contactB];
                    notStupidBool_t contactA_persistency, contactB_persistency;
                    if (persistency_affect) {
                        contactA_persistency = persistency[contactA];
                        contactB_persistency = persistency[contactB];
                    }

                    if (contactA_idB == contactB_idB && contactA_cntType == contactB_cntType) {
                        // If so, they are the same contact, we have to remove one
                        if (!persistency_affect) {
                            // If persistency does not matter, we remove the one with bigger index, in case there are
                            // 3-way duplicates
                            retain_list[contactB] = 0;
                        } else {
                            if (contactA_persistency == contactB_persistency) {
                                // If both are persistent or both are non-persistent, we remove the one with bigger
                                // index, in case there are 3-way duplicates
                                retain_list[contactB] = 0;
                            } else if (contactB_persistency == CONTACT_NOT_PERSISTENT) {
                                retain_list[contactB] = 0;
                            } else {
                                retain_list[contactA] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void extractPatchInvolvedContactPatchIDPairs(patchIDPair_t* contactPatchPairs,
                                                        contact_t* contactTypePrimitive,
                                                        bodyID_t* idPrimitiveA,
                                                        bodyID_t* idPrimitiveB,
                                                        bodyID_t* triPatchID,
                                                        size_t nContacts) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContacts) {
        bodyID_t bodyA = idPrimitiveA[myID];
        bodyID_t bodyB = idPrimitiveB[myID];
        switch (contactTypePrimitive[myID]) {
            case SPHERE_TRIANGLE_CONTACT: {
                bodyID_t patchB = triPatchID[bodyB];
                // For sphere-triangle contact: triangle has patch ID, sphere does not but its geoID serves the same
                // purpose. We ensure sphere is A.
                contactPatchPairs[myID] =
                    encodeType<patchIDPair_t, bodyID_t>(bodyA, patchB);  // Input bodyID_t, return patchIDPair_t
                break;
            }
            case TRIANGLE_ANALYTICAL_CONTACT: {
                bodyID_t patchA = triPatchID[bodyA];
                // For mesh-analytical contact: mesh has patch ID, analytical object does not but its geoID serves the
                // same purpose. We ensure triangle is A.
                contactPatchPairs[myID] = encodeType<patchIDPair_t, bodyID_t>(patchA, bodyB);
                break;
            }
            case TRIANGLE_TRIANGLE_CONTACT: {
                bodyID_t patchA = triPatchID[bodyA];
                bodyID_t patchB = triPatchID[bodyB];
                // For triangle-triangle contact: both triangles have patch IDs. Keeps original order.
                contactPatchPairs[myID] = encodeType<patchIDPair_t, bodyID_t>(patchA, patchB);
                break;
            }
            default:
                // In other sph-sph and sph-anal, for now, we just use geoID as patchIDs. Keeps original order.
                contactPatchPairs[myID] = encodeType<patchIDPair_t, bodyID_t>(bodyA, bodyB);
                break;
        }
    }
}

// Decode unique patch pairs into separate idPatchA/idPatchB arrays
__global__ void decodePatchPairsToSeparateArrays(patchIDPair_t* uniquePatchPairs,
                                                 bodyID_t* idPatchA,
                                                 bodyID_t* idPatchB,
                                                 size_t numUnique) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < numUnique) {
        patchIDPair_t patchPair = uniquePatchPairs[myID];
        // decodeTypeA returns the value stored in high bits, and decodeTypeB returns the value stored in low bits. This
        // decoding does not change the order as they were encoded in encodeType.
        idPatchA[myID] = decodeTypeA<patchIDPair_t, bodyID_t>(patchPair);
        idPatchB[myID] = decodeTypeB<patchIDPair_t, bodyID_t>(patchPair);
    }
}

// Build geomToPatchMap by detecting boundaries of unique patch pairs in sorted array
// and using prefix scan on "is first of group" flags.
// For sorted patch pairs, mark 1 at each position where a new unique value starts, then prefix scan.
__global__ void markNewPatchPairGroups(patchIDPair_t* sortedPatchPairs, contactPairs_t* isNewGroup, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        if (myID == 0) {
            isNewGroup[myID] = 0;
        } else {
            // Compare with previous element - if different, it's a new group
            isNewGroup[myID] = (sortedPatchPairs[myID] != sortedPatchPairs[myID - 1]) ? 1 : 0;
        }
    }
}

__global__ void markNewPatchPairGroupsByType(const patchIDPair_t* sortedPatchPairs,
                                             const contact_t* sortedContactTypes,
                                             contactPairs_t* isNewGroup,
                                             size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        if (myID == 0) {
            isNewGroup[myID] = 0;
        } else {
            const bool new_pair = sortedPatchPairs[myID] != sortedPatchPairs[myID - 1];
            const bool new_type = sortedContactTypes[myID] != sortedContactTypes[myID - 1];
            isNewGroup[myID] = (new_pair || new_type) ? 1 : 0;
        }
    }
}

__global__ void setFirstFlagToOne(contactPairs_t* flags, size_t n) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && n > 0) {
        flags[0] = 1;
    }
}

__global__ void lineNumbers(contactPairs_t* arr, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] = myID;
    }
}

template <typename T>
__global__ void gatherByIndex(const T* in, T* out, const contactPairs_t* idx, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        out[myID] = in[idx[myID]];
    }
}

__global__ void buildGroupPrimitiveKeys(const contactPairs_t* groupIndex,
                                        const bodyID_t* primitiveIDs,
                                        uint64_t* keys,
                                        size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const bodyID_t phys_id = primitiveIDs[myID];
        keys[myID] = (static_cast<uint64_t>(groupIndex[myID]) << 32) | static_cast<uint64_t>(phys_id);
    }
}

__global__ void extractGroupIndexFromKey(const uint64_t* keys, contactPairs_t* groupIndex, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        groupIndex[myID] = static_cast<contactPairs_t>(keys[myID] >> 32);
    }
}

__global__ void scatterGroupCounts(const contactPairs_t* groupIDs,
                                   const contactPairs_t* counts,
                                   contactPairs_t* groupCounts,
                                   size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        groupCounts[groupIDs[myID]] = counts[myID];
    }
}

__global__ void computeGroupWinners(const contact_t* groupTypes,
                                    const bodyID_t* groupPrimA,
                                    const bodyID_t* groupPrimB,
                                    const contactPairs_t* countA,
                                    const contactPairs_t* countB,
                                    const bodyID_t* ownerTriMesh,
                                    const notStupidBool_t* ownerMeshConvex,
                                    const notStupidBool_t* ownerMeshNeverWinner,
                                    notStupidBool_t* winnerIsA,
                                    notStupidBool_t* winnerIsTri,
                                    notStupidBool_t* forceSingleIsland,
                                    size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const contact_t ctype = groupTypes[myID];
        const geoType_t typeA = decodeTypeA<contact_t, geoType_t>(ctype);
        const geoType_t typeB = decodeTypeB<contact_t, geoType_t>(ctype);
        const contactPairs_t nA = countA[myID];
        const contactPairs_t nB = countB[myID];
        const bool A_is_tri = (typeA == GEO_T_TRIANGLE);
        const bool B_is_tri = (typeB == GEO_T_TRIANGLE);
        bool A_convex = false;
        bool B_convex = false;
        bool A_never = false;
        bool B_never = false;
        if (A_is_tri) {
            const bodyID_t triA = groupPrimA[myID];
            const bodyID_t ownerA = ownerTriMesh[triA];
            if (ownerA != NULL_BODYID) {
                A_convex = (ownerMeshConvex[ownerA] != 0);
                A_never = (ownerMeshNeverWinner[ownerA] != 0);
            }
        }
        if (B_is_tri) {
            const bodyID_t triB = groupPrimB[myID];
            const bodyID_t ownerB = ownerTriMesh[triB];
            if (ownerB != NULL_BODYID) {
                B_convex = (ownerMeshConvex[ownerB] != 0);
                B_never = (ownerMeshNeverWinner[ownerB] != 0);
            }
        }
        const bool single_island = (A_is_tri && B_is_tri && A_convex && B_convex);
        forceSingleIsland[myID] = single_island ? 1 : 0;

        notStupidBool_t pickA = 0;
        if (ctype == SPHERE_TRIANGLE_CONTACT) {
            pickA = 0;
        } else if (A_never && B_never) {
            pickA = 0;
        } else if (A_never && !B_never) {
            pickA = 0;
        } else if (B_never && !A_never) {
            pickA = 1;
        } else if (nA > nB) {
            pickA = 1;
        } else if (nA < nB) {
            pickA = 0;
        } else {
            if (A_is_tri && B_is_tri) {
                if (A_convex != B_convex) {
                    pickA = A_convex ? 0 : 1;
                } else {
                    pickA = 0;
                }
            } else if (A_is_tri && !B_is_tri) {
                pickA = 1;
            } else if (B_is_tri && !A_is_tri) {
                pickA = 0;
            } else {
                pickA = 0;
            }
        }
        winnerIsA[myID] = pickA;
        if (single_island) {
            winnerIsTri[myID] = 0;
        } else {
            const geoType_t winnerType = (pickA ? typeA : typeB);
            winnerIsTri[myID] = (winnerType == GEO_T_TRIANGLE) ? 1 : 0;
        }
    }
}

__global__ void selectWinnerPrimitive(const contactPairs_t* groupIndex,
                                      const bodyID_t* idA,
                                      const bodyID_t* idB,
                                      const notStupidBool_t* groupWinnerIsA,
                                      const notStupidBool_t* groupWinnerIsTri,
                                      const notStupidBool_t* groupForceSingleIsland,
                                      bodyID_t* winnerID,
                                      notStupidBool_t* winnerIsTri,
                                      size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const contactPairs_t grp = groupIndex[myID];
        if (groupForceSingleIsland[grp] != 0) {
            winnerID[myID] = 0;
            winnerIsTri[myID] = 0;
            return;
        }
        const bool pickA = (groupWinnerIsA[grp] != 0);
        winnerID[myID] = pickA ? idA[myID] : idB[myID];
        winnerIsTri[myID] = groupWinnerIsTri[grp];
    }
}

__global__ void buildActiveTriKeys(const contactPairs_t* groupIndex,
                                   const bodyID_t* winnerID,
                                   const notStupidBool_t* winnerIsTri,
                                   uint64_t* keys,
                                   notStupidBool_t* flags,
                                   size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const notStupidBool_t is_tri = winnerIsTri[myID];
        flags[myID] = is_tri;
        if (is_tri) {
            keys[myID] = (static_cast<uint64_t>(groupIndex[myID]) << 32) | static_cast<uint64_t>(winnerID[myID]);
        } else {
            keys[myID] = 0;
        }
    }
}

__global__ void initActiveTriLabels(const uint64_t* keys, bodyID_t* labels, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        labels[myID] = static_cast<bodyID_t>(keys[myID] & 0xffffffffull);
    }
}

__global__ void countActiveTriPerGroup(const uint64_t* keys, contactPairs_t* groupCounts, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const contactPairs_t grp = static_cast<contactPairs_t>(keys[myID] >> 32);
        atomicAdd(&groupCounts[grp], (contactPairs_t)1);
    }
}

constexpr contactPairs_t DEME_INVALID_ACTIVE_TRI_POS = 0xffffffffu;

__global__ void buildActiveTriNeighborPos(const uint64_t* keys,
                                          const contactPairs_t* groupStart,
                                          const contactPairs_t* groupCount,
                                          const bodyID_t* triNeighborIndex,
                                          const bodyID_t* triNeighbor1,
                                          const bodyID_t* triNeighbor2,
                                          const bodyID_t* triNeighbor3,
                                          contactPairs_t* outNeighborPos,
                                          size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const uint64_t key = keys[myID];
        const contactPairs_t grp = static_cast<contactPairs_t>(key >> 32);
        const bodyID_t triID = static_cast<bodyID_t>(key & 0xffffffffull);
        const contactPairs_t start = groupStart[grp];
        const contactPairs_t count = groupCount[grp];

        outNeighborPos[3 * myID + 0] = DEME_INVALID_ACTIVE_TRI_POS;
        outNeighborPos[3 * myID + 1] = DEME_INVALID_ACTIVE_TRI_POS;
        outNeighborPos[3 * myID + 2] = DEME_INVALID_ACTIVE_TRI_POS;

        const bodyID_t nb_idx = triNeighborIndex[triID];
        if (nb_idx == NULL_BODYID || count == 0) {
            return;
        }

        const bodyID_t nbs[3] = {triNeighbor1[nb_idx], triNeighbor2[nb_idx], triNeighbor3[nb_idx]};
        for (int e = 0; e < 3; ++e) {
            const bodyID_t nb = nbs[e];
            if (nb == NULL_BODYID) {
                continue;
            }
            const uint64_t target = (static_cast<uint64_t>(grp) << 32) | static_cast<uint64_t>(nb);
            contactPairs_t left = 0;
            contactPairs_t right = count;
            while (left < right) {
                const contactPairs_t mid = left + (right - left) / 2;
                if (keys[start + mid] < target) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            if (left < count && keys[start + left] == target) {
                outNeighborPos[3 * myID + e] = start + left;
            }
        }
    }
}

__global__ void propagateActiveTriLabelsFromNeighborPos(const bodyID_t* labelsIn,
                                                        bodyID_t* labelsOut,
                                                        const contactPairs_t* neighborPos,
                                                        contactPairs_t* changedFlag,
                                                        size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int blockChanged;
    if (threadIdx.x == 0) {
        blockChanged = 0;
    }
    __syncthreads();

    if (myID < n) {
        const bodyID_t oldLabel = labelsIn[myID];
        bodyID_t label = oldLabel;
        const contactPairs_t p0 = neighborPos[3 * myID + 0];
        const contactPairs_t p1 = neighborPos[3 * myID + 1];
        const contactPairs_t p2 = neighborPos[3 * myID + 2];

        if (p0 != DEME_INVALID_ACTIVE_TRI_POS && labelsIn[p0] < label) {
            label = labelsIn[p0];
        }
        if (p1 != DEME_INVALID_ACTIVE_TRI_POS && labelsIn[p1] < label) {
            label = labelsIn[p1];
        }
        if (p2 != DEME_INVALID_ACTIVE_TRI_POS && labelsIn[p2] < label) {
            label = labelsIn[p2];
        }

        labelsOut[myID] = label;
        if (changedFlag != nullptr && label != oldLabel) {
            atomicExch(&blockChanged, 1);
        }
    }

    __syncthreads();
    if (changedFlag != nullptr && threadIdx.x == 0 && blockChanged) {
        atomicExch(changedFlag, (contactPairs_t)1);
    }
}

__global__ void assignContactIslandLabel(const contactPairs_t* groupIndex,
                                         const bodyID_t* winnerID,
                                         const notStupidBool_t* winnerIsTri,
                                         const uint64_t* activeKeys,
                                         const bodyID_t* activeLabels,
                                         const contactPairs_t* groupStart,
                                         const contactPairs_t* groupCount,
                                         bodyID_t* outLabels,
                                         size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const bodyID_t prim = winnerID[myID];
        if (winnerIsTri[myID] == 0) {
            outLabels[myID] = prim;
            return;
        }
        const contactPairs_t grp = groupIndex[myID];
        const contactPairs_t start = groupStart[grp];
        const contactPairs_t count = groupCount[grp];
        if (count == 0) {
            outLabels[myID] = prim;
            return;
        }
        const uint64_t target = (static_cast<uint64_t>(grp) << 32) | static_cast<uint64_t>(prim);
        contactPairs_t left = 0;
        contactPairs_t right = count;
        while (left < right) {
            const contactPairs_t mid = left + (right - left) / 2;
            if (activeKeys[start + mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        outLabels[myID] = (left < count && activeKeys[start + left] == target) ? activeLabels[start + left] : prim;
    }
}

__global__ void copyBodyIDArray(const bodyID_t* in, bodyID_t* out, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        out[myID] = in[myID];
    }
}

__global__ void buildIslandCompositeKeyParts(const patchIDPair_t* patchPairs,
                                             const contact_t* contactTypes,
                                             const bodyID_t* labels,
                                             uint64_t* key_hi,
                                             uint64_t* key_lo,
                                             size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const patchIDPair_t pair = patchPairs[myID];
        const uint64_t hi = static_cast<uint64_t>(pair >> 32);
        const uint64_t lo = static_cast<uint64_t>(pair & 0xffffffffull);
        key_hi[myID] = (static_cast<uint64_t>(contactTypes[myID]) << 32) | hi;
        key_lo[myID] = (lo << 32) | static_cast<uint64_t>(labels[myID]);
    }
}

__global__ void buildPrimitiveContactKeyParts(const bodyID_t* idA,
                                              const bodyID_t* idB,
                                              const contact_t* contactTypes,
                                              uint64_t* key_hi,
                                              uint64_t* key_lo,
                                              size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        key_hi[myID] = (static_cast<uint64_t>(contactTypes[myID]) << 32) | static_cast<uint64_t>(idA[myID]);
        key_lo[myID] = static_cast<uint64_t>(idB[myID]);
    }
}

__global__ void buildStableIslandVotes(const bodyID_t* curr_idA,
                                       const bodyID_t* curr_idB,
                                       const contact_t* curr_types,
                                       const contactPairs_t* curr_geomToPatchMap,
                                       const uint64_t* prev_key_hi,
                                       const uint64_t* prev_key_lo,
                                       const bodyID_t* prev_primitiveIsland,
                                       uint64_t* voteKeys,
                                       notStupidBool_t* voteFlags,
                                       size_t numCurr,
                                       size_t numPrev) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID >= numCurr) {
        return;
    }
    voteFlags[myID] = 0;
    voteKeys[myID] = 0;
    if (numPrev == 0 || !usesStablePatchIslandHistory(curr_types[myID])) {
        return;
    }

    const uint64_t curr_hi = (static_cast<uint64_t>(curr_types[myID]) << 32) | static_cast<uint64_t>(curr_idA[myID]);
    const uint64_t curr_lo = static_cast<uint64_t>(curr_idB[myID]);
    size_t left = 0;
    size_t right = numPrev;
    while (left < right) {
        const size_t mid = left + (right - left) / 2;
        const uint64_t mid_hi = prev_key_hi[mid];
        const uint64_t mid_lo = prev_key_lo[mid];
        if (mid_hi < curr_hi || (mid_hi == curr_hi && mid_lo < curr_lo)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    if (left < numPrev && prev_key_hi[left] == curr_hi && prev_key_lo[left] == curr_lo) {
        const bodyID_t prev_label = prev_primitiveIsland[left];
        if (prev_label != NULL_BODYID) {
            const contactPairs_t patch_idx = curr_geomToPatchMap[myID];
            voteKeys[myID] = (static_cast<uint64_t>(patch_idx) << 32) | static_cast<uint64_t>(prev_label);
            voteFlags[myID] = 1;
        }
    }
}

__global__ void scatterBestStableIslandVote(const uint64_t* uniqueVoteKeys,
                                            const contactPairs_t* counts,
                                            unsigned long long* bestPacked,
                                            size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const uint64_t vote_key = uniqueVoteKeys[myID];
        const contactPairs_t patch_idx = static_cast<contactPairs_t>(vote_key >> 32);
        const bodyID_t label = static_cast<bodyID_t>(vote_key & 0xffffffffull);
        const unsigned long long packed =
            (static_cast<unsigned long long>(counts[myID]) << 32) |
            static_cast<unsigned long long>(0xffffffffull - static_cast<unsigned long long>(label));
        atomicMax(&bestPacked[patch_idx], packed);
    }
}

__global__ void applyBestStableIslandVotes(const unsigned long long* bestPacked,
                                           bodyID_t* contactPatchIsland,
                                           size_t nPatchContacts) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nPatchContacts) {
        const unsigned long long packed = bestPacked[myID];
        if ((packed >> 32) != 0ull) {
            contactPatchIsland[myID] = static_cast<bodyID_t>(0xffffffffull - (packed & 0xffffffffull));
        }
    }
}

__global__ void markNewCompositeGroups64(const uint64_t* key_hi,
                                         const uint64_t* key_lo,
                                         contactPairs_t* isNewGroup,
                                         size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        if (myID == 0) {
            isNewGroup[myID] = 0;
        } else {
            const bool new_hi = key_hi[myID] != key_hi[myID - 1];
            const bool new_lo = key_lo[myID] != key_lo[myID - 1];
            isNewGroup[myID] = (new_hi || new_lo) ? 1 : 0;
        }
    }
}

// Set NULL_MAPPING_PARTNER for contacts of a specific type segment.
// Used when current step has contacts of this type but previous step does not.
//
// Parameters:
//   contactMapping: Output mapping array (entire array, not segment)
//   curr_start: Starting index in current arrays for this contact type segment
//   curr_count: Number of contacts of this type in current step
__global__ void setNullMappingForType(contactPairs_t* contactMapping,
                                      contactPairs_t curr_start,
                                      contactPairs_t curr_count) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < curr_count) {
        contactPairs_t curr_idx = curr_start + myID;
        contactMapping[curr_idx] = NULL_MAPPING_PARTNER;
    }
}

inline __host__ __device__ uint64_t makePatchHistoryPairKey(bodyID_t patchA, bodyID_t patchB, contact_t type) {
    if (type == TRIANGLE_TRIANGLE_CONTACT && patchB < patchA) {
        const bodyID_t tmp = patchA;
        patchA = patchB;
        patchB = tmp;
    }
    return (static_cast<uint64_t>(patchA) << 32) | static_cast<uint64_t>(patchB);
}

__global__ void buildPatchHistoryPairKeys(const bodyID_t* idPatchA,
                                          const bodyID_t* idPatchB,
                                          uint64_t* pairKeys,
                                          contactPairs_t* relativeIndices,
                                          contactPairs_t start,
                                          contactPairs_t count,
                                          contact_t type) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < count) {
        const contactPairs_t idx = start + myID;
        pairKeys[myID] = makePatchHistoryPairKey(idPatchA[idx], idPatchB[idx], type);
        relativeIndices[myID] = myID;
    }
}

__global__ void buildPatchContactMappingByHistoryKeys(bodyID_t* curr_idPatchA,
                                                      bodyID_t* curr_idPatchB,
                                                      bodyID_t* curr_patchIsland,
                                                      bodyID_t* prev_patchIsland,
                                                      const uint64_t* prev_pairKeys_sorted,
                                                      const contactPairs_t* prev_relativeIndices_sorted,
                                                      contactPairs_t* contactMapping,
                                                      contactPairs_t curr_start,
                                                      contactPairs_t curr_count,
                                                      contactPairs_t prev_start,
                                                      contactPairs_t prev_count,
                                                      contact_t type) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < curr_count) {
        const contactPairs_t curr_idx = curr_start + myID;
        const uint64_t curr_key = makePatchHistoryPairKey(curr_idPatchA[curr_idx], curr_idPatchB[curr_idx], type);
        const bodyID_t curr_label = curr_patchIsland[curr_idx];
        contactPairs_t my_partner = NULL_MAPPING_PARTNER;

        contactPairs_t left = 0;
        contactPairs_t right = prev_count;
        while (left < right) {
            const contactPairs_t mid = left + (right - left) / 2;
            if (prev_pairKeys_sorted[mid] < curr_key) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        for (contactPairs_t i = left; i < prev_count && prev_pairKeys_sorted[i] == curr_key; i++) {
            const contactPairs_t prev_idx = prev_start + prev_relativeIndices_sorted[i];
            if (prev_patchIsland[prev_idx] == curr_label) {
                my_partner = prev_idx;
                break;
            }
        }

        contactMapping[curr_idx] = my_partner;
    }
}

// Build patch-based contact mapping for a single contact type segment.
// This kernel operates on a specific segment of the contact arrays identified by start offsets and counts.
// Each thread processes one current contact and searches for a match in the previous contacts of the same type.
//
// Parameters:
//   curr_idPatchA, curr_idPatchB: Current step's patch contact arrays (entire arrays, not segment)
//   prev_idPatchA, prev_idPatchB: Previous step's patch contact arrays (entire arrays, not segment)
//   contactMapping: Output mapping array (entire array, not segment)
//   curr_start: Starting index in current arrays for this contact type segment
//   curr_count: Number of contacts of this type in current step
//   prev_start: Starting index in previous arrays for this contact type segment
//   prev_count: Number of contacts of this type in previous step
__global__ void buildPatchContactMappingForType(bodyID_t* curr_idPatchA,
                                                bodyID_t* curr_idPatchB,
                                                bodyID_t* curr_patchIsland,
                                                bodyID_t* prev_idPatchA,
                                                bodyID_t* prev_idPatchB,
                                                bodyID_t* prev_patchIsland,
                                                contactPairs_t* contactMapping,
                                                contactPairs_t curr_start,
                                                contactPairs_t curr_count,
                                                contactPairs_t prev_start,
                                                contactPairs_t prev_count) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < curr_count) {
        // Absolute index in the full contact array
        contactPairs_t curr_idx = curr_start + myID;

        bodyID_t curr_A = curr_idPatchA[curr_idx];
        bodyID_t curr_B = curr_idPatchB[curr_idx];
        bodyID_t curr_L = curr_patchIsland[curr_idx];

        // Default: no match found
        contactPairs_t my_partner = NULL_MAPPING_PARTNER;

        // Binary search within the previous type segment for the matching A/B pair
        // The segment is sorted by the combined patch ID pair
        contactPairs_t left = 0;
        contactPairs_t right = prev_count;
        while (left < right) {
            contactPairs_t mid = left + (right - left) / 2;
            contactPairs_t prev_idx = prev_start + mid;
            bodyID_t prev_A = prev_idPatchA[prev_idx];
            bodyID_t prev_B = prev_idPatchB[prev_idx];

            // Compare (A, B, label) triples lexicographically.
            if (prev_A < curr_A ||
                (prev_A == curr_A && (prev_B < curr_B || (prev_B == curr_B && prev_patchIsland[prev_idx] < curr_L)))) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Check if we found a match at position left
        if (left < prev_count) {
            contactPairs_t prev_idx = prev_start + left;
            bodyID_t prev_A = prev_idPatchA[prev_idx];
            bodyID_t prev_B = prev_idPatchB[prev_idx];
            bodyID_t prev_L = prev_patchIsland[prev_idx];
            if (prev_A == curr_A && prev_B == curr_B && prev_L == curr_L) {
                my_partner = prev_idx;
            }
        }

        contactMapping[curr_idx] = my_partner;
    }
}

__global__ void buildPatchContactMappingForStableType(bodyID_t* curr_idPatchA,
                                                      bodyID_t* curr_idPatchB,
                                                      bodyID_t* curr_patchIsland,
                                                      bodyID_t* prev_idPatchA,
                                                      bodyID_t* prev_idPatchB,
                                                      bodyID_t* prev_patchIsland,
                                                      contactPairs_t* contactMapping,
                                                      contactPairs_t curr_start,
                                                      contactPairs_t curr_count,
                                                      contactPairs_t prev_start,
                                                      contactPairs_t prev_count) {
    // Stable-label rewriting preserves the existing (patch A, patch B) ordering but may scramble island labels within
    // one patch pair. Binary-search the pair range first, then scan only that usually-small range for the label.
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < curr_count) {
        const contactPairs_t curr_idx = curr_start + myID;
        const bodyID_t curr_A = curr_idPatchA[curr_idx];
        const bodyID_t curr_B = curr_idPatchB[curr_idx];
        const bodyID_t curr_L = curr_patchIsland[curr_idx];
        contactPairs_t my_partner = NULL_MAPPING_PARTNER;

        contactPairs_t left = 0;
        contactPairs_t right = prev_count;
        while (left < right) {
            const contactPairs_t mid = left + (right - left) / 2;
            const contactPairs_t prev_idx = prev_start + mid;
            const bodyID_t prev_A = prev_idPatchA[prev_idx];
            const bodyID_t prev_B = prev_idPatchB[prev_idx];
            if (prev_A < curr_A || (prev_A == curr_A && prev_B < curr_B)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        for (contactPairs_t i = left; i < prev_count; i++) {
            const contactPairs_t prev_idx = prev_start + i;
            const bodyID_t prev_A = prev_idPatchA[prev_idx];
            const bodyID_t prev_B = prev_idPatchB[prev_idx];
            if (prev_A != curr_A || prev_B != curr_B) {
                break;
            }
            if (prev_patchIsland[prev_idx] == curr_L) {
                my_partner = prev_idx;
                break;
            }
        }
        contactMapping[curr_idx] = my_partner;
    }
}

// Build patch-based contact mapping between current and previous patch contact arrays.
// Both arrays are sorted by contact type, then by combined patch ID pair within each type.
// For each current contact, we use binary search to find the matching contact in the previous array.
__global__ void buildPatchContactMapping(bodyID_t* curr_idPatchA,
                                         bodyID_t* curr_idPatchB,
                                         contact_t* curr_contactTypePatch,
                                         bodyID_t* prev_idPatchA,
                                         bodyID_t* prev_idPatchB,
                                         contact_t* previous_contactTypePatch,
                                         contactPairs_t* contactMapping,
                                         size_t numCurrContacts,
                                         size_t numPrevContacts) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < numCurrContacts) {
        bodyID_t curr_A = curr_idPatchA[myID];
        bodyID_t curr_B = curr_idPatchB[myID];
        contact_t curr_type = curr_contactTypePatch[myID];

        // Default: no match found
        contactPairs_t my_partner = NULL_MAPPING_PARTNER;

        // Find the segment in the previous array with the same contact type
        // Since arrays are sorted by type first, we can find the type segment bounds using binary search

        // Find the lower bound (first element with type >= curr_type)
        size_t left = 0;
        size_t right = numPrevContacts;
        while (left < right) {
            size_t mid = left + (right - left) / 2;
            if (previous_contactTypePatch[mid] < curr_type) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        size_t type_start = left;

        // Find the upper bound (first element with type > curr_type)
        // Note: we intentionally reuse 'left' from the lower_bound result (= type_start)
        // since we know upper_bound >= lower_bound, this is an optimization
        right = numPrevContacts;
        while (left < right) {
            size_t mid = left + (right - left) / 2;
            if (previous_contactTypePatch[mid] <= curr_type) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        size_t type_end = left;

        // Within this type segment, use binary search to find the matching A/B pair
        // The segment is sorted by the combined patch ID pair (A in high bits, B in low bits)
        // The encoding ensures that (smaller_A, larger_B) pattern creates a sortable value
        left = type_start;
        right = type_end;
        while (left < right) {
            size_t mid = left + (right - left) / 2;
            bodyID_t prev_A = prev_idPatchA[mid];
            bodyID_t prev_B = prev_idPatchB[mid];

            // Compare (A, B) pairs lexicographically
            // Since they're sorted by patch ID pair where smaller ID is in high bits
            if (prev_A < curr_A || (prev_A == curr_A && prev_B < curr_B)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Check if we found a match at position left
        if (left < type_end) {
            bodyID_t prev_A = prev_idPatchA[left];
            bodyID_t prev_B = prev_idPatchB[left];
            if (prev_A == curr_A && prev_B == curr_B) {
                my_partner = left;
            }
        }

        contactMapping[myID] = my_partner;
    }
}

// Build a sortable 64-bit key from (idB, contactType, persistency_preference).
// If prefer_persistent is true, persistent contacts sort before non-persistent ones.
__global__ void buildKeyBTypePersist(const bodyID_t* idB,
                                     const contact_t* contactType,
                                     const notStupidBool_t* persistency,
                                     unsigned long long* keys,
                                     size_t n,
                                     bool prefer_persistent) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const unsigned long long high = (static_cast<unsigned long long>(idB[myID]) << 32);
        const unsigned long long type_part = (static_cast<unsigned long long>(contactType[myID]) << 1);
        const unsigned long long persist_part =
            prefer_persistent ? static_cast<unsigned long long>(persistency[myID] == CONTACT_NOT_PERSISTENT) : 0ull;
        keys[myID] = high | type_part | persist_part;
    }
}

// Mark (idA, idB, contactType) duplicates in a lexicographically sorted contact list.
__global__ void markUniqueTriples(const bodyID_t* idA,
                                  const bodyID_t* idB,
                                  const contact_t* contactType,
                                  notStupidBool_t* retain_flags,
                                  size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        if (myID == 0) {
            retain_flags[myID] = (notStupidBool_t)1;
            return;
        }
        const bool is_dup = (idA[myID] == idA[myID - 1]) && (idB[myID] == idB[myID - 1]) &&
                            (contactType[myID] == contactType[myID - 1]);
        retain_flags[myID] = is_dup ? (notStupidBool_t)0 : (notStupidBool_t)1;
    }
}

__global__ void fillPrimitivePatchIslandLabels(const contactPairs_t* geomToPatchMap,
                                               const bodyID_t* contactPatchIsland,
                                               const contact_t* primitiveContactTypes,
                                               bodyID_t* primitivePatchIsland,
                                               size_t nPrimitive) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nPrimitive) {
        primitivePatchIsland[myID] = usesStablePatchIslandHistory(primitiveContactTypes[myID])
                                         ? contactPatchIsland[geomToPatchMap[myID]]
                                         : NULL_BODYID;
    }
}

// The rest are old kernels only used in DEME2.x for primitive-primitive contact mapping
/*
__global__ void fillRunLengthArray(primitivesPrimTouches_t* runlength_full,
                                   bodyID_t* unique_ids,
                                   primitivesPrimTouches_t* runlength,
                                   size_t numUnique) {
    bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < numUnique) {
        bodyID_t i = unique_ids[myID];
        runlength_full[i] = runlength[myID];
    }
}

__global__ void buildPersistentMap(primitivesPrimTouches_t* new_idA_runlength_full,
                                   primitivesPrimTouches_t* old_idA_runlength_full,
                                   contactPairs_t* new_idA_scanned_runlength,
                                   contactPairs_t* old_idA_scanned_runlength,
                                   contactPairs_t* mapping,
                                   DEMDataKT* granData,
                                   size_t nGeoSafe) {
    bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nGeoSafe) {
        primitivesPrimTouches_t new_cnt_count = new_idA_runlength_full[myID];
        primitivesPrimTouches_t old_cnt_count = old_idA_runlength_full[myID];
        // If this idA has non-zero runlength in new: a potential enduring sphere
        if (new_cnt_count > 0) {
            // Where should I start looking? Grab the offset.
            contactPairs_t new_cnt_offset = new_idA_scanned_runlength[myID];
            contactPairs_t old_cnt_offset = old_idA_scanned_runlength[myID];
            for (primitivesPrimTouches_t i = 0; i < new_cnt_count; i++) {
                // Current contact number we are inspecting
                contactPairs_t this_contact = new_cnt_offset + i;
                bodyID_t new_idB = granData->idPrimitiveB[this_contact];
                contact_t new_cntType = granData->contactTypePrimitive[this_contact];
                // Mark it as no matching pair found, being a new contact; modify it later
                contactPairs_t my_partner = NULL_MAPPING_PARTNER;
                // If this is a fake contact, we can move on
                if (new_cntType == NOT_A_CONTACT) {
                    mapping[this_contact] = my_partner;
                    continue;
                }
                // Loop through the old idB to see if there is a match
                for (primitivesPrimTouches_t j = 0; j < old_cnt_count; j++) {
                    bodyID_t old_idB = granData->previous_idPrimitiveB[old_cnt_offset + j];
                    contact_t old_cntType = granData->previous_contactTypePrimitive[old_cnt_offset + j];
                    // If both idB and contact type match, then it is an enduring contact, write it to the mapping
                    // array
                    if (new_idB == old_idB && new_cntType == old_cntType) {
                        my_partner = old_cnt_offset + j;
                        break;
                    }
                }
                // If old_cnt_count == 0, it is automatically NULL_MAPPING_PARTNER
                mapping[this_contact] = my_partner;
            }
        }
    }
}

__global__ void lineNumbers(contactPairs_t* arr, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] = myID;
    }
}

__global__ void convertToAndFrom(contactPairs_t* old_arr_unsort_to_sort_map,
                                 contactPairs_t* converted_map,
                                 size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        contactPairs_t map_from = old_arr_unsort_to_sort_map[myID];
        converted_map[map_from] = myID;
    }
}

__global__ void rearrangeMapping(contactPairs_t* map_sorted,
                                 contactPairs_t* old_arr_unsort_to_sort_map,
                                 size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        contactPairs_t map_to = map_sorted[myID];
        if (map_to != NULL_MAPPING_PARTNER)
            map_sorted[myID] = old_arr_unsort_to_sort_map[map_to];
    }
}
*/

}  // namespace deme

#endif
