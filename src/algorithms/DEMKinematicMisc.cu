// DEM history mapping related custom kernels
#include <DEM/Defines.h>
#include <kernel/DEMHelperKernels.cuh>

#include <stdio.h>

// Kernel to add a constant offset to each element of an array
__global__ void addOffsetToArray(deme::contactPairs_t* arr, deme::contactPairs_t offset, size_t n) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] += offset;
    }
}

// Kernel to fill a contact_t array with a constant value
__global__ void fillContactTypeArray(deme::contact_t* arr, deme::contact_t val, size_t n) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] = val;
    }
}

__global__ void fillRunLengthArray(deme::primitivesPrimTouches_t* runlength_full,
                                   deme::bodyID_t* unique_ids,
                                   deme::primitivesPrimTouches_t* runlength,
                                   size_t numUnique) {
    deme::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < numUnique) {
        deme::bodyID_t i = unique_ids[myID];
        runlength_full[i] = runlength[myID];
    }
}

__global__ void buildPersistentMap(deme::primitivesPrimTouches_t* new_idA_runlength_full,
                                   deme::primitivesPrimTouches_t* old_idA_runlength_full,
                                   deme::contactPairs_t* new_idA_scanned_runlength,
                                   deme::contactPairs_t* old_idA_scanned_runlength,
                                   deme::contactPairs_t* mapping,
                                   deme::DEMDataKT* granData,
                                   size_t nGeoSafe) {
    deme::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nGeoSafe) {
        deme::primitivesPrimTouches_t new_cnt_count = new_idA_runlength_full[myID];
        deme::primitivesPrimTouches_t old_cnt_count = old_idA_runlength_full[myID];
        // If this idA has non-zero runlength in new: a potential enduring sphere
        if (new_cnt_count > 0) {
            // Where should I start looking? Grab the offset.
            deme::contactPairs_t new_cnt_offset = new_idA_scanned_runlength[myID];
            deme::contactPairs_t old_cnt_offset = old_idA_scanned_runlength[myID];
            for (deme::primitivesPrimTouches_t i = 0; i < new_cnt_count; i++) {
                // Current contact number we are inspecting
                deme::contactPairs_t this_contact = new_cnt_offset + i;
                deme::bodyID_t new_idB = granData->idPrimitiveB[this_contact];
                deme::contact_t new_cntType = granData->contactTypePrimitive[this_contact];
                // Mark it as no matching pair found, being a new contact; modify it later
                deme::contactPairs_t my_partner = deme::NULL_MAPPING_PARTNER;
                // If this is a fake contact, we can move on
                if (new_cntType == deme::NOT_A_CONTACT) {
                    mapping[this_contact] = my_partner;
                    continue;
                }
                // Loop through the old idB to see if there is a match
                for (deme::primitivesPrimTouches_t j = 0; j < old_cnt_count; j++) {
                    deme::bodyID_t old_idB = granData->previous_idPrimitiveB[old_cnt_offset + j];
                    deme::contact_t old_cntType = granData->previous_contactTypePrimitive[old_cnt_offset + j];
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

__global__ void lineNumbers(deme::contactPairs_t* arr, size_t n) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] = myID;
    }
}

__global__ void convertToAndFrom(deme::contactPairs_t* old_arr_unsort_to_sort_map,
                                 deme::contactPairs_t* converted_map,
                                 size_t n) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        deme::contactPairs_t map_from = old_arr_unsort_to_sort_map[myID];
        converted_map[map_from] = myID;
    }
}

__global__ void rearrangeMapping(deme::contactPairs_t* map_sorted,
                                 deme::contactPairs_t* old_arr_unsort_to_sort_map,
                                 size_t n) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        deme::contactPairs_t map_to = map_sorted[myID];
        if (map_to != deme::NULL_MAPPING_PARTNER)
            map_sorted[myID] = old_arr_unsort_to_sort_map[map_to];
    }
}

__global__ void markBoolIf(deme::notStupidBool_t* bool_arr,
                           deme::notStupidBool_t* value_arr,
                           deme::notStupidBool_t val,
                           size_t n) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        deme::notStupidBool_t my_val = value_arr[myID];
        if (my_val == val) {
            bool_arr[myID] = 1;
        } else {
            bool_arr[myID] = 0;
        }
    }
}

__global__ void setArr(deme::notStupidBool_t* arr, size_t n, deme::notStupidBool_t val) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] = val;
    }
}

__global__ void markDuplicateContacts(deme::primitivesPrimTouches_t* idA_runlength,
                                      deme::contactPairs_t* idA_scanned_runlength,
                                      deme::bodyID_t* idB,
                                      deme::contact_t* contactTypePrimitive,
                                      deme::notStupidBool_t* persistency,
                                      deme::notStupidBool_t* retain_list,
                                      size_t n,
                                      bool persistency_affect) {
    deme::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        deme::primitivesPrimTouches_t cnt_count = idA_runlength[myID];
        // If this idA has non-zero runlength in new: a potential removal needed
        if (cnt_count > 0) {
            // Where should I start looking? Grab the offset.
            deme::contactPairs_t cnt_offset = idA_scanned_runlength[myID];
            for (deme::primitivesPrimTouches_t i = 0; i < cnt_count - 1; i++) {
                for (deme::primitivesPrimTouches_t j = i + 1; j < cnt_count; j++) {
                    // Current contact numbers we are inspecting
                    deme::contactPairs_t contactA = cnt_offset + i;
                    deme::contactPairs_t contactB = cnt_offset + j;
                    deme::bodyID_t contactA_idB = idB[contactA];
                    deme::contact_t contactA_cntType = contactTypePrimitive[contactA];
                    deme::bodyID_t contactB_idB = idB[contactB];
                    deme::contact_t contactB_cntType = contactTypePrimitive[contactB];
                    deme::notStupidBool_t contactA_persistency, contactB_persistency;
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
                            } else if (contactB_persistency == deme::CONTACT_NOT_PERSISTENT) {
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

__global__ void extractPatchInvolvedContactPatchIDPairs(deme::patchIDPair_t* contactPatchPairs,
                                                        deme::contact_t* contactTypePrimitive,
                                                        deme::bodyID_t* idPrimitiveA,
                                                        deme::bodyID_t* idPrimitiveB,
                                                        deme::bodyID_t* triPatchID,
                                                        size_t nContacts) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContacts) {
        deme::bodyID_t bodyA = idPrimitiveA[myID];
        deme::bodyID_t bodyB = idPrimitiveB[myID];
        switch (contactTypePrimitive[myID]) {
            case deme::SPHERE_TRIANGLE_CONTACT: {
                deme::bodyID_t patchB = triPatchID[bodyB];
                // For sphere-triangle contact: triangle has patch ID, sphere does not but its geoID serves the same
                // purpose. We ensure sphere is A.
                contactPatchPairs[myID] = deme::encodeType<deme::patchIDPair_t, deme::bodyID_t>(
                    bodyA, patchB);  // Input bodyID_t, return patchIDPair_t
                break;
            }
            case deme::TRIANGLE_ANALYTICAL_CONTACT: {
                deme::bodyID_t patchA = triPatchID[bodyA];
                // For mesh-analytical contact: mesh has patch ID, analytical object does not but its geoID serves the
                // same purpose. We ensure triangle is A.
                contactPatchPairs[myID] = deme::encodeType<deme::patchIDPair_t, deme::bodyID_t>(patchA, bodyB);
                break;
            }
            case deme::TRIANGLE_TRIANGLE_CONTACT: {
                deme::bodyID_t patchA = triPatchID[bodyA];
                deme::bodyID_t patchB = triPatchID[bodyB];
                // For triangle-triangle contact: both triangles have patch IDs. Keeps original order.
                contactPatchPairs[myID] = deme::encodeType<deme::patchIDPair_t, deme::bodyID_t>(patchA, patchB);
                break;
            }
            default:
                // In other sph-sph and sph-anal, for now, we just use geoID as patchIDs. Keeps original order.
                contactPatchPairs[myID] = deme::encodeType<deme::patchIDPair_t, deme::bodyID_t>(bodyA, bodyB);
                break;
        }
    }
}

// Decode unique patch pairs into separate idPatchA/idPatchB arrays
__global__ void decodePatchPairsToSeparateArrays(deme::patchIDPair_t* uniquePatchPairs,
                                                 deme::bodyID_t* idPatchA,
                                                 deme::bodyID_t* idPatchB,
                                                 size_t numUnique) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < numUnique) {
        deme::patchIDPair_t patchPair = uniquePatchPairs[myID];
        // decodeTypeA returns the value stored in high bits, and decodeTypeB returns the value stored in low bits. This
        // decoding does not change the order as they were encoded in encodeType.
        idPatchA[myID] = deme::decodeTypeA<deme::patchIDPair_t, deme::bodyID_t>(patchPair);
        idPatchB[myID] = deme::decodeTypeB<deme::patchIDPair_t, deme::bodyID_t>(patchPair);
    }
}

// Build geomToPatchMap by detecting boundaries of unique patch pairs in sorted array
// and using prefix scan on "is first of group" flags.
// For sorted patch pairs, mark 1 at each position where a new unique value starts, then prefix scan.
__global__ void markNewPatchPairGroups(deme::patchIDPair_t* sortedPatchPairs,
                                       deme::contactPairs_t* isNewGroup,
                                       size_t n) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        if (myID == 0) {
            isNewGroup[myID] = 0;
        } else {
            // Compare with previous element - if different, it's a new group
            isNewGroup[myID] = (sortedPatchPairs[myID] != sortedPatchPairs[myID - 1]) ? 1 : 0;
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
__global__ void setNullMappingForType(deme::contactPairs_t* contactMapping,
                                      deme::contactPairs_t curr_start,
                                      deme::contactPairs_t curr_count) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < curr_count) {
        deme::contactPairs_t curr_idx = curr_start + myID;
        contactMapping[curr_idx] = deme::NULL_MAPPING_PARTNER;
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
__global__ void buildPatchContactMappingForType(deme::bodyID_t* curr_idPatchA,
                                                deme::bodyID_t* curr_idPatchB,
                                                deme::bodyID_t* prev_idPatchA,
                                                deme::bodyID_t* prev_idPatchB,
                                                deme::contactPairs_t* contactMapping,
                                                deme::contactPairs_t curr_start,
                                                deme::contactPairs_t curr_count,
                                                deme::contactPairs_t prev_start,
                                                deme::contactPairs_t prev_count) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < curr_count) {
        // Absolute index in the full contact array
        deme::contactPairs_t curr_idx = curr_start + myID;

        deme::bodyID_t curr_A = curr_idPatchA[curr_idx];
        deme::bodyID_t curr_B = curr_idPatchB[curr_idx];

        // Default: no match found
        deme::contactPairs_t my_partner = deme::NULL_MAPPING_PARTNER;

        // Binary search within the previous type segment for the matching A/B pair
        // The segment is sorted by the combined patch ID pair
        deme::contactPairs_t left = 0;
        deme::contactPairs_t right = prev_count;
        while (left < right) {
            deme::contactPairs_t mid = left + (right - left) / 2;
            deme::contactPairs_t prev_idx = prev_start + mid;
            deme::bodyID_t prev_A = prev_idPatchA[prev_idx];
            deme::bodyID_t prev_B = prev_idPatchB[prev_idx];

            // Compare (A, B) pairs lexicographically
            if (prev_A < curr_A || (prev_A == curr_A && prev_B < curr_B)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Check if we found a match at position left
        if (left < prev_count) {
            deme::contactPairs_t prev_idx = prev_start + left;
            deme::bodyID_t prev_A = prev_idPatchA[prev_idx];
            deme::bodyID_t prev_B = prev_idPatchB[prev_idx];
            if (prev_A == curr_A && prev_B == curr_B) {
                my_partner = prev_idx;
            }
        }

        contactMapping[curr_idx] = my_partner;
    }
}

// Build patch-based contact mapping between current and previous patch contact arrays.
// Both arrays are sorted by contact type, then by combined patch ID pair within each type.
// For each current contact, we use binary search to find the matching contact in the previous array.
__global__ void buildPatchContactMapping(deme::bodyID_t* curr_idPatchA,
                                         deme::bodyID_t* curr_idPatchB,
                                         deme::contact_t* curr_contactTypePatch,
                                         deme::bodyID_t* prev_idPatchA,
                                         deme::bodyID_t* prev_idPatchB,
                                         deme::contact_t* previous_contactTypePatch,
                                         deme::contactPairs_t* contactMapping,
                                         size_t numCurrContacts,
                                         size_t numPrevContacts) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < numCurrContacts) {
        deme::bodyID_t curr_A = curr_idPatchA[myID];
        deme::bodyID_t curr_B = curr_idPatchB[myID];
        deme::contact_t curr_type = curr_contactTypePatch[myID];

        // Default: no match found
        deme::contactPairs_t my_partner = deme::NULL_MAPPING_PARTNER;

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
            deme::bodyID_t prev_A = prev_idPatchA[mid];
            deme::bodyID_t prev_B = prev_idPatchB[mid];

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
            deme::bodyID_t prev_A = prev_idPatchA[left];
            deme::bodyID_t prev_B = prev_idPatchB[left];
            if (prev_A == curr_A && prev_B == curr_B) {
                my_partner = left;
            }
        }

        contactMapping[myID] = my_partner;
    }
}
