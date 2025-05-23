// DEM history mapping related custom kernels
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>
_kernelIncludes_;

__global__ void fillRunLengthArray(deme::geoSphereTouches_t* runlength_full,
                                   deme::bodyID_t* unique_ids,
                                   deme::geoSphereTouches_t* runlength,
                                   size_t numUnique) {
    deme::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < numUnique) {
        deme::bodyID_t i = unique_ids[myID];
        runlength_full[i] = runlength[myID];
    }
}

__global__ void buildPersistentMap(deme::geoSphereTouches_t* new_idA_runlength_full,
                                   deme::geoSphereTouches_t* old_idA_runlength_full,
                                   deme::contactPairs_t* new_idA_scanned_runlength,
                                   deme::contactPairs_t* old_idA_scanned_runlength,
                                   deme::contactPairs_t* mapping,
                                   deme::DEMDataKT* granData,
                                   size_t nSpheresSafe) {
    deme::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nSpheresSafe) {
        deme::geoSphereTouches_t new_cnt_count = new_idA_runlength_full[myID];
        deme::geoSphereTouches_t old_cnt_count = old_idA_runlength_full[myID];
        // If this idA has non-zero runlength in new: a potential enduring sphere
        if (new_cnt_count > 0) {
            // Where should I start looking? Grab the offset.
            deme::contactPairs_t new_cnt_offset = new_idA_scanned_runlength[myID];
            deme::contactPairs_t old_cnt_offset = old_idA_scanned_runlength[myID];
            for (deme::geoSphereTouches_t i = 0; i < new_cnt_count; i++) {
                // Current contact number we are inspecting
                deme::contactPairs_t this_contact = new_cnt_offset + i;
                deme::bodyID_t new_idB = granData->idGeometryB[this_contact];
                deme::contact_t new_cntType = granData->contactType[this_contact];
                // Mark it as no matching pair found, being a new contact; modify it later
                deme::contactPairs_t my_partner = deme::NULL_MAPPING_PARTNER;
                // If this is a fake contact, we can move on
                if (new_cntType == deme::NOT_A_CONTACT) {
                    mapping[this_contact] = my_partner;
                    continue;
                }
                // Loop through the old idB to see if there is a match
                for (deme::geoSphereTouches_t j = 0; j < old_cnt_count; j++) {
                    deme::bodyID_t old_idB = granData->previous_idGeometryB[old_cnt_offset + j];
                    deme::contact_t old_cntType = granData->previous_contactType[old_cnt_offset + j];
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

__global__ void markDuplicateContacts(deme::geoSphereTouches_t* idA_runlength,
                                      deme::contactPairs_t* idA_scanned_runlength,
                                      deme::bodyID_t* idB,
                                      deme::contact_t* contactType,
                                      deme::notStupidBool_t* persistency,
                                      deme::notStupidBool_t* retain_list,
                                      size_t n) {
    deme::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        deme::geoSphereTouches_t cnt_count = idA_runlength[myID];
        // If this idA has non-zero runlength in new: a potential removal needed
        if (cnt_count > 0) {
            // Where should I start looking? Grab the offset.
            deme::contactPairs_t cnt_offset = idA_scanned_runlength[myID];
            for (deme::geoSphereTouches_t i = 0; i < cnt_count - 1; i++) {
                for (deme::geoSphereTouches_t j = i + 1; j < cnt_count; j++) {
                    // Current contact numbers we are inspecting
                    deme::contactPairs_t contactA = cnt_offset + i;
                    deme::contactPairs_t contactB = cnt_offset + j;
                    deme::bodyID_t contactA_idB = idB[contactA];
                    deme::contact_t contactA_cntType = contactType[contactA];
                    deme::notStupidBool_t contactA_persistency = persistency[contactA];
                    deme::bodyID_t contactB_idB = idB[contactB];
                    deme::contact_t contactB_cntType = contactType[contactB];
                    deme::notStupidBool_t contactB_persistency = persistency[contactB];

                    if (contactA_idB == contactB_idB && contactA_cntType == contactB_cntType) {
                        // If so, they are the same contact, we have to remove one
                        if (contactA_persistency && contactB_persistency) {
                            // This is a weird situation: Both are persistent, it should not exist. But still, we remove
                            // the first one
                            retain_list[contactA] = 0;
                        } else {
                            // If one is persistent, we remove the non-persistent one
                            if (contactA_persistency == deme::CONTACT_NOT_PERSISTENT) {
                                retain_list[contactA] = 0;
                            } else {
                                retain_list[contactB] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}
