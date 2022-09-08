// DEM history mapping related custom kernels
#include <DEM/DEMDefines.h>
#include <kernel/DEMHelperKernels.cu>

__global__ void fillRunLengthArray(smug::geoSphereTouches_t* runlength_full,
                                   smug::bodyID_t* unique_ids,
                                   smug::geoSphereTouches_t* runlength,
                                   size_t numUnique) {
    smug::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < numUnique) {
        smug::bodyID_t i = unique_ids[myID];
        runlength_full[i] = runlength[myID];
    }
}

__global__ void buildPersistentMap(smug::geoSphereTouches_t* new_idA_runlength_full,
                                   smug::geoSphereTouches_t* old_idA_runlength_full,
                                   smug::contactPairs_t* new_idA_scanned_runlength,
                                   smug::contactPairs_t* old_idA_scanned_runlength,
                                   smug::contactPairs_t* mapping,
                                   smug::DEMDataKT* granData,
                                   size_t nSpheresSafe) {
    smug::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nSpheresSafe) {
        smug::geoSphereTouches_t new_cnt_count = new_idA_runlength_full[myID];
        smug::geoSphereTouches_t old_cnt_count = old_idA_runlength_full[myID];
        // If this idA has non-zero runlength in new: a potential persistent sphere
        if (new_cnt_count > 0) {
            // Where should I start looking? Grab the offset.
            smug::contactPairs_t new_cnt_offset = new_idA_scanned_runlength[myID];
            smug::contactPairs_t old_cnt_offset = old_idA_scanned_runlength[myID];
            for (smug::geoSphereTouches_t i = 0; i < new_cnt_count; i++) {
                // Current contact number we are inspecting
                smug::contactPairs_t this_contact = new_cnt_offset + i;
                smug::bodyID_t new_idB = granData->idGeometryB[this_contact];
                smug::contact_t new_cntType = granData->contactType[this_contact];
                // Mark it as no matching pair found, being a new contact; modify it later
                smug::contactPairs_t my_partner = smug::DEM_NULL_MAPPING_PARTNER;
                // If this is a fake contact, we can move on
                if (new_cntType == smug::DEM_NOT_A_CONTACT) {
                    mapping[this_contact] = my_partner;
                    continue;
                }
                // Loop through the old idB to see if there is a match
                for (smug::geoSphereTouches_t j = 0; j < old_cnt_count; j++) {
                    smug::bodyID_t old_idB = granData->previous_idGeometryB[old_cnt_offset + j];
                    smug::contact_t old_cntType = granData->previous_contactType[old_cnt_offset + j];
                    // If both idB and contact type match, then it is a persistent contact, write it to the mapping
                    // array
                    if (new_idB == old_idB && new_cntType == old_cntType) {
                        my_partner = old_cnt_offset + j;
                        break;
                    }
                }
                // If old_cnt_count == 0, it is automatically DEM_NULL_MAPPING_PARTNER
                mapping[this_contact] = my_partner;
            }
        }
    }
}
