// DEM history mapping related custom kernels
#include <DEM/DEMDefines.h>
#include <kernel/DEMHelperKernels.cu>

__global__ void fillRunLengthArray(sgps::geoSphereTouches_t* runlength_full,
                                   sgps::bodyID_t* unique_ids,
                                   sgps::geoSphereTouches_t* runlength,
                                   size_t numUnique) {
    sgps::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < numUnique) {
        sgps::bodyID_t i = unique_ids[myID];
        runlength_full[i] = runlength[myID];
    }
}

__global__ void buildPersistentMap(sgps::geoSphereTouches_t* new_idA_runlength_full,
                                   sgps::geoSphereTouches_t* old_idA_runlength_full,
                                   sgps::contactPairs_t* new_idA_scanned_runlength,
                                   sgps::contactPairs_t* old_idA_scanned_runlength,
                                   sgps::contactPairs_t* mapping,
                                   sgps::DEMDataKT* granData,
                                   size_t nSpheresSafe) {
    sgps::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nSpheresSafe) {
        // If this idA has non-zero runlength both in old and new: a potential persistent sphere
        sgps::geoSphereTouches_t new_cnt_count = new_idA_runlength_full[myID];
        sgps::geoSphereTouches_t old_cnt_count = old_idA_runlength_full[myID];
        if (new_cnt_count > 0 && old_cnt_count > 0) {
            // Where should I start looking? Grab the offset.
            sgps::contactPairs_t new_cnt_offset = new_idA_scanned_runlength[myID];
            sgps::contactPairs_t old_cnt_offset = old_idA_scanned_runlength[myID];
            for (sgps::geoSphereTouches_t i = 0; i < new_cnt_count; i++) {
                // Current contact number we are inspecting
                sgps::contactPairs_t this_contact = new_cnt_offset + i;
                sgps::bodyID_t new_idB = granData->idGeometryB[this_contact];
                sgps::contact_t new_cntType = granData->contactType[this_contact];
                // Mark it as no matching pair found, being a new contact; modify it later
                sgps::contactPairs_t my_partner = sgps::DEM_NULL_MAPPING_PARTNER;
                // If this is a fake contact, we can move on
                if (new_cntType == sgps::DEM_NOT_A_CONTACT) {
                    mapping[this_contact] = my_partner;
                    continue;
                }
                // Loop through the old idB to see if there is a match
                for (sgps::geoSphereTouches_t j = 0; j < old_cnt_count; j++) {
                    sgps::bodyID_t old_idB = granData->previous_idGeometryB[old_cnt_offset + j];
                    sgps::contact_t old_cntType = granData->previous_contactType[old_cnt_offset + j];
                    // If both idB and contact type match, then it is a persistent contact, write it to the mapping
                    // array
                    if (new_idB == old_idB && new_cntType == old_cntType) {
                        my_partner = old_cnt_offset + j;
                        break;
                    }
                }
                mapping[this_contact] = my_partner;
            }
        }
    }
}
