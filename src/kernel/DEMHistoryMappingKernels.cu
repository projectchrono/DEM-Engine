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
