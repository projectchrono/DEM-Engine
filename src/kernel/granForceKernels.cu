// Granular force computation related custom kernels
#include <granular/DataStructs.h>

inline __device__ void applyGravity(unsigned int thisClump,
                                    sgps::GranSimParams* simParams,
                                    sgps::GranDataDT* granData) {
    // Actually, h should be JITCed into the kernel itself
    // granData->h2aX[thisClump] += simParams->h * simParams->h * simParams->Gx;
}

__global__ void deriveClumpForces(sgps::GranSimParams* simParams, sgps::GranDataDT* granData) {
    unsigned int thisClump = blockIdx.x * blockDim.x + threadIdx.x;
    printf("thisClump: %u\n", thisClump);
    unsigned int a = simParams->nClumpBodies;
    printf("nClump: %u\n", a);
    if (thisClump < simParams->nClumpBodies) {
        // applyGravity(thisClump, simParams, granData);
    }
}
