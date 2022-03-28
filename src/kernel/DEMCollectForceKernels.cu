// DEM force computation related custom kernels
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>
#include <kernel/DEMHelperKernels.cu>

__global__ void cashInOwnerIndex(sgps::bodyID_t* idOwner,
                                 sgps::bodyID_t* id,
                                 sgps::bodyID_t* ownerClumpBody,
                                 size_t nContactPairs) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        sgps::bodyID_t thisBodyID = id[myID];
        idOwner[myID] = ownerClumpBody[thisBodyID];
    }
}

__global__ void cashInMassIndex(float* massOwner,
                                sgps::clumpBodyInertiaOffset_t* inertiaPropOffsets,
                                sgps::bodyID_t* idOwner,
                                double l,
                                double h,
                                size_t nContactPairs,
                                float* massClumpBody,
                                sgps::clumpBodyInertiaOffset_t nDistinctClumpBodyTopologies) {
    extern __shared__ float ClumpMasses[];
    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < nDistinctClumpBodyTopologies; i++) {
            ClumpMasses[i] = massClumpBody[i];
        }
    }
    __syncthreads();

    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        sgps::bodyID_t thisOwnerID = idOwner[myID];
        sgps::clumpBodyInertiaOffset_t myMassOffset = inertiaPropOffsets[thisOwnerID];
        // Later, to get h2A we divide force by (m*l/h/h)
        massOwner[myID] = (double)ClumpMasses[myMassOffset] * l / h / h;
    }
}

// TODO: make it a template
__global__ void elemDivide(float3* out, float3* a, float* b, float modifier, size_t n) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        auto num = a[myID];
        auto den = b[myID];
        out[myID] = num / den * modifier;
    }
}

// Place information to an array based on an index array and a value array
__global__ void stashElem(float* out1, float* out2, float* out3, sgps::bodyID_t* index, float3* value, size_t n) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        sgps::bodyID_t my_index = index[myID];
        float3 my_value = value[myID];
        out1[my_index] += my_value.x;
        out2[my_index] += my_value.y;
        out3[my_index] += my_value.z;
    }
}
