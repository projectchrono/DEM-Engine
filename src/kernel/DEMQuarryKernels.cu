// DEM kernels used for quarrying (statistical) information from the current simulation system
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>

__global__ void computeKE(sgps::DEMDataDT* granData, float* KE) {
    // CUDA does not support initializing shared arrays, so we have to manually load them
    __shared__ float ClumpMasses[_nDistinctClumpBodyTopologies_];
    __shared__ float moiX[_nDistinctClumpBodyTopologies_];
    __shared__ float moiY[_nDistinctClumpBodyTopologies_];
    __shared__ float moiZ[_nDistinctClumpBodyTopologies_];
    if (threadIdx.x < _nActiveLoadingThreads_) {
        const float jitifiedMass[_nDistinctClumpBodyTopologies_] = {_ClumpMasses_};
        const float jitifiedMoiX[_nDistinctClumpBodyTopologies_] = {_moiX_};
        const float jitifiedMoiY[_nDistinctClumpBodyTopologies_] = {_moiY_};
        const float jitifiedMoiZ[_nDistinctClumpBodyTopologies_] = {_moiZ_};
        for (sgps::clumpBodyInertiaOffset_t i = threadIdx.x; i < _nDistinctClumpBodyTopologies_;
             i += _nActiveLoadingThreads_) {
            ClumpMasses[i] = jitifiedMass[i];
            moiX[i] = jitifiedMoiX[i];
            moiY[i] = jitifiedMoiY[i];
            moiZ[i] = jitifiedMoiZ[i];
        }
    }
    __syncthreads();
    sgps::bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < _nOwnerBodies_) {
        sgps::clumpBodyInertiaOffset_t myMassOffset = granData->inertiaPropOffsets[myID];
        float myMass = ClumpMasses[myMassOffset];
        float myMOIX = moiX[myMassOffset];
        float myMOIY = moiY[myMassOffset];
        float myMOIZ = moiZ[myMassOffset];
        // First lin energy
        float myVX = granData->vX[myID];
        float myVY = granData->vY[myID];
        float myVZ = granData->vZ[myID];
        float myKE = 0.5 * myMass * (myVX * myVX + myVY * myVY + myVZ * myVZ);
        // Then rot energy
        myVX = granData->omgBarX[myID];
        myVY = granData->omgBarY[myID];
        myVZ = granData->omgBarZ[myID];
        myKE += 0.5 * (myMOIX * myVX * myVX + myMOIY * myVY * myVY + myMOIZ * myVZ * myVZ);
        KE[myID] = myKE;
    }
}
