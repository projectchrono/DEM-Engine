// DEM kernels used for quarrying (statistical) information from the current simulation system
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>

__global__ void computeKE(sgps::DEMDataDT* granData, float* KE) {
    // CUDA does not support initializing shared arrays, so we have to manually load them
    __shared__ float ClumpMasses[_nTotalBodyTopologies_];
    __shared__ float moiX[_nTotalBodyTopologies_];
    __shared__ float moiY[_nTotalBodyTopologies_];
    __shared__ float moiZ[_nTotalBodyTopologies_];
    if (threadIdx.x < _nActiveLoadingThreads_) {
        const float jitifiedMass[_nTotalBodyTopologies_] = {_ClumpMasses_};
        const float jitifiedMoiX[_nTotalBodyTopologies_] = {_moiX_};
        const float jitifiedMoiY[_nTotalBodyTopologies_] = {_moiY_};
        const float jitifiedMoiZ[_nTotalBodyTopologies_] = {_moiZ_};
        for (sgps::clumpBodyInertiaOffset_t i = threadIdx.x; i < _nTotalBodyTopologies_; i += _nActiveLoadingThreads_) {
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
