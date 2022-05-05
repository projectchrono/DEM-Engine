// DEM kernels used for quarrying (statistical) information from the current simulation system
#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>

__global__ void computeKE(sgps::DEMDataDT* granData, float* KE) {
    // _nTotalBodyTopologies_  elements are in these arrays
    const float ClumpMasses[] = {_ClumpMasses_};
    const float moiX[] = {_moiX_};
    const float moiY[] = {_moiY_};
    const float moiZ[] = {_moiZ_};

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
