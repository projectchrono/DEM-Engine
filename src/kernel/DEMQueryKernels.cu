// DEM kernels used for quarrying (statistical) information from the current simulation system
#include <DEM/DEMDefines.h>

// Mass properties are below, if jitified mass properties are in use
_massDefs_;
_moiDefs_;

__global__ void computeKE(sgps::DEMDataDT* granData, sgps::bodyID_t nOwnerBodies, double* KE) {
    sgps::bodyID_t myOwner = blockIdx.x * blockDim.x + threadIdx.x;
    if (myOwner < nOwnerBodies) {
        float myMass;
        float3 myMOI;
        // Get my mass info from either jitified arrays or global memory
        // Outputs myMass
        // Use an input named exactly `myOwner' which is the id of this owner
        { _massAcqStrat_; }

        // Get my mass info from either jitified arrays or global memory
        // Outputs myMOI
        // Use an input named exactly `myOwner' which is the id of this owner
        { _moiAcqStrat_; }

        // First lin energy
        double myVX = granData->vX[myOwner];
        double myVY = granData->vY[myOwner];
        double myVZ = granData->vZ[myOwner];
        double myKE = 0.5 * myMass * (myVX * myVX + myVY * myVY + myVZ * myVZ);
        // Then rot energy
        myVX = granData->omgBarX[myOwner];
        myVY = granData->omgBarY[myOwner];
        myVZ = granData->omgBarZ[myOwner];
        myKE += 0.5 * ((double)myMOI.x * myVX * myVX + (double)myMOI.y * myVY * myVY + (double)myMOI.z * myVZ * myVZ);
        KE[myOwner] = myKE;
    }
}
