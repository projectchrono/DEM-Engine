// DEM force computation related custom kernels
#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>
_kernelIncludes_;

// For analytical entities' owners
__constant__ __device__ deme::bodyID_t objOwner[] = {_objOwner_};
// Mass properties are below, if jitified mass properties are in use
_massDefs_;
_moiDefs_;

// computes a ./ b
__global__ void forceToAcc(deme::DEMDataDT* granData, size_t n) {
    deme::contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        deme::contact_t thisCntType = granData->contactType[myID];
        const float3 F = granData->contactForces[myID];

        // Take care of A
        {
            float myMass;
            float3 myMOI;
            const deme::bodyID_t idGeo = granData->idGeometryA[myID];
            const float3 myCntPnt = granData->contactPointGeometryA[myID];

            const deme::bodyID_t myOwner = granData->ownerClumpBody[idGeo];
            // Get my mass info from either jitified arrays or global memory
            // Outputs myMass
            // Use an input named exactly `myOwner' which is the id of this owner
            {
                _massAcqStrat_;
                _moiAcqStrat_;
            }

            atomicAdd(granData->aX + myOwner, F.x / myMass);
            atomicAdd(granData->aY + myOwner, F.y / myMass);
            atomicAdd(granData->aZ + myOwner, F.z / myMass);

            // Then ang acc
            const deme::oriQ_t myOriQw = granData->oriQw[myOwner];
            const deme::oriQ_t myOriQx = granData->oriQx[myOwner];
            const deme::oriQ_t myOriQy = granData->oriQy[myOwner];
            const deme::oriQ_t myOriQz = granData->oriQz[myOwner];

            // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque only,
            // not linear velocity
            float3 myF = (F + granData->contactTorque_convToForce[myID]);
            // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
            applyOriQToVector3<float, deme::oriQ_t>(myF.x, myF.y, myF.z, myOriQw, -myOriQx, -myOriQy, -myOriQz);
            const float3 angAcc = cross(myCntPnt, myF) / myMOI;
            atomicAdd(granData->alphaX + myOwner, angAcc.x);
            atomicAdd(granData->alphaY + myOwner, angAcc.y);
            atomicAdd(granData->alphaZ + myOwner, angAcc.z);
        }

        // Take care of B
        {
            float myMass;
            float3 myMOI;
            const deme::bodyID_t idGeo = granData->idGeometryB[myID];
            const float3 myCntPnt = granData->contactPointGeometryB[myID];

            deme::bodyID_t myOwner;
            if (thisCntType == deme::SPHERE_SPHERE_CONTACT) {
                myOwner = granData->ownerClumpBody[idGeo];
            } else if (thisCntType == deme::SPHERE_MESH_CONTACT) {
                myOwner = granData->ownerMesh[idGeo];
            } else {
                // This is a sphere--analytical geometry contact, its owner is jitified
                myOwner = objOwner[idGeo];
            }

            // Get my mass info from either jitified arrays or global memory
            // Outputs myMass
            // Use an input named exactly `myOwner' which is the id of this owner
            {
                _massAcqStrat_;
                _moiAcqStrat_;
            }

            atomicAdd(granData->aX + myOwner, -F.x / myMass);
            atomicAdd(granData->aY + myOwner, -F.y / myMass);
            atomicAdd(granData->aZ + myOwner, -F.z / myMass);

            // Then ang acc
            const deme::oriQ_t myOriQw = granData->oriQw[myOwner];
            const deme::oriQ_t myOriQx = granData->oriQx[myOwner];
            const deme::oriQ_t myOriQy = granData->oriQy[myOwner];
            const deme::oriQ_t myOriQz = granData->oriQz[myOwner];

            // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque only,
            // not linear velocity
            float3 myF = -1.f * (F + granData->contactTorque_convToForce[myID]);
            // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
            applyOriQToVector3<float, deme::oriQ_t>(myF.x, myF.y, myF.z, myOriQw, -myOriQx, -myOriQy, -myOriQz);
            const float3 angAcc = cross(myCntPnt, myF) / myMOI;
            atomicAdd(granData->alphaX + myOwner, angAcc.x);
            atomicAdd(granData->alphaY + myOwner, angAcc.y);
            atomicAdd(granData->alphaZ + myOwner, angAcc.z);
        }
    }
}
