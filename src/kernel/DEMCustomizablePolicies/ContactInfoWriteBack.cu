granData->contactPointGeometryA[myPatchContactID] = locCPA;
granData->contactPointGeometryB[myPatchContactID] = locCPB;
granData->contactForces[myPatchContactID] = force;
granData->contactTorque_convToForce[myPatchContactID] = torque_only_force;
if (simParams->storeNormal) {
    granData->contactNormals[myPatchContactID] = B2A;
}
