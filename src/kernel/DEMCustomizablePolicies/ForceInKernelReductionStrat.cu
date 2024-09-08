// Take care of A
{
    atomicAdd(granData->aX + AOwner, force.x / AOwnerMass);
    atomicAdd(granData->aY + AOwner, force.y / AOwnerMass);
    atomicAdd(granData->aZ + AOwner, force.z / AOwnerMass);

    // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque
    // only, not linear velocity
    float3 myF = (force + torque_only_force);
    // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
    applyOriQToVector3<float, deme::oriQ_t>(myF.x, myF.y, myF.z, AOriQ.w, -AOriQ.x, -AOriQ.y, -AOriQ.z);
    const float3 angAcc = cross(locCPA, myF) / AOwnerMOI;
    atomicAdd(granData->alphaX + AOwner, angAcc.x);
    atomicAdd(granData->alphaY + AOwner, angAcc.y);
    atomicAdd(granData->alphaZ + AOwner, angAcc.z);
}

// Take care of B
{
    atomicAdd(granData->aX + BOwner, -force.x / BOwnerMass);
    atomicAdd(granData->aY + BOwner, -force.y / BOwnerMass);
    atomicAdd(granData->aZ + BOwner, -force.z / BOwnerMass);

    // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque
    // only, not linear velocity
    float3 myF = -1.f * (force + torque_only_force);
    // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
    applyOriQToVector3<float, deme::oriQ_t>(myF.x, myF.y, myF.z, BOriQ.w, -BOriQ.x, -BOriQ.y, -BOriQ.z);
    const float3 angAcc = cross(locCPB, myF) / BOwnerMOI;
    atomicAdd(granData->alphaX + BOwner, angAcc.x);
    atomicAdd(granData->alphaY + BOwner, angAcc.y);
    atomicAdd(granData->alphaZ + BOwner, angAcc.z);
}
