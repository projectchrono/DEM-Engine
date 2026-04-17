float3 forceA = force;
float3 forceB = make_float3(-force.x, -force.y, -force.z);
float3 torqueA = torque_only_force;
float3 torqueB = make_float3(-torque_only_force.x, -torque_only_force.y, -torque_only_force.z);

// Take care of A
{
    atomicAdd(granData->aX + AOwner, forceA.x / AOwnerMass);
    atomicAdd(granData->aY + AOwner, forceA.y / AOwnerMass);
    atomicAdd(granData->aZ + AOwner, forceA.z / AOwnerMass);

    const float cpA_sq = locCPA.x * locCPA.x + locCPA.y * locCPA.y + locCPA.z * locCPA.z;
    const bool bad_cp = !isfinite3(locCPA);
    const bool bad_vec = !isfinite3(forceA) || !isfinite3(torqueA);
    if (!(bad_cp || bad_vec)) {
        // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque
        // only, not linear velocity
        float3 myF = (forceA + torqueA);
        // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
        applyOriQToVector3(myF, make_float4(-AOriQ.x, -AOriQ.y, -AOriQ.z, AOriQ.w));
        const float3 angAcc = cross(locCPA, myF) / AOwnerMOI;
        atomicAdd(granData->alphaX + AOwner, angAcc.x);
        atomicAdd(granData->alphaY + AOwner, angAcc.y);
        atomicAdd(granData->alphaZ + AOwner, angAcc.z);
    }
}

// Take care of B
{
    atomicAdd(granData->aX + BOwner, forceB.x / BOwnerMass);
    atomicAdd(granData->aY + BOwner, forceB.y / BOwnerMass);
    atomicAdd(granData->aZ + BOwner, forceB.z / BOwnerMass);

    const float cpB_sq = locCPB.x * locCPB.x + locCPB.y * locCPB.y + locCPB.z * locCPB.z;
    const bool bad_cp = !isfinite3(locCPB);
    const bool bad_vec = !isfinite3(forceB) || !isfinite3(torqueB);
    if (!(bad_cp || bad_vec)) {
        // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque
        // only, not linear velocity
        float3 myF = (forceB + torqueB);
        // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
        applyOriQToVector3(myF, make_float4(-BOriQ.x, -BOriQ.y, -BOriQ.z, BOriQ.w));
        const float3 angAcc = cross(locCPB, myF) / BOwnerMOI;
        atomicAdd(granData->alphaX + BOwner, angAcc.x);
        atomicAdd(granData->alphaY + BOwner, angAcc.y);
        atomicAdd(granData->alphaZ + BOwner, angAcc.z);
    }
}
