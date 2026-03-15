float3 forceA = force;
float3 forceB = make_float3(-force.x, -force.y, -force.z);
float3 torqueA = torque_only_force;
float3 torqueB = make_float3(-torque_only_force.x, -torque_only_force.y, -torque_only_force.z);

if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f) {
    if (wrapA) {
        const float sin_span = -wrapA_sin;
        const float cos_span = wrapA_cos;
        forceA = cylPeriodicRotate(forceA, make_float3(0.f, 0.f, 0.f), simParams->cylPeriodicAxisVec,
                                   simParams->cylPeriodicU, simParams->cylPeriodicV, cos_span, sin_span);
        torqueA = cylPeriodicRotate(torqueA, make_float3(0.f, 0.f, 0.f), simParams->cylPeriodicAxisVec,
                                    simParams->cylPeriodicU, simParams->cylPeriodicV, cos_span, sin_span);
    }
    if (wrapB) {
        const float sin_span = -wrapB_sin;
        const float cos_span = wrapB_cos;
        forceB = cylPeriodicRotate(forceB, make_float3(0.f, 0.f, 0.f), simParams->cylPeriodicAxisVec,
                                   simParams->cylPeriodicU, simParams->cylPeriodicV, cos_span, sin_span);
        torqueB = cylPeriodicRotate(torqueB, make_float3(0.f, 0.f, 0.f), simParams->cylPeriodicAxisVec,
                                    simParams->cylPeriodicU, simParams->cylPeriodicV, cos_span, sin_span);
    }
}

// Take care of A
{
    atomicAdd(granData->aX + AOwner, forceA.x / AOwnerMass);
    atomicAdd(granData->aY + AOwner, forceA.y / AOwnerMass);
    atomicAdd(granData->aZ + AOwner, forceA.z / AOwnerMass);

    float max_local_lever = DEME_HUGE_FLOAT;
    if (granData->ownerBoundRadius && AOwner != deme::NULL_BODYID && AOwner < simParams->nOwnerBodies) {
        const float bound_r = fmaxf(granData->ownerBoundRadius[AOwner], 0.f);
        if (isfinite(bound_r) && bound_r > DEME_TINY_FLOAT) {
            const float geom_tol = fmaxf(simParams->dyn.beta + simParams->maxFamilyExtraMargin, 0.f) + 1e-4f;
            max_local_lever = fmaxf(bound_r + geom_tol, 1e-3f);
        }
    }
    const float cpA_sq = locCPA.x * locCPA.x + locCPA.y * locCPA.y + locCPA.z * locCPA.z;
    const bool bad_cp = !isfinite(locCPA.x) || !isfinite(locCPA.y) || !isfinite(locCPA.z) || !isfinite(cpA_sq) ||
                        cpA_sq > max_local_lever * max_local_lever;
    const bool bad_vec = !isfinite(forceA.x) || !isfinite(forceA.y) || !isfinite(forceA.z) || !isfinite(torqueA.x) ||
                         !isfinite(torqueA.y) || !isfinite(torqueA.z);
    if (!(bad_cp || bad_vec)) {
        // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque
        // only, not linear velocity
        float3 myF = (forceA + torqueA);
        // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
        applyOriQToVector3<float, deme::oriQ_t>(myF.x, myF.y, myF.z, AOriQ.w, -AOriQ.x, -AOriQ.y, -AOriQ.z);
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

    float max_local_lever = DEME_HUGE_FLOAT;
    if (granData->ownerBoundRadius && BOwner != deme::NULL_BODYID && BOwner < simParams->nOwnerBodies) {
        const float bound_r = fmaxf(granData->ownerBoundRadius[BOwner], 0.f);
        if (isfinite(bound_r) && bound_r > DEME_TINY_FLOAT) {
            const float geom_tol = fmaxf(simParams->dyn.beta + simParams->maxFamilyExtraMargin, 0.f) + 1e-4f;
            max_local_lever = fmaxf(bound_r + geom_tol, 1e-3f);
        }
    }
    const float cpB_sq = locCPB.x * locCPB.x + locCPB.y * locCPB.y + locCPB.z * locCPB.z;
    const bool bad_cp = !isfinite(locCPB.x) || !isfinite(locCPB.y) || !isfinite(locCPB.z) || !isfinite(cpB_sq) ||
                        cpB_sq > max_local_lever * max_local_lever;
    const bool bad_vec = !isfinite(forceB.x) || !isfinite(forceB.y) || !isfinite(forceB.z) || !isfinite(torqueB.x) ||
                         !isfinite(torqueB.y) || !isfinite(torqueB.z);
    if (!(bad_cp || bad_vec)) {
        // torque_inForceForm is usually the contribution of rolling resistance and it contributes to torque
        // only, not linear velocity
        float3 myF = (forceB + torqueB);
        // F is in global frame, but it needs to be in local to coordinate with moi and cntPnt
        applyOriQToVector3<float, deme::oriQ_t>(myF.x, myF.y, myF.z, BOriQ.w, -BOriQ.x, -BOriQ.y, -BOriQ.z);
        const float3 angAcc = cross(locCPB, myF) / BOwnerMOI;
        atomicAdd(granData->alphaX + BOwner, angAcc.x);
        atomicAdd(granData->alphaY + BOwner, angAcc.y);
        atomicAdd(granData->alphaZ + BOwner, angAcc.z);
    }
}
