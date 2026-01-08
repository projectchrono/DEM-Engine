// DEM force calculation strategies, modifiable

if (overlapDepth > 0) {
    // Material properties
    float E_cnt, CoR_cnt;
    {
        matProxy2ContactParam<float>(E_cnt, E[bodyAMatType], nu[bodyAMatType], E[bodyBMatType], nu[bodyBMatType]);
        // CoR is pair-wise, so obtain it this way
        CoR_cnt = CoR[bodyAMatType][bodyBMatType];
    }

    float3 rotVelCPA = make_float3(0.f, 0.f, 0.f);
    float3 rotVelCPB = make_float3(0.f, 0.f, 0.f);
    if constexpr (AType != deme::GEO_T_SPHERE || BType != deme::GEO_T_SPHERE) {
        // We also need the relative velocity between A and B in global frame to use in the damping terms
        // To get that, we need contact points' rotational velocity in GLOBAL frame
        // This is local rotational velocity (the portion of linear vel contributed by rotation)
        rotVelCPA = cross(ARotVel, locCPA);
        rotVelCPB = cross(BRotVel, locCPB);
        // This is mapping from local rotational velocity to global
        applyOriQToVector3<float, deme::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AOriQ.w, AOriQ.x, AOriQ.y,
                                                AOriQ.z);
        applyOriQToVector3<float, deme::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BOriQ.w, BOriQ.x, BOriQ.y,
                                                BOriQ.z);
    } else {
        if (simParams->useAngVelMargin) {
            rotVelCPA = cross(ARotVel, locCPA);
            rotVelCPB = cross(BRotVel, locCPB);
            applyOriQToVector3<float, deme::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AOriQ.w, AOriQ.x,
                                                    AOriQ.y, AOriQ.z);
            applyOriQToVector3<float, deme::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BOriQ.w, BOriQ.x,
                                                    BOriQ.y, BOriQ.z);
        }
    }

    // The (total) relative linear velocity of A relative to B
    const float3 velB2A = (ALinVel + rotVelCPA) - (BLinVel + rotVelCPB);
    const float projection = dot(velB2A, B2A);

    const float mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);

    // Contact radius: use area only when a triangle is involved, otherwise fall back to classic Hertz form
    const bool tri_involved = (AType == deme::GEO_T_TRIANGLE) || (BType == deme::GEO_T_TRIANGLE);
    float cnt_rad;
    if (tri_involved) {
        cnt_rad = sqrtf(overlapArea / deme::PI);
    } else {
        const float effective_radius =
            (BType == deme::GEO_T_ANALYTICAL) ? ARadius : (ARadius * BRadius) / (ARadius + BRadius);
        cnt_rad = sqrtf(overlapDepth * effective_radius);
    }
    const float Sn = 2.f * E_cnt * cnt_rad;

    const float loge = (CoR_cnt < DEME_TINY_FLOAT) ? log(DEME_TINY_FLOAT) : log(CoR_cnt);
    const float beta = loge / sqrt(loge * loge + deme::PI_SQUARED);

    const float k_n = (2.f / 3.f) * Sn;
    const float gamma_n = deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrtf(Sn * mass_eff);

    force += (k_n * overlapDepth + gamma_n * projection) * B2A;
}
