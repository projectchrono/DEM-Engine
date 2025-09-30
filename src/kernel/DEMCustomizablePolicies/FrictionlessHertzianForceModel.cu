// DEM force calculation strategies, modifiable

if (overlapDepth > 0) {
    // Material properties
    float E_cnt, CoR_cnt;
    {
        matProxy2ContactParam<float>(E_cnt, E[bodyAMatType], nu[bodyAMatType], E[bodyBMatType], nu[bodyBMatType]);
        // CoR is pair-wise, so obtain it this way
        CoR_cnt = CoR[bodyAMatType][bodyBMatType];
    }

    float3 rotVelCPA, rotVelCPB;
    {
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
    }

    // The (total) relative linear velocity of A relative to B
    const float3 velB2A = (ALinVel + rotVelCPA) - (BLinVel + rotVelCPB);
    const float projection = dot(velB2A, B2A);

    const float mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);
    float sqrt_Rd = sqrt(overlapDepth * (ARadius * BRadius) / (ARadius + BRadius));
    const float Sn = 2. * E_cnt * sqrt_Rd;

    const float loge = (CoR_cnt < DEME_TINY_FLOAT) ? log(DEME_TINY_FLOAT) : log(CoR_cnt);
    float beta = loge / sqrt(loge * loge + deme::PI_SQUARED);

    const float k_n = deme::TWO_OVER_THREE * Sn;
    const float gamma_n = deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

    // normal force (that A feels)
    force += (k_n * overlapDepth + gamma_n * projection) * B2A;

    // printf("A linear vel is (%.9g, %.9g, %.9g)\n", ALinVel.x, ALinVel.y, ALinVel.z);
    // printf("A rotational vel is (%.9g, %.9g, %.9g)\n", ARotVel.x, ARotVel.y, ARotVel.z);
    // printf("locCPA is (%.9g, %.9g, %.9g)\n", locCPA.x, locCPA.y, locCPA.z);
    // printf("Force is (%.9g, %.9g, %.9g) on body %d\n", force.x, force.y, force.z, AOwner);
    // printf("CoR_cnt is %.9g\n", CoR_cnt);
    // printf("Sn is %.9g, sqrt_Rd is %.9g\n", Sn, sqrt_Rd);
    // printf("k_n is %.9g, gamma_n is %.9g\n", k_n, gamma_n);
    // printf("overlapDepth is %.9g, projection is %.9g\n", overlapDepth, projection);
}
