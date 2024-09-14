// A force model that each particle has a non-local effect that pulls other particles towards itself (gravity well)

// Normal force. But you know, if you are simulating a solver system like the demo that uses this force model, then
// the normal force does not matter unless 2 planets collide...
if (overlapDepth > 0) {
    // Material properties represent a frictionless scenario
    float E_cnt, G_cnt, CoR_cnt;
    {
        float E_A = E[bodyAMatType];
        float nu_A = nu[bodyAMatType];
        float E_B = E[bodyBMatType];
        float nu_B = nu[bodyBMatType];
        matProxy2ContactParam<float>(E_cnt, G_cnt, E_A, nu_A, E_B, nu_B);
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
    const float sqrt_Rd = sqrt(overlapDepth * (ARadius * BRadius) / (ARadius + BRadius));
    const float Sn = 2. * E_cnt * sqrt_Rd;

    const float loge = (CoR_cnt < DEME_TINY_FLOAT) ? log(DEME_TINY_FLOAT) : log(CoR_cnt);
    const float beta = loge / sqrt(loge * loge + deme::PI_SQUARED);

    const float k_n = deme::TWO_OVER_THREE * Sn;
    const float gamma_n = deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

    force += (k_n * overlapDepth + gamma_n * projection) * B2A;
}

// A non-local gravitational pull
{
    // In AU and day
    const double Gconst = 6.674e-11 * 86400 * 86400 / 1.496e+11 / 1.496e+11 / 1.496e+11;
    const double ABdist2 = dot(bodyAPos - bodyBPos, bodyAPos - bodyBPos);
    // To A, gravity pulls it towards B, so -B2A direction
    force += Gconst * my_mass_A[AGeo] * my_mass_B[BGeo] / ABdist2 * (-B2A);
}
