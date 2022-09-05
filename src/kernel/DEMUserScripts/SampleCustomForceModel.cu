// DEM force calculation strategies, modifiable

// Material properties and time (user referrable)
float E_cnt, G_cnt, CoR_cnt, mu_cnt, Crr_cnt, h;
{
    h = simParams->h;
    float E_A = E[bodyAMatType];
    float nu_A = nu[bodyAMatType];
    float CoR_A = CoR[bodyAMatType];
    float mu_A = mu[bodyAMatType];
    float Crr_A = Crr[bodyAMatType];
    float E_B = E[bodyBMatType];
    float nu_B = nu[bodyBMatType];
    float CoR_B = CoR[bodyBMatType];
    float mu_B = mu[bodyBMatType];
    float Crr_B = Crr[bodyBMatType];
    matProxy2ContactParam<float>(E_cnt, G_cnt, CoR_cnt, mu_cnt, Crr_cnt, E_A, nu_A, CoR_A, mu_A, Crr_A, E_B, nu_B,
                                 CoR_B, mu_B, Crr_B);
}

{
    // We also need the relative velocity between A and B in global frame to use in the damping terms
    // To get that, we need contact points' rotational velocity in GLOBAL frame
    // This is local rotational velocity (the portion of linear vel contributed by rotation)
    rotVelCPA = cross(ARotVel, locCPA);
    rotVelCPB = cross(BRotVel, locCPB);
    // This is mapping from local rotational velocity to global
    applyOriQToVector3<float, sgps::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AoriQ0, AoriQ1, AoriQ2, AoriQ3);
    applyOriQToVector3<float, sgps::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BoriQ0, BoriQ1, BoriQ2, BoriQ3);
    // Get contact history from global memory
    delta_tan = granData->contactHistory[myContactID];
    // Get contact duration time from global memory
    delta_time = granData->contactDuration[myContactID];
}

// A few re-usables
float mass_eff, sqrt_Rd, beta;
float3 vrel_tan;

// Normal force part
{
    // The (total) relative linear velocity of A relative to B
    const float3 velB2A = (ALinVel + rotVelCPA) - (BLinVel + rotVelCPB);
    const float projection = dot(velB2A, B2A);
    vrel_tan = velB2A - projection * B2A;

    // Now we already have sufficient info to update contact history
    {
        delta_tan += h * vrel_tan;
        const float disp_proj = dot(delta_tan, B2A);
        delta_tan -= disp_proj * B2A;
        delta_time += h;
    }

    mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);
    sqrt_Rd = sqrt(overlapDepth * (ARadius * BRadius) / (ARadius + BRadius));
    const float Sn = 2. * E_cnt * sqrt_Rd;

    const float loge = (CoR_cnt < SGPS_DEM_TINY_FLOAT) ? log(SGPS_DEM_TINY_FLOAT) : log(CoR_cnt);
    beta = loge / sqrt(loge * loge + sgps::PI_SQUARED);

    const float k_n = sgps::TWO_OVER_THREE * Sn;
    const float gamma_n = sgps::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

    force += (k_n * overlapDepth + gamma_n * projection) * B2A;
    // printf("normal force: %f, %f, %f\n", force.x, force.y, force.z);
}

// Tangential force part
if (mu_cnt > 0.0) {
    const float kt = 8. * G_cnt * sqrt_Rd;
    const float gt = -sgps::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(mass_eff * kt);
    float3 tangent_force = -kt * delta_tan - gt * vrel_tan;
    const float ft = length(tangent_force);
    if (ft > SGPS_DEM_TINY_FLOAT) {
        // Reverse-engineer to get tangential displacement
        const float ft_max = length(force) * mu_cnt;
        if (ft > ft_max) {
            tangent_force = (ft_max / ft) * tangent_force;
            delta_tan = (tangent_force + gt * vrel_tan) / (-kt);
        }
    } else {
        tangent_force = make_float3(0, 0, 0);
    }
    // Use force to collect tangent_force
    force += tangent_force;
    // printf("tangent force: %f, %f, %f\n", tangent_force.x, tangent_force.y, tangent_force.z);
}