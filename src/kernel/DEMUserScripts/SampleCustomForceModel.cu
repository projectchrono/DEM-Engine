// A custom DEM force model which does not have rolling resistance, and its friction coefficient is determined
// externally

float E_cnt, G_cnt, CoR_cnt;
{
    float mu_cnt, Crr_cnt;  // Dummies, we don't use'em in force calculation
    float E_A = E[bodyAMatType];
    float nu_A = nu[bodyAMatType];
    float CoR_A = CoR[bodyAMatType];
    float E_B = E[bodyBMatType];
    float nu_B = nu[bodyBMatType];
    float CoR_B = CoR[bodyBMatType];
    matProxy2ContactParam<float>(E_cnt, G_cnt, CoR_cnt, mu_cnt, Crr_cnt, E_A, nu_A, CoR_A, 0, 0, E_B, nu_B, CoR_B, 0,
                                 0);
}

float3 rotVelCPA, rotVelCPB;
{
    // We also need the relative velocity between A and B in global frame to use in the damping terms
    // To get that, we need contact points' rotational velocity in GLOBAL frame
    // This is local rotational velocity (the portion of linear vel contributed by rotation)
    rotVelCPA = cross(ARotVel, locCPA);
    rotVelCPB = cross(BRotVel, locCPB);
    // This is mapping from local rotational velocity to global
    applyOriQToVector3<float, deme::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AOriQ.w, AOriQ.x, AOriQ.y, AOriQ.z);
    applyOriQToVector3<float, deme::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BOriQ.w, BOriQ.x, BOriQ.y, BOriQ.z);
}

// A few re-usables
float mass_eff, sqrt_Rd, beta;
float3 vrel_tan;
float3 delta_tan = make_float3(delta_tan_x, delta_tan_y, delta_tan_z);

// Normal force part
{
    // The (total) relative linear velocity of A relative to B
    const float3 velB2A = (ALinVel + rotVelCPA) - (BLinVel + rotVelCPB);
    const float projection = dot(velB2A, B2A);
    vrel_tan = velB2A - projection * B2A;

    // Now we already have sufficient info to update contact history
    {
        delta_tan += ts * vrel_tan;
        const float disp_proj = dot(delta_tan, B2A);
        delta_tan -= disp_proj * B2A;
    }

    mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);
    sqrt_Rd = sqrt(overlapDepth * (ARadius * BRadius) / (ARadius + BRadius));
    const float Sn = 2. * E_cnt * sqrt_Rd;

    const float loge = (CoR_cnt < DEME_TINY_FLOAT) ? log(DEME_TINY_FLOAT) : log(CoR_cnt);
    beta = loge / sqrt(loge * loge + deme::PI_SQUARED);

    const float k_n = deme::TWO_OVER_THREE * Sn;
    const float gamma_n = deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

    force += (k_n * overlapDepth + gamma_n * projection) * B2A;
}

// Tangential force part
{
    // mu is obtained from a `owner wildcard', which is a custom array where the user can associated any quantity with
    // each owner. In this case, it is mu_custom.
    float mu = DEME_MAX(mu_custom[AOwner], mu_custom[BOwner]);
    const float kt = 8. * G_cnt * sqrt_Rd;
    const float gt = -deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(mass_eff * kt);
    float3 tangent_force = -kt * delta_tan - gt * vrel_tan;
    const float ft = length(tangent_force);
    if (ft > DEME_TINY_FLOAT) {
        // Reverse-engineer to get tangential displacement
        const float ft_max = length(force) * mu;
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

// Finally, make sure we update those wildcards (in this case, contact history)
delta_tan_x = delta_tan.x;
delta_tan_y = delta_tan.y;
delta_tan_z = delta_tan.z;