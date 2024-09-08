/////////////////////////////////////////////////////////////
// The first part is just the standard full Hertzian--Mindlin
/////////////////////////////////////////////////////////////

// No need to do any contact force calculation if no contact. And it can happen,
// since we added extra contact margin for adding electrostatic forces before
// physical contacts emerge.
if (overlapDepth > 0) {
    // Material properties
    float E_cnt, G_cnt, CoR_cnt, mu_cnt, Crr_cnt;
    {
        // E and nu are associated with each material, so obtain them this way
        float E_A = E[bodyAMatType];
        float nu_A = nu[bodyAMatType];
        float E_B = E[bodyBMatType];
        float nu_B = nu[bodyBMatType];
        matProxy2ContactParam<float>(E_cnt, G_cnt, E_A, nu_A, E_B, nu_B);
        // CoR, mu and Crr are pair-wise, so obtain them this way
        CoR_cnt = CoR[bodyAMatType][bodyBMatType];
        mu_cnt = mu[bodyAMatType][bodyBMatType];
        Crr_cnt = Crr[bodyAMatType][bodyBMatType];
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
            delta_time += ts;
        }

        mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);
        sqrt_Rd = sqrt(overlapDepth * (ARadius * BRadius) / (ARadius + BRadius));
        const float Sn = 2. * E_cnt * sqrt_Rd;

        const float loge = (CoR_cnt < DEME_TINY_FLOAT) ? log(DEME_TINY_FLOAT) : log(CoR_cnt);
        beta = loge / sqrt(loge * loge + deme::PI_SQUARED);

        const float k_n = deme::TWO_OVER_THREE * Sn;
        const float gamma_n = deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

        force += (k_n * overlapDepth + gamma_n * projection) * B2A;
        // printf("normal force: %f, %f, %f\n", force.x, force.y, force.z);
    }

    // Rolling resistance part
    if (Crr_cnt > 0.0) {
        // Figure out if we should apply rolling resistance force
        bool should_add_rolling_resistance = true;
        {
            const float R_eff = sqrtf((ARadius * BRadius) / (ARadius + BRadius));
            const float kn_simple = deme::FOUR_OVER_THREE * E_cnt * sqrtf(R_eff);
            const float gn_simple = -2.f * sqrtf(deme::FIVE_OVER_THREE * mass_eff * E_cnt) * beta * powf(R_eff, 0.25f);

            const float d_coeff = gn_simple / (2.f * sqrtf(kn_simple * mass_eff));

            if (d_coeff < 1.0) {
                float t_collision = deme::PI * sqrtf(mass_eff / (kn_simple * (1.f - d_coeff * d_coeff)));
                if (delta_time <= t_collision) {
                    should_add_rolling_resistance = false;
                }
            }
        }
        // If should, then compute it (using Schwartz model)
        if (should_add_rolling_resistance) {
            // Tangential velocity (only rolling contribution) of B relative to A, at contact point, in global
            const float3 v_rot = rotVelCPB - rotVelCPA;
            // This v_rot is only used for identifying resistance direction
            const float v_rot_mag = length(v_rot);
            if (v_rot_mag > DEME_TINY_FLOAT) {
                // You should know that Crr_cnt * normal_force is the underlying formula, and in our model,
                // it is a `force' that produces torque only, instead of also cancelling out friction.
                // Its direction is that it `resists' rotation, see picture in
                // https://en.wikipedia.org/wiki/Rolling_resistance.
                torque_only_force = (v_rot / v_rot_mag) * (Crr_cnt * length(force));
            }
        }
    }

    // Tangential force part
    if (mu_cnt > 0.0) {
        const float kt = 8. * G_cnt * sqrt_Rd;
        const float gt = -deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(mass_eff * kt);
        float3 tangent_force = -kt * delta_tan - gt * vrel_tan;
        const float ft = length(tangent_force);
        if (ft > DEME_TINY_FLOAT) {
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

    // Finally, make sure we update those wildcards (in this case, contact history)
    delta_tan_x = delta_tan.x;
    delta_tan_y = delta_tan.y;
    delta_tan_z = delta_tan.z;
} else {
    // This is to be more rigorous. If in fact no physical contact, then contact wildcards (such as contact history)
    // should be cleared (they will also be automatically cleared if the contact is no longer detected).
    delta_time = 0;
    delta_tan_x = 0;
    delta_tan_y = 0;
    delta_tan_z = 0;
}

/////////////////////////////////////////////////
// Now we add an extra electrostatic force
////////////////////////////////////////////////
{
    const float k = 8.99e9;
    const double ABdist2 = dot(bodyAPos - bodyBPos, bodyAPos - bodyBPos);
    // If Q_A and Q_B are the same sign, then the force pushes A away from B, so B2A is the direction.
    force += k * Q_A[AGeo] * Q_B[BGeo] / ABdist2 * (B2A);
    // Fun part: we can modify the electric charge on the fly. But we have to use atomic, since multiple contacts
    // can modify the same Q.
    // But this is not recommend unless you understand what you are doing, and there are a lot of details related to it.
    // For example, although the charges transfer between geometries, the geometries within one clump cannot
    // re-distribute elec charges among them, since no contact among geometries in one clump. Still, you could write
    // your own subroutine to further modify those geometry and/or own wildcards in your script, or within the force
    // model.
    // On the other hand, if you do not need to modify the wildcards, you just need to use them for calculating
    // the force, then that is probably easier and with less strings attached to it. I can see this being more
    // useful.
    if (overlapDepth > 0) {  // Exchange the elec charge only in physical contact
        float avg_Q = (Q_A[AGeo] + Q_B[BGeo]) / 2.;
        float A_change_dir = (abs(avg_Q - Q_A[AGeo]) > 1e-11) ? (avg_Q - Q_A[AGeo]) / abs(avg_Q - Q_A[AGeo]) : 0.;
        // Modify the charge they carry... the rate is 1e-8 per second
        atomicAdd(Q_A + AGeo, A_change_dir * 1e-8 * ts);
        atomicAdd(Q_B + BGeo, -A_change_dir * 1e-8 * ts);
    }
}