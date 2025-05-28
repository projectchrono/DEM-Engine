// DEM force calculation strategies for a model with a cohesive force.
// Note the force is in effect only when two entities are in contact

// No need to do any contact force calculation if no contact. And it can happen,
// when we added extra contact margins
if (overlapDepth > 0) {
    // Material properties
    float E_cnt, G_cnt, CoR_cnt, mu_cnt, Crr_cnt, Cohesion_coeff;
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
        Cohesion_coeff = Cohesion[bodyAMatType][bodyBMatType];
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
                // printf("torque force: %f, %f, %f\n", torque_only_force.x, torque_only_force.y, torque_only_force.z);
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

    // Make sure we update those wildcards (in this case, contact history)
    delta_tan_x = delta_tan.x;
    delta_tan_y = delta_tan.y;
    delta_tan_z = delta_tan.z;

    // Finally add a simple cohesion model.
    // The cohesion model is as simple as simple as Cohesion_coeff times effective mass.
    // And cohesion pulls A towards B, so direction is -B2A
    force += Cohesion_coeff * mass_eff * (-B2A);

} else {
    // This is to be more rigorous. If in fact no physical contact, then contact wildcards (such as contact history)
    // should be cleared (they will also be automatically cleared if the contact is no longer detected).
    delta_time = 0;
    delta_tan_x = 0;
    delta_tan_y = 0;
    delta_tan_z = 0;
}
