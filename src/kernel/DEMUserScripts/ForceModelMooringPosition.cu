// DEM force calculation strategies for grain breakage
float E_cnt, G_cnt, CoR_cnt, mu_cnt, Crr_cnt, E_A, E_B;
{
    // E and nu are associated with each material, so obtain them this way
    E_A = E[bodyAMatType];
    float nu_A = nu[bodyAMatType];
    E_B = E[bodyBMatType];
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
    applyOriQToVector3<float, deme::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AOriQ.w, AOriQ.x, AOriQ.y, AOriQ.z);
    applyOriQToVector3<float, deme::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BOriQ.w, BOriQ.x, BOriQ.y, BOriQ.z);
}
float mass_eff, sqrt_Rd, beta;
float3 vrel_tan;
float3 delta_tan = make_float3(delta_tan_x, delta_tan_y, delta_tan_z);

// The (total) relative linear velocity of A relative to B
const float3 velB2A = (ALinVel + rotVelCPA) - (BLinVel + rotVelCPB);
const float projection = dot(velB2A, B2A);
vrel_tan = velB2A - projection * B2A;

const float3 v_rot = rotVelCPB - rotVelCPA;
// This v_rot is only used for identifying resistance direction
const float v_rot_mag = length(v_rot);
mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);

// Now we already have sufficient info to update contact history
{
    delta_tan += ts * vrel_tan;
    const float disp_proj = dot(delta_tan, B2A);
    delta_tan -= disp_proj * B2A;
    delta_time += ts;
}

// If this contact is marked as not broken (unbroken is 1 means it stands for the bond between the components of a
// unbroken particle in grain breakage simulation), then it takes this force model which is a strong cohesion.
if (innerInteraction > DEME_TINY_FLOAT) {
    const double ABdist = length(bodyAPos - bodyBPos);
    initialLength = (innerInteraction > 1.0) ? ABdist : initialLength;  // reusing a variable that survives the loop
    innerInteraction = (innerInteraction > 1.0) ? 1.0 : innerInteraction;
    tension = 0.0;
    float deltaD = (ABdist - initialLength > DEME_TINY_FLOAT) ? (ABdist - initialLength) : 0.0;
    if (deltaD > DEME_TINY_FLOAT) {
        float kn = 20.0 / initialLength;
        // float massCyl=initialLength*3.141592*ARadius*ARadius*800;
        // float c = 0.03 * 2.0 * sqrt(massCyl * kn);
        float epsilon = deltaD / initialLength - 1;
        float Cint = -0.80 * 0.25f * 3.141592 * (ARadius * ARadius) * epsilon;
        float3 A2B = -1.0 * (B2A);  // Unit vector pointing from A to B, which is also the segment direction
        tension = kn * deltaD;
        float3 velAveFluid = -1.0 * (ALinVel + BLinVel) / 2.0;  // fluid velocity taken as segment motion
                                                                /*         if (length(velAveFluid) > DEME_TINY_FLOAT) {
                                                                            float velNormal = dot(velAveFluid, A2B);
                                                                            float velTangential = length(velAveFluid) - velNormal;
                                                                            // normal
                                                                            float3 projVonA2B = A2B * dot(velAveFluid, A2B) / dot(A2B, A2B);
                                                                            float3 normalToSegment = velAveFluid - projVonA2B;
                                                                            normalToSegment = normalToSegment / length(normalToSegment);
                                                                            float drag_normal = -0.50 * 0.5f * 1000.0 * (2.0 * ARadius) * initialLength * velNormal;
                                                                            float drag_tangential = -0.10 * 0.5f * 1000.0 * (2.0 * ARadius) * initialLength * velTangential;
                                                                            force += normalToSegment * drag_normal + A2B * drag_tangential;
                                                                            // printf("%f ", length(A2B));
                                                                        } */

        force += A2B * (tension + Cint);  // - c * velB2A;
    }

} else {  // If unbroken == 0, then the bond is broken, and the grain is broken, components are now separate particles,
          // so they just follow Hertzian contact law.

    // Whether or not the bonding force among particle components exists, contact forces are always active if two
    // particles are in contact: This is a part of our model.
    if (overlapDepth > 0.0) {
        // Material properties
        tension = -1;
        float temp = overlapDepth;
        if (temp > 0.0) {
            float3 rotVelCPA, rotVelCPB;
            {
                // We also need the relative velocity between A and B in global frame to use in the damping terms
                // To get that, we need contact points' rotational velocity in GLOBAL frame
                // This is local rotational velocity (the portion of linear vel contributed by rotation)
                rotVelCPA = cross(ARotVel, locCPA);
                rotVelCPB = cross(BRotVel, locCPB);
                // This is mapping from local rotational velocity to global
                applyOriQToVector3<float, deme::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AOriQ.w, AOriQ.x,
                                                        AOriQ.y, AOriQ.z);
                applyOriQToVector3<float, deme::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BOriQ.w, BOriQ.x,
                                                        BOriQ.y, BOriQ.z);
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
                sqrt_Rd = sqrt(temp * (ARadius * BRadius) / (ARadius + BRadius));
                const float Sn = 2. * E_cnt * sqrt_Rd;

                const float loge = (CoR_cnt < DEME_TINY_FLOAT) ? log(DEME_TINY_FLOAT) : log(CoR_cnt);
                beta = loge / sqrt(loge * loge + deme::PI_SQUARED);

                const float k_n = deme::TWO_OVER_THREE * Sn;
                const float gamma_n = deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

                force += (k_n * temp + gamma_n * projection) * B2A;
                // printf("normal force: %f, %f, %f\n", force.x, force.y, force.z);
            }

            // Rolling resistance part
            if (Crr_cnt > 0.0) {
                // Figure out if we should apply rolling resistance force
                bool should_add_rolling_resistance = true;
                {
                    const float R_eff = sqrtf((ARadius * BRadius) / (ARadius + BRadius));
                    const float kn_simple = deme::FOUR_OVER_THREE * E_cnt * sqrtf(R_eff);
                    const float gn_simple =
                        -2.f * sqrtf(deme::FIVE_OVER_THREE * mass_eff * E_cnt) * beta * powf(R_eff, 0.25f);

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
                        // printf("torque force: %f, %f, %f\n", torque_only_force.x, torque_only_force.y,
                        // torque_only_force.z);
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
        }
    }
}