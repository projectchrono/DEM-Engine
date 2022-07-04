//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#ifndef SGPS_DEM_MODEL_STASH
#define SGPS_DEM_MODEL_STASH

#include <cstring>
#include <cmath>
#include <vector>

#include <DEM/DEMDefines.h>
#include <DEM/DEMStructs.h>
#include <DEM/HostSideHelpers.hpp>

namespace sgps {

// TODO: Mark those user referrables
inline std::string DEM_HERTZIAN_FORCE_MODEL() {
    std::string model;
    model =
        R"V0G0N(
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
            const float Sn = 2. * E * sqrt_Rd;

            const float loge = (CoR < SGPS_DEM_TINY_FLOAT) ? log(SGPS_DEM_TINY_FLOAT) : log(CoR);
            beta = loge / sqrt(loge * loge + SGPS_PI_SQUARED);

            const float k_n = SGPS_TWO_OVER_THREE * Sn;
            const float gamma_n = SGPS_TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

            force += (k_n * overlapDepth + gamma_n * projection) * B2A;
        }

        // Rolling resistance part
        if (Crr > 0.0) {
            // Figure out if we should apply rolling resistance force
            bool should_add_rolling_resistance = true;
            {
                const float R_eff = sqrtf((ARadius * BRadius) / (ARadius + BRadius));
                const float kn_simple = SGPS_FOUR_OVER_THREE * E * sqrtf(R_eff);
                const float gn_simple = -2.f * sqrtf(SGPS_FIVE_OVER_THREE * mass_eff * E) * beta * powf(R_eff, 0.25f);

                const float d_coeff = gn_simple / (2.f * sqrtf(kn_simple * mass_eff));

                if (d_coeff < 1.0) {
                    float t_collision = SGPS_PI * sqrtf(mass_eff / (kn_simple * (1.f - d_coeff * d_coeff)));
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
                if (v_rot_mag > SGPS_DEM_TINY_FLOAT) {
                    // You should know that Crr * normal_force is the underlying formula, and in our model,
                    // it is a `force' that produces torque only, instead of also cancelling out friction.
                    // Its direction is that it `resists' rotation, see picture in 
                    // https://en.wikipedia.org/wiki/Rolling_resistance.
                    torque_only_force = (v_rot / v_rot_mag) * (Crr * length(force));
                }
            }
        }

        // Tangential force part
        if (mu > 0.0) {
            const float kt = 8. * G * sqrt_Rd;
            const float gt = -SGPS_TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(mass_eff * kt);
            float3 tangent_force = -kt * delta_tan - gt * vrel_tan;
            const float ft = length(tangent_force);
            if (ft > SGPS_DEM_TINY_FLOAT) {
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
        }    
    )V0G0N";

    return model;
}

inline std::string DEM_HERTZIAN_FORCE_MODEL_FRICTIONLESS() {
    std::string model;
    model =
        R"V0G0N(
        // The (total) relative linear velocity of A relative to B
        const float3 velB2A = (ALinVel + rotVelCPA) - (BLinVel + rotVelCPB);
        const float projection = dot(velB2A, B2A);
        // vrel_tan = velB2A - projection * B2A;

        const float mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);
        float sqrt_Rd = sqrt(overlapDepth * (ARadius * BRadius) / (ARadius + BRadius));
        const float Sn = 2. * E * sqrt_Rd;

        const float loge = (CoR < SGPS_DEM_TINY_FLOAT) ? log(SGPS_DEM_TINY_FLOAT) : log(CoR);
        float beta = loge / sqrt(loge * loge + SGPS_PI_SQUARED);

        const float k_n = SGPS_TWO_OVER_THREE * Sn;
        const float gamma_n = SGPS_TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

        // normal force (that A feels)
        // printf("overlapDepth: %f\n", overlapDepth);
        // printf("kn * overlapDepth: %f\n", k_n * overlapDepth);
        // printf("gn * projection: %f\n", gamma_n * projection);
        force += (k_n * overlapDepth + gamma_n * projection) * B2A;
    )V0G0N";

    return model;
}

inline std::string DEM_CLUMP_COMPONENT_ACQUISITION_ALL_JITIFIED() {
    std::string model;
    model =
        R"V0G0N(
        sgps::clumpComponentOffset_t myCompOffset = granData->clumpComponentOffset[sphereID];
        myRelPosX = CDRelPosX[myCompOffset];
        myRelPosY = CDRelPosY[myCompOffset];
        myRelPosZ = CDRelPosZ[myCompOffset];
        myRadius = Radii[myCompOffset];
    )V0G0N";
    return model;
}

inline std::string DEM_CLUMP_COMPONENT_ACQUISITION_PARTIALLY_JITIFIED() {
    std::string model;
    model =
        R"V0G0N(
        sgps::clumpComponentOffset_t myCompOffset = granData->clumpComponentOffset[sphereID];
        if (myCompOffset != sgps::DEM_RESERVED_CLUMP_COMPONENT_OFFSET) {
            myRelPosX = CDRelPosX[myCompOffset];
            myRelPosY = CDRelPosY[myCompOffset];
            myRelPosZ = CDRelPosZ[myCompOffset];
            myRadius = Radii[myCompOffset];
        } else {
            // Look for my components in global memory
            sgps::clumpComponentOffsetExt_t myCompOffsetExt = granData->clumpComponentOffsetExt[sphereID];
            myRelPosX = granData->relPosSphereX[myCompOffsetExt];
            myRelPosY = granData->relPosSphereY[myCompOffsetExt];
            myRelPosZ = granData->relPosSphereZ[myCompOffsetExt];
            myRadius = granData->radiiSphere[myCompOffsetExt];
        }
    )V0G0N";
    return model;
}

}  // namespace sgps

#endif
