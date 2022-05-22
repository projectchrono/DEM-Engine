//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#ifndef SGPS_DEM_MODEL_STASH
#define SGPS_DEM_MODEL_STASH

#pragma once
#include <cstring>
#include <cmath>
#include <vector>

#include <DEM/DEMDefines.h>
#include <DEM/DEMStructs.h>
#include <DEM/HostSideHelpers.cpp>

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
            const float projection = dot(velB2A, B2A);
            vrel_tan = velB2A - projection * B2A;

            mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);
            sqrt_Rd = sqrt(overlapDepth * (ARadius * BRadius) / (ARadius + BRadius));
            const float Sn = 2. * E * sqrt_Rd;

            const float loge = (CoR < SGPS_DEM_TINY_FLOAT) ? log(SGPS_DEM_TINY_FLOAT) : log(CoR);
            beta = loge / sqrt(loge * loge + SGPS_PI_SQUARED);

            const float k_n = SGPS_TWO_OVER_THREE * Sn;
            const float gamma_n = SGPS_TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

            force = (k_n * overlapDepth + gamma_n * projection) * B2A;
        }

        // Tangential force part
        {
            {
                delta_tan += h * vrel_tan;
                const float disp_proj = dot(delta_tan, B2A);
                delta_tan -= disp_proj * B2A;
            }
            const float kt = 8. * G * sqrt_Rd;
            const float gt = -SGPS_TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(mass_eff * kt);
            float3 tangent_force = -kt * delta_tan - gt * vrel_tan;
            const float ft = length(tangent_force);
            if (ft > SGPS_DEM_TINY_FLOAT) {
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

inline std::string DEM_HERTZIAN_FORCE_MODEL_HISTORYLESS() {
    std::string model;
    model =
        R"V0G0N(
        // normal component of relative velocity
        const float projection = dot(velB2A, B2A);
        float3 vrel_tan = velB2A - projection * B2A;  // May want to report this for tangent force calculation

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
        force = (k_n * overlapDepth + gamma_n * projection) * B2A;
    )V0G0N";

    return model;
}

}  // namespace sgps

#endif
