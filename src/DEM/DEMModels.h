//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#ifndef SGPS_DEM_MODEL_STASH
#define SGPS_DEM_MODEL_STASH

#include <cstring>
#include <cmath>
#include <vector>
#include <filesystem>

#include <DEM/DEMDefines.h>
#include <DEM/DEMStructs.h>
#include <DEM/HostSideHelpers.hpp>

namespace sgps {

// TODO: Mark those user referrables
inline std::string DEM_HERTZIAN_FORCE_MODEL() {
    // We now do not use R"V0G0N string to store the force mode; we instead use a file to store it for better
    // readability std::string model = R"V0G0N( ... )V0G0N";

    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "ForceCalcStrategy.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The force model file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
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
        float beta = loge / sqrt(loge * loge + sgps::PI_SQUARED);

        const float k_n = sgps::TWO_OVER_THREE * Sn;
        const float gamma_n = sgps::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

        // normal force (that A feels)
        // printf("overlapDepth: %f\n", overlapDepth);
        // printf("kn * overlapDepth: %f\n", k_n * overlapDepth);
        // printf("gn * projection: %f\n", gamma_n * projection);
        force += (k_n * overlapDepth + gamma_n * projection) * B2A;
    )V0G0N";

    return model;
}

inline std::string DEM_CLUMP_COMPONENT_ACQUISITION_ALL_JITIFIED() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "ClumpCompAcqStratAllJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The clump component jitification strategy file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string DEM_CLUMP_COMPONENT_ACQUISITION_PARTIALLY_JITIFIED() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "ClumpCompAcqStratPartialJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The clump component jitification strategy file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

}  // namespace sgps

#endif
