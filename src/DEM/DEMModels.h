//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

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

////////////////////////////////////////////////////////////////////////////////
// Contact force model storage files
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
// Clump template definition and acquisition strategy files
////////////////////////////////////////////////////////////////////////////////

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

inline std::string DEM_CLUMP_COMPONENT_DEFINITIONS_JITIFIED() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "ClumpCompDefJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The jitified clump component array file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string DEM_CLUMP_COMPONENT_ACQUISITION_FLATTENED() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "ClumpCompAcqStratAllFlatten.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The clump component loading strategy file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

////////////////////////////////////////////////////////////////////////////////
// Analytical object definition files
////////////////////////////////////////////////////////////////////////////////

inline std::string DEM_ANALYTICAL_COMPONENT_DEFINITIONS_JITIFIED() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "AnalyticalCompDefJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The jitified analytical object component array file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

////////////////////////////////////////////////////////////////////////////////
// Material definition files
////////////////////////////////////////////////////////////////////////////////

inline std::string DEM_MATERIAL_DEFINITIONS_JITIFIED() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "MaterialDefJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The jitified material proxy array file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

////////////////////////////////////////////////////////////////////////////////
// Mass and MOI definition and acquisition strategy files
////////////////////////////////////////////////////////////////////////////////

inline std::string DEM_MASS_DEFINITIONS_JITIFIED() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "MassDefJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The mass property array file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string DEM_MOI_DEFINITIONS_JITIFIED() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "MOIDefJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The MOI property array file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string DEM_MASS_ACQUISITION_JITIFIED() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "MassAcqStratJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The mass loading strategy array file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string DEM_MASS_ACQUISITION_FLATTENED() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "MassAcqStratFlatten.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The mass loading strategy array file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string DEM_MOI_ACQUISITION_JITIFIED() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "MOIAcqStratJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The MOI loading strategy array file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string DEM_MOI_ACQUISITION_FLATTENED() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "MOIAcqStratFlatten.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The MOI loading strategy array file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

}  // namespace sgps

#endif
