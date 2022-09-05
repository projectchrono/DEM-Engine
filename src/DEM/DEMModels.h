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
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "FullHertzianForceModel.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The force model file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string DEM_HERTZIAN_FORCE_MODEL_FRICTIONLESS() {
    std::filesystem::path sourcefile = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel" /
                                       "DEMCustomizablePolicies" / "FrictionlessHertzianForceModel.cu";
    if (!std::filesystem::exists(sourcefile)) {
        SGPS_DEM_ERROR("The force model file %s is not found.", sourcefile.c_str());
    }
    return read_file_to_string(sourcefile);
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

////////////////////////////////////////////////////////////////////////////////
// Ingredient definition and acquisition module in DEM force models
////////////////////////////////////////////////////////////////////////////////

// Sweep through all ingredients...
inline void equip_force_model_ingr_acq(std::string& definition,
                                       std::string& acquisition_A,
                                       std::string& acquisition_B,
                                       const std::string& model) {
    if (any_whole_word_match(model, {"ts"})) {
        definition += "float ts = simParams->h;\n";
    }
    if (any_whole_word_match(model, {"AOwnerFamily", "BOwnerFamily"})) {
        definition += "sgps::family_t AOwnerFamily, BOwnerFamily;\n";
        acquisition_A += "AOwnerFamily = granData->familyID[myOwner];\n";
        acquisition_B += "BOwnerFamily = granData->familyID[myOwner];\n";
    }
    if (any_whole_word_match(model, {"ALinVel", "BLinVel"})) {
        definition += "float3 ALinVel, BLinVel;\n";
        acquisition_A += R"V0G0N(ALinVel.x = granData->vX[myOwner];
                                 ALinVel.y = granData->vY[myOwner];
                                 ALinVel.z = granData->vZ[myOwner];
                         )V0G0N";
        acquisition_B += R"V0G0N(BLinVel.x = granData->vX[myOwner];
                                 BLinVel.y = granData->vY[myOwner];
                                 BLinVel.z = granData->vZ[myOwner];
                         )V0G0N";
    }
    if (any_whole_word_match(model, {"ARotVel", "BRotVel"})) {
        definition += "float3 ARotVel, BRotVel;\n";
        acquisition_A += R"V0G0N(ARotVel.x = granData->omgBarX[myOwner];
                                 ARotVel.y = granData->omgBarY[myOwner];
                                 ARotVel.z = granData->omgBarZ[myOwner];
                         )V0G0N";
        acquisition_B += R"V0G0N(BRotVel.x = granData->omgBarX[myOwner];
                                 BRotVel.y = granData->omgBarY[myOwner];
                                 BRotVel.z = granData->omgBarZ[myOwner];
                         )V0G0N";
    }
}

}  // namespace sgps

#endif
