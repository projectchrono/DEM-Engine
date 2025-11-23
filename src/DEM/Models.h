//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_MODEL_STASH
#define DEME_MODEL_STASH

#include <cstring>
#include <cmath>
#include <vector>
#include <filesystem>

#include "Defines.h"
#include "Structs.h"
#include "HostSideHelpers.hpp"
#include "../core/utils/RuntimeData.h"
#include "../core/utils/DEMEPaths.h"

namespace deme {

////////////////////////////////////////////////////////////////////////////////
// Contact force model storage files
////////////////////////////////////////////////////////////////////////////////

inline std::string HERTZIAN_FORCE_MODEL() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "FullHertzianForceModel.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The force model file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string HERTZIAN_FORCE_MODEL_FRICTIONLESS() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "FrictionlessHertzianForceModel.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The force model file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string FORCE_REDUCTION_RIGHT_AFTER_CALC_STRAT() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "ForceInKernelReductionStrat.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("A strategy file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string FORCE_INFO_WRITE_BACK_STRAT() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "ContactInfoWriteBack.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("A strategy file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

////////////////////////////////////////////////////////////////////////////////
// Clump template definition and acquisition strategy files
////////////////////////////////////////////////////////////////////////////////

inline std::string CLUMP_COMPONENT_ACQUISITION_ALL_JITIFIED() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "ClumpCompAcqStratAllJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The clump component jitification strategy file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string CLUMP_COMPONENT_ACQUISITION_PARTIALLY_JITIFIED() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "ClumpCompAcqStratPartialJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The clump component jitification strategy file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string CLUMP_COMPONENT_DEFINITIONS_JITIFIED() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "ClumpCompDefJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The jitified clump component array file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string CLUMP_COMPONENT_ACQUISITION_FLATTENED() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "ClumpCompAcqStratAllFlatten.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The clump component loading strategy file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

////////////////////////////////////////////////////////////////////////////////
// Analytical object definition files
////////////////////////////////////////////////////////////////////////////////

inline std::string ANALYTICAL_COMPONENT_DEFINITIONS_JITIFIED() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "AnalyticalCompDefJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The jitified analytical object component array file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

////////////////////////////////////////////////////////////////////////////////
// Mass and MOI definition and acquisition strategy files
////////////////////////////////////////////////////////////////////////////////

inline std::string MASS_DEFINITIONS_JITIFIED() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "MassDefJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The mass property array file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string MOI_DEFINITIONS_JITIFIED() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "MOIDefJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The MOI property array file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string MASS_ACQUISITION_JITIFIED() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "MassAcqStratJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The mass loading strategy array file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string MASS_ACQUISITION_FLATTENED() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "MassAcqStratFlatten.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The mass loading strategy array file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string MOI_ACQUISITION_JITIFIED() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "MOIAcqStratJitify.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The MOI loading strategy array file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string MOI_ACQUISITION_FLATTENED() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "MOIAcqStratFlatten.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The MOI loading strategy array file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

////////////////////////////////////////////////////////////////////////////////
// Data path and other paths
////////////////////////////////////////////////////////////////////////////////

inline std::filesystem::path GET_DATA_PATH() {
    return DEME_data_path;
}

////////////////////////////////////////////////////////////////////////////////
// Integration policies
////////////////////////////////////////////////////////////////////////////////

inline std::string VEL_TO_PASS_ON_FORWARD_EULER() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "IntegrationVelPassOnForwardEuler.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The integration strategy file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string VEL_TO_PASS_ON_EXTENDED_TAYLOR() {
    std::filesystem::path sourcefile = DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" /
                                       "IntegrationVelPassOnExtendedTaylor.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The integration strategy file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

inline std::string VEL_TO_PASS_ON_CENTERED_DIFF() {
    std::filesystem::path sourcefile =
        DEMERuntimeDataHelper::data_path / "kernel" / "DEMCustomizablePolicies" / "IntegrationVelPassOnCenteredDiff.cu";
    if (!std::filesystem::exists(sourcefile)) {
        DEME_ERROR("The integration strategy file %s is not found.", sourcefile.string().c_str());
    }
    return read_file_to_string(sourcefile);
}

////////////////////////////////////////////////////////////////////////////////
// Ingredient definition and acquisition module in DEM force models
////////////////////////////////////////////////////////////////////////////////

inline void add_force_model_ingr(std::unordered_map<std::string, bool>& added_ingredients, const std::string& str) {
    added_ingredients[str] = true;
}
inline void scan_force_model_ingr(std::unordered_map<std::string, bool>& added_ingredients, const std::string& model) {
    if (any_whole_word_match(model, {"ts"})) {
        added_ingredients["ts"] = true;
    }
    if (any_whole_word_match(model, {"time"})) {
        added_ingredients["time"] = true;
    }
    if (any_whole_word_match(model, {"AOwnerFamily"})) {
        added_ingredients["AOwnerFamily"] = true;
    }
    if (any_whole_word_match(model, {"BOwnerFamily"})) {
        added_ingredients["BOwnerFamily"] = true;
    }
    if (any_whole_word_match(model, {"ALinVel", "BLinVel"})) {
        added_ingredients["ALinVel"] = true;
        added_ingredients["BLinVel"] = true;
    }
    if (any_whole_word_match(model, {"ARotVel", "BRotVel"})) {
        added_ingredients["ARotVel"] = true;
        added_ingredients["BRotVel"] = true;
    }
    if (any_whole_word_match(model, {"AOwner", "BOwner"})) {
        added_ingredients["AOwner"] = true;
        added_ingredients["BOwner"] = true;
    }
    if (any_whole_word_match(model, {"AGeo", "BGeo"})) {
        added_ingredients["AGeo"] = true;
        added_ingredients["BGeo"] = true;
    }
    if (any_whole_word_match(model, {"AOwnerMOI", "BOwnerMOI"})) {
        added_ingredients["AOwnerMOI"] = true;
        added_ingredients["BOwnerMOI"] = true;
    }
}
// Sweep through all ingredients...
inline void equip_force_model_ingr_acq(std::string& definition,
                                       std::string& acquisition_A,
                                       std::string& acquisition_B,
                                       std::unordered_map<std::string, bool>& added_ingredients) {
    if (added_ingredients["ts"]) {
        definition += "float ts = simParams->h;\n";
    }
    if (added_ingredients["time"]) {
        definition += "float time = simParams->timeElapsed;\n";
    }
    if (added_ingredients["AOwnerFamily"]) {
        definition += "deme::family_t AOwnerFamily;\n";
        acquisition_A += "AOwnerFamily = granData->familyID[myOwner];\n";
    }
    if (added_ingredients["BOwnerFamily"]) {
        definition += "deme::family_t BOwnerFamily;\n";
        acquisition_B += "BOwnerFamily = granData->familyID[myOwner];\n";
    }
    if (added_ingredients["ALinVel"] || added_ingredients["BLinVel"]) {
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
    if (added_ingredients["ARotVel"] || added_ingredients["BRotVel"]) {
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
    if (added_ingredients["AOwner"] || added_ingredients["BOwner"]) {
        definition += "deme::bodyID_t AOwner, BOwner;\n";
        acquisition_A += "AOwner = myOwner;";
        acquisition_B += "BOwner = myOwner;";
    }
    if (added_ingredients["AGeo"] || added_ingredients["BGeo"]) {
        definition += "deme::bodyID_t AGeo, BGeo;\n";
        acquisition_A += "AGeo = sphereID;";
        // BGeo can be sphere, tri or analytical, but they are all named sphereID in the force kernel.
        acquisition_B += "BGeo = sphereID;";
    }
    if (added_ingredients["AOwnerMOI"] || added_ingredients["BOwnerMOI"]) {
        definition += "float3 AOwnerMOI, BOwnerMOI;\n";
        acquisition_A += R"V0G0N(float3 myMOI;
                                 _moiAcqStrat_;
                                 AOwnerMOI = myMOI;
                        )V0G0N";
        acquisition_B += R"V0G0N(float3 myMOI;
                                _moiAcqStrat_;
                                BOwnerMOI = myMOI;
                        )V0G0N";
    }
}

// Sweep through all ingredients...
inline void equip_owner_wildcards(std::string& definition,
                                  std::string& acquisition_A,
                                  std::string& acquisition_B,
                                  std::string& writeback,
                                  const std::set<std::string>& added_ingredients) {
    unsigned int i = 0;
    for (const auto& name : added_ingredients) {
        // Owner wildcards are given as alias, because unlike contact wildcard which is pair-wise, there is some
        // ambiguity in which contact pair should update a owner property. So, this management is completely given to
        // the user, and they may choose to atomically update them if needed.
        definition += "float* " + name + " = granData->ownerWildcards[" + std::to_string(i) + "];\n";
        // Those _A and _B variances are alias only. Since for owner wildcards, A and B geometries can share.
        definition += "float* " + name + "_A = granData->ownerWildcards[" + std::to_string(i) + "];\n";
        definition += "float* " + name + "_B = granData->ownerWildcards[" + std::to_string(i) + "];\n";
        // Because of using alias, no acquisition or write-back needed
        /*
        acquisition_A += name + "_A = granData->ownerWildcards[" + std::to_string(i) + "][myOwner];\n";
        acquisition_B += name + "_B = granData->ownerWildcards[" + std::to_string(i) + "][myOwner];\n";
        writeback += "granData->ownerWildcards[" + std::to_string(i) + "][AOwner] = " + name + "_A;\n" +
                     "granData->ownerWildcards[" + std::to_string(i) + "][BOwner] = " + name + "_B;\n";
        */
        i++;
    }
}

// Sweep through all ingredients...
inline void equip_geo_wildcards(std::string& definition,
                                std::string& acquisition_A,
                                std::string& acquisition_B_sph,
                                std::string& acquisition_B_tri,
                                std::string& acquisition_B_anal,
                                const std::set<std::string>& added_ingredients) {
    unsigned int i = 0;
    for (const auto& name : added_ingredients) {
        definition += "float* " + name + "_A, *" + name + "_B;\n";
        acquisition_A += name + "_A = granData->sphereWildcards[" + std::to_string(i) + "];\n";
        acquisition_B_sph += name + "_B = granData->sphereWildcards[" + std::to_string(i) + "];\n";
        acquisition_B_tri += name + "_B = granData->triWildcards[" + std::to_string(i) + "];\n";
        acquisition_B_anal += name + "_B = granData->analWildcards[" + std::to_string(i) + "];\n";
        i++;
    }
}

// Massage contact wildcards (by that I mean those contact history arrays)
inline void equip_contact_wildcards(std::string& acquisition,
                                    std::string& write_back,
                                    std::string& destroy_record,
                                    const std::set<std::string>& names) {
    unsigned int i = 0;
    for (const auto& name : names) {
        // Rigth now, supports float arrays only...
        // Getting it from global mem
        acquisition += "float " + name + " = granData->contactWildcards[" + std::to_string(i) + "][myContactID];\n";
        // Write it back to global mem
        write_back += "granData->contactWildcards[" + std::to_string(i) + "][myContactID] = " + name + ";\n";
        // Destroy it (set to 0) if it is a fake contact
        destroy_record += name + " = 0;\n";
        i++;
    }
}

}  // namespace deme

#endif
