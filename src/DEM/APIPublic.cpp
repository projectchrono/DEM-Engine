//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <core/ApiVersion.h>
#include <DEM/API.h>
#include <DEM/Defines.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/AuxClasses.h>

#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstring>
#include <limits>
#include <algorithm>

namespace deme {

DEMSolver::DEMSolver(unsigned int nGPUs) {
    dTkT_InteractionManager = new ThreadManager();
    kTMain_InteractionManager = new WorkerReportChannel();
    dTMain_InteractionManager = new WorkerReportChannel();

    // 2 means 2 threads (nGPUs is currently not used)
    dTkT_GpuManager = new GpuManager(2);

    dT = new DEMDynamicThread(dTMain_InteractionManager, dTkT_InteractionManager, dTkT_GpuManager);
    kT = new DEMKinematicThread(kTMain_InteractionManager, dTkT_InteractionManager, dTkT_GpuManager);

    // Make friends
    dT->kT = kT;
    kT->dT = dT;
}

DEMSolver::~DEMSolver() {
    delete kT;
    delete dT;
    delete kTMain_InteractionManager;
    delete dTMain_InteractionManager;
    delete dTkT_InteractionManager;
    delete dTkT_GpuManager;
}

float3 DEMSolver::GetOwnerPosition(bodyID_t ownerID) const {
    return dT->getOwnerPos(ownerID);
}
float3 DEMSolver::GetOwnerAngVel(bodyID_t ownerID) const {
    return dT->getOwnerAngVel(ownerID);
}
float3 DEMSolver::GetOwnerVelocity(bodyID_t ownerID) const {
    return dT->getOwnerVel(ownerID);
}
float4 DEMSolver::GetOwnerOriQ(bodyID_t ownerID) const {
    return dT->getOwnerOriQ(ownerID);
}
float3 DEMSolver::GetOwnerAcc(bodyID_t ownerID) const {
    return dT->getOwnerAcc(ownerID);
}
float3 DEMSolver::GetOwnerAngAcc(bodyID_t ownerID) const {
    return dT->getOwnerAngAcc(ownerID);
}

void DEMSolver::SetOwnerPosition(bodyID_t ownerID, float3 pos) {
    dT->setOwnerPos(ownerID, pos);
}
void DEMSolver::SetOwnerAngVel(bodyID_t ownerID, float3 angVel) {
    dT->setOwnerAngVel(ownerID, angVel);
}
void DEMSolver::SetOwnerVelocity(bodyID_t ownerID, float3 vel) {
    dT->setOwnerVel(ownerID, vel);
}
void DEMSolver::SetOwnerOriQ(bodyID_t ownerID, float4 oriQ) {
    dT->setOwnerOriQ(ownerID, oriQ);
}
void DEMSolver::SetTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& triangles, bool overwrite) {
    dT->setTriNodeRelPos(start, triangles, overwrite);
    kT->setTriNodeRelPos(start, triangles, overwrite);
}

void DEMSolver::InstructBoxDomainDimension(float x, float y, float z, SPATIAL_DIR dir_exact) {
    m_user_boxSize.x = x;
    m_user_boxSize.y = y;
    m_user_boxSize.z = z;
    m_box_dir_length_is_exact = dir_exact;
    explicit_nv_override = false;
}

void DEMSolver::InstructBoxDomainNumVoxel(unsigned char x, unsigned char y, unsigned char z, float len_unit) {
    if (x + y + z != sizeof(voxelID_t) * DEME_BITS_PER_BYTE) {
        DEME_ERROR("Please give voxel numbers (as powers of 2) along each direction such that they add up to %zu.",
                   sizeof(voxelID_t) * DEME_BITS_PER_BYTE);
    }
    l = len_unit;
    nvXp2 = x;
    nvYp2 = y;
    nvZp2 = z;

    // Calculating `world' size by the input nvXp2 and l
    m_voxelSize = (double)((size_t)1 << VOXEL_RES_POWER2) * (double)l;
    m_boxX = m_voxelSize * (double)((size_t)1 << x);
    m_boxY = m_voxelSize * (double)((size_t)1 << y);
    m_boxZ = m_voxelSize * (double)((size_t)1 << z);
    // In this debug case, user domain size is the same as actual domain size
    m_user_boxSize.x = m_boxX;
    m_user_boxSize.y = m_boxY;
    m_user_boxSize.z = m_boxZ;
    explicit_nv_override = true;
}

std::shared_ptr<DEMForceModel> DEMSolver::DefineContactForceModel(const std::string& model) {
    DEMForceModel force_model;  // Custom
    force_model.DefineCustomModel(model);
    m_force_model = std::make_shared<DEMForceModel>(std::move(force_model));
    return m_force_model;
}

std::shared_ptr<DEMForceModel> DEMSolver::ReadContactForceModel(const std::string& filename) {
    DEMForceModel force_model;  // Custom
    std::filesystem::path sourcefile = USER_SCRIPT_PATH / filename;
    if (!force_model.ReadCustomModelFile(sourcefile)) {
        // If not in that folder, then maybe the user meant an absolute path
        if (!force_model.ReadCustomModelFile(std::filesystem::path(filename)))
            DEME_ERROR("The force model file %s is not found.", filename.c_str());
    }
    m_force_model = std::make_shared<DEMForceModel>(std::move(force_model));
    return m_force_model;
}

std::shared_ptr<DEMForceModel> DEMSolver::UseFrictionalHertzianModel() {
    m_force_model->SetForceModelType(FORCE_MODEL::HERTZIAN);
    return m_force_model;
}

std::shared_ptr<DEMForceModel> DEMSolver::UseFrictionlessHertzianModel() {
    m_force_model->SetForceModelType(FORCE_MODEL::HERTZIAN_FRICTIONLESS);
    return m_force_model;
}

void DEMSolver::SetFamilyFixed(unsigned int ID) {
    assertSysNotInit("SetFamilyFixed");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;
    preInfo.linVelX = "0";
    preInfo.linVelY = "0";
    preInfo.linVelZ = "0";
    preInfo.rotVelX = "0";
    preInfo.rotVelY = "0";
    preInfo.rotVelZ = "0";
    preInfo.linVelPrescribed = true;
    preInfo.rotVelPrescribed = true;
    preInfo.rotPosPrescribed = true;
    preInfo.linPosPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::ChangeFamilyWhen(unsigned int ID_from, unsigned int ID_to, const std::string& condition) {
    assertSysNotInit("ChangeFamilyWhen");
    if (ID_from > std::numeric_limits<family_t>::max() || ID_to > std::numeric_limits<family_t>::max()) {
        DEME_ERROR(
            "You instructed family number %u should change to %u, but family number should not be larger than %u.",
            ID_from, ID_to, std::numeric_limits<family_t>::max());
    }

    // If one such user call is made, then the solver needs to prepare for per-step family number-changing sweeps
    famnum_can_change_conditionally = true;
    familyPair_t a_pair;
    a_pair.ID1 = ID_from;
    a_pair.ID2 = ID_to;

    m_family_change_pairs.push_back(a_pair);
    m_family_change_conditions.push_back(condition);
}

void DEMSolver::ChangeFamily(unsigned int ID_from, unsigned int ID_to) {
    assertSysInit("ChangeFamily");
    // if (!check_exist(unique_user_families, ID_from)) {
    //     DEME_WARNING(
    //         "Family %u (from-family) has no prior reference before a ChangeFamily call, therefore no work is done.",
    //         ID_from);
    //     return;
    // }
    // if (!check_exist(unique_user_families, ID_to)) {
    //     DEME_ERROR(
    //         "Family %u (to-family) has no prior reference before a ChangeFamily call.\nThis is currently not allowed,
    //         " "as creating a family in mid-simulation usually requires re-jitification anyway.\nIf a family with no "
    //         "special prescription or masking is needed, then you can forward-declare this family via InsertFamily "
    //         "before initialization.",
    //         ID_to);
    //     return;
    // }
    if (ID_from > std::numeric_limits<family_t>::max() || ID_to > std::numeric_limits<family_t>::max()) {
        DEME_ERROR(
            "You instructed family number %u should change to %u, but family number should not be larger than %u.",
            ID_from, ID_to, std::numeric_limits<family_t>::max());
    }

    dT->changeFamily(ID_from, ID_to);
    kT->changeFamily(ID_from, ID_to);
}

void DEMSolver::SetFamilyPrescribedLinVel(unsigned int ID,
                                          const std::string& velX,
                                          const std::string& velY,
                                          const std::string& velZ,
                                          bool dictate) {
    assertSysNotInit("SetFamilyPrescribedLinVel");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;
    preInfo.linVelX = velX;
    preInfo.linVelY = velY;
    preInfo.linVelZ = velZ;

    preInfo.linVelPrescribed = dictate;
    preInfo.rotVelPrescribed = dictate;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::SetFamilyPrescribedLinVel(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedLinVel");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.linVelPrescribed = true;
    preInfo.rotVelPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::SetFamilyPrescribedAngVel(unsigned int ID,
                                          const std::string& velX,
                                          const std::string& velY,
                                          const std::string& velZ,
                                          bool dictate) {
    assertSysNotInit("SetFamilyPrescribedAngVel");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;
    preInfo.rotVelX = velX;
    preInfo.rotVelY = velY;
    preInfo.rotVelZ = velZ;

    preInfo.linVelPrescribed = dictate;
    preInfo.rotVelPrescribed = dictate;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::SetFamilyPrescribedAngVel(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedAngVel");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.linVelPrescribed = true;
    preInfo.rotVelPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::SetFamilyPrescribedPosition(unsigned int ID,
                                            const std::string& X,
                                            const std::string& Y,
                                            const std::string& Z) {
    assertSysNotInit("SetFamilyPrescribedPosition");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;
    preInfo.linPosX = X;
    preInfo.linPosY = Y;
    preInfo.linPosZ = Z;
    // Both rot and lin pos are fixed. Use other methods if this is not intended.
    preInfo.rotPosPrescribed = true;
    preInfo.linPosPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::SetFamilyPrescribedPosition(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedPosition");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;
    // Both rot and lin pos are fixed.
    preInfo.rotPosPrescribed = true;
    preInfo.linPosPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::SetFamilyPrescribedQuaternion(unsigned int ID, const std::string& q_formula) {
    assertSysNotInit("SetFamilyPrescribedQuaternion");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
}

void DEMSolver::SetOwnerWildcardValue(const std::string& name, float val) {
    assertSysInit("SetOwnerWildcardValue");
    if (m_owner_wc_num.find(name) == m_owner_wc_num.end()) {
        DEME_ERROR(
            "No owner wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerOwnerWildcards first.",
            name.c_str());
    }
    dT->setOwnerWildcardValue(m_owner_wc_num.at(name), val);
}
void DEMSolver::SetFamilyOwnerWildcardValue(unsigned int N, const std::string& name, float val) {
    assertSysInit("SetFamilyOwnerWildcardValue");
    if (m_owner_wc_num.find(name) == m_owner_wc_num.end()) {
        DEME_ERROR(
            "No owner wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerOwnerWildcards first.",
            name.c_str());
    }
    dT->setFamilyOwnerWildcardValue(N, m_owner_wc_num.at(name), val);
}

void DEMSolver::DisableFamilyOutput(unsigned int ID) {
    assertSysNotInit("DisableFamilyOutput");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You tried to disable output for family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    m_no_output_families.insert(ID);
}

std::shared_ptr<DEMMaterial> DEMSolver::LoadMaterial(const std::unordered_map<std::string, float>& mat_prop) {
    // Register material property names that appeared in this call
    for (const auto& a_pair : mat_prop) {
        m_material_prop_names.insert(a_pair.first);
        if (a_pair.first == "CoR") {
            if (a_pair.second < DEME_TINY_FLOAT)
                DEME_WARNING("Material type %u is set to have 0 restitution. Please make sure this is intentional.",
                             m_loaded_materials.size());
            if (a_pair.second > 1.f)
                DEME_WARNING(
                    "Material type %u is set to have a restitution coefficient larger than 1. This is typically not "
                    "physical and should destabilize the simulation.",
                    m_loaded_materials.size());
        }
        // Material names cannot have spaces in them
        if (match_pattern(a_pair.first, " ")) {
            DEME_ERROR("Material property %s is not valid: no spaces allowed in its name.", a_pair.first.c_str());
        }
    }
    DEMMaterial a_material(mat_prop);
    std::shared_ptr<DEMMaterial> ptr = std::make_shared<DEMMaterial>(std::move(a_material));
    ptr->load_order = m_loaded_materials.size();
    m_loaded_materials.push_back(ptr);
    nMaterialsLoad++;
    return m_loaded_materials.back();
}

std::shared_ptr<DEMClumpTemplate> DEMSolver::LoadClumpType(DEMClumpTemplate& clump) {
    if (clump.nComp != clump.radii.size() || clump.nComp != clump.relPos.size() ||
        clump.nComp != clump.materials.size()) {
        DEME_ERROR(
            "Radii, relative positions and material arrays defining a clump topology, must all have the same length "
            "(%zu, as indicated by nComp).\nHowever it seems that their lengths are %zu, %zu, %zu, respectively.\nIf "
            "you constructed a DEMClumpTemplate struct yourself, you may need to carefully check if their lengths "
            "agree with nComp.",
            clump.nComp, clump.radii.size(), clump.relPos.size(), clump.materials.size());
    }
    if (clump.mass < DEME_TINY_FLOAT || length(clump.MOI) < DEME_TINY_FLOAT) {
        DEME_WARNING(
            "A type of clump is instructed to have 0 mass or moment of inertia. This will most likely destabilize the "
            "simulation.");
    }

    // Print the mark to this clump template
    unsigned int offset = m_templates.size();
    clump.mark = offset;

    // Give it a default name
    if (clump.m_name == "NULL") {
        char my_name[200];
        sprintf(my_name, "%04d", nClumpTemplateLoad);
        clump.AssignName(std::string(my_name));
    }

    std::shared_ptr<DEMClumpTemplate> ptr = std::make_shared<DEMClumpTemplate>(std::move(clump));
    m_templates.push_back(ptr);
    nClumpTemplateLoad++;
    return m_templates.back();
}

std::shared_ptr<DEMClumpTemplate> DEMSolver::LoadClumpType(float mass,
                                                           float3 moi,
                                                           const std::string filename,
                                                           const std::shared_ptr<DEMMaterial>& sp_material) {
    DEMClumpTemplate clump;
    clump.mass = mass;
    clump.MOI = moi;
    clump.ReadComponentFromFile(filename);
    std::vector<std::shared_ptr<DEMMaterial>> sp_materials(clump.nComp, sp_material);
    clump.materials = sp_materials;
    return LoadClumpType(clump);
}

std::shared_ptr<DEMClumpTemplate> DEMSolver::LoadClumpType(
    float mass,
    float3 moi,
    const std::string filename,
    const std::vector<std::shared_ptr<DEMMaterial>>& sp_materials) {
    DEMClumpTemplate clump;
    clump.mass = mass;
    clump.MOI = moi;
    clump.ReadComponentFromFile(filename);
    clump.materials = sp_materials;
    return LoadClumpType(clump);
}

std::shared_ptr<DEMClumpTemplate> DEMSolver::LoadClumpType(
    float mass,
    float3 moi,
    const std::vector<float>& sp_radii,
    const std::vector<float3>& sp_locations_xyz,
    const std::vector<std::shared_ptr<DEMMaterial>>& sp_materials) {
    DEMClumpTemplate clump;
    clump.mass = mass;
    clump.MOI = moi;
    clump.radii = sp_radii;
    clump.relPos = sp_locations_xyz;
    clump.materials = sp_materials;
    clump.nComp = sp_radii.size();
    return LoadClumpType(clump);
}

std::shared_ptr<DEMClumpTemplate> DEMSolver::LoadClumpType(float mass,
                                                           float3 moi,
                                                           const std::vector<float>& sp_radii,
                                                           const std::vector<float3>& sp_locations_xyz,
                                                           const std::shared_ptr<DEMMaterial>& sp_material) {
    unsigned int num_comp = sp_radii.size();
    std::vector<std::shared_ptr<DEMMaterial>> sp_materials(num_comp, sp_material);
    return LoadClumpType(mass, moi, sp_radii, sp_locations_xyz, sp_materials);
}

std::shared_ptr<DEMClumpTemplate> DEMSolver::LoadSphereType(float mass,
                                                            float radius,
                                                            const std::shared_ptr<DEMMaterial>& material) {
    float3 I = make_float3(2.0 / 5.0 * mass * radius * radius);
    float3 pos = make_float3(0);
    return LoadClumpType(mass, I, std::vector<float>(1, radius), std::vector<float3>(1, pos),
                         std::vector<std::shared_ptr<DEMMaterial>>(1, material));
}

std::shared_ptr<DEMExternObj> DEMSolver::AddExternalObject() {
    DEMExternObj an_obj;
    std::shared_ptr<DEMExternObj> ptr = std::make_shared<DEMExternObj>(std::move(an_obj));
    // load_order should beits position in the cache array, not nExtObjLoad
    ptr->load_order = cached_extern_objs.size();
    // But we still need to record a ext obj loaded
    nExtObjLoad++;
    cached_extern_objs.push_back(ptr);
    return cached_extern_objs.back();
}

std::shared_ptr<DEMExternObj> DEMSolver::AddBCPlane(const float3 pos,
                                                    const float3 normal,
                                                    const std::shared_ptr<DEMMaterial>& material) {
    std::shared_ptr<DEMExternObj> ptr = AddExternalObject();
    // TODO: make the owner of this BC to have the same CoM as this BC
    ptr->AddPlane(pos, normal, material);
    return ptr;
}

void DEMSolver::DisableContactBetweenFamilies(unsigned int ID1, unsigned int ID2) {
    if (ID1 > std::numeric_limits<family_t>::max() || ID2 > std::numeric_limits<family_t>::max()) {
        DEME_ERROR(
            "You tried to disable contact between family number %u and %u, but family number should not be larger than "
            "%u.",
            ID1, ID2, std::numeric_limits<family_t>::max());
    }
    if (!sys_initialized) {
        familyPair_t a_pair;
        a_pair.ID1 = ID1;
        a_pair.ID2 = ID2;
        m_input_no_contact_pairs.push_back(a_pair);
    } else {
        // If initialized, directly pass this info to workers
        unsigned int posInMat = locateMaskPair<unsigned int>(ID1, ID2);
        kT->familyMaskMatrix.at(posInMat) = PREVENT_CONTACT;
        dT->familyMaskMatrix.at(posInMat) = PREVENT_CONTACT;
    }
}

void DEMSolver::EnableContactBetweenFamilies(unsigned int ID1, unsigned int ID2) {
    if (ID1 > std::numeric_limits<family_t>::max() || ID2 > std::numeric_limits<family_t>::max()) {
        DEME_ERROR(
            "You tried to disable contact between family number %u and %u, but family number should not be larger than "
            "%u.",
            ID1, ID2, std::numeric_limits<family_t>::max());
    }
    if (!sys_initialized) {
        DEME_ERROR(
            "There is no need to call EnableContactBetweenFamilies before system initialization.\nAll families have "
            "contacts with each other by default. Just do not disable them if you need that contact.");
    } else {
        // If initialized, directly pass this info to workers
        unsigned int posInMat = locateMaskPair<unsigned int>(ID1, ID2);
        kT->familyMaskMatrix.at(posInMat) = DONT_PREVENT_CONTACT;
        dT->familyMaskMatrix.at(posInMat) = DONT_PREVENT_CONTACT;
    }
}

void DEMSolver::ClearCache() {
    deallocate_array(cached_input_clump_batches);
    deallocate_array(cached_extern_objs);
    deallocate_array(cached_mesh_objs);

    // m_input_no_contact_pairs can be removed, if the system is initialized. After initialization, family mask can be
    // directly transferred to workers on user call.
    deallocate_array(m_input_no_contact_pairs);

    // Should not remove tracked objs: we don't want to break trackers when new entities are loaded
    // deallocate_array(m_tracked_objs);

    // Rigth now, there is no way to re-define the following arrays without re-starting the simulation
    // m_loaded_materials;
    // m_templates;
    // m_input_family_prescription;
    // m_no_output_families;
    // m_family_change_pairs;
    // m_family_change_conditions;
}

float DEMSolver::GetTotalKineticEnergy() const {
    if (nOwnerBodies == 0) {
        return 0.0;
    }
    return dT->getKineticEnergy();
}

std::shared_ptr<DEMClumpBatch> DEMSolver::AddClumps(DEMClumpBatch& input_batch) {
    // load_order should beits position in the cache array, not nBatchClumpsLoad
    input_batch.load_order = cached_input_clump_batches.size();
    // But we still need to record a batch loaded
    nBatchClumpsLoad++;
    cached_input_clump_batches.push_back(std::make_shared<DEMClumpBatch>(std::move(input_batch)));
    return cached_input_clump_batches.back();
}

std::shared_ptr<DEMClumpBatch> DEMSolver::AddClumps(const std::vector<std::shared_ptr<DEMClumpTemplate>>& input_types,
                                                    const std::vector<float3>& input_xyz) {
    if (input_types.size() != input_xyz.size()) {
        DEME_ERROR("Arrays in the call AddClumps must all have the same length.");
    }
    size_t nClumps = input_types.size();
    // We did not create defaults for families, and if the user did not specify families then they will be added at
    // initialization, and a warning will be given

    DEMClumpBatch a_batch(nClumps);
    a_batch.SetTypes(input_types);
    a_batch.SetPos(input_xyz);
    return AddClumps(a_batch);
}

std::shared_ptr<DEMMeshConnected> DEMSolver::AddWavefrontMeshObject(DEMMeshConnected& mesh) {
    if (mesh.GetNumTriangles() == 0) {
        DEME_WARNING("It seems that a mesh contains 0 triangle facet at the time it is loaded.");
    }
    // load_order should be its position in the cache array, not nTriObjLoad
    mesh.load_order = cached_mesh_objs.size();

    // But we still need to record a tri-mesh loaded
    nTriObjLoad++;

    cached_mesh_objs.push_back(std::make_shared<DEMMeshConnected>(std::move(mesh)));
    return cached_mesh_objs.back();
}

std::shared_ptr<DEMMeshConnected> DEMSolver::AddWavefrontMeshObject(const std::string& filename,
                                                                    const std::shared_ptr<DEMMaterial>& mat,
                                                                    bool load_normals,
                                                                    bool load_uv) {
    DEMMeshConnected mesh;
    bool flag = mesh.LoadWavefrontMesh(filename, load_normals, load_uv);
    if (!flag) {
        DEME_ERROR("Failed to load in mesh file %s.", filename.c_str());
    }
    mesh.SetMaterial(mat);
    return AddWavefrontMeshObject(mesh);
}

std::shared_ptr<DEMMeshConnected> DEMSolver::AddWavefrontMeshObject(const std::string& filename,
                                                                    bool load_normals,
                                                                    bool load_uv) {
    DEMMeshConnected mesh;
    bool flag = mesh.LoadWavefrontMesh(filename, load_normals, load_uv);
    if (!flag) {
        DEME_ERROR("Failed to load in mesh file %s.", filename.c_str());
    }
    return AddWavefrontMeshObject(mesh);
}

std::shared_ptr<DEMTracker> DEMSolver::Track(std::shared_ptr<DEMExternObj>& obj) {
    // Create a middle man: DEMTrackedObj. The reason we use it is because a simple struct should be used to transfer to
    // dT for owner-number processing. If we cut the middle man and use things such as DEMExtObj, there will not be a
    // universal treatment that dT can apply, besides we may have some include-related issues.
    DEMTrackedObj tracked_obj;
    tracked_obj.load_order = obj->load_order;
    tracked_obj.type = OWNER_TYPE::ANALYTICAL;
    m_tracked_objs.push_back(std::make_shared<DEMTrackedObj>(std::move(tracked_obj)));

    // Create a Tracker for this tracked object
    DEMTracker tracker(this);
    tracker.obj = m_tracked_objs.back();
    return std::make_shared<DEMTracker>(std::move(tracker));
}

std::shared_ptr<DEMTracker> DEMSolver::Track(std::shared_ptr<DEMMeshConnected>& obj) {
    DEMTrackedObj tracked_obj;
    tracked_obj.load_order = obj->load_order;
    tracked_obj.type = OWNER_TYPE::MESH;
    m_tracked_objs.push_back(std::make_shared<DEMTrackedObj>(std::move(tracked_obj)));

    // Create a Tracker for this tracked object
    DEMTracker tracker(this);
    tracker.obj = m_tracked_objs.back();
    return std::make_shared<DEMTracker>(std::move(tracker));
}

std::shared_ptr<DEMTracker> DEMSolver::Track(std::shared_ptr<DEMClumpBatch>& obj) {
    DEMTrackedObj tracked_obj;
    tracked_obj.load_order = obj->load_order;
    tracked_obj.type = OWNER_TYPE::CLUMP;
    m_tracked_objs.push_back(std::make_shared<DEMTrackedObj>(std::move(tracked_obj)));

    // Create a Tracker for this tracked object
    DEMTracker tracker(this);
    tracker.obj = m_tracked_objs.back();
    return std::make_shared<DEMTracker>(std::move(tracker));
}

std::shared_ptr<DEMInspector> DEMSolver::CreateInspector(const std::string& quantity) {
    DEMInspector insp(this, quantity);
    m_inspectors.push_back(std::make_shared<DEMInspector>(std::move(insp)));
    return m_inspectors.back();
}

std::shared_ptr<DEMInspector> DEMSolver::CreateInspector(const std::string& quantity, const std::string& region) {
    DEMInspector insp(this, quantity, region);
    m_inspectors.push_back(std::make_shared<DEMInspector>(std::move(insp)));
    return m_inspectors.back();
}

void DEMSolver::WriteSphereFile(const std::string& outfilename) const {
    switch (m_out_format) {
        case (OUTPUT_FORMAT::CHPF): {
            std::ofstream ptFile(outfilename, std::ios::out | std::ios::binary);
            dT->writeSpheresAsChpf(ptFile);
            break;
        }
        case (OUTPUT_FORMAT::CSV): {
            std::ofstream ptFile(outfilename, std::ios::out);
            dT->writeSpheresAsCsv(ptFile);
            break;
        }
        case (OUTPUT_FORMAT::BINARY): {
            std::ofstream ptFile(outfilename, std::ios::out | std::ios::binary);
            //// TODO: Implement it
            break;
        }
        default:
            DEME_ERROR("Sphere output file format is unknown. Please set it via SetOutputFormat.");
    }
}

void DEMSolver::WriteClumpFile(const std::string& outfilename, unsigned int accuracy) const {
    switch (m_out_format) {
        case (OUTPUT_FORMAT::CHPF): {
            std::ofstream ptFile(outfilename, std::ios::out | std::ios::binary);
            dT->writeClumpsAsChpf(ptFile, accuracy);
            break;
        }
        case (OUTPUT_FORMAT::CSV): {
            std::ofstream ptFile(outfilename, std::ios::out);
            dT->writeClumpsAsCsv(ptFile, accuracy);
            break;
        }
        case (OUTPUT_FORMAT::BINARY): {
            std::ofstream ptFile(outfilename, std::ios::out | std::ios::binary);
            //// TODO: Implement it
            break;
        }
        default:
            DEME_ERROR("Clump output file format is unknown. Please set it via SetOutputFormat.");
    }
}

void DEMSolver::WriteContactFile(const std::string& outfilename, float force_thres) const {
    if (no_recording_contact_forces) {
        DEME_WARNING(
            "The solver is instructed to not record contact force info, so no work is done in a WriteContactFile "
            "call.");
        return;
    }
    switch (m_cnt_out_format) {
        case (OUTPUT_FORMAT::CSV): {
            std::ofstream ptFile(outfilename, std::ios::out);
            dT->writeContactsAsCsv(ptFile, force_thres);
            break;
        }
        default:
            DEME_ERROR(
                "Contact pair output file format is unknown or not implemented. Please re-set it via SetOutputFormat.");
    }
}

void DEMSolver::WriteMeshFile(const std::string& outfilename) const {
    switch (m_mesh_out_format) {
        case (MESH_FORMAT::VTK): {
            std::ofstream ptFile(outfilename, std::ios::out);
            dT->writeMeshesAsVtk(ptFile);
            break;
        }
        default:
            DEME_ERROR(
                "Mesh output file format is unknown or not implemented. Please re-set it via SetMeshOutputFormat.");
    }
}

// The method should be called after user inputs are in place, and before starting the simulation. It figures out a part
// of the required simulation information such as the scale of the poblem domain, and makes sure these info live in
// managed memory.
void DEMSolver::Initialize() {
    // A few checks first
    validateUserInputs();

    // Call the JIT compiler generator to make prep for this simulation
    generateEntityResources();
    generatePolicyResources();  // Policy info such as family policies needs entity info
    postResourceGen();

    // Transfer user-specified solver preference/instructions to workers
    transferSolverParams();
    // Transfer some simulation params to implementation level
    transferSimParams();

    // Allocate and populate kT dT managed arrays
    allocateGPUArrays();
    initializeGPUArrays();

    // Put sim data array pointers in place
    packDataPointers();

    // Compile some of the kernels
    jitifyKernels();

    // Notify the user how jitification goes
    reportInitStats();

    // Release the memory for those flattened arrays, as they are only used for transfers between workers and
    // jitification
    ReleaseFlattenedArrays();

    //// TODO: Give a warning if sys_initialized is true and the system is re-initialized: in that case, the user should
    /// know what they are doing
    sys_initialized = true;

    // Initialization is critical
    dT->announceCritical();
}

void DEMSolver::ShowTimingStats() {
    std::vector<std::string> kT_timer_names, dT_timer_names;
    std::vector<double> kT_timer_vals, dT_timer_vals;
    double kT_total_time, dT_total_time;
    kT->getTiming(kT_timer_names, kT_timer_vals);
    dT->getTiming(dT_timer_names, dT_timer_vals);
    kT_total_time = vector_sum<double>(kT_timer_vals);
    dT_total_time = vector_sum<double>(dT_timer_vals);
    DEME_PRINTF("\n~~ kT TIMING STATISTICS ~~\n");
    DEME_PRINTF("kT total active time: %.9g seconds\n", kT_total_time);
    for (unsigned int i = 0; i < kT_timer_names.size(); i++) {
        DEME_PRINTF("%s: %.9g seconds, %.6g%% of kT total runtime\n", kT_timer_names.at(i).c_str(), kT_timer_vals.at(i),
                    kT_timer_vals.at(i) / kT_total_time * 100.);
    }
    DEME_PRINTF("\n~~ dT TIMING STATISTICS ~~\n");
    DEME_PRINTF("dT total active time: %.9g seconds\n", dT_total_time);
    for (unsigned int i = 0; i < dT_timer_names.size(); i++) {
        DEME_PRINTF("%s: %.9g seconds, %.6g%% of dT total runtime\n", dT_timer_names.at(i).c_str(), dT_timer_vals.at(i),
                    dT_timer_vals.at(i) / dT_total_time * 100.);
    }
    DEME_PRINTF("--------------------------\n");
}

void DEMSolver::ClearTimingStats() {
    kT->resetTimers();
    dT->resetTimers();
}

void DEMSolver::ReleaseFlattenedArrays() {
    deallocate_array(m_family_mask_matrix);

    deallocate_array(m_template_number_name_map);

    deallocate_array(m_input_ext_obj_xyz);
    deallocate_array(m_input_ext_obj_rot);
    deallocate_array(m_input_ext_obj_family);

    deallocate_array(m_input_mesh_obj_xyz);
    deallocate_array(m_input_mesh_obj_rot);
    deallocate_array(m_input_mesh_obj_family);

    deallocate_array(m_unique_family_prescription);
    deallocate_array(m_input_clump_family);
    deallocate_array(m_anal_owner);
    deallocate_array(m_anal_materials);
    deallocate_array(m_anal_comp_pos);
    deallocate_array(m_anal_comp_rot);
    deallocate_array(m_anal_size_1);
    deallocate_array(m_anal_size_2);
    deallocate_array(m_anal_size_3);
    deallocate_array(m_anal_types);
    deallocate_array(m_anal_normals);

    deallocate_array(m_mesh_facet_owner);
    deallocate_array(m_mesh_facet_materials);
    deallocate_array(m_mesh_facets);

    deallocate_array(m_template_clump_mass);
    deallocate_array(m_template_clump_moi);
    deallocate_array(m_template_sp_mat_ids);
    deallocate_array(m_template_sp_radii);
    deallocate_array(m_template_sp_relPos);
    deallocate_array(m_template_clump_volume);

    deallocate_array(m_ext_obj_mass);
    deallocate_array(m_ext_obj_moi);
    deallocate_array(m_ext_obj_comp_num);

    deallocate_array(m_mesh_obj_mass);
    deallocate_array(m_mesh_obj_moi);

    nExtraContacts = 0;
}

void DEMSolver::resetWorkerThreads() {
    // The user won't be calling this when dT is working, so our only problem is that kT may be spinning in the inner
    // loop. So let's release kT.
    std::unique_lock<std::mutex> lock(kTMain_InteractionManager->mainCanProceed);
    kT->breakWaitingStatus();
    while (!kTMain_InteractionManager->userCallDone) {
        kTMain_InteractionManager->cv_mainCanProceed.wait(lock);
    }
    // Reset to make ready for next user call, don't forget it
    kTMain_InteractionManager->userCallDone = false;

    // Finally, reset the thread stats and wait for potential new user calls
    kT->resetUserCallStat();
    dT->resetUserCallStat();
}

/// When simulation parameters are updated by the user, they can call this method to transfer them to the GPU-side in
/// mid-simulation. This is relatively light-weight, designed only to change solver behavior and no array re-allocation
/// and re-compilation will happen.
void DEMSolver::UpdateSimParams() {
    transferSolverParams();
    transferSimParams();
    //// TODO: inspect what sim params should be transferred and what should not

    // Updating sim environment is critical
    dT->announceCritical();
}

void DEMSolver::UpdateClumps() {
    if (!sys_initialized) {
        DEME_ERROR(
            "Please call UpdateClumps only after the system is initialized, because it is for adding additional clumps "
            "to an initialized DEM system.");
    }
    if (nLastTimeExtObjLoad != nExtObjLoad) {
        DEME_ERROR(
            "UpdateClumps cannot be used after loading new analytical objects. Consider re-initializing at this "
            "point.\nNumber of analytical objects at last initialization: %u\nNumber of analytical objects now: %u",
            nLastTimeExtObjLoad, nExtObjLoad);
    }
    if (nLastTimeClumpTemplateLoad != nClumpTemplateLoad) {
        DEME_ERROR(
            "UpdateClumps cannot be used after loading new clump templates. Consider re-initializing at this "
            "point.\nNumber of clump templates at last initialization: %u\nNumber of clump templates now: %u",
            nLastTimeClumpTemplateLoad, nClumpTemplateLoad);
    }
    DEME_WARNING(
        "UpdateClumps will add all currently cached clumps to the simulation.\nYou may want to ClearCache first, then "
        "AddClumps, then call this method, so the clumps cached earlier are forgotten before this method takes place.");

    // This method requires kT and dT are sync-ed
    // resetWorkerThreads();

    // Record the number of entities, before adding to the system
    size_t nOwners_old = nOwnerBodies;
    size_t nClumps_old = nOwnerClumps;
    size_t nSpheres_old = nSpheresGM;
    size_t nTriMesh_old = nTriMeshes;
    size_t nFacets_old = nTriGM;

    preprocessClumps();
    preprocessClumpTemplates();
    //// TODO: This method should also work on newly added meshes
    updateTotalEntityNum();
    allocateGPUArrays();
    // `Update' method needs to know the number of existing clumps and spheres (before this addition)
    updateClumpMeshArrays(nOwners_old, nClumps_old, nSpheres_old, nTriMesh_old, nFacets_old);
    packDataPointers();
    ReleaseFlattenedArrays();
    // Updating clumps is very critical
    dT->announceCritical();

    // This method should not introduce new material or clump template or family prescription, let's check that
    if (nLastTimeMatNum != m_loaded_materials.size() || nLastTimeFamilyPreNum != m_input_family_prescription.size()) {
        DEME_ERROR(
            "UpdateClumps should not be used if you introduce new material types or family prescription (which will "
            "need re-jitification).\nWe used to have %u materials, now we have %u.\nWe used to have %u family "
            "prescription, now we have %u.",
            nLastTimeMatNum, m_loaded_materials.size(), nLastTimeFamilyPreNum, m_input_family_prescription.size());
    }
}

void DEMSolver::ChangeClumpSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors) {
    if (!sys_initialized) {
        DEME_ERROR(
            "ChangeClumpSizes operates on device-side arrays directly, so it requires the system to be initialized "
            "first.");
    }
    if (jitify_clump_templates || jitify_mass_moi) {
        DEME_ERROR(
            "ChangeClumpSizes only works when the clump components are flattened (not jitified).\nConsider calling "
            "SetJitifyClumpTemplates(false) and SetJitifyMassProperties(false).");
    }

    // This method requires kT and dT are sync-ed
    // resetWorkerThreads();

    std::thread dThread = std::move(std::thread([this, IDs, factors]() { this->dT->changeOwnerSizes(IDs, factors); }));
    std::thread kThread = std::move(std::thread([this, IDs, factors]() { this->kT->changeOwnerSizes(IDs, factors); }));
    dThread.join();
    kThread.join();

    // Size changes are critical
    dT->announceCritical();
}

/// Removes all entities associated with a family from the arrays (to save memory space). This method should only be
/// called periodically because it gives a large overhead. This is only used in long simulations where if the
/// `phased-out' entities do not get cleared, we won't have enough memory space.
/// TODO: Implement it.
void DEMSolver::PurgeFamily(unsigned int family_num) {}

void DEMSolver::DoDynamics(double thisCallDuration) {
    // Is it needed here??
    // dT->packDataPointers(kT->granData);

    // TODO: Return if nSphere == 0
    // TODO: Check if initialized

    // Tell dT how long this call is
    dT->setCycleDuration(thisCallDuration);

    dT->startThread();
    kT->startThread();

    // Wait till dT is done
    std::unique_lock<std::mutex> lock(dTMain_InteractionManager->mainCanProceed);
    while (!dTMain_InteractionManager->userCallDone) {
        dTMain_InteractionManager->cv_mainCanProceed.wait(lock);
    }
    // Reset to make ready for next user call, don't forget it. We don't do a `deep' reset using resetUserCallStat,
    // since that's only used when kT and dT sync.
    dTMain_InteractionManager->userCallDone = false;
}

void DEMSolver::DoDynamicsThenSync(double thisCallDuration) {
    // Based on async calls
    DoDynamics(thisCallDuration);

    // dT is finished, but the user asks us to sync, so we have to make kT sync with dT. This can be done by calling
    // resetWorkerThreads.
    resetWorkerThreads();
}

void DEMSolver::ShowThreadCollaborationStats() {
    DEME_PRINTF("\n~~ kT--dT CO-OP STATISTICS ~~\n");
    DEME_PRINTF("Number of steps dynamic executed: %zu\n", dT->nTotalSteps);
    DEME_PRINTF("Number of updates dynamic gets: %zu\n",
                (dTkT_InteractionManager->schedulingStats.nDynamicUpdates).load());
    DEME_PRINTF("Number of updates kinematic gets: %zu\n",
                (dTkT_InteractionManager->schedulingStats.nKinematicUpdates).load());
    if ((dTkT_InteractionManager->schedulingStats.nKinematicUpdates).load() > 0)
        DEME_PRINTF("Average steps per dynamic update: %.7g\n",
                    (double)(dT->nTotalSteps) / (dTkT_InteractionManager->schedulingStats.nKinematicUpdates).load());
    // DEME_PRINTF("Number of times dynamic loads buffer: %zu\n",
    //                 (dTkT_InteractionManager->schedulingStats.nDynamicReceives).load());
    // DEME_PRINTF("Number of times kinematic loads buffer: %zu\n",
    //                 (dTkT_InteractionManager->schedulingStats.nKinematicReceives).load());
    DEME_PRINTF("Number of times dynamic held back: %zu\n",
                (dTkT_InteractionManager->schedulingStats.nTimesDynamicHeldBack).load());
    DEME_PRINTF("Number of times kinematic held back: %zu\n",
                (dTkT_InteractionManager->schedulingStats.nTimesKinematicHeldBack).load());
    DEME_PRINTF("-----------------------------\n");
}

void DEMSolver::ClearThreadCollaborationStats() {
    dTkT_InteractionManager->schedulingStats.nDynamicUpdates = 0;
    dTkT_InteractionManager->schedulingStats.nKinematicUpdates = 0;
    // dTkT_InteractionManager->schedulingStats.nDynamicReceives = 0;
    // dTkT_InteractionManager->schedulingStats.nKinematicReceives = 0;
    dTkT_InteractionManager->schedulingStats.nTimesDynamicHeldBack = 0;
    dTkT_InteractionManager->schedulingStats.nTimesKinematicHeldBack = 0;
    dT->nTotalSteps = 0;
}

float DEMSolver::dTInspectReduce(const std::shared_ptr<jitify::Program>& inspection_kernel,
                                 const std::string& kernel_name,
                                 INSPECT_ENTITY_TYPE thing_to_insp,
                                 CUB_REDUCE_FLAVOR reduce_flavor,
                                 bool all_domain) {
    size_t n;
    switch (thing_to_insp) {
        case (INSPECT_ENTITY_TYPE::SPHERE):
            n = nSpheresGM;
            break;
        case (INSPECT_ENTITY_TYPE::CLUMP):
            n = nOwnerClumps;
            break;
    }
    float* pRes = dT->inspectCall(inspection_kernel, kernel_name, n, reduce_flavor, all_domain);
    return (float)(*pRes);
}

}  // namespace deme
