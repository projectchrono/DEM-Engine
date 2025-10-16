//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <core/ApiVersion.h>
#include "API.h"
#include "Defines.h"
#include "HostSideHelpers.hpp"
#include "AuxClasses.h"
#include "../kernel/DEMHelperKernels.cuh"

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

    // Set default solver params
    setDefaultSolverParams();

    // Thread-based worker creation may be needed as the workers allocate DualStructs on construction
    std::thread dT_construct([&]() {
        // Get a device/stream ID to use from the GPU Manager
        const GpuManager::StreamInfo dT_stream_info = dTkT_GpuManager->getAvailableStream();
        DEME_GPU_CALL(cudaSetDevice(dT_stream_info.device));
        dT = new DEMDynamicThread(dTMain_InteractionManager, dTkT_InteractionManager, dT_stream_info);
    });

    std::thread kT_construct([&]() {
        const GpuManager::StreamInfo kT_stream_info = dTkT_GpuManager->getAvailableStream();
        DEME_GPU_CALL(cudaSetDevice(kT_stream_info.device));
        kT = new DEMKinematicThread(kTMain_InteractionManager, dTkT_InteractionManager, kT_stream_info);
    });

    dT_construct.join();
    kT_construct.join();

    // Make friends
    dT->kT = kT;
    kT->dT = dT;

    // Specify force model
    m_force_model[DEFAULT_FORCE_MODEL_NAME] =
        std::make_shared<DEMForceModel>(std::move(DEMForceModel(FORCE_MODEL::HERTZIAN)));
}

DEMSolver::~DEMSolver() {
    if (sys_initialized)
        DoDynamicsThenSync(0.0);
    delete kT;
    delete dT;
    delete kTMain_InteractionManager;
    delete dTMain_InteractionManager;
    delete dTkT_InteractionManager;
    delete dTkT_GpuManager;
}

void DEMSolver::SetVerbosity(const std::string& verbose) {
    std::string u_verbose = str_to_upper(verbose);
    switch (hash_charr(u_verbose.c_str())) {
        case ("QUIET"_):
            verbosity = VERBOSITY::QUIET;
            break;
        case ("ERROR"_):
            verbosity = VERBOSITY::DEME_ERROR;
            break;
        case ("WARNING"_):
            verbosity = VERBOSITY::WARNING;
            break;
        case ("INFO"_):
            verbosity = VERBOSITY::INFO;
            break;
        case ("STEP_ANOMALY"_):
            verbosity = VERBOSITY::STEP_ANOMALY;
            break;
        case ("STEP_METRIC"_):
            verbosity = VERBOSITY::STEP_METRIC;
            break;
        case ("DEBUG"_):
            verbosity = VERBOSITY::DEBUG;
            break;
        case ("STEP_DEBUG"_):
            verbosity = VERBOSITY::STEP_DEBUG;
            break;
        default:
            DEME_ERROR("Instruction %s is unknown in SetVerbosity call.", verbose.c_str());
    }
}
void DEMSolver::SetOutputFormat(const std::string& format) {
    std::string u_format = str_to_upper(format);
    switch (hash_charr(u_format.c_str())) {
        case ("CSV"_):
            m_out_format = OUTPUT_FORMAT::CSV;
            break;
        case ("BINARY"_):
            m_out_format = OUTPUT_FORMAT::BINARY;
            break;
        case ("CHPF"_):
#ifdef DEME_USE_CHPF
            m_out_format = OUTPUT_FORMAT::CHPF;
            break;
#else
            DEME_ERROR("ChPF is not enabled when the code was compiled.");
#endif
        default:
            DEME_ERROR("Instruction %s is unknown in SetOutputFormat call.", format.c_str());
    }
}
void DEMSolver::SetContactOutputFormat(const std::string& format) {
    std::string u_format = str_to_upper(format);
    switch (hash_charr(u_format.c_str())) {
        case ("CSV"_):
            m_cnt_out_format = OUTPUT_FORMAT::CSV;
            break;
        case ("BINARY"_):
            m_cnt_out_format = OUTPUT_FORMAT::BINARY;
            break;
        case ("CHPF"_):
#ifdef DEME_USE_CHPF
            m_cnt_out_format = OUTPUT_FORMAT::CHPF;
            break;
#else
            DEME_ERROR("ChPF is not enabled when the code was compiled.");
#endif
        default:
            DEME_ERROR("Instruction %s is unknown in SetContactOutputFormat call.", format.c_str());
    }
}
void DEMSolver::SetMeshOutputFormat(const std::string& format) {
    std::string u_format = str_to_upper(format);
    switch (hash_charr(u_format.c_str())) {
        case ("VTK"_):
            m_mesh_out_format = MESH_FORMAT::VTK;
            break;
        case ("OBJ"_):
            m_mesh_out_format = MESH_FORMAT::OBJ;
            break;
        default:
            DEME_ERROR("Instruction %s is unknown in SetMeshOutputFormat call.", format.c_str());
    }
}

void DEMSolver::SetOutputContent(const std::vector<std::string>& content) {
    std::vector<std::string> u_content(content.size());
    for (unsigned int i = 0; i < content.size(); i++) {
        u_content[i] = str_to_upper(content[i]);
    }
    m_out_content = OUTPUT_CONTENT::XYZ;
    for (unsigned int i = 0; i < content.size(); i++) {
        switch (hash_charr(u_content[i].c_str())) {
            case ("XYZ"_):
                m_out_content = m_out_content | OUTPUT_CONTENT::XYZ;
                break;
            case ("QUAT"_):
                m_out_content = m_out_content | OUTPUT_CONTENT::QUAT;
                break;
            case ("ABSV"_):
                m_out_content = m_out_content | OUTPUT_CONTENT::ABSV;
                break;
            case ("VEL"_):
                m_out_content = m_out_content | OUTPUT_CONTENT::VEL;
                break;
            case ("ANG_VEL"_):
                m_out_content = m_out_content | OUTPUT_CONTENT::ANG_VEL;
                break;
            case ("ABS_ACC"_):
                m_out_content = m_out_content | OUTPUT_CONTENT::ABS_ACC;
                break;
            case ("ACC"_):
                m_out_content = m_out_content | OUTPUT_CONTENT::ACC;
                break;
            case ("ANG_ACC"_):
                m_out_content = m_out_content | OUTPUT_CONTENT::ANG_ACC;
                break;
            case ("FAMILY"_):
                m_out_content = m_out_content | OUTPUT_CONTENT::FAMILY;
                break;
            case ("MAT"_):
                m_out_content = m_out_content | OUTPUT_CONTENT::MAT;
                break;
            case ("OWNER_WILDCARD"_):
                m_out_content = m_out_content | OUTPUT_CONTENT::OWNER_WILDCARD;
                break;
            case ("GEO_WILDCARD"_):
                m_out_content = m_out_content | OUTPUT_CONTENT::GEO_WILDCARD;
                break;
            default:
                DEME_ERROR("Instruction %s is unknown in SetOutputContent call.", content[i].c_str());
        }
    }
}
void DEMSolver::SetContactOutputContent(const std::vector<std::string>& content) {
    std::vector<std::string> u_content(content.size());
    for (unsigned int i = 0; i < content.size(); i++) {
        u_content[i] = str_to_upper(content[i]);
    }
    m_cnt_out_content = CNT_OUTPUT_CONTENT::CNT_TYPE;
    for (unsigned int i = 0; i < content.size(); i++) {
        switch (hash_charr(u_content[i].c_str())) {
            case ("CNT_TYPE"_):
                m_cnt_out_content = m_cnt_out_content | CNT_OUTPUT_CONTENT::CNT_TYPE;
                break;
            case ("FORCE"_):
                m_cnt_out_content = m_cnt_out_content | CNT_OUTPUT_CONTENT::FORCE;
                break;
            case ("POINT"_):
                m_cnt_out_content = m_cnt_out_content | CNT_OUTPUT_CONTENT::CNT_POINT;
                break;
            case ("COMPONENT"_):
                m_cnt_out_content = m_cnt_out_content | CNT_OUTPUT_CONTENT::COMPONENT;
                break;
            case ("NORMAL"_):
                m_cnt_out_content = m_cnt_out_content | CNT_OUTPUT_CONTENT::NORMAL;
                break;
            case ("TORQUE"_):
                m_cnt_out_content = m_cnt_out_content | CNT_OUTPUT_CONTENT::TORQUE;
                break;
            case ("CNT_WILDCARD"_):
                m_cnt_out_content = m_cnt_out_content | CNT_OUTPUT_CONTENT::CNT_WILDCARD;
                break;
            case ("OWNER"_):
                m_cnt_out_content = m_cnt_out_content | CNT_OUTPUT_CONTENT::OWNER;
                break;
            case ("GEO_ID"_):
                m_cnt_out_content = m_cnt_out_content | CNT_OUTPUT_CONTENT::GEO_ID;
                break;
            case ("NICKNAME"_):
                m_cnt_out_content = m_cnt_out_content | CNT_OUTPUT_CONTENT::NICKNAME;
                break;
            default:
                DEME_ERROR("Instruction %s is unknown in SetContactOutputContent call.", content[i].c_str());
        }
    }
}

void DEMSolver::SyncMemoryTransfer() {
    dT->syncMemoryTransfer();
    kT->syncMemoryTransfer();
}

std::vector<bodyID_t> DEMSolver::GetOwnerContactClumps(bodyID_t ownerID) const {
    // Is this owner a clump?
    ownerType_t this_type = dT->ownerTypes[ownerID];  // ownerTypes has no way to change on device
    std::vector<bodyID_t> geo_to_watch;               // geo IDs that need to scan

    // Get device-major info to host first
    dT->idGeometryA.toHostAsync(dT->streamInfo.stream);
    dT->idGeometryB.toHostAsync(dT->streamInfo.stream);
    dT->contactType.toHostAsync(dT->streamInfo.stream);

    // These arrays can't change on device
    switch (this_type) {
        case OWNER_T_CLUMP:
            for (bodyID_t i = 0; i < nSpheresGM; i++) {
                if (ownerID == dT->ownerClumpBody[i])
                    geo_to_watch.push_back(i);
            }
            break;
        case OWNER_T_ANALYTICAL:
            for (bodyID_t i = 0; i < nAnalGM; i++) {
                if (ownerID == dT->ownerAnalBody[i])
                    geo_to_watch.push_back(i);
            }
            break;
        case OWNER_T_MESH:
            for (bodyID_t i = 0; i < nTriGM; i++) {
                if (ownerID == dT->ownerMesh[i])
                    geo_to_watch.push_back(i);
            }
            break;
    }

    // Try overlapping mem transfer with allocation...
    dT->syncMemoryTransfer();

    std::vector<bodyID_t> clumps_in_cnt;
    // If this is not clump, then checking idB for it is enough
    if (this_type != OWNER_T_CLUMP) {
        for (size_t i = 0; i < dT->getNumContacts(); i++) {
            auto idA = dT->idGeometryA[i];
            auto idB = dT->idGeometryB[i];
            if (!check_exist(geo_to_watch, idB))
                continue;
            auto cnt_type = dT->contactType[i];
            // If it is a mesh facet, then contact type needs to match
            if (this_type == OWNER_T_MESH) {
                if (cnt_type == SPHERE_MESH_CONTACT) {
                    clumps_in_cnt.push_back(dT->ownerClumpBody[idA]);
                }
            } else {  // If analytical, then contact type larger than PLANE is fine
                if (cnt_type >= SPHERE_PLANE_CONTACT) {
                    clumps_in_cnt.push_back(dT->ownerClumpBody[idA]);
                }
            }
        }
    } else {  // If a clump, then both idA and idB need to be checked
        for (size_t i = 0; i < dT->getNumContacts(); i++) {
            auto idA = dT->idGeometryA[i];
            auto idB = dT->idGeometryB[i];
            auto cnt_type = dT->contactType[i];
            if (check_exist(geo_to_watch, idA)) {
                if (cnt_type == SPHERE_SPHERE_CONTACT) {
                    clumps_in_cnt.push_back(dT->ownerClumpBody[idB]);
                }
            } else if (check_exist(geo_to_watch, idB)) {
                if (cnt_type == SPHERE_SPHERE_CONTACT) {
                    clumps_in_cnt.push_back(dT->ownerClumpBody[idA]);
                }
            }
        }
    }
    return clumps_in_cnt;
}

std::shared_ptr<DEMMaterial> DEMSolver::Duplicate(const std::shared_ptr<DEMMaterial>& ptr) {
    // Make a copy
    DEMMaterial obj = *ptr;
    return this->LoadMaterial(obj);
}
std::shared_ptr<DEMClumpTemplate> DEMSolver::Duplicate(const std::shared_ptr<DEMClumpTemplate>& ptr) {
    // Make a copy
    DEMClumpTemplate obj = *ptr;
    return this->LoadClumpType(obj);
}
std::shared_ptr<DEMClumpBatch> DEMSolver::Duplicate(const std::shared_ptr<DEMClumpBatch>& ptr) {
    // Make a copy
    DEMClumpBatch obj = *ptr;
    return this->AddClumps(obj);
}

std::vector<std::pair<bodyID_t, bodyID_t>> DEMSolver::GetClumpContacts() const {
    std::vector<bodyID_t> idA_tmp, idB_tmp;
    std::vector<family_t> famA_tmp, famB_tmp;
    std::vector<contact_t> cnt_type_tmp;
    // Getting sphere contacts is enough
    getContacts_impl(idA_tmp, idB_tmp, cnt_type_tmp, famA_tmp, famB_tmp,
                     [](contact_t type) { return type == SPHERE_SPHERE_CONTACT; });
    auto idx = hostSortIndices(idA_tmp);
    std::vector<std::pair<bodyID_t, bodyID_t>> out_pair(idx.size());
    for (size_t i = 0; i < idx.size(); i++) {
        out_pair[i] = std::pair<bodyID_t, bodyID_t>(idA_tmp[idx[i]], idB_tmp[idx[i]]);
    }
    return out_pair;
}

std::vector<std::pair<bodyID_t, bodyID_t>> DEMSolver::GetClumpContacts(
    const std::set<family_t>& family_to_include) const {
    std::vector<bodyID_t> idA_tmp, idB_tmp;
    std::vector<family_t> famA_tmp, famB_tmp;
    std::vector<contact_t> cnt_type_tmp;
    // Getting sphere contacts is enough
    getContacts_impl(idA_tmp, idB_tmp, cnt_type_tmp, famA_tmp, famB_tmp,
                     [](contact_t type) { return type == SPHERE_SPHERE_CONTACT; });
    // Exclude the families that are not in the set
    std::vector<bool> elem_to_remove(idA_tmp.size(), false);
    for (size_t i = 0; i < idA_tmp.size(); i++) {
        if (!check_exist(family_to_include, famA_tmp.at(i)) || !check_exist(family_to_include, famB_tmp.at(i))) {
            elem_to_remove[i] = true;
        }
    }
    idA_tmp = hostRemoveElem(idA_tmp, elem_to_remove);
    idB_tmp = hostRemoveElem(idB_tmp, elem_to_remove);
    famA_tmp = hostRemoveElem(famA_tmp, elem_to_remove);
    famB_tmp = hostRemoveElem(famB_tmp, elem_to_remove);
    cnt_type_tmp = hostRemoveElem(cnt_type_tmp, elem_to_remove);

    auto idx = hostSortIndices(idA_tmp);
    std::vector<std::pair<bodyID_t, bodyID_t>> out_pair(idx.size());
    for (size_t i = 0; i < idx.size(); i++) {
        out_pair[i] = std::pair<bodyID_t, bodyID_t>(idA_tmp[idx[i]], idB_tmp[idx[i]]);
    }
    return out_pair;
}

std::vector<std::pair<bodyID_t, bodyID_t>> DEMSolver::GetClumpContacts(
    std::vector<std::pair<family_t, family_t>>& family_pair) const {
    std::vector<bodyID_t> idA_tmp, idB_tmp;
    std::vector<family_t> famA_tmp, famB_tmp;
    std::vector<contact_t> cnt_type_tmp;
    // Getting sphere contacts is enough
    getContacts_impl(idA_tmp, idB_tmp, cnt_type_tmp, famA_tmp, famB_tmp,
                     [](contact_t type) { return type == SPHERE_SPHERE_CONTACT; });
    auto idx = hostSortIndices(idA_tmp);
    std::vector<std::pair<bodyID_t, bodyID_t>> out_pair(idx.size());
    family_pair.resize(idx.size());
    for (size_t i = 0; i < idx.size(); i++) {
        out_pair[i] = std::pair<bodyID_t, bodyID_t>(idA_tmp[idx[i]], idB_tmp[idx[i]]);
        family_pair[i] = std::pair<family_t, family_t>(famA_tmp[idx[i]], famB_tmp[idx[i]]);
    }
    return out_pair;
}

std::vector<std::pair<bodyID_t, bodyID_t>> DEMSolver::GetContacts() const {
    std::vector<bodyID_t> idA_tmp, idB_tmp;
    std::vector<family_t> famA_tmp, famB_tmp;
    std::vector<contact_t> cnt_type_tmp;
    // Getting sphere contacts is enough
    getContacts_impl(idA_tmp, idB_tmp, cnt_type_tmp, famA_tmp, famB_tmp, [](contact_t type) { return true; });
    auto idx = hostSortIndices(idA_tmp);
    std::vector<std::pair<bodyID_t, bodyID_t>> out_pair(idx.size());
    for (size_t i = 0; i < idx.size(); i++) {
        out_pair[i] = std::pair<bodyID_t, bodyID_t>(idA_tmp[idx[i]], idB_tmp[idx[i]]);
    }
    return out_pair;
}

std::vector<std::pair<bodyID_t, bodyID_t>> DEMSolver::GetContacts(const std::set<family_t>& family_to_include) const {
    std::vector<bodyID_t> idA_tmp, idB_tmp;
    std::vector<family_t> famA_tmp, famB_tmp;
    std::vector<contact_t> cnt_type_tmp;
    // Getting sphere contacts is enough
    getContacts_impl(idA_tmp, idB_tmp, cnt_type_tmp, famA_tmp, famB_tmp, [](contact_t type) { return true; });
    // Exclude the families that are not in the set
    std::vector<bool> elem_to_remove(idA_tmp.size(), false);
    for (size_t i = 0; i < idA_tmp.size(); i++) {
        if (!check_exist(family_to_include, famA_tmp.at(i)) || !check_exist(family_to_include, famB_tmp.at(i))) {
            elem_to_remove[i] = true;
        }
    }
    idA_tmp = hostRemoveElem(idA_tmp, elem_to_remove);
    idB_tmp = hostRemoveElem(idB_tmp, elem_to_remove);
    famA_tmp = hostRemoveElem(famA_tmp, elem_to_remove);
    famB_tmp = hostRemoveElem(famB_tmp, elem_to_remove);
    cnt_type_tmp = hostRemoveElem(cnt_type_tmp, elem_to_remove);

    auto idx = hostSortIndices(idA_tmp);
    std::vector<std::pair<bodyID_t, bodyID_t>> out_pair(idx.size());
    for (size_t i = 0; i < idx.size(); i++) {
        out_pair[i] = std::pair<bodyID_t, bodyID_t>(idA_tmp[idx[i]], idB_tmp[idx[i]]);
    }
    return out_pair;
}

std::vector<std::pair<bodyID_t, bodyID_t>> DEMSolver::GetContacts(
    std::vector<std::pair<family_t, family_t>>& family_pair) const {
    std::vector<bodyID_t> idA_tmp, idB_tmp;
    std::vector<family_t> famA_tmp, famB_tmp;
    std::vector<contact_t> cnt_type_tmp;
    // Getting sphere contacts is enough
    getContacts_impl(idA_tmp, idB_tmp, cnt_type_tmp, famA_tmp, famB_tmp, [](contact_t type) { return true; });
    auto idx = hostSortIndices(idA_tmp);
    std::vector<std::pair<bodyID_t, bodyID_t>> out_pair(idx.size());
    family_pair.resize(idx.size());
    for (size_t i = 0; i < idx.size(); i++) {
        out_pair[i] = std::pair<bodyID_t, bodyID_t>(idA_tmp[idx[i]], idB_tmp[idx[i]]);
        family_pair[i] = std::pair<family_t, family_t>(famA_tmp[idx[i]], famB_tmp[idx[i]]);
    }
    return out_pair;
}

std::shared_ptr<ContactInfoContainer> DEMSolver::GetContactDetailedInfo(float force_thres) const {
    return dT->generateContactInfo(force_thres);
}

std::vector<float3> DEMSolver::GetOwnerPosition(bodyID_t ownerID, bodyID_t n) const {
    return dT->getOwnerPos(ownerID, n);
}
std::vector<float3> DEMSolver::GetOwnerAngVel(bodyID_t ownerID, bodyID_t n) const {
    return dT->getOwnerAngVel(ownerID, n);
}
std::vector<float3> DEMSolver::GetOwnerVelocity(bodyID_t ownerID, bodyID_t n) const {
    return dT->getOwnerVel(ownerID, n);
}
std::vector<float4> DEMSolver::GetOwnerOriQ(bodyID_t ownerID, bodyID_t n) const {
    return dT->getOwnerOriQ(ownerID, n);
}
std::vector<float3> DEMSolver::GetOwnerAcc(bodyID_t ownerID, bodyID_t n) const {
    return dT->getOwnerAcc(ownerID, n);
}
std::vector<float3> DEMSolver::GetOwnerAngAcc(bodyID_t ownerID, bodyID_t n) const {
    return dT->getOwnerAngAcc(ownerID, n);
}
std::vector<unsigned int> DEMSolver::GetOwnerFamily(bodyID_t ownerID, bodyID_t n) const {
    return dT->getOwnerFamily(ownerID, n);
}

std::vector<float> DEMSolver::GetOwnerWildcardValue(bodyID_t ownerID, const std::string& name, bodyID_t n) {
    assertSysInit("GetOwnerWildcardValue");
    if (m_owner_wc_num.find(name) == m_owner_wc_num.end()) {
        DEME_ERROR(
            "No owner wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerOwnerWildcards first.",
            name.c_str());
    }
    return dT->getOwnerWildcardValue(ownerID, m_owner_wc_num.at(name), n);
}

std::vector<float> DEMSolver::GetAllOwnerWildcardValue(const std::string& name) {
    assertSysInit("GetAllOwnerWildcardValue");
    if (m_owner_wc_num.find(name) == m_owner_wc_num.end()) {
        DEME_ERROR(
            "No owner wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerOwnerWildcards first.",
            name.c_str());
    }
    std::vector<float> res;
    dT->getAllOwnerWildcardValue(res, m_owner_wc_num.at(name));
    return res;
}

std::vector<float> DEMSolver::GetFamilyOwnerWildcardValue(unsigned int N, const std::string& name) {
    assertSysInit("GetFamilyOwnerWildcardValue");
    if (m_owner_wc_num.find(name) == m_owner_wc_num.end()) {
        DEME_ERROR(
            "No owner wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerOwnerWildcards first.",
            name.c_str());
    }
    std::vector<float> res;
    dT->getFamilyOwnerWildcardValue(res, N, m_owner_wc_num.at(name));
    return res;
}

std::vector<float> DEMSolver::GetTriWildcardValue(bodyID_t geoID, const std::string& name, size_t n) {
    assertSysInit("GetTriWildcardValue");
    if (m_geo_wc_num.find(name) == m_geo_wc_num.end()) {
        DEME_ERROR(
            "No geometry wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerGeometryWildcards in the force model first.",
            name.c_str());
    }
    std::vector<float> res;
    dT->getTriWildcardValue(res, geoID, m_geo_wc_num.at(name), n);
    return res;
}

std::vector<float> DEMSolver::GetSphereWildcardValue(bodyID_t geoID, const std::string& name, size_t n) {
    assertSysInit("GetSphereWildcardValue");
    if (m_geo_wc_num.find(name) == m_geo_wc_num.end()) {
        DEME_ERROR(
            "No geometry wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerGeometryWildcards in the force model first.",
            name.c_str());
    }
    std::vector<float> res;
    dT->getSphereWildcardValue(res, geoID, m_geo_wc_num.at(name), n);
    return res;
}

std::vector<float> DEMSolver::GetAnalWildcardValue(bodyID_t geoID, const std::string& name, size_t n) {
    assertSysInit("GetAnalWildcardValue");
    if (m_geo_wc_num.find(name) == m_geo_wc_num.end()) {
        DEME_ERROR(
            "No geometry wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerGeometryWildcards in the force model first.",
            name.c_str());
    }
    std::vector<float> res;
    dT->getAnalWildcardValue(res, geoID, m_geo_wc_num.at(name), n);
    return res;
}

size_t DEMSolver::GetOwnerContactForces(const std::vector<bodyID_t>& ownerIDs,
                                        std::vector<float3>& points,
                                        std::vector<float3>& forces) {
    return dT->getOwnerContactForces(ownerIDs, points, forces);
}
size_t DEMSolver::GetOwnerContactForces(const std::vector<bodyID_t>& ownerIDs,
                                        std::vector<float3>& points,
                                        std::vector<float3>& forces,
                                        std::vector<float3>& torques,
                                        bool torque_in_local) {
    return dT->getOwnerContactForces(ownerIDs, points, forces, torques, torque_in_local);
}

std::vector<float> DEMSolver::GetOwnerMass(bodyID_t ownerID, bodyID_t n) const {
    std::vector<float> res(n);
    for (bodyID_t i = 0; i < n; i++) {
        // No mechanism to change mass properties on device, so [] operator is fine
        if (jitify_mass_moi) {
            inertiaOffset_t offset = dT->inertiaPropOffsets[ownerID + i];
            res[i] = dT->massOwnerBody[offset];
        } else {
            res[i] = dT->massOwnerBody[ownerID + i];
        }
    }
    return res;
}

std::vector<float3> DEMSolver::GetOwnerMOI(bodyID_t ownerID, bodyID_t n) const {
    std::vector<float3> res(n);
    for (bodyID_t i = 0; i < n; i++) {
        // No mechanism to change mass properties on device, so [] operator is fine
        if (jitify_mass_moi) {
            inertiaOffset_t offset = dT->inertiaPropOffsets[ownerID + i];
            float m1 = dT->mmiXX[offset];
            float m2 = dT->mmiYY[offset];
            float m3 = dT->mmiZZ[offset];
            res[i] = make_float3(m1, m2, m3);
        } else {
            float m1 = dT->mmiXX[ownerID + i];
            float m2 = dT->mmiYY[ownerID + i];
            float m3 = dT->mmiZZ[ownerID + i];
            res[i] = make_float3(m1, m2, m3);
        }
    }
    return res;
}

void DEMSolver::AddOwnerNextStepAcc(bodyID_t ownerID, const std::vector<float3>& acc) {
    dT->addOwnerNextStepAcc(ownerID, acc);
}
void DEMSolver::AddOwnerNextStepAngAcc(bodyID_t ownerID, const std::vector<float3>& angAcc) {
    dT->addOwnerNextStepAngAcc(ownerID, angAcc);
}
void DEMSolver::SetOwnerPosition(bodyID_t ownerID, const std::vector<float3>& pos) {
    dT->setOwnerPos(ownerID, pos);
}
void DEMSolver::SetOwnerAngVel(bodyID_t ownerID, const std::vector<float3>& angVel) {
    dT->setOwnerAngVel(ownerID, angVel);
}
void DEMSolver::SetOwnerVelocity(bodyID_t ownerID, const std::vector<float3>& vel) {
    dT->setOwnerVel(ownerID, vel);
}
void DEMSolver::SetOwnerOriQ(bodyID_t ownerID, const std::vector<float4>& oriQ) {
    dT->setOwnerOriQ(ownerID, oriQ);
}
void DEMSolver::SetOwnerFamily(bodyID_t ownerID, unsigned int fam, bodyID_t n) {
    if (fam > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You called SetOwnerFamily with family number %u, but family number should not be larger than %u.",
                   fam, std::numeric_limits<family_t>::max());
    }
    kT->setOwnerFamily(ownerID, static_cast<family_t>(fam), n);
    dT->setOwnerFamily(ownerID, static_cast<family_t>(fam), n);
}

void DEMSolver::SetTriNodeRelPos(size_t owner, size_t triID, const std::vector<float3>& new_nodes) {
    auto& mesh = m_meshes.at(m_owner_mesh_map.at(owner));
    if (mesh->GetNumNodes() != new_nodes.size()) {
        DEME_ERROR(
            "To deform a mesh, provided vector must have the same length as the number of nodes in mesh.\nThe mesh has "
            "%zu nodes, yet the provided vector has length %zu.",
            mesh->GetNumNodes(), new_nodes.size());
    }
    // We actually modify the cached mesh... since it has implications in output
    for (size_t i = 0; i < mesh->GetNumNodes(); i++) {
        mesh->m_vertices[i] = new_nodes[i];
    }
    std::vector<DEMTriangle> new_triangles(mesh->GetNumTriangles());
    for (size_t i = 0; i < mesh->GetNumTriangles(); i++) {
        new_triangles[i] = mesh->GetTriangle(i);
    }
    dT->setTriNodeRelPos(triID, new_triangles);
    dT->solverFlags.willMeshDeform = true;

    // kT just receives update from dT, to avoid mem hazards
    // kT->setTriNodeRelPos(triID, new_triangles);
}
void DEMSolver::UpdateTriNodeRelPos(size_t owner, size_t triID, const std::vector<float3>& updates) {
    auto& mesh = m_meshes.at(m_owner_mesh_map.at(owner));
    if (mesh->GetNumNodes() != updates.size()) {
        DEME_ERROR(
            "To deform a mesh, provided vector must have the same length as the number of nodes in mesh.\nThe mesh has "
            "%zu nodes, yet the provided vector has length %zu.",
            mesh->GetNumNodes(), updates.size());
    }
    // We actually modify the cached mesh... since it has implications in output
    for (size_t i = 0; i < mesh->GetNumNodes(); i++) {
        mesh->m_vertices[i] += updates[i];
    }
    // No need to worry about RHR: that's taken care of at init
    std::vector<DEMTriangle> new_triangles(mesh->GetNumTriangles());
    for (size_t i = 0; i < mesh->GetNumTriangles(); i++) {
        new_triangles[i] = mesh->GetTriangle(i);
    }
    // This is correct to use setTriNodeRelPos, as mesh is already modified in this method
    dT->setTriNodeRelPos(triID, new_triangles);
    dT->solverFlags.willMeshDeform = true;

    // kT just receives update from dT, to avoid mem hazards
    // kT->setTriNodeRelPos(triID, new_triangles);
}
std::shared_ptr<DEMMeshConnected>& DEMSolver::GetCachedMesh(bodyID_t ownerID) {
    if (m_owner_mesh_map.find(ownerID) == m_owner_mesh_map.end()) {
        DEME_ERROR("Owner %zu is not a mesh, you therefore cannot retrive a handle to mesh using it.", (size_t)ownerID);
    }
    return m_meshes.at(m_owner_mesh_map.at(ownerID));
}
std::vector<float3> DEMSolver::GetMeshNodesGlobal(bodyID_t ownerID) {
    if (m_owner_mesh_map.find(ownerID) == m_owner_mesh_map.end()) {
        DEME_ERROR("Owner %zu is not a mesh, you therefore cannot get its nodes' coordinates.", (size_t)ownerID);
    }
    float3 mesh_pos = dT->getOwnerPos(ownerID)[0];
    float4 mesh_oriQ = dT->getOwnerOriQ(ownerID)[0];
    std::vector<float3> nodes(m_meshes.at(m_owner_mesh_map.at(ownerID))->GetCoordsVertices());
    for (auto& pnt : nodes) {
        applyFrameTransformLocalToGlobal<float3, float3, float4>(pnt, mesh_pos, mesh_oriQ);
    }
    return nodes;
}

double DEMSolver::GetSimTime() const {
    return dT->getSimTime();
}

void DEMSolver::SetSimTime(double time) {
    dT->setSimTime(time);
}

float DEMSolver::GetUpdateFreq() const {
    return dT->getUpdateFreq();
}

void DEMSolver::SetIntegrator(const std::string& intg) {
    std::string u_intg = str_to_upper(intg);
    switch (hash_charr(u_intg.c_str())) {
        case ("FORWARD_EULER"_):
            m_integrator = TIME_INTEGRATOR::FORWARD_EULER;
            break;
        case ("CENTERED_DIFFERENCE"_):
            m_integrator = TIME_INTEGRATOR::CENTERED_DIFFERENCE;
            break;
        case ("EXTENDED_TAYLOR"_):
            m_integrator = TIME_INTEGRATOR::EXTENDED_TAYLOR;
            break;
        default:
            DEME_ERROR("Integration type %s is unknown. Please select another via SetIntegrator.", intg.c_str());
    }
}

void DEMSolver::SetAdaptiveTimeStepType(const std::string& type) {
    DEME_WARNING(
        "SetAdaptiveTimeStepType is currently not implemented and has no effect, time step size is still fixed.");
    switch (hash_charr(type.c_str())) {
        case ("none"_):
            adapt_ts_type = ADAPT_TS_TYPE::NONE;
            break;
        case ("max_vel"_):
            adapt_ts_type = ADAPT_TS_TYPE::MAX_VEL;
            break;
        case ("int_diff"_):
            adapt_ts_type = ADAPT_TS_TYPE::INT_DIFF;
            break;
        default:
            DEME_ERROR("Adaptive time step type %s is unknown. Please select another via SetAdaptiveTimeStepType.",
                       type.c_str());
    }
}

void DEMSolver::SetCDNumStepsMaxDriftHistorySize(unsigned int n) {
    if (n > NUM_STEPS_RESERVED_AFTER_RENEWING_FREQ_TUNER) {
        max_drift_gauge_history_size = n;
    } else {
        DEME_WARNING(
            "SetCDNumStepsMaxDriftHistorySize has no effect, since the argument supplied needs to be larger than %u.",
            NUM_STEPS_RESERVED_AFTER_RENEWING_FREQ_TUNER);
    }
}

void DEMSolver::SetMaxVelocity(float max_vel) {
    m_approx_max_vel = max_vel;
}

void DEMSolver::SetExpandSafetyType(const std::string& insp_type) {
    if (insp_type == "auto") {
        m_max_v_finder_type = MARGIN_FINDER_TYPE::DEFAULT;
        use_user_defined_expand_factor = false;
    } else {
        DEME_ERROR("Unknown string input \"%s\" for SetExpandSafetyType.", insp_type.c_str());
    }
}

void DEMSolver::InstructBoxDomainDimension(float x, float y, float z, const std::string& dir_exact) {
    m_user_box_min = make_float3(-x / 2., -y / 2., -z / 2.);
    m_user_box_max = make_float3(x / 2., y / 2., z / 2.);

    float3 enlarge_amount =
        make_float3(x * DEFAULT_BOX_DOMAIN_ENLARGE_RATIO / 2., y * DEFAULT_BOX_DOMAIN_ENLARGE_RATIO / 2.,
                    z * DEFAULT_BOX_DOMAIN_ENLARGE_RATIO / 2.);
    m_target_box_min = m_user_box_min - enlarge_amount;
    m_target_box_max = m_user_box_max + enlarge_amount;

    if (str_to_upper(dir_exact) == "X") {
        m_box_dir_length_is_exact = SPATIAL_DIR::X;
        m_target_box_min.x = m_user_box_min.x;
        m_target_box_max.x = m_user_box_max.x;
    } else if (str_to_upper(dir_exact) == "Y") {
        m_box_dir_length_is_exact = SPATIAL_DIR::Y;
        m_target_box_min.y = m_user_box_min.y;
        m_target_box_max.y = m_user_box_max.y;
    } else if (str_to_upper(dir_exact) == "Z") {
        m_box_dir_length_is_exact = SPATIAL_DIR::Z;
        m_target_box_min.z = m_user_box_min.z;
        m_target_box_max.z = m_user_box_max.z;
    } else if (str_to_upper(dir_exact) == "NONE") {
        m_box_dir_length_is_exact = SPATIAL_DIR::NONE;
    } else {
        DEME_ERROR("Unknown '%s' parameter in InstructBoxDomainDimension call.", dir_exact.c_str());
    }
}

void DEMSolver::InstructBoxDomainDimension(const std::pair<float, float>& x,
                                           const std::pair<float, float>& y,
                                           const std::pair<float, float>& z,
                                           const std::string& dir_exact) {
    m_user_box_min = make_float3(DEME_MIN(x.first, x.second), DEME_MIN(y.first, y.second), DEME_MIN(z.first, z.second));
    m_user_box_max = make_float3(DEME_MAX(x.second, x.first), DEME_MAX(y.second, y.first), DEME_MAX(z.second, z.first));

    float3 enlarge_amount =
        make_float3(std::abs(x.second - x.first), std::abs(y.second - y.first), std::abs(z.second - z.first));
    enlarge_amount *= DEFAULT_BOX_DOMAIN_ENLARGE_RATIO / 2.;
    m_target_box_min = m_user_box_min - enlarge_amount;
    m_target_box_max = m_user_box_max + enlarge_amount;

    if (str_to_upper(dir_exact) == "X") {
        m_box_dir_length_is_exact = SPATIAL_DIR::X;
        m_target_box_min.x = m_user_box_min.x;
        m_target_box_max.x = m_user_box_max.x;
    } else if (str_to_upper(dir_exact) == "Y") {
        m_box_dir_length_is_exact = SPATIAL_DIR::Y;
        m_target_box_min.y = m_user_box_min.y;
        m_target_box_max.y = m_user_box_max.y;
    } else if (str_to_upper(dir_exact) == "Z") {
        m_box_dir_length_is_exact = SPATIAL_DIR::Z;
        m_target_box_min.z = m_user_box_min.z;
        m_target_box_max.z = m_user_box_max.z;
    } else if (str_to_upper(dir_exact) == "NONE") {
        m_box_dir_length_is_exact = SPATIAL_DIR::NONE;
    } else {
        DEME_ERROR("Unknown '%s' parameter in InstructBoxDomainDimension call.", dir_exact.c_str());
    }
}

std::shared_ptr<DEMForceModel> DEMSolver::DefineContactForceModel(const std::string& model) {
    DEMForceModel force_model(FORCE_MODEL::CUSTOM);  // Custom
    force_model.DefineCustomModel(model);
    m_force_model[DEFAULT_FORCE_MODEL_NAME] = std::make_shared<DEMForceModel>(std::move(force_model));

    return m_force_model[DEFAULT_FORCE_MODEL_NAME];
}

std::shared_ptr<DEMForceModel> DEMSolver::ReadContactForceModel(const std::string& filename) {
    DEMForceModel force_model(FORCE_MODEL::CUSTOM);  // Custom
    std::filesystem::path sourcefile = USER_SCRIPT_PATH / filename;
    if (force_model.ReadCustomModelFile(std::filesystem::path(filename))) {
        // If not in that folder, then maybe the user meant an absolute path
        if (force_model.ReadCustomModelFile(sourcefile))
            DEME_ERROR("The force model file %s is not found.", filename.c_str());
    }
    m_force_model[DEFAULT_FORCE_MODEL_NAME] = std::make_shared<DEMForceModel>(std::move(force_model));

    return m_force_model[DEFAULT_FORCE_MODEL_NAME];
}

std::shared_ptr<DEMForceModel> DEMSolver::UseFrictionalHertzianModel() {
    m_force_model[DEFAULT_FORCE_MODEL_NAME]->SetForceModelType(FORCE_MODEL::HERTZIAN);

    return m_force_model[DEFAULT_FORCE_MODEL_NAME];
}

std::shared_ptr<DEMForceModel> DEMSolver::UseFrictionlessHertzianModel() {
    m_force_model[DEFAULT_FORCE_MODEL_NAME]->SetForceModelType(FORCE_MODEL::HERTZIAN_FRICTIONLESS);

    return m_force_model[DEFAULT_FORCE_MODEL_NAME];
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

void DEMSolver::SetFamilyFixed(unsigned int ID) {
    assertSysNotInit("SetFamilyFixed");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.linVelXPrescribed = true;
    preInfo.linVelYPrescribed = true;
    preInfo.linVelZPrescribed = true;
    preInfo.rotVelXPrescribed = true;
    preInfo.rotVelYPrescribed = true;
    preInfo.rotVelZPrescribed = true;

    preInfo.rotPosPrescribed = true;
    preInfo.linPosXPrescribed = true;
    preInfo.linPosYPrescribed = true;
    preInfo.linPosZPrescribed = true;

    preInfo.linVelX = "0";
    preInfo.linVelY = "0";
    preInfo.linVelZ = "0";
    preInfo.rotVelX = "0";
    preInfo.rotVelY = "0";
    preInfo.rotVelZ = "0";

    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::SetFamilyPrescribedLinVel(unsigned int ID,
                                          const std::string& velX,
                                          const std::string& velY,
                                          const std::string& velZ,
                                          bool dictate,
                                          const std::string& pre) {
    assertSysNotInit("SetFamilyPrescribedLinVel");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.linVelXPrescribed = dictate;
    preInfo.linVelYPrescribed = dictate;
    preInfo.linVelZPrescribed = dictate;
    // By default, lin vel prescription fixes rotation if dictate == true
    preInfo.rotVelXPrescribed = dictate;
    preInfo.rotVelYPrescribed = dictate;
    preInfo.rotVelZPrescribed = dictate;

    preInfo.linVelX = velX;
    preInfo.linVelY = velY;
    preInfo.linVelZ = velZ;
    // If the user does specify, then `Set' methods dictate motion
    if (velX != "none") {
        preInfo.linVelXPrescribed = true;
    }
    if (velY != "none") {
        preInfo.linVelYPrescribed = true;
    }
    if (velZ != "none") {
        preInfo.linVelZPrescribed = true;
    }

    preInfo.linVelPre = pre;

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

    preInfo.linVelXPrescribed = true;
    preInfo.linVelYPrescribed = true;
    preInfo.linVelZPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}
void DEMSolver::SetFamilyPrescribedLinVelX(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedLinVelX");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.linVelXPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}
void DEMSolver::SetFamilyPrescribedLinVelY(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedLinVelY");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.linVelYPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}
void DEMSolver::SetFamilyPrescribedLinVelZ(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedLinVelZ");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.linVelZPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::SetFamilyPrescribedAngVel(unsigned int ID,
                                          const std::string& velX,
                                          const std::string& velY,
                                          const std::string& velZ,
                                          bool dictate,
                                          const std::string& pre) {
    assertSysNotInit("SetFamilyPrescribedAngVel");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    // By default, rot vel prescription fixes linear motions if dictate == true
    preInfo.linVelXPrescribed = dictate;
    preInfo.linVelYPrescribed = dictate;
    preInfo.linVelZPrescribed = dictate;

    preInfo.rotVelXPrescribed = dictate;
    preInfo.rotVelYPrescribed = dictate;
    preInfo.rotVelZPrescribed = dictate;

    preInfo.rotVelX = velX;
    preInfo.rotVelY = velY;
    preInfo.rotVelZ = velZ;
    // If the user does specify, then `Set' methods dictate motion
    if (velX != "none") {
        preInfo.rotVelXPrescribed = true;
    }
    if (velY != "none") {
        preInfo.rotVelYPrescribed = true;
    }
    if (velZ != "none") {
        preInfo.rotVelZPrescribed = true;
    }

    preInfo.rotVelPre = pre;

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

    preInfo.rotVelXPrescribed = true;
    preInfo.rotVelYPrescribed = true;
    preInfo.rotVelZPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}
void DEMSolver::SetFamilyPrescribedAngVelX(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedAngVelX");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.rotVelXPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}
void DEMSolver::SetFamilyPrescribedAngVelY(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedAngVelY");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.rotVelYPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}
void DEMSolver::SetFamilyPrescribedAngVelZ(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedAngVelZ");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.rotVelZPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::SetFamilyPrescribedPosition(unsigned int ID,
                                            const std::string& X,
                                            const std::string& Y,
                                            const std::string& Z,
                                            bool dictate,
                                            const std::string& pre) {
    assertSysNotInit("SetFamilyPrescribedPosition");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    // If dictate, this method also dictate quaternion
    preInfo.rotPosPrescribed = dictate;
    preInfo.linPosXPrescribed = dictate;
    preInfo.linPosYPrescribed = dictate;
    preInfo.linPosZPrescribed = dictate;

    preInfo.linPosX = X;
    preInfo.linPosY = Y;
    preInfo.linPosZ = Z;

    // If the user does specify, then `Set' methods dictate motion
    if (X != "none") {
        preInfo.linPosXPrescribed = true;
    }
    if (Y != "none") {
        preInfo.linPosYPrescribed = true;
    }
    if (Z != "none") {
        preInfo.linPosZPrescribed = true;
    }

    preInfo.linPosPre = pre;

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

    preInfo.linPosXPrescribed = true;
    preInfo.linPosYPrescribed = true;
    preInfo.linPosZPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}
void DEMSolver::SetFamilyPrescribedPositionX(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedPositionX");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.linPosXPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}
void DEMSolver::SetFamilyPrescribedPositionY(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedPositionY");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.linPosYPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}
void DEMSolver::SetFamilyPrescribedPositionZ(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedPositionZ");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.linPosZPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::SetFamilyPrescribedQuaternion(unsigned int ID, const std::string& q_formula, bool dictate) {
    assertSysNotInit("SetFamilyPrescribedQuaternion");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.rotPosPrescribed = dictate;
    // If dictate, this method also dictate linear position
    preInfo.linPosXPrescribed = dictate;
    preInfo.linPosYPrescribed = dictate;
    preInfo.linPosZPrescribed = dictate;

    // Make sure there is return
    if (!match_whole_word(q_formula, "return")) {
        DEME_ERROR(
            "CorrectFamilyQuaternion call must supply a code string that returns a float4, like 'return "
            "make_float4(0,0,0,1)'.");
    }

    preInfo.oriQ = q_formula;

    // If the user does specify, then `Set' methods dictate motion
    preInfo.rotPosPrescribed = true;

    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}
void DEMSolver::SetFamilyPrescribedQuaternion(unsigned int ID) {
    assertSysNotInit("SetFamilyPrescribedQuaternion");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied prescribed motion to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.rotPosPrescribed = true;
    preInfo.used = true;

    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::ShowMemStats() const {
    DEME_PRINTF("kT host memory usage: %s\n", pretty_format_bytes(GetHostMemUsageKinematic()).c_str());
    DEME_PRINTF("kT device memory usage: %s\n", pretty_format_bytes(GetDeviceMemUsageKinematic()).c_str());
    DEME_PRINTF("dT host memory usage: %s\n", pretty_format_bytes(GetHostMemUsageDynamic()).c_str());
    DEME_PRINTF("dT device memory usage: %s\n", pretty_format_bytes(GetDeviceMemUsageDynamic()).c_str());
}

void DEMSolver::AddFamilyPrescribedAcc(unsigned int ID,
                                       const std::string& X,
                                       const std::string& Y,
                                       const std::string& Z,
                                       const std::string& pre) {
    assertSysNotInit("AddFamilyPrescribedAcc");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You added acceleration to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.accX = X;
    preInfo.accY = Y;
    preInfo.accZ = Z;
    preInfo.accPre = pre;

    preInfo.used = true;
    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::AddFamilyPrescribedAngAcc(unsigned int ID,
                                          const std::string& X,
                                          const std::string& Y,
                                          const std::string& Z,
                                          const std::string& pre) {
    assertSysNotInit("AddFamilyPrescribedAngAcc");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You added angular acceleration to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.angAccX = X;
    preInfo.angAccY = Y;
    preInfo.angAccZ = Z;
    preInfo.angAccPre = pre;

    preInfo.used = true;
    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::CorrectFamilyLinVel(unsigned int ID,
                                    const std::string& X,
                                    const std::string& Y,
                                    const std::string& Z,
                                    const std::string& pre) {
    assertSysNotInit("CorrectFamilyLinVel");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied motion correction to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.linVelX = X;
    preInfo.linVelY = Y;
    preInfo.linVelZ = Z;
    preInfo.linVelPre = pre;

    // Correction methods do not resist true simulation physics

    preInfo.used = true;
    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::CorrectFamilyAngVel(unsigned int ID,
                                    const std::string& X,
                                    const std::string& Y,
                                    const std::string& Z,
                                    const std::string& pre) {
    assertSysNotInit("CorrectFamilyAngVel");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied motion correction to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.rotVelX = X;
    preInfo.rotVelY = Y;
    preInfo.rotVelZ = Z;
    preInfo.rotVelPre = pre;

    // Correction methods do not resist true simulation physics

    preInfo.used = true;
    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::CorrectFamilyPosition(unsigned int ID,
                                      const std::string& X,
                                      const std::string& Y,
                                      const std::string& Z,
                                      const std::string& pre) {
    assertSysNotInit("CorrectFamilyPosition");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied motion correction to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    preInfo.linPosX = X;
    preInfo.linPosY = Y;
    preInfo.linPosZ = Z;
    preInfo.linPosPre = pre;

    // Correction methods do not resist true simulation physics

    preInfo.used = true;
    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::CorrectFamilyQuaternion(unsigned int ID, const std::string& q_formula) {
    assertSysNotInit("CorrectFamilyQuaternion");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You applied motion correction to family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    familyPrescription_t preInfo;
    preInfo.family = ID;

    // Make sure there is return
    if (!match_whole_word(q_formula, "return")) {
        DEME_ERROR(
            "CorrectFamilyQuaternion call must supply a code string that returns a float4, like 'return "
            "make_float4(0,0,0,1)'.");
    }

    preInfo.oriQ = q_formula;

    // Correction methods do not resist true simulation physics

    preInfo.used = true;
    m_input_family_prescription.push_back(preInfo);
}

void DEMSolver::SetTriWildcardValue(bodyID_t geoID, const std::string& name, const std::vector<float>& vals) {
    assertSysInit("SetTriWildcardValue");
    if (m_geo_wc_num.find(name) == m_geo_wc_num.end()) {
        DEME_ERROR(
            "No geometry wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerGeometryWildcards in the force model first.",
            name.c_str());
    }
    dT->setTriWildcardValue(geoID, m_geo_wc_num.at(name), vals);
}

void DEMSolver::SetSphereWildcardValue(bodyID_t geoID, const std::string& name, const std::vector<float>& vals) {
    assertSysInit("SetSphereWildcardValue");
    if (m_geo_wc_num.find(name) == m_geo_wc_num.end()) {
        DEME_ERROR(
            "No geometry wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerGeometryWildcards in the force model first.",
            name.c_str());
    }
    dT->setSphWildcardValue(geoID, m_geo_wc_num.at(name), vals);
}

void DEMSolver::SetAnalWildcardValue(bodyID_t geoID, const std::string& name, const std::vector<float>& vals) {
    assertSysInit("SetAnalWildcardValue");
    if (m_geo_wc_num.find(name) == m_geo_wc_num.end()) {
        DEME_ERROR(
            "No geometry wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerGeometryWildcards in the force model first.",
            name.c_str());
    }
    dT->setAnalWildcardValue(geoID, m_geo_wc_num.at(name), vals);
}

void DEMSolver::SetOwnerWildcardValue(bodyID_t ownerID, const std::string& name, const std::vector<float>& vals) {
    assertSysInit("SetOwnerWildcardValue");
    if (m_owner_wc_num.find(name) == m_owner_wc_num.end()) {
        DEME_ERROR(
            "No owner wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerOwnerWildcards in the force model first.",
            name.c_str());
    }
    dT->setOwnerWildcardValue(ownerID, m_owner_wc_num.at(name), vals);
}

void DEMSolver::SetFamilyContactWildcardValueEither(unsigned int N, const std::string& name, float val) {
    assertSysInit("SetFamilyContactWildcardValueEither");
    if (m_cnt_wc_num.find(name) == m_cnt_wc_num.end()) {
        DEME_ERROR(
            "No contact wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerContactWildcards in the force model first.",
            name.c_str());
    }
    dT->setFamilyContactWildcardValueEither(N, m_cnt_wc_num.at(name), val);
}
void DEMSolver::SetFamilyContactWildcardValueBoth(unsigned int N, const std::string& name, float val) {
    assertSysInit("SetFamilyContactWildcardValueBoth");
    if (m_cnt_wc_num.find(name) == m_cnt_wc_num.end()) {
        DEME_ERROR(
            "No contact wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerContactWildcards in the force model first.",
            name.c_str());
    }
    dT->setFamilyContactWildcardValueBoth(N, m_cnt_wc_num.at(name), val);
}
void DEMSolver::SetFamilyContactWildcardValue(unsigned int N1, unsigned int N2, const std::string& name, float val) {
    assertSysInit("SetFamilyContactWildcardValue");
    if (m_cnt_wc_num.find(name) == m_cnt_wc_num.end()) {
        DEME_ERROR(
            "No contact wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerContactWildcards in the force model first.",
            name.c_str());
    }
    dT->setFamilyContactWildcardValue(N1, N2, m_cnt_wc_num.at(name), val);
}
void DEMSolver::SetContactWildcardValue(const std::string& name, float val) {
    assertSysInit("SetContactWildcardValue");
    if (m_cnt_wc_num.find(name) == m_cnt_wc_num.end()) {
        DEME_ERROR(
            "No contact wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerContactWildcards in the force model first.",
            name.c_str());
    }
    dT->setContactWildcardValue(m_cnt_wc_num.at(name), val);
}

void DEMSolver::SetFamilyClumpMaterial(unsigned int N, const std::shared_ptr<DEMMaterial>& mat) {
    assertSysInit("SetFamilyClumpMaterial");
    dT->setFamilyClumpMaterial(N, mat->load_order);
}
void DEMSolver::SetFamilyMeshMaterial(unsigned int N, const std::shared_ptr<DEMMaterial>& mat) {
    assertSysInit("SetFamilyMeshMaterial");
    dT->setFamilyMeshMaterial(N, mat->load_order);
}

void DEMSolver::SetFamilyOwnerWildcardValue(unsigned int N, const std::string& name, const std::vector<float>& vals) {
    assertSysInit("SetFamilyOwnerWildcardValue");
    if (m_owner_wc_num.find(name) == m_owner_wc_num.end()) {
        DEME_ERROR(
            "No owner wildcard in the force model is named %s.\nIf you need to use it, declare it via "
            "SetPerOwnerWildcards first.",
            name.c_str());
    }
    dT->setFamilyOwnerWildcardValue(N, m_owner_wc_num.at(name), vals);
}

void DEMSolver::SetContactWildcards(const std::set<std::string>& wildcards) {
    m_force_model[DEFAULT_FORCE_MODEL_NAME]->SetPerContactWildcards(wildcards);
}

void DEMSolver::SetOwnerWildcards(const std::set<std::string>& wildcards) {
    m_force_model[DEFAULT_FORCE_MODEL_NAME]->SetPerOwnerWildcards(wildcards);
}

void DEMSolver::SetGeometryWildcards(const std::set<std::string>& wildcards) {
    m_force_model[DEFAULT_FORCE_MODEL_NAME]->SetPerGeometryWildcards(wildcards);
}

void DEMSolver::DisableFamilyOutput(unsigned int ID) {
    assertSysNotInit("DisableFamilyOutput");
    if (ID > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You tried to disable output for family %u, but family number should not be larger than %u.", ID,
                   std::numeric_limits<family_t>::max());
    }
    m_no_output_families.insert(ID);
}

void DEMSolver::AddKernelInclude(const std::string& lib_name) {
    kernel_includes += "#include <" + lib_name + ">\n";
}

void DEMSolver::MarkFamilyPersistentContactEither(unsigned int N) {
    assertSysInit("MarkFamilyPersistentContactEither");
    assignFamilyPersistentContactEither(N, CONTACT_IS_PERSISTENT);
}
void DEMSolver::RemoveFamilyPersistentContactEither(unsigned int N) {
    assertSysInit("RemoveFamilyPersistentContactEither");
    assignFamilyPersistentContactEither(N, CONTACT_NOT_PERSISTENT);
}

void DEMSolver::MarkFamilyPersistentContactBoth(unsigned int N) {
    assertSysInit("MarkFamilyPersistentContactBoth");
    assignFamilyPersistentContactBoth(N, CONTACT_IS_PERSISTENT);
}
void DEMSolver::RemoveFamilyPersistentContactBoth(unsigned int N) {
    assertSysInit("RemoveFamilyPersistentContactBoth");
    assignFamilyPersistentContactBoth(N, CONTACT_NOT_PERSISTENT);
}

void DEMSolver::MarkFamilyPersistentContact(unsigned int N1, unsigned int N2) {
    assertSysInit("MarkFamilyPersistentContact");
    assignFamilyPersistentContact(N1, N2, CONTACT_IS_PERSISTENT);
}
void DEMSolver::RemoveFamilyPersistentContact(unsigned int N1, unsigned int N2) {
    assertSysInit("RemoveFamilyPersistentContact");
    assignFamilyPersistentContact(N1, N2, CONTACT_NOT_PERSISTENT);
}

void DEMSolver::MarkPersistentContact() {
    assertSysInit("MarkPersistentContact");
    assignPersistentContact(CONTACT_IS_PERSISTENT);
}
void DEMSolver::RemovePersistentContact() {
    assertSysInit("RemovePersistentContact");
    assignPersistentContact(CONTACT_NOT_PERSISTENT);
}

std::shared_ptr<DEMMaterial> DEMSolver::LoadMaterial(DEMMaterial& a_material) {
    std::shared_ptr<DEMMaterial> ptr = std::make_shared<DEMMaterial>(std::move(a_material));
    ptr->load_order = m_loaded_materials.size();
    m_loaded_materials.push_back(ptr);
    nMaterialsLoad++;
    return m_loaded_materials.back();
}

std::shared_ptr<DEMMaterial> DEMSolver::LoadMaterial(const std::unordered_map<std::string, float>& mat_prop) {
    // Register material property names that appeared in this call
    for (const auto& a_pair : mat_prop) {
        m_material_prop_names.insert(a_pair.first);
        if (a_pair.first == "CoR") {
            if (a_pair.second < DEME_TINY_FLOAT)
                DEME_WARNING(
                    "Material type %zu is set to have near-zero restitution (%.9g). Please make sure this is "
                    "intentional.",
                    m_loaded_materials.size(), a_pair.second);
            if (a_pair.second > 1.f)
                DEME_WARNING(
                    "Material type %zu is set to have a restitution coefficient larger than 1 (%.9g). This is "
                    "typically not physical and should destabilize the simulation.",
                    m_loaded_materials.size(), a_pair.second);
        }
        // Material names cannot have spaces in them
        if (match_pattern(a_pair.first, " ")) {
            DEME_ERROR("Material property %s is not valid: no spaces allowed in its name.", a_pair.first.c_str());
        }
    }
    DEMMaterial a_material(mat_prop);
    return LoadMaterial(a_material);
}

void DEMSolver::SetMaterialPropertyPair(const std::string& name,
                                        const std::shared_ptr<DEMMaterial>& mat1,
                                        const std::shared_ptr<DEMMaterial>& mat2,
                                        float val) {
    m_pairwise_material_prop_names.insert(name);
    if (!check_exist(m_material_prop_names, name)) {
        DEME_WARNING(
            "Material property %s in SetMaterialPropertyPair call never appeared in previous LoadMaterial calls.\nIf "
            "two objects having the same material collide, this property is likely to be considered 0.",
            name.c_str());
    }
    std::pair<unsigned int, unsigned int> mat_load_ids =
        std::pair<unsigned int, unsigned int>(mat1->load_order, mat2->load_order);
    std::pair<std::pair<unsigned int, unsigned int>, float> pair_val =
        std::pair<std::pair<unsigned int, unsigned int>, float>(mat_load_ids, val);
    m_pairwise_matprop[name].push_back(pair_val);
}

std::shared_ptr<DEMClumpTemplate> DEMSolver::LoadClumpType(DEMClumpTemplate& clump) {
    if (clump.nComp != clump.radii.size() || clump.nComp != clump.relPos.size() ||
        clump.nComp != clump.materials.size()) {
        DEME_ERROR(
            "Radii, relative positions and material arrays defining a clump topology, must all have the same length "
            "(%u, as indicated by nComp).\nHowever it seems that their lengths are %zu, %zu, %zu, respectively.\nIf "
            "you constructed a DEMClumpTemplate struct yourself, you may need to carefully check if their lengths "
            "agree with nComp.",
            clump.nComp, clump.radii.size(), clump.relPos.size(), clump.materials.size());
    }
    if (clump.mass < 1e-15 || length(clump.MOI) < 1e-15) {
        DEME_WARNING(
            "A type of clump is instructed to have near-zero (or negative) mass or moment of inertia (mass: %.9g, MOI "
            "magnitude: %.9g). This could destabilize the simulation.\nPlease make sure this is intentional.",
            clump.mass, length(clump.MOI));
    }

    // Print the mark to this clump template
    unsigned int offset = m_templates.size();
    clump.mark = offset;

    // Give it a default name
    if (clump.m_name == DEME_NULL_CLUMP_NAME) {
        char my_name[200];
        sprintf(my_name, "%04d", (int)nClumpTemplateLoad);
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
    ptr->SetInitPos(pos);
    ptr->AddPlane(make_float3(0), normal, material);
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
        kT->familyMaskMatrix.setVal(PREVENT_CONTACT, posInMat);
        dT->familyMaskMatrix.setVal(PREVENT_CONTACT, posInMat);
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
        kT->familyMaskMatrix.setVal(DONT_PREVENT_CONTACT, posInMat);
        dT->familyMaskMatrix.setVal(DONT_PREVENT_CONTACT, posInMat);
    }
}

void DEMSolver::SetFamilyExtraMargin(unsigned int N, float extra_size) {
    if (N > std::numeric_limits<family_t>::max()) {
        DEME_ERROR("You are adding an extra margin to family %u, but family number should not be larger than %u.", N,
                   std::numeric_limits<family_t>::max());
    }
    if (extra_size < 0.) {
        DEME_ERROR("You are adding an extra margin of size %.7g, but the size should not be smaller than 0.",
                   extra_size);
    }
    kT->familyExtraMarginSize.setVal(extra_size, N);
    dT->familyExtraMarginSize.setVal(extra_size, N);
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

std::shared_ptr<DEMClumpBatch> DEMSolver::AddClumps(DEMClumpBatch& input_batch) {
    // load_order should be its position in the cache array, not nBatchClumpsLoad
    input_batch.load_order = cached_input_clump_batches.size();
    // But we still need to record a batch loaded
    nBatchClumpsLoad++;
    cached_input_clump_batches.push_back(std::make_shared<DEMClumpBatch>(std::move(input_batch)));

    for (size_t i = 0; i < cached_input_clump_batches.back()->GetNumClumps(); i++) {
        unsigned int nComp = cached_input_clump_batches.back()->types.at(i)->nComp;
        // Keep tab for itself
        cached_input_clump_batches.back()->nSpheres += nComp;
    }
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

std::shared_ptr<DEMInspector> DEMSolver::CreateInspector(const std::string& quantity) {
    DEMInspector insp(this, this->dT, quantity);
    m_inspectors.push_back(std::make_shared<DEMInspector>(std::move(insp)));
    return m_inspectors.back();
}

std::shared_ptr<DEMInspector> DEMSolver::CreateInspector(const std::string& quantity, const std::string& region) {
    DEMInspector insp(this, this->dT, quantity, region);
    m_inspectors.push_back(std::make_shared<DEMInspector>(std::move(insp)));
    return m_inspectors.back();
}

void DEMSolver::WriteSphereFile(const std::string& outfilename) const {
    switch (m_out_format) {
#ifdef DEME_USE_CHPF
        case (OUTPUT_FORMAT::CHPF): {
            std::ofstream ptFile(outfilename, std::ios::out | std::ios::binary);
            dT->writeSpheresAsChpf(ptFile);
            ptFile.close();
            break;
        }
#endif
        case (OUTPUT_FORMAT::CSV): {
            std::ofstream ptFile(outfilename, std::ios::out);
            dT->writeSpheresAsCsv(ptFile);
            ptFile.close();
            break;
        }
        case (OUTPUT_FORMAT::BINARY): {
            // std::ofstream ptFile(outfilename, std::ios::out | std::ios::binary);
            //// TODO: Implement it
            std::ofstream ptFile(outfilename, std::ios::out);
            DEME_WARNING("Binary sphere output is not implemented yet, using CSV...");
            dT->writeSpheresAsCsv(ptFile);
            ptFile.close();
            break;
        }
        default:
            DEME_ERROR("Sphere output file format is unknown. Please set it via SetOutputFormat.");
    }
}

void DEMSolver::WriteClumpFile(const std::string& outfilename, unsigned int accuracy) const {
    switch (m_out_format) {
#ifdef DEME_USE_CHPF
        case (OUTPUT_FORMAT::CHPF): {
            std::ofstream ptFile(outfilename, std::ios::out | std::ios::binary);
            dT->writeClumpsAsChpf(ptFile, accuracy);
            ptFile.close();
            break;
        }
#endif
        case (OUTPUT_FORMAT::CSV): {
            std::ofstream ptFile(outfilename, std::ios::out);
            dT->writeClumpsAsCsv(ptFile, accuracy);
            ptFile.close();
            break;
        }
        case (OUTPUT_FORMAT::BINARY): {
            // std::ofstream ptFile(outfilename, std::ios::out | std::ios::binary);
            //// TODO: Implement it
            std::ofstream ptFile(outfilename, std::ios::out);
            DEME_WARNING("Binary clump output is not implemented yet, using CSV...");
            dT->writeClumpsAsCsv(ptFile, accuracy);
            ptFile.close();
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
            ptFile.close();
            break;
        }
        case (OUTPUT_FORMAT::BINARY): {
            // std::ofstream ptFile(outfilename, std::ios::out | std::ios::binary);
            //// TODO: Implement it
            DEME_WARNING("Binary contact pair output is not implemented yet, using CSV...");
            std::ofstream ptFile(outfilename, std::ios::out);
            dT->writeContactsAsCsv(ptFile, force_thres);
            ptFile.close();
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
            ptFile.close();
            break;
        }
        default:
            DEME_ERROR(
                "Mesh output file format is unknown or not implemented. Please re-set it via SetMeshOutputFormat.");
    }
}

size_t DEMSolver::ChangeClumpFamily(unsigned int fam_num,
                                    const std::pair<double, double>& X,
                                    const std::pair<double, double>& Y,
                                    const std::pair<double, double>& Z,
                                    const std::set<unsigned int>& orig_fam) {
    float3 L = make_float3(X.first, Y.first, Z.first);
    float3 U = make_float3(X.second, Y.second, Z.second);
    size_t count = 0;

    // And get those device-major data from device
    if (dT->solverFlags.canFamilyChangeOnDevice) {
        dT->familyID.toHost();
        kT->familyID.toHost();
    }
    dT->voxelID.toHost();
    dT->locX.toHost();
    dT->locY.toHost();
    dT->locZ.toHost();

    for (bodyID_t ownerID = 0; ownerID < nOwnerBodies; ownerID++) {
        // ownerTypes has no way to change on device
        const ownerType_t this_type = dT->ownerTypes[ownerID];
        if (this_type != OWNER_T_CLUMP)
            continue;
        float3 CoM;
        voxelID_t voxel = dT->voxelID[ownerID];
        subVoxelPos_t subVoxX = dT->locX[ownerID];
        subVoxelPos_t subVoxY = dT->locY[ownerID];
        subVoxelPos_t subVoxZ = dT->locZ[ownerID];
        voxelIDToPosition<float, voxelID_t, subVoxelPos_t>(CoM.x, CoM.y, CoM.z, voxel, subVoxX, subVoxY, subVoxZ,
                                                           dT->simParams->nvXp2, dT->simParams->nvYp2,
                                                           dT->simParams->voxelSize, dT->simParams->l);
        CoM.x += dT->simParams->LBFX;
        CoM.y += dT->simParams->LBFY;
        CoM.z += dT->simParams->LBFZ;

        // In region. This can be generalized in future versions.
        if (isBetween(CoM, L, U)) {
            if (orig_fam.size() == 0) {
                dT->familyID[ownerID] = fam_num;
                kT->familyID[ownerID] = fam_num;  // Must do both for dT and kT
                count++;
            } else {
                unsigned int old_fam = dT->familyID[ownerID];
                if (check_exist(orig_fam, old_fam)) {
                    dT->familyID[ownerID] = fam_num;
                    kT->familyID[ownerID] = fam_num;
                    count++;
                }
            }
        }
    }

    dT->familyID.toDevice();
    kT->familyID.toDevice();
    return count;
}

// The method should be called after user inputs are in place, and before starting the simulation. It figures out a part
// of the required simulation information such as the scale of the problem domain, and makes sure these info live in
// GPU memory.
void DEMSolver::Initialize(bool dry_run) {
    // A few checks first
    validateUserInputs();

    // Call the JIT compiler generator to make prep for this simulation
    generateEntityResources();
    generatePolicyResources();  // Policy info such as family policies needs entity info
    postResourceGen();

    // Transfer user-specified solver preference/instructions to workers
    setSolverParams();
    // Transfer some simulation params to implementation level
    setSimParams();

    // Allocate and populate kT dT arrays
    allocateGPUArrays();
    initializeGPUArrays();

    // Put sim data array pointers in place
    packDataPointers();

    // Now that all params prepared, and all data pointers packed on host side, we need to migrate that imformation to
    // the device
    migrateSimParamsToDevice();
    migrateArrayDataToDevice();

    // Compile some of the kernels
    jitifyKernels();

    // Notify the user how jitification goes
    reportInitStats();

    // Release the memory for those flattened arrays, as they are only used for transfers between workers and
    // jitification
    ReleaseFlattenedArrays();

    // Initialization is critical
    dT->announceCritical();

    // Always clear cache after init
    ClearCache();

    //// TODO: Give a warning if sys_initialized is true and the system is re-initialized: in that case, the user should
    /// know what they are doing
    sys_initialized = true;

    if (dry_run) {
        // Do a dry-run: It establishes contact pairs. It helps to locate obvious problems at the start (like, too many
        // contact pairs), and if the user needs to modify the contact wildcards before simulation starts, this step is
        // meaningful. Dry-run is automatically done if advancing the simulation by 0 or a negative amount of time.
        DoDynamicsThenSync(-1.0);
    }
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
    if (kT_total_time == 0.)
        kT_total_time = DEME_TINY_FLOAT;
    for (unsigned int i = 0; i < kT_timer_names.size(); i++) {
        DEME_PRINTF("%s: %.9g seconds, %.6g%% of kT total runtime\n", kT_timer_names.at(i).c_str(), kT_timer_vals.at(i),
                    kT_timer_vals.at(i) / kT_total_time * 100.);
    }
    DEME_PRINTF("\n~~ dT TIMING STATISTICS ~~\n");
    DEME_PRINTF("dT total active time: %.9g seconds\n", dT_total_time);
    if (dT_total_time == 0.)
        dT_total_time = DEME_TINY_FLOAT;
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
    {
        std::unique_lock<std::mutex> lock(kTMain_InteractionManager->mainCanProceed);
        kT->breakWaitingStatus();
        while (!kTMain_InteractionManager->userCallDone) {
            kTMain_InteractionManager->cv_mainCanProceed.wait(lock);
        }
        // Reset to make ready for next user call, don't forget it
        kTMain_InteractionManager->userCallDone = false;
    }

    // Finally, reset the thread stats and wait for potential new user calls
    kT->resetUserCallStat();
    dT->resetUserCallStat();
}

/// When simulation parameters are updated by the user, they can call this method to transfer them to the GPU-side in
/// mid-simulation. This is a relatively deep reset. If you just need to update step size, don't use this.
void DEMSolver::UpdateSimParams() {
    // Bin size will be re-calculated (in case you wish to switch to manual).
    decideBinSize();
    decideCDMarginStrat();

    setSolverParams();
    setSimParams();

    // Now that all params prepared, we need to migrate that imformation to the device
    migrateSimParamsToDevice();

    // JItify may require a defined device to decide the arch
    std::thread dT_build([&]() {
        DEME_GPU_CALL(cudaSetDevice(dT->streamInfo.device));
        // Jitify max vel finder, in case the policy there changed
        m_approx_max_vel_func->Initialize(m_subs, m_jitify_options, true);
        dT->approxMaxVelFunc = m_approx_max_vel_func;
        // Updating sim environment is critical
        dT->announceCritical();
    });
    dT_build.join();
}

void DEMSolver::UpdateStepSize(double ts) {
    m_ts_size = ts;
    // We for now store ts as float on devices...
    dT->simParams->h = ts;
    kT->simParams->h = ts;
    // dT->simParams.syncMemberToDevice<float>(offsetof(DEMSimParams, h));
    // kT->simParams.syncMemberToDevice<float>(offsetof(DEMSimParams, h));
    dT->simParams.toDevice();
    kT->simParams.toDevice();
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
            "point.\nNumber of clump templates at last initialization: %zu\nNumber of clump templates now: %zu",
            nLastTimeClumpTemplateLoad, nClumpTemplateLoad);
    }
    // DEME_WARNING(
    //     "UpdateClumps will add all currently cached clumps to the simulation.\nYou may want to ClearCache first,"
    //     "then AddClumps, then call this method, so the clumps cached earlier are forgotten before this method takes"
    //     "place.");

    // This method requires kT and dT are sync-ed
    // resetWorkerThreads();

    // NOTE!!! This step in UpdateClumps is extremely important, as we'll soon modify device-major arrays on host!
    migrateArrayDataToHost();

    // Record the number of entities, before adding to the system
    size_t nOwners_old = nOwnerBodies;
    size_t nClumps_old = nOwnerClumps;
    size_t nSpheres_old = nSpheresGM;
    size_t nTriMesh_old = nTriMeshes;
    size_t nFacets_old = nTriGM;
    unsigned int nAnalGM_old = nAnalGM;
    unsigned int nExtObj_old = nExtObj;

    preprocessClumps();
    preprocessClumpTemplates();
    //// TODO: This method should also work on newly added meshes
    updateTotalEntityNum();
    allocateGPUArrays();
    // `Update' method needs to know the number of existing clumps and spheres (before this addition)
    updateClumpMeshArrays(nOwners_old, nClumps_old, nSpheres_old, nTriMesh_old, nFacets_old, nExtObj_old, nAnalGM_old);
    packDataPointers();

    // Now that all params prepared, and all data pointers packed on host side, we need to migrate that imformation to
    // the device
    migrateSimParamsToDevice();
    migrateArrayDataToDevice();

    ReleaseFlattenedArrays();
    // Updating clumps is very critical
    dT->announceCritical();

    // This method should not introduce new material or clump template or family prescription, let's check that
    if (nLastTimeMatNum != m_loaded_materials.size() || nLastTimeFamilyPreNum != m_input_family_prescription.size()) {
        DEME_ERROR(
            "UpdateClumps should not be used if you introduce new material types or family prescription (which will "
            "need re-jitification).\nWe used to have %u materials, now we have %zu.\nWe used to have %u family "
            "prescription, now we have %zu.",
            nLastTimeMatNum, m_loaded_materials.size(), nLastTimeFamilyPreNum, m_input_family_prescription.size());
    }

    // After Initialize or UpdateClumps, we should clear host-side initialization object cache
    ClearCache();
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
    if (!sys_initialized) {
        Initialize();
    }

    // Tell dT how long this call is
    dT->setCycleDuration(thisCallDuration);

    dT->startThread();
    kT->startThread();

    // Wait till dT is done
    {
        std::unique_lock<std::mutex> lock(dTMain_InteractionManager->mainCanProceed);
        while (!dTMain_InteractionManager->userCallDone) {
            dTMain_InteractionManager->cv_mainCanProceed.wait(lock);
        }
        // Reset to make ready for next user call, don't forget it. We don't do a `deep' reset using resetUserCallStat,
        // since that's only used when kT and dT sync.
        dTMain_InteractionManager->userCallDone = false;
    }
}

void DEMSolver::DoDynamicsThenSync(double thisCallDuration) {
    // Based on async calls
    DoDynamics(thisCallDuration);

    // dT is finished, but the user asks us to sync, so we have to make kT sync with dT. This can be done by calling
    // resetWorkerThreads.
    resetWorkerThreads();

    // This is hardly needed
    // SyncMemoryTransfer();
}

void DEMSolver::ShowThreadCollaborationStats() {
    DEME_PRINTF("\n~~ kT--dT CO-OP STATISTICS ~~\n");
    DEME_PRINTF("Number of steps dynamic executed: %zu\n", dT->nTotalSteps);
    DEME_PRINTF("Number of updates dynamic gets: %zu\n",
                (dTkT_InteractionManager->schedulingStats.nDynamicUpdates).load());
    DEME_PRINTF("Number of updates kinematic gets: %zu\n",
                (dTkT_InteractionManager->schedulingStats.nKinematicUpdates).load());
    if ((dTkT_InteractionManager->schedulingStats.nKinematicUpdates).load() > 0) {
        // The numerator is not currentStampOfDynamic since it got cleared when the user syncs.
        DEME_PRINTF("Average steps per dynamic update: %.7g\n",
                    (double)(dT->nTotalSteps) / (dTkT_InteractionManager->schedulingStats.nKinematicUpdates).load());
        DEME_PRINTF("Average steps contact detection lags behind: %.7g\n",
                    (double)(dTkT_InteractionManager->schedulingStats.accumKinematicLagSteps).load() /
                        (dTkT_InteractionManager->schedulingStats.nKinematicUpdates).load());
    }
    // DEME_PRINTF("Number of times dynamic loads buffer: %zu\n",
    //                 (dTkT_InteractionManager->schedulingStats.nDynamicReceives).load());
    // DEME_PRINTF("Number of times kinematic loads buffer: %zu\n",
    //                 (dTkT_InteractionManager->schedulingStats.nKinematicReceives).load());
    DEME_PRINTF("Number of times dynamic held back: %zu\n",
                (dTkT_InteractionManager->schedulingStats.nTimesDynamicHeldBack).load());
    // DEME_PRINTF("Number of times kinematic held back: %zu\n",
    //             (dTkT_InteractionManager->schedulingStats.nTimesKinematicHeldBack).load());
    DEME_PRINTF("-----------------------------\n");
}

void DEMSolver::ShowAnomalies() {
    DEME_PRINTF("\n~~ Simulation anomaly report ~~\n");
    bool there_is_anomaly = goThroughWorkerAnomalies();
    if (!there_is_anomaly) {
        DEME_PRINTF("There is no simulation anomalies on record.\n");
    }
    DEME_PRINTF("-----------------------------\n");
    kT->anomalies.Clear();
    dT->anomalies.Clear();
}

void DEMSolver::ClearThreadCollaborationStats() {
    dTkT_InteractionManager->schedulingStats.nDynamicUpdates = 0;
    dTkT_InteractionManager->schedulingStats.nKinematicUpdates = 0;
    // dTkT_InteractionManager->schedulingStats.nDynamicReceives = 0;
    // dTkT_InteractionManager->schedulingStats.nKinematicReceives = 0;
    dTkT_InteractionManager->schedulingStats.nTimesDynamicHeldBack = 0;
    dTkT_InteractionManager->schedulingStats.nTimesKinematicHeldBack = 0;
    dTkT_InteractionManager->schedulingStats.accumKinematicLagSteps = 0;
    dT->nTotalSteps = 0;
}

float DEMSolver::dTInspectReduce(const std::shared_ptr<jitify::Program>& inspection_kernel,
                                 const std::string& kernel_name,
                                 INSPECT_ENTITY_TYPE thing_to_insp,
                                 CUB_REDUCE_FLAVOR reduce_flavor,
                                 bool all_domain) {
    // Note they are currently running in the device associated with the main, but it's not a big issue
    //// TODO: Think about the implication on using more than 2 GPUs
    float* pRes = dT->inspectCall(inspection_kernel, kernel_name, thing_to_insp, reduce_flavor, all_domain);
    return (float)(*pRes);
}

float* DEMSolver::dTInspectNoReduce(const std::shared_ptr<jitify::Program>& inspection_kernel,
                                    const std::string& kernel_name,
                                    INSPECT_ENTITY_TYPE thing_to_insp,
                                    CUB_REDUCE_FLAVOR reduce_flavor,
                                    bool all_domain) {
    // Note they are currently running in the device associated with the main, but it's not a big issue
    //// TODO: Think about the implication on using more than 2 GPUs
    float* pRes = dT->inspectCall(inspection_kernel, kernel_name, thing_to_insp, reduce_flavor, all_domain);
    return pRes;
}

}  // namespace deme
