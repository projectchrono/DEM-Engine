//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <granular/ApiSystem.h>
#include <granular/GranularDefines.h>
#include <granular/HostSideHelpers.cpp>

#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstring>

namespace sgps {

DEMSolver::DEMSolver(unsigned int nGPUs) {
    dTkT_InteractionManager = new ThreadManager();
    // dTkT_InteractionManager->dynamicRequestedUpdateFrequency = m_updateFreq;

    dTkT_GpuManager = new GpuManager(nGPUs);

    kT = new DEMKinematicThread(dTkT_InteractionManager, dTkT_GpuManager);
    dT = new DEMDynamicThread(dTkT_InteractionManager, dTkT_GpuManager);

    // dT->setNDynamicCycles(nDynamicCycles);

    kT->primeDynamic();
}

DEMSolver::~DEMSolver() {
    delete kT;
    delete dT;
    delete dTkT_InteractionManager;
    delete dTkT_GpuManager;
}

void DEMSolver::InstructBoxDomainNumVoxel(unsigned char x, unsigned char y, unsigned char z, float len_unit, float3 O) {
    if (x + y + z != sizeof(voxelID_t) * SGPS_BITS_PER_BYTE) {
        SGPS_ERROR("Please give voxel numbers (as powers of 2) along each direction such that they add up to %zu.",
                   sizeof(voxelID_t) * SGPS_BITS_PER_BYTE);
    }
    l = len_unit;
    nvXp2 = x;
    nvYp2 = y;
    nvZp2 = z;
    m_boxLBF = O;

    // Calculating ``world'' size by the input nvXp2 and l
    m_voxelSize = (double)((size_t)1 << VOXEL_RES_POWER2) * (double)l;
    m_boxX = m_voxelSize * (double)((size_t)1 << x);
    m_boxY = m_voxelSize * (double)((size_t)1 << y);
    m_boxZ = m_voxelSize * (double)((size_t)1 << z);
    explicit_nv_override = true;
}

float3 DEMSolver::CenterCoordSys() {
    float3 O;
    O.x = -(m_boxX) / 2.0;
    O.y = -(m_boxY) / 2.0;
    O.z = -(m_boxZ) / 2.0;
    m_boxLBF = O;
    return O;
}

void DEMSolver::SetExpandFactor(float beta) {
    m_expand_factor = beta;
}

void DEMSolver::SuggestExpandFactor(float max_vel, float max_time_per_CD) {
    m_expand_factor = max_vel * max_time_per_CD;
}

void DEMSolver::SuggestExpandFactor(float max_vel) {
    if (m_ts_size <= 0.0) {
        SGPS_ERROR(
            "Please set the constant time step size before calling this method, or supplying both the maximum expect "
            "velocity AND maximum time between contact detections as arguments.");
    }
    if (m_updateFreq == 0) {
        SGPS_ERROR(
            "Please set contact detection frequency via SetCDUpdateFreq before calling this method, or supplying both "
            "the maximum expect velocity AND maximum time between contact detections as arguments.");
    }
    DEMSolver::SuggestExpandFactor(max_vel, m_ts_size * m_updateFreq);
}

void DEMSolver::SuggestExpandSafetyParam(float param) {
    m_expand_safety_param = param;
}

void DEMSolver::SetGravitationalAcceleration(float3 g) {
    G = g;
}

void DEMSolver::SetTimeStepSize(double ts_size) {
    m_ts_size = ts_size;
}

unsigned int DEMSolver::LoadMaterialType(float E, float nu, float CoR, float density) {
    unsigned int mat_num = m_sp_materials.size();
    if (CoR < SGPS_DEM_TINY_FLOAT) {
        std::cout << "\nWARNING! Material type " << mat_num
                  << " is set (or defaulted) to have 0 restitution. Please make sure this is intentional.\n"
                  << std::endl;
    }
    if (CoR > 1.f) {
        std::cout << "\nWARNING! Material type " << mat_num
                  << " is set to have a restitution coefficient larger than 1. This is typically not physical and "
                     "should destabilize the simulation.\n"
                  << std::endl;
    }

    struct DEMMaterial a_material;
    a_material.density = density;
    a_material.E = E;
    a_material.nu = nu;
    a_material.CoR = CoR;

    m_sp_materials.push_back(a_material);
    return mat_num;
}

unsigned int DEMSolver::LoadClumpType(float mass,
                                      float3 moi,
                                      const std::vector<float>& sp_radii,
                                      const std::vector<float3>& sp_locations_xyz,
                                      const std::vector<unsigned int>& sp_material_ids) {
    auto len = sp_radii.size();
    if (len != sp_locations_xyz.size() || len != sp_material_ids.size()) {
        SGPS_ERROR("Arrays defining a clump topology type must all have the same length.");
    }

    m_template_mass.push_back(mass);
    m_template_moi.push_back(moi);
    m_template_sp_radii.push_back(sp_radii);
    m_template_sp_relPos.push_back(sp_locations_xyz);
    m_template_sp_mat_ids.push_back(sp_material_ids);

    return m_template_mass.size() - 1;
}

unsigned int DEMSolver::LoadClumpSimpleSphere(float mass, float radius, unsigned int material_id) {
    float3 I = make_float3(2.0 / 5.0 * mass * radius * radius);
    float3 pos = make_float3(0);
    return LoadClumpType(mass, I, std::vector<float>(1, radius), std::vector<float3>(1, pos),
                         std::vector<unsigned int>(1, material_id));
}

voxelID_t DEMSolver::GetClumpVoxelID(unsigned int i) const {
    return dT->voxelID.at(i);
}

void DEMSolver::figureOutNV() {
    if (m_boxX <= 0.f || m_boxY <= 0.f || m_boxZ <= 0.f) {
        SGPS_ERROR(
            "The size of the simulation world is set to be (or default to be) %f by %f by %f. It is impossibly small.",
            m_boxX, m_boxY, m_boxZ);
    }
}

void DEMSolver::decideDefaultBinSize() {
    // find the smallest radius
    for (auto elem : m_template_sp_radii) {
        for (auto radius : elem) {
            if (radius < m_smallest_radius) {
                m_smallest_radius = radius;
            }
        }
    }

    // What should be a default bin size?
    m_binSize = 1.0 * m_smallest_radius;
}

void DEMSolver::figureOutMaterialProxies() {
    // Use the info in m_sp_materials to populate API-side proxy arrays
    // These arrays are later passed to kTdT in populateManagedArrays
    unsigned int count = (1 + m_sp_materials.size()) * m_sp_materials.size() / 2;
    m_E_proxy.resize(count);
    m_G_proxy.resize(count);
    m_CoR_proxy.resize(count);
    for (unsigned int i = 0; i < m_sp_materials.size(); i++) {
        for (unsigned int j = i; j < m_sp_materials.size(); j++) {
            auto Mat1 = m_sp_materials.at(i);
            auto Mat2 = m_sp_materials.at(j);
            float E_eff, G_eff;
            materialProxyMaterixCalculator(E_eff, G_eff, Mat1.E, Mat1.nu, Mat2.E, Mat2.nu);
            unsigned int entry_num = locateMatPair<unsigned int>(i, j);
            m_E_proxy.at(entry_num) = E_eff;
            m_G_proxy.at(entry_num) = G_eff;
            m_CoR_proxy.at(entry_num) = std::min(Mat1.CoR, Mat2.CoR);
        }
    }
}

void DEMSolver::generateJITResources() {
    /*
    // Dan and Ruochun decided not to extract unique input values.
    // Instead, we trust users: we simply store all clump template info users give.
    // So the unique-value-extractor block is disabled and commented.
    size_t input_num_clump_types = m_template_mass.size();
    // Put unique clump mass values in a set.
    m_template_mass_types.insert(m_template_mass.begin(), m_template_mass.end());
    for (size_t i = 0; i < input_num_clump_types; i++) {
        // Put unique sphere radii values in a set.
        m_template_sp_radii_types.insert(m_template_sp_radii.at(i).begin(), m_template_sp_radii.at(i).end());
        // Put unique clump sphere component locations in a set.
        m_clumps_sp_location_types.insert(m_template_sp_relPos.at(i).begin(), m_template_sp_relPos.at(i).end());
    }
    // Now rearrange so the original input mass and sphere radii are now stored as the offsets to their respective
    // uniques sets.
    for (size_t i = 0; i < input_num_clump_types; i++) {
        m_template_mass_type_offset.push_back(
            std::distance(m_template_mass_types.begin(), m_template_mass_types.find(m_template_mass.at(i))));
        std::vector<distinctSphereRadiiOffset_default_t> sp_radii_type_offset(m_template_sp_radii.at(i).size(), 0);
        std::vector<distinctSphereRelativePositions_default_t> sp_location_type_offset(
            m_template_sp_relPos.at(i).size(), 0);
        for (size_t j = 0; j < sp_radii_type_offset.size(); j++) {
            sp_radii_type_offset.at(j) = std::distance(m_template_sp_radii_types.begin(),
                                                       m_template_sp_radii_types.find(m_template_sp_radii.at(i).at(j)));
            sp_location_type_offset.at(j) =
                std::distance(m_clumps_sp_location_types.begin(),
                              m_clumps_sp_location_types.find(m_template_sp_relPos.at(i).at(j)));
        }
        m_template_sp_radii_type_offset.push_back(sp_radii_type_offset);
        m_clumps_sp_location_type_offset.push_back(sp_location_type_offset);
    }

    nDistinctClumpBodyTopologies_computed = m_template_mass_types.size();
    nMatTuples_computed = m_sp_materials.size();

    nDistinctSphereRadii_computed = m_template_sp_radii_types.size();
    nDistinctSphereRelativePositions_computed = m_clumps_sp_location_types.size();
    */

    // Compile the magic number header.
    nDistinctClumpComponents_computed = 0;
    nDistinctClumpBodyTopologies_computed = m_template_mass.size();
    for (unsigned int i = 0; i < nDistinctClumpBodyTopologies_computed; i++) {
        nDistinctClumpComponents_computed += m_template_sp_radii.at(i).size();
    }
    nMatTuples_computed = m_sp_materials.size();
    // IF these "computed" numbers are larger than types like materialsOffset_t can hold, then we should error out and
    // let the user re-compile (or, should we somehow change the header automatically?)

    // nDistinctSphereRadii_computed = m_template_sp_radii_types.size();
    // nDistinctSphereRelativePositions_computed = m_clumps_sp_location_types.size();
    // std::cout << nDistinctClumpBodyTopologies_computed << std::endl;
    // std::cout << nDistinctSphereRadii_computed << std::endl;
    // std::cout << nDistinctSphereRelativePositions_computed << std::endl;
    // for (int i = 0; i < m_clumps_sp_location_type_offset.size(); i++) {
    //     for (int j = 0; j < m_clumps_sp_location_type_offset.at(i).size(); j++) {
    //        std::cout << m_clumps_sp_location_type_offset.at(i).at(j) << " ";
    //    }
    //    std::cout << std::endl;
    // }

    // Figure out the parameters related to the simulation ``world'', if need to
    if (!explicit_nv_override) {
        figureOutNV();
    }
    decideDefaultBinSize();
    std::cout << "The dimension of the simulation world: " << m_boxX << ", " << m_boxY << ", " << m_boxZ << std::endl;
    std::cout << "The length unit in this simulation is: " << l << std::endl;
    std::cout << "The edge length of a voxel: " << m_voxelSize << std::endl;

    nbX = (binID_t)(m_voxelSize * (double)((size_t)1 << nvXp2) / m_binSize) + 1;
    nbY = (binID_t)(m_voxelSize * (double)((size_t)1 << nvYp2) / m_binSize) + 1;
    nbZ = (binID_t)(m_voxelSize * (double)((size_t)1 << nvZp2) / m_binSize) + 1;
    uint64_t num_bins = (uint64_t)nbX * (uint64_t)nbY * (uint64_t)nbZ;
    // It's better to compute num of bins this way, rather than...
    // (uint64_t)(m_boxX / m_binSize + 1) * (uint64_t)(m_boxY / m_binSize + 1) * (uint64_t)(m_boxZ / m_binSize + 1);
    // because the space bins and voxels can cover may be larger than the user-defined sim domain
    std::cout << "The edge length of a bin: " << m_binSize << std::endl;
    std::cout << "The total number of bins: " << num_bins << std::endl;
    // TODO: should check if num_bins is larger than uint32, if uint32 is selected for storing binID

    // Figure out the initial profile/status of clumps, and related quantities, if need to
    nClumpBodies = m_input_clump_types.size();
    std::cout << "The current number of clumps: " << nClumpBodies << std::endl;

    // now a quick hack using for loop
    // I'll have to change that to be more... efficient
    nSpheresGM = 0;
    for (size_t i = 0; i < m_input_clump_types.size(); i++) {
        auto this_type_num = m_input_clump_types.at(i);
        auto this_radii = m_template_sp_radii.at(this_type_num);
        nSpheresGM += this_radii.size();
    }

    // Enlarge the expand factor if the user tells us to
    m_expand_factor *= m_expand_safety_param;
    if (m_expand_factor > 0.0) {
        std::cout << "All geometries are enlarged/thickened by " << m_expand_factor << " for contact detection purpose"
                  << std::endl;
        std::cout << "This in the case of smallest sphere, means enlarging radius by "
                  << (m_expand_factor / m_smallest_radius) * 100.0 << "%" << std::endl;
    }

    // Process the loaded materials
    std::cout << "The number of material types: " << nMatTuples_computed << std::endl;
    figureOutMaterialProxies();
}

void DEMSolver::SetClumps(const std::vector<unsigned int>& types, const std::vector<float3>& xyz) {
    if (types.size() != xyz.size()) {
        SGPS_ERROR("Arrays in the call SetClumps must all have the same length.");
    }

    // clump_xyz are effectively the xyz of the CoM
    m_input_clump_types.insert(m_input_clump_types.end(), types.begin(), types.end());
    m_input_clump_xyz.insert(m_input_clump_xyz.end(), xyz.begin(), xyz.end());
}

void DEMSolver::SetClumpVels(const std::vector<float3>& vel) {
    m_input_clump_vel.insert(m_input_clump_vel.end(), vel.begin(), vel.end());
}

void DEMSolver::SetClumpFamily(const std::vector<unsigned int>& code) {
    m_input_clump_family.insert(m_input_clump_family.end(), code.begin(), code.end());
}

void DEMSolver::WriteFileAsSpheres(const std::string& outfilename) const {
    std::ofstream ptFile(outfilename, std::ios::out);  // std::ios::binary?
    dT->WriteCsvAsSpheres(ptFile);
}

/// Transfer (CPU-side) cached simulation data (about sim world) to the GPU-side. It is called automatically during
/// system initialization.
void DEMSolver::transferSimParams() {
    dT->setSimParams(nvXp2, nvYp2, nvZp2, l, m_voxelSize, m_binSize, nbX, nbY, nbZ, m_boxLBF, G, m_ts_size,
                     m_expand_factor);
    kT->setSimParams(nvXp2, nvYp2, nvZp2, l, m_voxelSize, m_binSize, nbX, nbY, nbZ, m_boxLBF, G, m_ts_size,
                     m_expand_factor);
}

/// Transfer (CPU-side) cached clump templates info and initial clump type/position info to GPU-side arrays
void DEMSolver::initializeArrays() {
    // Resize managed arrays based on the statistical data we had from the previous step
    dT->allocateManagedArrays(nClumpBodies, nSpheresGM, nDistinctClumpBodyTopologies_computed,
                              nDistinctClumpComponents_computed, nMatTuples_computed);
    kT->allocateManagedArrays(nClumpBodies, nSpheresGM, nDistinctClumpBodyTopologies_computed,
                              nDistinctClumpComponents_computed, nMatTuples_computed);

    // Now that the CUDA-related functions and data types are JITCompiled, we can feed those GPU-side arrays with the
    // cached API-level simulation info.
    m_input_clump_vel.resize(m_input_clump_xyz.size(), make_float3(0));
    m_input_clump_family.resize(m_input_clump_xyz.size(), 0);
    dT->populateManagedArrays(m_input_clump_types, m_input_clump_xyz, m_input_clump_vel, m_input_clump_family,
                              m_template_sp_mat_ids, m_template_mass, m_template_moi, m_template_sp_radii,
                              m_template_sp_relPos, m_E_proxy, m_G_proxy, m_CoR_proxy);
    kT->populateManagedArrays(m_input_clump_types, m_input_clump_xyz, m_input_clump_vel, m_input_clump_family,
                              m_template_mass, m_template_sp_radii, m_template_sp_relPos);
}

void DEMSolver::packDataPointers() {
    dT->packDataPointers();
    kT->packDataPointers();
    // Each worker thread needs pointers used for data transfering. Note this step must be done after packDataPointers
    // are called, so each thread has its own pointers packed.
    dT->packTransferPointers(kT);
    kT->packTransferPointers(dT);
}

void DEMSolver::validateUserInputs() {
    if (m_sp_materials.size() == 0) {
        SGPS_ERROR("Before initializing the system, at least one material type should be loaded via LoadMaterialType.");
    }
    if (m_ts_size <= 0.0 && ts_size_is_const) {
        SGPS_ERROR(
            "Time step size is set to be %f. Please supply a positive number via SetTimeStepSize, or define the "
            "variable stepping properly.",
            m_ts_size);
    }
    if (m_expand_factor * m_expand_safety_param <= 0.0 && m_updateFreq > 0) {
        std::cout << "\nWARNING! You instructed that the physics can stretch " << m_updateFreq
                  << " time steps into the future, but did not instruct the geometries to expand via "
                     "SuggestExpandFactor. The contact detection procedure will likely fail to detect some contact "
                     "events before it is too late, hindering the simulation accuracy and stability."
                  << std::endl;
    }
    if (m_updateFreq < 0) {
        std::cout << "\nWARNING! The physics of the granular system can drift into the future as much as it wants "
                     "compared to contact detections, because SetCDUpdateFreq was called with a negative argument. "
                     "Please make sure this is intended."
                  << std::endl;
    }

    // TODO: Add check for inputs sizes (nClumps, nSpheres, nMat, nTopo...)
}

void DEMSolver::jitifyKernels() {
    std::unordered_map<std::string, std::string> templateSubs, simParamSubs, massMatSubs, familySubs;
    equipClumpTemplates(templateSubs);
    equipSimParams(simParamSubs);
    equipClumpMassMat(massMatSubs);
    kT->jitifyKernels(templateSubs, simParamSubs, massMatSubs, familySubs);
    dT->jitifyKernels(templateSubs, simParamSubs, massMatSubs, familySubs);
}

// The method should be called after user inputs are in place, and before starting the simulation. It figures out a part
// of the required simulation information such as the scale of the poblem domain, and makes sure these info live in
// managed memory.
int DEMSolver::Initialize() {
    // A few checks first.
    validateUserInputs();

    // Call the JIT compiler generator to make prep for this simulation.
    generateJITResources();

    // Transfer some simulation params to implementation level
    transferSimParams();

    // Allocate and populate kT dT managed arrays
    initializeArrays();

    // Put sim data array pointers in place
    packDataPointers();

    // Compile some of the kernels
    jitifyKernels();

    sys_initialized = true;
    return 0;
}

// TODO: it seems that for variable step size, it is the best not to do the computation of n cycles here; rather we
// should use a while loop to control that loop in worker threads.
inline size_t DEMSolver::computeDTCycles(double thisCallDuration) {
    return (size_t)std::round(thisCallDuration / m_ts_size);
}

/// Designed such that when (CPU-side) cached simulation data (about sim world, and clump templates) are updated by the
/// user, they can call this method to transfer them to the GPU-side in mid-simulation.
void DEMSolver::UpdateSimParams() {
    // TODO: transferSimParams() only transfers sim world info, not clump template info. Clump info transformation is
    // now in populateManagedArrays! Need to resolve that.
    transferSimParams();
}

void DEMSolver::waitOnThreads() {
    while (!(kT->isUserCallDone() & dT->isUserCallDone())) {
        std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_GRANULARITY_MS));
    }
    // Reset UserDone to false, make ready for the next user AdvanceSim call.
    kT->resetUserCallStat();
    dT->resetUserCallStat();
}

int DEMSolver::LaunchThreads(double thisCallDuration) {
    // Is it needed here??
    // dT->packDataPointers(kT->granData);

    // TODO: Return if nSphere==0

    // Tell dT how many iterations to go
    size_t nDTIters = computeDTCycles(thisCallDuration);
    dT->setNDynamicCycles(nDTIters);

    // Make sure dT kT understand the lock--waiting policy of this run
    dTkT_InteractionManager->dynamicRequestedUpdateFrequency = m_updateFreq;

    dT->startThread();
    kT->startThread();

    // We have to wait until these 2 threads finish their job before moving on.
    waitOnThreads();

    /*
    // Sim statistics
    std::cout << "\n~~ SIM STATISTICS ~~\n";
    std::cout << "Number of dynamic updates: " << dTkT_InteractionManager->schedulingStats.nDynamicUpdates << std::endl;
    std::cout << "Number of kinematic updates: " << dTkT_InteractionManager->schedulingStats.nKinematicUpdates
              << std::endl;
    std::cout << "Number of times dynamic held back: " << dTkT_InteractionManager->schedulingStats.nTimesDynamicHeldBack
              << std::endl;
    std::cout << "Number of times kinematic held back: "
              << dTkT_InteractionManager->schedulingStats.nTimesKinematicHeldBack << std::endl;
    */

    return 0;
}

inline void DEMSolver::equipClumpMassMat(std::unordered_map<std::string, std::string>& strMap) {
    strMap["_nDistinctClumpBodyTopologies_"] = std::to_string(nDistinctClumpBodyTopologies_computed);
    strMap["_nDistinctClumpComponents_"] = std::to_string(nDistinctClumpComponents_computed);
    strMap["_nActiveLoadingThreads_"] = std::to_string(NUM_ACTIVE_TEMPLATE_LOADING_THREADS);
    std::string ClumpMasses, moiX, moiY, moiZ;
    // loop through all templates to find in the JIT info
    for (unsigned int i = 0; i < nDistinctClumpBodyTopologies_computed; i++) {
        ClumpMasses += to_string_with_precision(m_template_mass.at(i)) + ",";
        moiX += to_string_with_precision(m_template_moi.at(i).x) + ",";
        moiY += to_string_with_precision(m_template_moi.at(i).y) + ",";
        moiZ += to_string_with_precision(m_template_moi.at(i).z) + ",";
    }
    strMap["_ClumpMasses_"] = ClumpMasses;
    strMap["_moiX_"] = moiX;
    strMap["_moiY_"] = moiY;
    strMap["_moiZ_"] = moiZ;
}

inline void DEMSolver::equipClumpTemplates(std::unordered_map<std::string, std::string>& strMap) {
    strMap["_nDistinctClumpBodyTopologies_"] = std::to_string(nDistinctClumpBodyTopologies_computed);
    strMap["_nDistinctClumpComponents_"] = std::to_string(nDistinctClumpComponents_computed);
    strMap["_nActiveLoadingThreads_"] = std::to_string(NUM_ACTIVE_TEMPLATE_LOADING_THREADS);
    std::string CDRadii, Radii, CDRelPosX, CDRelPosY, CDRelPosZ;
    // loop through all templates to find in the JIT info
    for (unsigned int i = 0; i < nDistinctClumpBodyTopologies_computed; i++) {
        for (unsigned int j = 0; j < m_template_sp_radii.at(i).size(); j++) {
            Radii += to_string_with_precision(m_template_sp_radii.at(i).at(j)) + ",";
            CDRadii += to_string_with_precision(m_template_sp_radii.at(i).at(j) + m_expand_factor) + ",";
            CDRelPosX += to_string_with_precision(m_template_sp_relPos.at(i).at(j).x) + ",";
            CDRelPosY += to_string_with_precision(m_template_sp_relPos.at(i).at(j).y) + ",";
            CDRelPosZ += to_string_with_precision(m_template_sp_relPos.at(i).at(j).z) + ",";
        }
    }
    strMap["_Radii_"] = Radii;
    strMap["_CDRadii_"] = CDRadii;
    strMap["_CDRelPosX_"] = CDRelPosX;
    strMap["_CDRelPosY_"] = CDRelPosY;
    strMap["_CDRelPosZ_"] = CDRelPosZ;
}

inline void DEMSolver::equipSimParams(std::unordered_map<std::string, std::string>& strMap) {
    strMap["_nvXp2_"] = std::to_string(nvXp2);
    strMap["_nvYp2_"] = std::to_string(nvYp2);
    strMap["_nvZp2_"] = std::to_string(nvZp2);

    strMap["_nbX_"] = std::to_string(nbX);
    strMap["_nbY_"] = std::to_string(nbY);
    strMap["_nbZ_"] = std::to_string(nbZ);

    // This l needs to be more accurate
    strMap["_l_"] = to_string_with_precision(l, 17);
    strMap["_voxelSize_"] = to_string_with_precision(m_voxelSize);
    strMap["_binSize_"] = to_string_with_precision(m_binSize);

    strMap["_nClumpBodies_"] = std::to_string(nClumpBodies);
    strMap["_nSpheresGM_"] = std::to_string(nSpheresGM);
    strMap["_nDistinctClumpBodyTopologies_"] = std::to_string(nDistinctClumpBodyTopologies_computed);
    strMap["_nDistinctClumpComponents_"] = std::to_string(nDistinctClumpComponents_computed);
    strMap["_nMatTuples_"] = std::to_string(nMatTuples_computed);

    strMap["_LBFX_"] = to_string_with_precision(m_boxLBF.x);
    strMap["_LBFY_"] = to_string_with_precision(m_boxLBF.y);
    strMap["_LBFZ_"] = to_string_with_precision(m_boxLBF.z);
    strMap["_Gx_"] = to_string_with_precision(G.x);
    strMap["_Gy_"] = to_string_with_precision(G.y);
    strMap["_Gz_"] = to_string_with_precision(G.z);

    strMap["_beta_"] = to_string_with_precision(m_expand_factor);
}

}  // namespace sgps
