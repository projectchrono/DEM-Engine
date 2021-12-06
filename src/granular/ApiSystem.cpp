//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <granular/ApiSystem.h>
#include <granular/GranularDefines.h>

#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstring>

namespace sgps {

DEMSolver::DEMSolver(float rad) {
    dTkT_InteractionManager = new ThreadManager();
    dTkT_InteractionManager->dynamicRequestedUpdateFrequency = updateFreq;

    dTkT_GpuManager = new GpuManager(2);

    kT = new DEMKinematicThread(dTkT_InteractionManager, dTkT_GpuManager);
    dT = new DEMDynamicThread(dTkT_InteractionManager, dTkT_GpuManager);

    dT->setDynamicAverageTime(timeDynamicSide);
    dT->setNDynamicCycles(nDynamicCycles);

    kT->primeDynamic();
    kT->setKinematicAverageTime(timeKinematicSide);
}

DEMSolver::~DEMSolver() {
    delete kT;
    delete dT;
    delete dTkT_InteractionManager;
    delete dTkT_GpuManager;
}

void DEMSolver::InstructBoxDomainNumVoxel(unsigned char x, unsigned char y, unsigned char z, float len_unit, float3 O) {
    if (x + y + z != sizeof(voxelID_t) * BITS_PER_BYTE) {
        SGPS_ERROR("Please give voxel numbers (as powers of 2) along each direction such that they add up to %zu.",
                   sizeof(voxelID_t) * BITS_PER_BYTE);
    }
    l = len_unit;
    nvXp2 = x;
    nvYp2 = y;
    nvZp2 = z;
    m_boxLBF = O;

    // Calculating ``world'' size by the input nvXp2 and l
    m_voxelSize = (double)pow(2, VOXEL_RES_POWER2) * (double)l;
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

void DEMSolver::SetGravitationalAcceleration(float3 g) {
    G = g;
}

void DEMSolver::SetTimeStepSize(double ts_size) {
    m_ts_size = ts_size;
}

unsigned int DEMSolver::LoadMaterialType(float density, float E) {
    struct DEMMaterial a_material;
    a_material.density = density;
    a_material.E = E;

    m_sp_materials.push_back(a_material);
    return m_sp_materials.size() - 1;
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
    float smallest_radius = FLT_MAX;
    for (auto elem : m_template_sp_radii) {
        for (auto radius : elem) {
            if (radius < smallest_radius) {
                smallest_radius = radius;
            }
        }
    }

    // What should be a default bin size?
    m_binSize = 4.0 * smallest_radius;
}

int DEMSolver::generateJITResources() {
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
    nMatTuples_computed = m_template_sp_mat_ids.size();

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

    uint64_t num_bins =
        (uint64_t)(m_boxX / m_binSize + 1) * (uint64_t)(m_boxY / m_binSize + 1) * (uint64_t)(m_boxZ / m_binSize + 1);
    std::cout << "The edge length of a bin: " << m_binSize << std::endl;
    std::cout << "The total number of bins: " << num_bins << std::endl;

    // Figure out the initial profile/status of clumps, and related quantities, if need to
    nClumpBodies = m_input_clump_types.size();

    // now a quick hack using for loop
    // I'll have to change that to be more... efficient
    nSpheresGM = 0;
    for (size_t i = 0; i < m_input_clump_types.size(); i++) {
        auto this_type_num = m_input_clump_types.at(i);
        auto this_radii = m_template_sp_radii.at(this_type_num);
        nSpheresGM += this_radii.size();
    }

    // Compile the kernels if there are some can be compiled now

    return 0;
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

void DEMSolver::WriteFileAsSpheres(const std::string& outfilename) const {
    std::ofstream ptFile(outfilename, std::ios::out);  // std::ios::binary?
    dT->WriteCsvAsSpheres(ptFile);
}

/// Transfer (CPU-side) cached simulation data (about sim world) to the GPU-side. It is called automatically during
/// system initialization.
void DEMSolver::transferSimParams() {
    dT->setSimParams(nvXp2, nvYp2, nvZp2, l, m_voxelSize, m_binSize, m_boxLBF, G, m_ts_size, m_expand_factor);
    kT->setSimParams(nvXp2, nvYp2, nvZp2, l, m_voxelSize, m_binSize, m_boxLBF, G, m_ts_size, m_expand_factor);
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
    dT->populateManagedArrays(m_input_clump_types, m_input_clump_xyz, m_input_clump_vel, m_template_mass,
                              m_template_sp_radii, m_template_sp_relPos);
    kT->populateManagedArrays(m_input_clump_types, m_input_clump_xyz, m_input_clump_vel, m_template_mass,
                              m_template_sp_radii, m_template_sp_relPos);
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

    // TODO: Add check for inputs sizes (nClumps, nSpheres, nMat, nTopo...)
}

// The method should be called after user inputs are in place, and before starting the simulation. It figures out a part
// of the required simulation information such as the scale of the poblem domain, and makes sure these info live in
// managed memory.
int DEMSolver::Initialize() {
    // Call the JIT compiler generator to make prep for this simulation.
    generateJITResources();

    // A few checks first.
    validateUserInputs();

    // Transfer some simulation params to implementation level
    transferSimParams();

    // Allocate and populate kT dT managed arrays
    initializeArrays();

    // Put sim data array pointers in place
    packDataPointers();

    sys_initialized = true;
    return 0;
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

int DEMSolver::LaunchThreads() {
    // Is it needed here??
    // dT->packDataPointers(kT->granData);

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

}  // namespace sgps
