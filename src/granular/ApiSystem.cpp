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

    voxelID_default_t* pBuffer = kT->pBuffer_voxelID();
    dT->setDestinationBuffer_voxelID(pBuffer);
    dT->setDynamicAverageTime(timeDynamicSide);
    dT->setNDynamicCycles(nDynamicCycles);

    pBuffer = dT->pBuffer_voxelID();
    kT->setDestinationBuffer_voxelID(pBuffer);
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
    if (x + y + z != sizeof(voxelID_default_t) * BITS_PER_BYTE) {
        SGPS_ERROR("Please give voxel numbers (as powers of 2) along each direction such that they add up to %zu.",
                   sizeof(voxelID_default_t) * BITS_PER_BYTE);
    }
    l = len_unit;
    nvXp2 = x;
    nvYp2 = y;
    nvZp2 = z;
    m_boxLBF = O;

    // Calculating ``world'' size by the input nvXp2 and l
    m_voxelSize = (double)pow(2, VOXEL_RES_POWER2) * (double)l;
    m_boxX = m_voxelSize * (double)(1u << x);
    m_boxY = m_voxelSize * (double)(1u << y);
    m_boxZ = m_voxelSize * (double)(1u << z);
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

void DEMSolver::SetGravitationalAcceleration(float3 g) {
    G = g;
}

void DEMSolver::SetTimeStepSize(double ts_size) {
    m_ts_size = ts_size;
}

materialsOffset_default_t DEMSolver::LoadMaterialType(float density, float E) {
    struct Material a_material;
    a_material.density = density;
    a_material.E = E;

    m_sp_materials.push_back(a_material);
    return m_sp_materials.size() - 1;
}

clumpBodyInertiaOffset_default_t DEMSolver::LoadClumpType(
    float mass,
    float3 moi,
    const std::vector<float>& sp_radii,
    const std::vector<float3>& sp_locations_xyz,
    const std::vector<materialsOffset_default_t>& sp_material_ids) {
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

clumpBodyInertiaOffset_default_t DEMSolver::LoadClumpSimpleSphere(float mass,
                                                                  float radius,
                                                                  materialsOffset_default_t material_id) {
    float3 I = make_float3(2.0 / 5.0 * mass * radius * radius);
    float3 pos = make_float3(0);
    return LoadClumpType(mass, I, std::vector<float>(1, radius), std::vector<float3>(1, pos),
                         std::vector<materialsOffset_default_t>(1, material_id));
}

voxelID_default_t DEMSolver::GetClumpVoxelID(unsigned int i) const {
    return dT->voxelID.at(i);
}

// Figure out the unit length l and corresponding numbers of voxels along each direction, based on domain size X, Y, Z
int DEMSolver::figureOutNV() {
    if (m_boxX <= 0.f || m_boxY <= 0.f || m_boxZ <= 0.f) {
        SGPS_ERROR(
            "The size of the simulation world is set to be (or default to be) %f by %f by %f. It is impossibly small.",
            m_boxX, m_boxY, m_boxZ);
    }

    return 0;
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
    std::cout << "The dimension of the simulation world: " << m_boxX << ", " << m_boxY << ", " << m_boxZ << std::endl;
    std::cout << "The length unit in this simulation is: " << l << std::endl;
    std::cout << "The edge length of a voxel: " << m_voxelSize << std::endl;

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

void DEMSolver::SetClumps(const std::vector<clumpBodyInertiaOffset_default_t>& types, const std::vector<float3>& xyz) {
    if (types.size() != xyz.size()) {
        SGPS_ERROR("Arrays in the call SetClumps must all have the same length.");
    }

    // clump_xyz are effectively the xyz of the CoM
    m_input_clump_types.insert(m_input_clump_types.end(), types.begin(), types.end());
    m_input_clump_xyz.insert(m_input_clump_xyz.end(), xyz.begin(), xyz.end());
}

void DEMSolver::WriteFileAsSpheres(const std::string& outfilename) const {
    std::ofstream ptFile(outfilename, std::ios::out);  // std::ios::binary?
    dT->WriteCsvAsSpheres(ptFile);
}

void DEMSolver::transferSimParams() {
    dT->setSimParams(nvXp2, nvYp2, nvZp2, l, m_voxelSize, m_boxLBF, G, m_ts_size);
}

int DEMSolver::Initialize() {
    // a few error checks first
    if (m_sp_materials.size() == 0) {
        SGPS_ERROR("Before initializing the system, at least one material type should be loaded via LoadMaterialType.");
    }

    // Figure out a part of the required simulation information such as the scale of the poblem domain. Make sure these
    // info live in managed memory.

    // Call the JIT compiler generator to make prep for this simulation.
    generateJITResources();

    // Transfer some simulation params to implementation level
    transferSimParams();

    // Resize managed arrays based on the statistical data we had from the previous step
    dT->allocateManagedArrays(nClumpBodies, nSpheresGM, nDistinctClumpBodyTopologies_computed,
                              nDistinctClumpComponents_computed, nMatTuples_computed);

    // Now that the CUDA-related functions and data types are JITCompiled, we can feed those GPU-side arrays with the
    // cached API-level simulation info.
    dT->populateManagedArrays(m_input_clump_types, m_input_clump_xyz, m_template_mass, m_template_sp_radii,
                              m_template_sp_relPos);

    // Put sim data array pointers in place
    dT->packDataPointers();

    sys_initialized = true;
    return 0;
}

void DEMSolver::UpdateSimParams() {
    // TODO: maybe some massaging is needed here, for more sophisticated params updates
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
    dT->packDataPointers();

    dT->startThread();
    kT->startThread();

    // We have to wait until these 2 threads finish their job before moving on.
    waitOnThreads();

    // Sim statistics
    std::cout << "\n~~ SIM STATISTICS ~~\n";
    std::cout << "Number of dynamic updates: " << dTkT_InteractionManager->schedulingStats.nDynamicUpdates << std::endl;
    std::cout << "Number of kinematic updates: " << dTkT_InteractionManager->schedulingStats.nKinematicUpdates
              << std::endl;
    std::cout << "Number of times dynamic held back: " << dTkT_InteractionManager->schedulingStats.nTimesDynamicHeldBack
              << std::endl;
    std::cout << "Number of times kinematic held back: "
              << dTkT_InteractionManager->schedulingStats.nTimesKinematicHeldBack << std::endl;

    return 0;
}

}  // namespace sgps
