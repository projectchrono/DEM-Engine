//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <granular/ApiSystem.h>
#include <granular/GranularDefines.h>

#include <iostream>
#include <thread>
#include <cstring>

namespace sgps {

SGPS::SGPS(float rad) {
    dTkT_InteractionManager = new ThreadManager();
    dTkT_InteractionManager->dynamicRequestedUpdateFrequency = updateFreq;

    dTkT_GpuManager = new GpuManager(2);

    kT = new kinematicThread(dTkT_InteractionManager, dTkT_GpuManager);
    dT = new dynamicThread(dTkT_InteractionManager, dTkT_GpuManager);

    voxelID_default_t* pBuffer = kT->pBuffer_voxelID();
    dT->setDestinationBuffer_voxelID(pBuffer);
    dT->setDynamicAverageTime(timeDynamicSide);
    dT->setNDynamicCycles(nDynamicCycles);

    pBuffer = dT->pBuffer_voxelID();
    kT->setDestinationBuffer_voxelID(pBuffer);
    kT->primeDynamic();
    kT->setKinematicAverageTime(timeKinematicSide);
}

SGPS::~SGPS() {
    delete dTkT_InteractionManager;
    delete dTkT_GpuManager;
    delete kT;
    delete dT;
}

materialsOffset_default_t SGPS::LoadMaterialType(float density, float E) {
    struct Material a_material;
    a_material.density = density;
    a_material.E = E;

    m_sp_materials.push_back(a_material);
    return m_sp_materials.size() - 1;
}

clumpBodyInertiaOffset_default_t SGPS::LoadClumpType(float mass,
                                                     float3 moi,
                                                     const std::vector<float>& sp_radii,
                                                     const std::vector<float3>& sp_locations_xyz,
                                                     const std::vector<materialsOffset_default_t>& sp_material_ids) {
    auto l = sp_radii.size();
    if (l != sp_locations_xyz.size() || l != sp_material_ids.size()) {
        SGPS_ERROR("Arrays defining a clump topology type must all have the same length.");
    }

    m_clumps_mass.push_back(mass);
    m_clumps_moi.push_back(moi);
    m_clumps_sp_radii.push_back(sp_radii);
    m_clumps_sp_location_xyz.push_back(sp_locations_xyz);
    m_clumps_sp_material_ids.push_back(sp_material_ids);

    return m_clumps_mass.size() - 1;
}

clumpBodyInertiaOffset_default_t SGPS::LoadClumpSimpleSphere(float mass,
                                                             float radius,
                                                             materialsOffset_default_t material_id) {
    float3 I = make_float3(2.0 / 5.0 * mass * radius * radius);
    float3 pos = make_float3(0);
    return LoadClumpType(mass, I, std::vector<float>(1, radius), std::vector<float3>(1, pos),
                         std::vector<materialsOffset_default_t>(1, material_id));
}

voxelID_default_t SGPS::GetClumpVoxelID(unsigned int i) const {
    return dT->voxelID.at(i);
}

int SGPS::generateJITResources() {
    size_t num_clump_types = m_clumps_mass.size();
    // Put unique clump mass values in a set.
    m_clumps_mass_types.insert(m_clumps_mass.begin(), m_clumps_mass.end());
    for (size_t i = 0; i < num_clump_types; i++) {
        // Put unique sphere radii values in a set.
        m_clumps_sp_radii_types.insert(m_clumps_sp_radii.at(i).begin(), m_clumps_sp_radii.at(i).end());
        // Put unique clump sphere component locations in a set.
        m_clumps_sp_location_types.insert(m_clumps_sp_location_xyz.at(i).begin(), m_clumps_sp_location_xyz.at(i).end());
    }
    // Now rearrange so the original input mass and sphere radii are now stored as the offsets to their respective
    // uniques sets.
    for (size_t i = 0; i < num_clump_types; i++) {
        m_clumps_mass_type_offset.push_back(
            std::distance(m_clumps_mass_types.begin(), m_clumps_mass_types.find(m_clumps_mass.at(i))));
        std::vector<distinctSphereRadiiOffset_default_t> sp_radii_type_offset(m_clumps_sp_radii.at(i).size(), 0);
        std::vector<distinctSphereRelativePositions_default_t> sp_location_type_offset(
            m_clumps_sp_location_xyz.at(i).size(), 0);
        for (size_t j = 0; j < sp_radii_type_offset.size(); j++) {
            sp_radii_type_offset.at(j) = std::distance(m_clumps_sp_radii_types.begin(),
                                                       m_clumps_sp_radii_types.find(m_clumps_sp_radii.at(i).at(j)));
            sp_location_type_offset.at(j) =
                std::distance(m_clumps_sp_location_types.begin(),
                              m_clumps_sp_location_types.find(m_clumps_sp_location_xyz.at(i).at(j)));
        }
        m_clumps_sp_radii_type_offset.push_back(sp_radii_type_offset);
        m_clumps_sp_location_type_offset.push_back(sp_location_type_offset);
    }

    // Compile the magic number header.
    nDistinctSphereRadii_computed = m_clumps_sp_radii_types.size();
    nDistinctClumpBodyTopologies_computed = m_clumps_mass_types.size();
    nMatTuples_computed = m_sp_materials.size();
    nDistinctSphereRelativePositions_computed = m_clumps_sp_location_types.size();
    // std::cout << nDistinctClumpBodyTopologies_computed << std::endl;
    // std::cout << nDistinctSphereRadii_computed << std::endl;
    // std::cout << nDistinctSphereRelativePositions_computed << std::endl;
    // for (int i = 0; i < m_clumps_sp_location_type_offset.size(); i++) {
    //     for (int j = 0; j < m_clumps_sp_location_type_offset.at(i).size(); j++) {
    //        std::cout << m_clumps_sp_location_type_offset.at(i).at(j) << " ";
    //    }
    //    std::cout << std::endl;
    // }

    // Compile the kernels needed.

    return 0;
}

void SGPS::SetClumps(const std::vector<clumpBodyInertiaOffset_default_t>& types, const std::vector<float3>& xyz) {
    if (types.size() != xyz.size()) {
        SGPS_ERROR("Arrays in the call SetClumps must all have the same length.");
    }

    m_input_clump_types.insert(m_input_clump_types.end(), types.begin(), types.end());
    m_input_clump_xyz.insert(m_input_clump_xyz.end(), xyz.begin(), xyz.end());
}

int SGPS::Initialize() {
    // a few error checks first
    if (m_sp_materials.size() == 0) {
        SGPS_ERROR("Before initializing the system, at least one material type should be loaded via LoadMaterialType.");
    }

    // Figure out a part of the required simulation information such as the scale of the poblem domain. Make sure these
    // info live in managed memory.

    // Call the JIT compiler generator to make prep for this simulation.
    generateJITResources();

    // Now that the CUDA-related functions and data types are JITCompiled, we can feed those GPU-side arrays with the
    // cached API-level simulation info.
    dT->populateManagedArrays(m_input_clump_types, m_input_clump_xyz, m_clumps_mass_types, m_clumps_sp_radii_types,
                              m_clumps_sp_location_types, m_clumps_mass_type_offset, m_clumps_sp_radii_type_offset,
                              m_clumps_sp_location_type_offset);

    return 0;
}

int SGPS::LaunchThreads() {
    // get the threads going
    std::thread kThread(std::ref(*kT));
    std::thread dThread(std::ref(*dT));

    dThread.join();
    kThread.join();

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
