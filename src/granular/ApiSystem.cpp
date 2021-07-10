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

    voxelID_t* pBuffer = kT->pBuffer_voxelID();
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

materialsOffset_t SGPS::LoadMaterialType(float density, float E) {
    struct Material a_material;
    a_material.density = density;
    a_material.E = E;

    m_sp_materials.push_back(a_material);
    return m_sp_materials.size() - 1;
}

clumpBodyInertiaOffset_t SGPS::LoadClumpType(const std::vector<float>& sp_radii,
                                             const std::vector<float>& sp_locations_x,
                                             const std::vector<float>& sp_locations_y,
                                             const std::vector<float>& sp_locations_z,
                                             const std::vector<materialsOffset_t>& sp_material_ids) {
    m_clumps_sp_radii.push_back(sp_radii);

    return m_clumps_sp_radii.size() - 1;
}

voxelID_t SGPS::GetClumpVoxelID(unsigned int i) const {
    return dT->voxelID.at(i);
}

int SGPS::generateJITResources() {
    return 0;
}

int SGPS::Initialize() {
    generateJITResources();

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
