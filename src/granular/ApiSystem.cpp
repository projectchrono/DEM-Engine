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

SGPS_api::SGPS_api(float rad) {
    dTkT_InteractionManager = new ThreadManager();
    dTkT_InteractionManager->dynamicRequestedUpdateFrequency = updateFreq;

    kT = new kinematicThread(dTkT_InteractionManager);
    dT = new dynamicThread(dTkT_InteractionManager);

    // gpuManager = new GpuManager(1);

    voxelID_ts* pBuffer = kT->pBuffer_voxelID();
    dT->setDestinationBuffer_voxelID(pBuffer);
    dT->setDynamicAverageTime(timeDynamicSide);
    dT->setNDynamicCycles(nDynamicCycles);

    pBuffer = dT->pBuffer_voxelID();
    kT->setDestinationBuffer_voxelID(pBuffer);
    kT->primeDynamic();
    kT->setKinematicAverageTime(timeKinematicSide);
}

SGPS_api::~SGPS_api() {
    // delete m_sys;
}

unsigned int SGPS_api::LoadClumpType(std::vector<float> sp_radii,
                                     std::vector<float> sp_location_x,
                                     std::vector<float> sp_location_y,
                                     std::vector<float> sp_location_z,
                                     std::vector<float> sp_density) {
    m_clumps_sp_radii.push_back(sp_radii);

    return m_clumps_sp_radii.size() - 1;
}

voxelID_ts SGPS_api::GetClumpVoxelID(unsigned int i) {
    return dT->voxelID.at(i);
}

int SGPS_api::LaunchThreads() {
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
