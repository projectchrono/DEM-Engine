//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <cstring>
#include <granular/GranularDefines.h>
#include <granular/PhysicsSystem.h>
#include <iostream>
#include <thread>

namespace sgps {

SGPS_impl::SGPS_impl(float rad) : sphereUU(rad) {
    dTkT_InteractionManager = new ThreadManager();
    dTkT_InteractionManager->dynamicRequestedUpdateFrequency = updateFreq;

    kT = new kinematicThread(dTkT_InteractionManager);
    dT = new dynamicThread(dTkT_InteractionManager);

    int* pBuffer = kT->pDestinationBuffer();
    dT->setDestinationBuffer(pBuffer);
    dT->setDynamicAverageTime(timeDynamicSide);
    dT->setNDynamicCycles(nDynamicCycles);

    pBuffer = dT->pDestinationBuffer();
    kT->setDestinationBuffer(pBuffer);
    kT->primeDynamic();
    kT->setKinematicAverageTime(timeKinematicSide);

    // Sim statistics
    std::cout << "\n~~ SIM STATISTICS ~~\n";
    std::cout << "Number of dynamic updates: " << dTkT_InteractionManager->schedulingStats.nDynamicUpdates << std::endl;
    std::cout << "Number of kinematic updates: " << dTkT_InteractionManager->schedulingStats.nKinematicUpdates
              << std::endl;
    std::cout << "Number of times dynamic held back: " << dTkT_InteractionManager->schedulingStats.nTimesDynamicHeldBack
              << std::endl;
    std::cout << "Number of times kinematic held back: "
              << dTkT_InteractionManager->schedulingStats.nTimesKinematicHeldBack << std::endl;
}

SGPS_impl::~SGPS_impl(){};

int SGPS_impl::LaunchThreads() {
    // get the threads going
    std::thread kThread(std::ref(*kT));
    std::thread dThread(std::ref(*dT));

    dThread.join();
    kThread.join();

    return 0;    
}

void kinematicThread::primeDynamic() {
    // produce produce to prime the pipeline
    for (int j = 0; j < N_INPUT_ITEMS; j++) {
        product[j] += this->costlyProductionStep(j);
    }

    // transfer produce to dynamic buffer
    memcpy(pDynamicOwned_TransfBuffer, product, N_INPUT_ITEMS * sizeof(int));
    pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = true;
    pSchedSupport->schedulingStats.nDynamicUpdates++;
}

int kinematicThread::costlyProductionStep(int val) const {
    std::this_thread::sleep_for(std::chrono::milliseconds(kinematicAverageTime));
    return 2 * val + 1;
}

}  // namespace sgps
