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
  ThreadManager *dTkT_InteractionManager = new ThreadManager();
  dTkT_InteractionManager->dynamicRequestedUpdateFrequency = updateFreq;

  kinematicThread kT(dTkT_InteractionManager);
  dynamicThread dT(dTkT_InteractionManager);

  int *pBuffer = kT.pDestinationBuffer();
  dT.setDestinationBuffer(pBuffer);
  dT.setDynamicAverageTime(timeDynamicSide);
  dT.setNDynamicCycles(nDynamicCycles);

  pBuffer = dT.pDestinationBuffer();
  kT.setDestinationBuffer(pBuffer);
  kT.primeDynamic();
  kT.setKinematicAverageTime(timeKinematicSide);
  // get the threads going
  std::thread kThread(std::ref(kT));
  std::thread dThread(std::ref(dT));

  dThread.join();
  kThread.join();

  // Sim statistics
  std::cout << "\n~~ SIM STATISTICS ~~\n";
  std::cout << "Number of dynamic updates: "
            << dTkT_InteractionManager->schedulingStats.nDynamicUpdates
            << std::endl;
  std::cout << "Number of kinematic updates: "
            << dTkT_InteractionManager->schedulingStats.nKinematicUpdates
            << std::endl;
  std::cout << "Number of times dynamic held back: "
            << dTkT_InteractionManager->schedulingStats.nTimesDynamicHeldBack
            << std::endl;
  std::cout << "Number of times kinematic held back: "
            << dTkT_InteractionManager->schedulingStats.nTimesKinematicHeldBack
            << std::endl;
}

SGPS_impl::~SGPS_impl(){};

void kinematicThread::primeDynamic() {
  // produce produce to prime the pipeline
  for (int j = 0; j < N_INPUT_ITEMS; j++) {
    product[j] += this->costlyProductionStep(j);
  }

  // transfer produce to dynamic buffer
  memcpy(pDynamicOwned_TransfBuffer, product, N_INPUT_ITEMS * sizeof(int));
  pSchedSupport->consOwned_Prod2ConsBuffer_isFresh = true;
  pSchedSupport->schedulingStats.nDynamicUpdates++;
}

int kinematicThread::costlyProductionStep(int val) const {
  std::this_thread::sleep_for(std::chrono::milliseconds(prodAverageTime));
  return 2 * val + 1;
}

void kinematicThread::operator()() {
  // run a while loop producing stuff in each iteration;
  // once produced, it should be made available to the dynamic via memcpy
  while (!pSchedSupport->dynamicDone) {
    // before producing something, a new work order should be in place. Wait on
    // it
    if (!pSchedSupport->prodOwned_Cons2ProdBuffer_isFresh) {
      pSchedSupport->schedulingStats.nTimesKinematicHeldBack++;
      std::unique_lock<std::mutex> lock(pSchedSupport->kinematicCanProceed);
      while (!pSchedSupport->prodOwned_Cons2ProdBuffer_isFresh) {
        // loop to avoid spurious wakeups
        pSchedSupport->cv_KinematicCanProceed.wait(lock);
      }
      // getting here means that new "work order" data has been provided
      {
        // acquire lock and supply the dynamic with fresh produce
        std::lock_guard<std::mutex> lock(
            pSchedSupport->prodOwnedBuffer_AccessCoordination);
        memcpy(inputData, transferBuffer, N_INPUT_ITEMS * sizeof(int));
      }
    }

    // produce something here; fake stuff for now
    for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
      int indx = j % N_INPUT_ITEMS;
      product[j] += this->costlyProductionStep(j) + inputData[indx];
    }

    // make it clear that the data for most recent work order has
    // been used, in case there is interest in updating it
    pSchedSupport->prodOwned_Cons2ProdBuffer_isFresh = false;

    {
      // acquire lock and supply the dynamic with fresh produce
      std::lock_guard<std::mutex> lock(
          pSchedSupport->consOwnedBuffer_AccessCoordination);
      memcpy(pDynamicOwned_TransfBuffer, product,
             N_MANUFACTURED_ITEMS * sizeof(int));
    }
    pSchedSupport->consOwned_Prod2ConsBuffer_isFresh = true;
    pSchedSupport->schedulingStats.nDynamicUpdates++;

    // signal the dynamic that it has fresh produce
    pSchedSupport->cv_DynamicCanProceed.notify_all();
  }

  // in case the dynamic is hanging in there...
  pSchedSupport->cv_DynamicCanProceed.notify_all();
}

} // namespace sgps
