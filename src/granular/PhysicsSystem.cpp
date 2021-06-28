//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <cstring>
#include <granular/GranularDefines.h>
#include <granular/PhysicsSystem.h>
#include <thread>

namespace sgps {

SGPS_impl::SGPS_impl(float rad) : sphereUU(rad) {
  ThreadManager *dTkT_InteractionManager = new ThreadManager();
  dTkT_InteractionManager->consumerRequestedUpdateFrequency = updateFreq;

  kinematicThread kT(dTkT_InteractionManager);
  dynamicThread dT(dTkT_InteractionManager);

  int *pBuffer = kT.pDestinationBuffer();
  dT.setDestinationBuffer(pBuffer);
  dT.setConsumerAverageTime(timeConsumerSide);
  dT.setNConsumerCycles(nConsumerCycles);

  pBuffer = dT.pDestinationBuffer();
  kT.setDestinationBuffer(pBuffer);
  kT.primeConsumer();
  kT.setProducerAverageTime(timeProducerSide);
}

SGPS_impl::~SGPS_impl(){};

void kinematicThread::primeConsumer() {
  // produce produce to prime the pipeline
  for (int j = 0; j < N_INPUT_ITEMS; j++) {
    product[j] += this->costlyProductionStep(j);
  }

  // transfer produce to consumer buffer
  memcpy(pConsumerOwned_TransfBuffer, product, N_INPUT_ITEMS * sizeof(int));
  pSchedSupport->consOwned_Prod2ConsBuffer_isFresh = true;
  pSchedSupport->schedulingStats.nConsumerUpdates++;
}

int kinematicThread::costlyProductionStep(int val) const {
  std::this_thread::sleep_for(std::chrono::milliseconds(prodAverageTime));
  return 2 * val + 1;
}

void kinematicThread::operator()() {
  // run a while loop producing stuff in each iteration;
  // once produced, it should be made available to the consumer via memcpy
  while (!pSchedSupport->consumerDone) {
    // before producing something, a new work order should be in place. Wait on
    // it
    if (!pSchedSupport->prodOwned_Cons2ProdBuffer_isFresh) {
      pSchedSupport->schedulingStats.nTimesProducerHeldBack++;
      std::unique_lock<std::mutex> lock(pSchedSupport->producerCanProceed);
      while (!pSchedSupport->prodOwned_Cons2ProdBuffer_isFresh) {
        // loop to avoid spurious wakeups
        pSchedSupport->cv_ProducerCanProceed.wait(lock);
      }
      // getting here means that new "work order" data has been provided
      {
        // acquire lock and supply the consumer with fresh produce
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
      // acquire lock and supply the consumer with fresh produce
      std::lock_guard<std::mutex> lock(
          pSchedSupport->consOwnedBuffer_AccessCoordination);
      memcpy(pConsumerOwned_TransfBuffer, product,
             N_MANUFACTURED_ITEMS * sizeof(int));
    }
    pSchedSupport->consOwned_Prod2ConsBuffer_isFresh = true;
    pSchedSupport->schedulingStats.nConsumerUpdates++;

    // signal the consumer that it has fresh produce
    pSchedSupport->cv_ConsumerCanProceed.notify_all();
  }

  // in case the consumer is hanging in there...
  pSchedSupport->cv_ConsumerCanProceed.notify_all();
}

} // namespace sgps
