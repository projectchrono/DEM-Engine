#include <chrono>
#include <cstring>
#include <cuda.h>
#include <granular/PhysicsSystem.h>
#include <iostream>
#include <thread>

namespace sgps {

__global__ void writer() { printf("Consumer run\n"); }

void dynamicThread::operator()() {
  // acquire lock to prevent the producer to mess up
  // with the transfer buffer while the latter is used

  for (int cycle = 0; cycle < nConsumerCycles; cycle++) {
    // if the produce is fresh, use it
    if (pSchedSupport->consOwned_Prod2ConsBuffer_isFresh) {
      {
        // acquire lock and use the content of the consumer-owned transfer
        // buffer
        std::lock_guard<std::mutex> lock(
            pSchedSupport->consOwnedBuffer_AccessCoordination);
        memcpy(outcome, transferBuffer, N_MANUFACTURED_ITEMS * sizeof(int));
      }
      pSchedSupport->consOwned_Prod2ConsBuffer_isFresh = false;
      pSchedSupport->stampLastUpdateOfConsumer = cycle;
    }

    // if it's the case, it's important at this point to let the producer know
    // that this is the last consumer cycle; this is important otherwise the
    // producer will hang waiting for communication swith the consumer
    if (cycle == (nConsumerCycles - 1))
      pSchedSupport->consumerDone = true;

    // if the producer is idle, give it the opportunity to get busy again
    if (!pSchedSupport->prodOwned_Cons2ProdBuffer_isFresh) {
      // acquire lock and refresh the work order for the producer
      {
        std::lock_guard<std::mutex> lock(
            pSchedSupport->prodOwnedBuffer_AccessCoordination);
        memcpy(pProducerOwned_TransfBuffer, input4Producer,
               N_INPUT_ITEMS * sizeof(int));
      }
      pSchedSupport->prodOwned_Cons2ProdBuffer_isFresh = true;
      pSchedSupport->schedulingStats.nProducerUpdates++;
      // signal the producer that it has data for a new work order
      pSchedSupport->cv_ProducerCanProceed.notify_all();
    }

    // this is the fake place where produce is used;
    for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
      outcome[j] += this->localUse(j);
    }

    int totGPU;
    cudaGetDeviceCount(&totGPU);
    printf("Total device: %d\n", totGPU);

    std::cout << "Consumer side values. Cycle: " << cycle << std::endl;
    for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
      std::cout << outcome[j] << std::endl;
    }

    // consumer wrapped up one cycle
    pSchedSupport->currentStampOfConsumer++;

    // check if we need to wait; i.e., if consumer drifted too much into future
    if (pSchedSupport->consumerShouldWait()) {
      // wait for a signal from the producer to indicate that
      // the producer has caught up
      pSchedSupport->schedulingStats.nTimesConsumerHeldBack++;
      std::unique_lock<std::mutex> lock(pSchedSupport->consumerCanProceed);
      while (!pSchedSupport->consOwned_Prod2ConsBuffer_isFresh) {
        // loop to avoid spurious wakeups
        pSchedSupport->cv_ConsumerCanProceed.wait(lock);
      }
    }
  }
}

int dynamicThread::localUse(int val) {
  // std::this_thread::sleep_for(std::chrono::milliseconds(consAverageTime));
  writer<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 2 * val;
}

} // namespace sgps
