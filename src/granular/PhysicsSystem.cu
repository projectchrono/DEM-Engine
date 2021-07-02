#include <chrono>
#include <cstring>
#include <cuda.h>
#include <granular/PhysicsSystem.h>
#include <iostream>
#include <thread>

namespace sgps {

__global__ void writer() { printf("Dynamic run\n"); }

void dynamicThread::operator()() {
  // acquire lock to prevent the kinematic to mess up
  // with the transfer buffer while the latter is used

  for (int cycle = 0; cycle < nDynamicCycles; cycle++) {
    // if the produce is fresh, use it
    if (pSchedSupport->consOwned_Prod2ConsBuffer_isFresh) {
      {
        // acquire lock and use the content of the dynamic-owned transfer
        // buffer
        std::lock_guard<std::mutex> lock(
            pSchedSupport->consOwnedBuffer_AccessCoordination);
        memcpy(outcome, transferBuffer, N_MANUFACTURED_ITEMS * sizeof(int));
      }
      pSchedSupport->consOwned_Prod2ConsBuffer_isFresh = false;
      pSchedSupport->stampLastUpdateOfDynamic = cycle;
    }

    // if it's the case, it's important at this point to let the kinematic know
    // that this is the last dynamic cycle; this is important otherwise the
    // kinematic will hang waiting for communication swith the dynamic
    if (cycle == (nDynamicCycles - 1))
      pSchedSupport->dynamicDone = true;

    // if the kinematic is idle, give it the opportunity to get busy again
    if (!pSchedSupport->prodOwned_Cons2ProdBuffer_isFresh) {
      // acquire lock and refresh the work order for the kinematic
      {
        std::lock_guard<std::mutex> lock(
            pSchedSupport->prodOwnedBuffer_AccessCoordination);
        memcpy(pKinematicOwned_TransfBuffer, input4Kinematic,
               N_INPUT_ITEMS * sizeof(int));
      }
      pSchedSupport->prodOwned_Cons2ProdBuffer_isFresh = true;
      pSchedSupport->schedulingStats.nKinematicUpdates++;
      // signal the kinematic that it has data for a new work order
      pSchedSupport->cv_KinematicCanProceed.notify_all();
    }

    // this is the fake place where produce is used;
    for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
      outcome[j] += this->localUse(j);
    }

    int totGPU;
    cudaGetDeviceCount(&totGPU);
    printf("Total device: %d\n", totGPU);

    std::cout << "Dynamic side values. Cycle: " << cycle << std::endl;
    for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
      std::cout << outcome[j] << std::endl;
    }

    // dynamic wrapped up one cycle
    pSchedSupport->currentStampOfDynamic++;

    // check if we need to wait; i.e., if dynamic drifted too much into future
    if (pSchedSupport->dynamicShouldWait()) {
      // wait for a signal from the kinematic to indicate that
      // the kinematic has caught up
      pSchedSupport->schedulingStats.nTimesDynamicHeldBack++;
      std::unique_lock<std::mutex> lock(pSchedSupport->dynamicCanProceed);
      while (!pSchedSupport->consOwned_Prod2ConsBuffer_isFresh) {
        // loop to avoid spurious wakeups
        pSchedSupport->cv_DynamicCanProceed.wait(lock);
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
