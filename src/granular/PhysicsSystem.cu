#include <chrono>
#include <cstring>
#include <cuda.h>
#include <granular/PhysicsSystem.h>
#include <iostream>
#include <thread>

namespace sgps {

__global__ void dynamicTestKernel() { printf("Dynamic run\n"); }
__global__ void kinematicTestKernel() { printf("Kinematic run\n"); }

void kinematicThread::operator()() {
  // run a while loop producing stuff in each iteration;
  // once produced, it should be made available to the dynamic via memcpy
  while (!pSchedSupport->dynamicDone) {
    // before producing something, a new work order should be in place. Wait on
    // it
    if (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh) {
      pSchedSupport->schedulingStats.nTimesKinematicHeldBack++;
      std::unique_lock<std::mutex> lock(pSchedSupport->kinematicCanProceed);
      while (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh) {
        // loop to avoid spurious wakeups
        pSchedSupport->cv_KinematicCanProceed.wait(lock);
      }
      // getting here means that new "work order" data has been provided
      {
        // acquire lock and supply the dynamic with fresh produce
        std::lock_guard<std::mutex> lock(
            pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
        memcpy(inputData, transferBuffer, N_INPUT_ITEMS * sizeof(int));
      }
    }

    int totGPU;
    cudaGetDeviceCount(&totGPU);
    printf("Total device: %d\n", totGPU);

    // produce something here; fake stuff for now
    for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
      kinematicTestKernel<<<1, 1>>>();
      cudaDeviceSynchronize();

      int indx = j % N_INPUT_ITEMS;
      product[j] += this->costlyProductionStep(j) + inputData[indx];
    }

    // make it clear that the data for most recent work order has
    // been used, in case there is interest in updating it
    pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = false;

    {
      // acquire lock and supply the dynamic with fresh produce
      std::lock_guard<std::mutex> lock(
          pSchedSupport->dynamicOwnedBuffer_AccessCoordination);
      memcpy(pDynamicOwned_TransfBuffer, product,
             N_MANUFACTURED_ITEMS * sizeof(int));
    }
    pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = true;
    pSchedSupport->schedulingStats.nDynamicUpdates++;

    // signal the dynamic that it has fresh produce
    pSchedSupport->cv_DynamicCanProceed.notify_all();
  }

  // in case the dynamic is hanging in there...
  pSchedSupport->cv_DynamicCanProceed.notify_all();
}

void dynamicThread::operator()() {
  // acquire lock to prevent the kinematic to mess up
  // with the transfer buffer while the latter is used

  for (int cycle = 0; cycle < nDynamicCycles; cycle++) {
    // if the produce is fresh, use it
    if (pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
      {
        // acquire lock and use the content of the dynamic-owned transfer
        // buffer
        std::lock_guard<std::mutex> lock(
            pSchedSupport->dynamicOwnedBuffer_AccessCoordination);
        memcpy(outcome, transferBuffer, N_MANUFACTURED_ITEMS * sizeof(int));
      }
      pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = false;
      pSchedSupport->stampLastUpdateOfDynamic = cycle;
    }

    // if it's the case, it's important at this point to let the kinematic know
    // that this is the last dynamic cycle; this is important otherwise the
    // kinematic will hang waiting for communication swith the dynamic
    if (cycle == (nDynamicCycles - 1))
      pSchedSupport->dynamicDone = true;

    // if the kinematic is idle, give it the opportunity to get busy again
    if (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh) {
      // acquire lock and refresh the work order for the kinematic
      {
        std::lock_guard<std::mutex> lock(
            pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
        memcpy(pKinematicOwned_TransfBuffer, input4Kinematic,
               N_INPUT_ITEMS * sizeof(int));
      }
      pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = true;
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
      while (!pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
        // loop to avoid spurious wakeups
        pSchedSupport->cv_DynamicCanProceed.wait(lock);
      }
    }
  }
}

int dynamicThread::localUse(int val) {
  // std::this_thread::sleep_for(std::chrono::milliseconds(dynamicAverageTime));
  dynamicTestKernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 2 * val;
}

} // namespace sgps
