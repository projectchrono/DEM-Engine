//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <mutex>
#include <vector>

#include <core/ApiVersion.h>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/ThreadManager.h>
#include <granular/GranularDefines.h>

namespace sgps {

class kinematicThread {
private:
  ThreadManager *pSchedSupport;
  // this is where the consumer stores data that needs to be produced herein
  int *pConsumerOwned_TransfBuffer;
  int product[N_MANUFACTURED_ITEMS] = {1, 2, 3};
  int transferBuffer[N_INPUT_ITEMS] = {0, 0, 0, 0};
  int inputData[N_INPUT_ITEMS];
  int prodAverageTime;
  int costlyProductionStep(int) const;

public:
  kinematicThread(ThreadManager *pSchedSup) : pSchedSupport(pSchedSup) {
    prodAverageTime = 0;
    pConsumerOwned_TransfBuffer = NULL;
  }
  ~kinematicThread() {}

  void setProducerAverageTime(int val) { prodAverageTime = val; }
  void setDestinationBuffer(int *pCB) { pConsumerOwned_TransfBuffer = pCB; }
  int *pDestinationBuffer() { return transferBuffer; }
  void primeConsumer();
  void operator()();
};

class dynamicThread {
private:
  ThreadManager *pSchedSupport;
  // pointer to remote buffer where producer stores work-order data provided by
  // the consumer
  int *pProducerOwned_TransfBuffer;
  int transferBuffer[N_MANUFACTURED_ITEMS] = {0, 0, 0};
  int outcome[N_INPUT_ITEMS];
  int input4Producer[N_INPUT_ITEMS] = {-1, -2, -3, -4};

  int consAverageTime; // time required in the consumption process; fake lag
  int nConsumerCycles;

  int localUse(int val);

public:
  dynamicThread(ThreadManager *pSchedSup) : pSchedSupport(pSchedSup) {
    pProducerOwned_TransfBuffer = NULL;
    nConsumerCycles = 0;
    consAverageTime = 0;
  }
  ~dynamicThread() {}

  void setConsumerAverageTime(int val) { consAverageTime = val; }
  void setDestinationBuffer(int *pPB) { pProducerOwned_TransfBuffer = pPB; }
  void setNConsumerCycles(int val) { nConsumerCycles = val; }
  int *pDestinationBuffer() { return transferBuffer; }
  void operator()();
};

class SGPS_impl {
public:
  virtual ~SGPS_impl();
  friend class SGPS_api;

protected:
  SGPS_impl() = delete;
  SGPS_impl(float sphere_rad);
  float sphereUU;

  // Arrays in managed memory
  std::vector<float, cudallocator<float>> mass;
  std::vector<float, cudallocator<float>> mmiXX;

  int updateFreq = 1;
  int timeConsumerSide = 1;
  int timeProducerSide = 1;
  int nConsumerCycles = 1;

  ThreadManager *dTkT_InteractionManager;
};

} // namespace sgps
