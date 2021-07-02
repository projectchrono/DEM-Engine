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
  // this is where the dynamic thread stores data that needs to be produced
  // herein
  int *pDynamicOwned_TransfBuffer;
  int product[N_MANUFACTURED_ITEMS] = {1, 2, 3};
  int transferBuffer[N_INPUT_ITEMS] = {0, 0, 0, 0};
  int inputData[N_INPUT_ITEMS];
  int kinematicAverageTime;
  int costlyProductionStep(int) const;

public:
  kinematicThread(ThreadManager *pSchedSup) : pSchedSupport(pSchedSup) {
    kinematicAverageTime = 0;
    pDynamicOwned_TransfBuffer = NULL;
  }
  ~kinematicThread() {}

  void setKinematicAverageTime(int val) { kinematicAverageTime = val; }
  void setDestinationBuffer(int *pCB) { pDynamicOwned_TransfBuffer = pCB; }
  int *pDestinationBuffer() { return transferBuffer; }
  void primeDynamic();
  void operator()();
};

class dynamicThread {
private:
  ThreadManager *pSchedSupport;
  // pointer to remote buffer where kinematic thread stores work-order data
  // provided by the dynamic thread
  int *pKinematicOwned_TransfBuffer;
  int transferBuffer[N_MANUFACTURED_ITEMS] = {0, 0, 0};
  int outcome[N_INPUT_ITEMS];
  int input4Kinematic[N_INPUT_ITEMS] = {-1, -2, -3, -4};

  int dynamicAverageTime; // time required in the consumption process; fake lag
  int nDynamicCycles;

  int localUse(int val);

public:
  dynamicThread(ThreadManager *pSchedSup) : pSchedSupport(pSchedSup) {
    pKinematicOwned_TransfBuffer = NULL;
    nDynamicCycles = 0;
    dynamicAverageTime = 0;
  }
  ~dynamicThread() {}

  void setDynamicAverageTime(int val) { dynamicAverageTime = val; }
  void setDestinationBuffer(int *pPB) { pKinematicOwned_TransfBuffer = pPB; }
  void setNDynamicCycles(int val) { nDynamicCycles = val; }
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

  // Body-related arrays in managed memory

  // The mass
  std::vector<float, cudallocator<float>> mass;

  // The components of MOI
  std::vector<float, cudallocator<float>> mmiXX;
  std::vector<float, cudallocator<float>> mmiYY;
  std::vector<float, cudallocator<float>> mmiZZ;

  // The voxel ID (split into 3 parts, representing XYZ location)
  std::vector<voxelID_ts, cudallocator<voxelID_ts>> voxelID;

  // The XYZ local location inside a voxel
  std::vector<unsigned int, cudallocator<unsigned int>> locX;
  std::vector<unsigned int, cudallocator<unsigned int>> locY;
  std::vector<unsigned int, cudallocator<unsigned int>> locZ;

  // The unit quaternion
  std::vector<int, cudallocator<int>> oriQ0;
  std::vector<int, cudallocator<int>> oriQ1;
  std::vector<int, cudallocator<int>> oriQ2;
  std::vector<int, cudallocator<int>> oriQ3;

  // Velocity times ts size: hv
  std::vector<int, cudallocator<int>> hvX;
  std::vector<int, cudallocator<int>> hvY;
  std::vector<int, cudallocator<int>> hvZ;

  // The angular velocity
  std::vector<int, cudallocator<int>> omgBarX;
  std::vector<int, cudallocator<int>> omgBarY;
  std::vector<int, cudallocator<int>> omgBarZ;

  // Sphere-related arrays in managed memory
  // Belonged-body ID (default unsigned int type)
  std::vector<bodyID_t, cudallocator<bodyID_t>> bodyID;

  // The ID that maps this sphere's radius
  std::vector<radiusIndex_t, cudallocator<radiusIndex_t>> radiusID;

  // The ID that maps this sphere's material
  std::vector<materialIndex_t, cudallocator<materialIndex_t>> materialID;

  // The location of the sphere, relative to the LRF of the body it belongs to
  std::vector<float, cudallocator<float>> offsetX;
  std::vector<float, cudallocator<float>> offsetY;
  std::vector<float, cudallocator<float>> offsetZ;

  int updateFreq = 1;
  int timeDynamicSide = 1;
  int timeKinematicSide = 1;
  int nDynamicCycles = 1;

  ThreadManager *dTkT_InteractionManager;
};

} // namespace sgps
