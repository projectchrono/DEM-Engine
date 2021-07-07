//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <mutex>
#include <vector>

#include <core/ApiVersion.h>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/ThreadManager.h>
#include <core/utils/GpuManager.h>
#include <granular/GranularDefines.h>

namespace sgps {

class kinematicThread {
  private:
    ThreadManager* pSchedSupport;
    // this is where the dynamic thread stores data that needs to be produced
    // herein
    voxelID_ts* pDynamicOwnedBuffer_voxelID;
    int kinematicAverageTime;
    int costlyProductionStep(int) const;
    int device_id = 0;

    // Buffer arrays for storing info from the dT side.
    // dT modifies these arrays; kT uses them only.

    // buffer array for voxelID
    std::vector<voxelID_ts, ManagedAllocator<voxelID_ts>> transferBuffer_voxelID;

    // The voxel ID (split into 3 parts, representing XYZ location)
    std::vector<voxelID_ts, ManagedAllocator<voxelID_ts>> voxelID;

  public:
    kinematicThread(ThreadManager* pSchedSup) : pSchedSupport(pSchedSup) {
        kinematicAverageTime = 0;
        pDynamicOwnedBuffer_voxelID = NULL;

        transferBuffer_voxelID.resize(N_MANUFACTURED_ITEMS, 0);
        voxelID.resize(N_MANUFACTURED_ITEMS, 1);
    }
    ~kinematicThread() {}

    void setKinematicAverageTime(int val) { kinematicAverageTime = val; }

    // buffer exchange methods
    void setDestinationBuffer_voxelID(voxelID_ts* pCB) { pDynamicOwnedBuffer_voxelID = pCB; }
    voxelID_ts* pBuffer_voxelID() { return transferBuffer_voxelID.data(); }

    void primeDynamic();
    void operator()();
};

class dynamicThread {
  private:
    ThreadManager* pSchedSupport;
    // pointer to remote buffer where kinematic thread stores work-order data
    // provided by the dynamic thread
    voxelID_ts* pKinematicOwnedBuffer_voxelID;

    int dynamicAverageTime;  // time required in the consumption process; fake lag
    int nDynamicCycles;

    // Buffer arrays for storing info from the dT side.
    // kT modifies these arrays; dT uses them only.

    // buffer array for voxelID
    std::vector<voxelID_ts, ManagedAllocator<voxelID_ts>> transferBuffer_voxelID;

    // Pointers to dynamics-related arrays
    struct SysState {
        float* mass;
    };

    // Body-related arrays in managed memory

    // The mass
    std::vector<float, ManagedAllocator<float>> mass;

    // The components of MOI
    std::vector<float, ManagedAllocator<float>> mmiXX;
    std::vector<float, ManagedAllocator<float>> mmiYY;
    std::vector<float, ManagedAllocator<float>> mmiZZ;

    // The voxel ID (split into 3 parts, representing XYZ location)
    std::vector<voxelID_ts, ManagedAllocator<voxelID_ts>> voxelID;

    // The XYZ local location inside a voxel
    std::vector<unsigned int, ManagedAllocator<unsigned int>> locX;
    std::vector<unsigned int, ManagedAllocator<unsigned int>> locY;
    std::vector<unsigned int, ManagedAllocator<unsigned int>> locZ;

    // The unit quaternion
    std::vector<int, ManagedAllocator<int>> oriQ0;
    std::vector<int, ManagedAllocator<int>> oriQ1;
    std::vector<int, ManagedAllocator<int>> oriQ2;
    std::vector<int, ManagedAllocator<int>> oriQ3;

    // Velocity times ts size: hv
    std::vector<int, ManagedAllocator<int>> hvX;
    std::vector<int, ManagedAllocator<int>> hvY;
    std::vector<int, ManagedAllocator<int>> hvZ;

    // The angular velocity
    std::vector<int, ManagedAllocator<int>> omgBarX;
    std::vector<int, ManagedAllocator<int>> omgBarY;
    std::vector<int, ManagedAllocator<int>> omgBarZ;

    // Sphere-related arrays in managed memory
    // Belonged-body ID (default unsigned int type)
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> bodyID;

    // The ID that maps this sphere's radius
    std::vector<radiusIndex_t, ManagedAllocator<radiusIndex_t>> radiusID;

    // The ID that maps this sphere's material
    std::vector<materialIndex_t, ManagedAllocator<materialIndex_t>> materialID;

    // The location of the sphere, relative to the LRF of the body it belongs to
    std::vector<float, ManagedAllocator<float>> offsetX;
    std::vector<float, ManagedAllocator<float>> offsetY;
    std::vector<float, ManagedAllocator<float>> offsetZ;

    int localUse(int val);

  public:
    dynamicThread(ThreadManager* pSchedSup) : pSchedSupport(pSchedSup) {
        pKinematicOwnedBuffer_voxelID = NULL;
        nDynamicCycles = 0;
        dynamicAverageTime = 0;

        transferBuffer_voxelID.resize(N_MANUFACTURED_ITEMS, 0);
        voxelID.resize(N_MANUFACTURED_ITEMS, 0);
    }
    ~dynamicThread() {}

    void setDynamicAverageTime(int val) { dynamicAverageTime = val; }
    void setNDynamicCycles(int val) { nDynamicCycles = val; }

    // buffer exchange methods
    void setDestinationBuffer_voxelID(voxelID_ts* pPB) { pKinematicOwnedBuffer_voxelID = pPB; }
    voxelID_ts* pBuffer_voxelID() { return transferBuffer_voxelID.data(); }

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

    int updateFreq = 1;
    int timeDynamicSide = 1;
    int timeKinematicSide = 1;
    int nDynamicCycles = 5;

    ThreadManager* dTkT_InteractionManager;
    kinematicThread* kT;
    dynamicThread* dT;

    // GpuManager* dTkT_GpuManager;
    int LaunchThreads();
};

}  // namespace sgps
