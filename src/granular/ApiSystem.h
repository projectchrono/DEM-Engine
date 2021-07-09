//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <vector>
#include <core/ApiVersion.h>
#include <granular/PhysicsSystem.h>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/ThreadManager.h>
#include <core/utils/GpuManager.h>
#include <granular/GranularDefines.h>

namespace sgps {

// class SGPS_impl;
// class kinematicThread;
// class dynamicThread;
// class ThreadManager;

class SGPS_api {
  public:
    SGPS_api(float rad);
    virtual ~SGPS_api();

    unsigned int LoadClumpType(std::vector<float> sp_radii,
                               std::vector<float> sp_location_x,
                               std::vector<float> sp_location_y,
                               std::vector<float> sp_location_z,
                               std::vector<float> sp_density);
    // TODO: need to overload with (float radii, float3 location, object properties)

    // return the voxel ID of a clump by its number
    voxelID_ts GetClumpVoxelID(unsigned int i);

    // GpuManager* dTkT_GpuManager;
    int LaunchThreads();

/*
  protected:
    SGPS_api() : m_sys(nullptr) {}
    SGPS_impl* m_sys;
*/

  private:
    std::vector<std::vector<float>> m_clumps_sp_radii;

    float sphereUU;

    int updateFreq = 1;
    int timeDynamicSide = 1;
    int timeKinematicSide = 1;
    int nDynamicCycles = 5;

    ThreadManager* dTkT_InteractionManager;
    kinematicThread* kT;
    dynamicThread* dT;
};

}  // namespace sgps
