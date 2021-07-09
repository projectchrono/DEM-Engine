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

class SGPS {
  public:
    SGPS(float rad);
    virtual ~SGPS();

    clumpBodyInertiaOffset_t LoadClumpType(const std::vector<float>& sp_radii,
                                           const std::vector<float>& sp_locations_x,
                                           const std::vector<float>& sp_locations_y,
                                           const std::vector<float>& sp_locations_z,
                                           const std::vector<materialsOffset_t>& sp_material_ids);
    // TODO: need to overload with (float radii, float3 location, object properties)

    // load possible materials into the API-level cache
    materialsOffset_t LoadMaterialType(float density, float E);

    // return the voxel ID of a clump by its numbering
    voxelID_t GetClumpVoxelID(unsigned int i) const;

    int LaunchThreads();

    /*
      protected:
        SGPS() : m_sys(nullptr) {}
        SGPS_impl* m_sys;
    */

  private:
    // This is the cached material information.
    // It will be massaged into kernels upon Initialize.
    struct Material {
        float density;
        float E;
    };
    std::vector<Material> m_sp_materials;

    // This is the cached clump structure information.
    // It will be massaged into kernels upon Initialize.
    std::vector<std::vector<float>> m_clumps_sp_radii;
    std::vector<std::vector<float>> m_clumps_sp_location_x;

    float sphereUU;

    int updateFreq = 1;
    int timeDynamicSide = 1;
    int timeKinematicSide = 1;
    int nDynamicCycles = 5;

    GpuManager* dTkT_GpuManager;
    ThreadManager* dTkT_InteractionManager;
    kinematicThread* kT;
    dynamicThread* dT;
};

}  // namespace sgps
