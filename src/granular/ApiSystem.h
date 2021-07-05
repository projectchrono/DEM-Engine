//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <vector>
#include <core/ApiVersion.h>
#include <granular/PhysicsSystem.h>

namespace sgps {

class SGPS_impl;

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

    void LaunchThreads();

  protected:
    SGPS_api() : m_sys(nullptr) {}
    SGPS_impl* m_sys;

  private:
    std::vector<std::vector<float>> m_clumps_sp_radii;
};

}  // namespace sgps
