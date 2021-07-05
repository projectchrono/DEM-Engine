//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <granular/ApiSystem.h>

namespace sgps {

SGPS_api::SGPS_api(float rad) {
    m_sys = new SGPS_impl(rad);
}

SGPS_api::~SGPS_api() {
    delete m_sys;
}

unsigned int SGPS_api::LoadClumpType(std::vector<float> sp_radii,
                                     std::vector<float> sp_location_x,
                                     std::vector<float> sp_location_y,
                                     std::vector<float> sp_location_z,
                                     std::vector<float> sp_density) {
    m_clumps_sp_radii.push_back(sp_radii);

    return m_clumps_sp_radii.size() - 1;
}

void SGPS_api::LaunchThreads() {
    m_sys->LaunchThreads();
}

}  // namespace sgps
