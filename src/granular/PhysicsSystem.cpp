//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cstring>
#include <iostream>
#include <thread>

#include <core/ApiVersion.h>
#include <core/utils/Macros.h>
#include <granular/GranularDefines.h>
#include <granular/PhysicsSystem.h>

namespace sgps {

int kinematicThread::costlyProductionStep(int val) const {
    std::this_thread::sleep_for(std::chrono::milliseconds(kinematicAverageTime));
    return 2 * val + 1;
}

void dynamicThread::populateManagedArrays(
    const std::vector<clumpBodyInertiaOffset_default_t>& input_clump_types,
    const std::vector<float3>& input_clump_xyz,
    const std::set<float>& clumps_mass_types,
    const std::set<float>& clumps_sp_radii_types,
    const std::set<float3>& clumps_sp_location_types,
    const std::vector<clumpBodyInertiaOffset_default_t>& clumps_mass_type_offset,
    const std::vector<std::vector<distinctSphereRadiiOffset_default_t>>& clumps_sp_radii_type_offset,
    const std::vector<std::vector<distinctSphereRelativePositions_default_t>>& clumps_sp_location_type_offset) {
    TRACKED_VECTOR_RESIZE(mass, 5, "mass", 0);
}

}  // namespace sgps
