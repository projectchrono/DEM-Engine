// Copyright (c) 2021, SBEL GPU Development Team
// Copyright (c) 2021, University of Wisconsin - Madison
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <fstream>
#include <vector>

#include "../Defines.h"
#include "../VariableTypes.h"

namespace deme {

/// Host-side description of one analytical component in the current world frame.
struct AnalyticalOutputComponent {
    objType_t type;
    float3 position;
    float3 axis;
    float size_1;
    float size_2;
    float size_3;
    float normal_sign;
    bodyID_t owner;
    family_t family;
};

/// Tessellate analytical surfaces, clip them to the user domain, and write directly displayable VTK PolyData.
void writeAnalyticalAsVtk(std::ofstream& file,
                          const std::vector<AnalyticalOutputComponent>& components,
                          const float3 domain_min,
                          const float3 domain_max,
                          unsigned int circumferential_resolution);

}  // namespace deme
