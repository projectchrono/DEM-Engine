//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <DEM/Defines.h>

#include <array>
#include <vector>

namespace deme {

std::vector<std::array<bodyID_t, 3>> buildTriangleEdgeNeighbors(const std::vector<int3>& face_v_indices,
                                                                const std::vector<float3>& vertices);

}  // namespace deme
