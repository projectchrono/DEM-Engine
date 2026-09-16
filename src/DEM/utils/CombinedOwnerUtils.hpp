//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <DEM/Defines.h>

namespace deme {

float3 rotateDiagonalMOIToFrame(const float3& diagonal_moi, const float4& q);
float3 parallelAxisDiagonal(float mass, const float3& rel_pos);

}  // namespace deme
