//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <DEM/Defines.h>

namespace deme {

class DEMClumpTemplate;

bool computeClumpUnionMassPropsApprox(const DEMClumpTemplate& clump, double& volume, float3& center, float3& inertia);

}  // namespace deme
