//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

namespace deme {

double overlapDuration(double a0, double a1, double b0, double b1);
bool hasPendingWear(const std::vector<float>& pending_depth);

}  // namespace deme
