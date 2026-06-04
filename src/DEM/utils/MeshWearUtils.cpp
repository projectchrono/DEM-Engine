//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <DEM/utils/MeshWearUtils.hpp>

#include <algorithm>
#include <cmath>

namespace deme {

double overlapDuration(double a0, double a1, double b0, double b1) {
    const double lo = std::max(a0, b0);
    const double hi = std::min(a1, b1);
    return (hi > lo) ? (hi - lo) : 0.0;
}

bool hasPendingWear(const std::vector<float>& pending_depth) {
    for (float d : pending_depth) {
        if (d > 0.f && std::isfinite(d)) {
            return true;
        }
    }
    return false;
}

}  // namespace deme
