//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <DEM/utils/CombinedOwnerUtils.hpp>

#include <kernel/DEMHelperKernels.cuh>

namespace deme {

float3 rotateDiagonalMOIToFrame(const float3& diagonal_moi, const float4& q) {
    float3 x_axis = make_float3(1, 0, 0);
    float3 y_axis = make_float3(0, 1, 0);
    float3 z_axis = make_float3(0, 0, 1);
    applyOriQToVector3(x_axis, q);
    applyOriQToVector3(y_axis, q);
    applyOriQToVector3(z_axis, q);

    return make_float3(diagonal_moi.x * x_axis.x * x_axis.x + diagonal_moi.y * y_axis.x * y_axis.x +
                           diagonal_moi.z * z_axis.x * z_axis.x,
                       diagonal_moi.x * x_axis.y * x_axis.y + diagonal_moi.y * y_axis.y * y_axis.y +
                           diagonal_moi.z * z_axis.y * z_axis.y,
                       diagonal_moi.x * x_axis.z * x_axis.z + diagonal_moi.y * y_axis.z * y_axis.z +
                           diagonal_moi.z * z_axis.z * z_axis.z);
}

float3 parallelAxisDiagonal(float mass, const float3& rel_pos) {
    return mass * make_float3(rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z,
                              rel_pos.x * rel_pos.x + rel_pos.z * rel_pos.z,
                              rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y);
}

}  // namespace deme
