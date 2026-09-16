//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "DEM/Defines.h"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace deme {

class DEMMesh;

struct MeshCanonicalizationResult {
    bool transformed = false;
    bool moi_rescaled = false;
    bool moi_inconsistent = false;
    double center_norm = 0.0;
    double offdiag_rel = 0.0;
};

struct MeshMassJitKey {
    size_t template_mark;
    float mass;
    float moi_x;
    float moi_y;
    float moi_z;

    bool operator<(const MeshMassJitKey& other) const {
        if (template_mark != other.template_mark)
            return template_mark < other.template_mark;
        if (mass != other.mass)
            return mass < other.mass;
        if (moi_x != other.moi_x)
            return moi_x < other.moi_x;
        if (moi_y != other.moi_y)
            return moi_y < other.moi_y;
        return moi_z < other.moi_z;
    }
};

bool loadMeshByExtension(DEMMesh& mesh, const std::string& filename, bool load_normals, bool load_uv);

std::vector<std::array<bodyID_t, 3>> buildTriangleEdgeNeighbors(const std::vector<int3>& face_v_indices,
                                                                const std::vector<float3>& vertices);

MeshCanonicalizationResult canonicalizeMeshOwnerFrame(DEMMesh& mesh);

double overlapDuration(double a0, double a1, double b0, double b1);
bool hasPendingWear(const std::vector<float>& pending_depth);

}  // namespace deme
