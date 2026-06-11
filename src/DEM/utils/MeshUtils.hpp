//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "DEM/Defines.h"

#include <array>
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

bool loadMeshByExtension(DEMMesh& mesh, const std::string& filename, bool load_normals, bool load_uv);

std::vector<std::array<bodyID_t, 3>> buildTriangleEdgeNeighbors(const std::vector<int3>& face_v_indices,
                                                                const std::vector<float3>& vertices);

MeshCanonicalizationResult canonicalizeMeshOwnerFrame(DEMMesh& mesh);

double overlapDuration(double a0, double a1, double b0, double b1);
bool hasPendingWear(const std::vector<float>& pending_depth);

}  // namespace deme
