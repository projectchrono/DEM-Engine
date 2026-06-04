//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <DEM/utils/MeshTopologyUtils.hpp>

#include <DEM/utils/HostSideHelpers.hpp>

#include <algorithm>
#include <cstdint>
#include <unordered_map>

namespace deme {

namespace {

struct EdgeInfo {
    size_t tri = 0;
    int edge = 0;
};

inline uint64_t makeEdgeKey(int a, int b) {
    const uint32_t lo = static_cast<uint32_t>(std::min(a, b));
    const uint32_t hi = static_cast<uint32_t>(std::max(a, b));
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
}

}  // namespace

std::vector<std::array<bodyID_t, 3>> buildTriangleEdgeNeighbors(const std::vector<int3>& face_v_indices,
                                                                const std::vector<float3>& vertices) {
    const size_t n_faces = face_v_indices.size();
    std::vector<std::array<bodyID_t, 3>> neighbors(n_faces, {NULL_BODYID, NULL_BODYID, NULL_BODYID});
    if (n_faces == 0) {
        return neighbors;
    }

    std::vector<size_t> canon;
    if (!vertices.empty()) {
        const double eps = computeVertexQuantEps(vertices);
        canon = buildCanonicalVertexMap(vertices, eps);
    }

    std::unordered_map<uint64_t, std::vector<EdgeInfo>> edge_map;
    edge_map.reserve(n_faces * 3);

    for (size_t i = 0; i < n_faces; ++i) {
        const int3& face = face_v_indices[i];
        const int v0_raw = face.x;
        const int v1_raw = face.y;
        const int v2_raw = face.z;
        if (v0_raw < 0 || v1_raw < 0 || v2_raw < 0) {
            continue;
        }
        int v0 = v0_raw;
        int v1 = v1_raw;
        int v2 = v2_raw;
        if (!canon.empty()) {
            if (static_cast<size_t>(v0_raw) >= canon.size() || static_cast<size_t>(v1_raw) >= canon.size() ||
                static_cast<size_t>(v2_raw) >= canon.size()) {
                continue;
            }
            v0 = static_cast<int>(canon[static_cast<size_t>(v0_raw)]);
            v1 = static_cast<int>(canon[static_cast<size_t>(v1_raw)]);
            v2 = static_cast<int>(canon[static_cast<size_t>(v2_raw)]);
        }
        if (v0 == v1 || v1 == v2 || v2 == v0) {
            continue;
        }
        const uint64_t e0 = makeEdgeKey(v0, v1);
        const uint64_t e1 = makeEdgeKey(v1, v2);
        const uint64_t e2 = makeEdgeKey(v2, v0);
        edge_map[e0].push_back(EdgeInfo{i, 0});
        edge_map[e1].push_back(EdgeInfo{i, 1});
        edge_map[e2].push_back(EdgeInfo{i, 2});
    }

    for (const auto& entry : edge_map) {
        const auto& info = entry.second;
        if (info.size() == 2) {
            const EdgeInfo& a = info[0];
            const EdgeInfo& b = info[1];
            neighbors[a.tri][a.edge] = static_cast<bodyID_t>(b.tri);
            neighbors[b.tri][b.edge] = static_cast<bodyID_t>(a.tri);
        }
    }

    return neighbors;
}

}  // namespace deme
