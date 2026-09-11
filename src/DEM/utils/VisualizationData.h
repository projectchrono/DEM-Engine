// Copyright (c) 2021, SBEL GPU Development Team
// Copyright (c) 2021, University of Wisconsin - Madison
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_VISUALIZATION_DATA_H
#define DEME_VISUALIZATION_DATA_H

#include <cuda_runtime.h>

#include <vector>
#include <cstdint>

#include "../VariableTypes.h"

namespace deme {

/// Geometry in owner-local coordinates. IDs are stable indices in these arrays until the scene revision changes.
struct DEMVisualizationScene {
    struct Sphere {
        float3 offset;
        float radius;
        bodyID_t owner;
    };
    struct Triangle {
        float3 a, b, c;
        bodyID_t owner;
    };
    std::uint64_t revision = 0;
    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;
};

/// Compact host frame, indexed by solver owner ID. Geometry is obtained separately and cached by revision.
struct DEMVisualizationFrame {
    double simulation_time = 0.0;
    std::uint64_t revision = 0;
    std::vector<float3> positions;
    std::vector<float4> orientations;
    std::vector<family_t> families;
    // Empty unless requested; avoids transferring velocities for family/height coloring.
    std::vector<float3> velocities;
};

/// One sphere component in a host-side visualization snapshot.
struct DEMVisualizationSphere {
    float3 position;
    float radius;
    family_t family;
    bodyID_t owner;
};

/// One mesh facet in a host-side visualization snapshot.
struct DEMVisualizationTriangle {
    float3 a;
    float3 b;
    float3 c;
    family_t family;
    bodyID_t owner;
};

/// Host-side geometry captured at one solver time for rendering or external visualization.
struct DEMVisualizationSnapshot {
    double simulation_time = 0.0;
    std::vector<DEMVisualizationSphere> spheres;
    std::vector<DEMVisualizationTriangle> triangles;
};

}  // namespace deme

#endif
