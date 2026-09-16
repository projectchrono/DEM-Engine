// Copyright (c) 2021, SBEL GPU Development Team
// Copyright (c) 2021, University of Wisconsin - Madison
//
// SPDX-License-Identifier: BSD-3-Clause

#include "AnalyticalOutput.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>

#include "HostSideHelpers.hpp"

namespace deme {
namespace {

struct OutputTriangle {
    float3 a;
    float3 b;
    float3 c;
    bodyID_t owner;
    family_t family;
    objType_t type;
    float normal_sign;
};

float component(const float3& point, unsigned int axis) {
    return axis == 0 ? point.x : (axis == 1 ? point.y : point.z);
}

// Clip a convex polygon against one axis-aligned half-space. Applying this to all six box faces performs
// Sutherland-Hodgman clipping and preserves the analytical surface rather than adding box-cap geometry.
std::vector<float3> clipHalfSpace(const std::vector<float3>& polygon,
                                  unsigned int axis,
                                  float bound,
                                  bool keep_greater) {
    std::vector<float3> clipped;
    if (polygon.empty()) {
        return clipped;
    }
    auto inside = [axis, bound, keep_greater](const float3& point) {
        return keep_greater ? component(point, axis) >= bound : component(point, axis) <= bound;
    };
    for (size_t i = 0; i < polygon.size(); i++) {
        const float3 current = polygon[i];
        const float3 previous = polygon[(i + polygon.size() - 1) % polygon.size()];
        const bool current_inside = inside(current);
        const bool previous_inside = inside(previous);
        if (current_inside != previous_inside) {
            const float denominator = component(current, axis) - component(previous, axis);
            const float fraction = (bound - component(previous, axis)) / denominator;
            clipped.push_back(previous + (current - previous) * fraction);
        }
        if (current_inside) {
            clipped.push_back(current);
        }
    }
    return clipped;
}

void appendClippedTriangle(std::vector<OutputTriangle>& output,
                           const float3& a,
                           const float3& b,
                           const float3& c,
                           const AnalyticalOutputComponent& component_data,
                           const float3& domain_min,
                           const float3& domain_max) {
    std::vector<float3> polygon = {a, b, c};
    for (unsigned int axis = 0; axis < 3; axis++) {
        polygon = clipHalfSpace(polygon, axis, component(domain_min, axis), true);
        polygon = clipHalfSpace(polygon, axis, component(domain_max, axis), false);
    }
    for (size_t i = 1; i + 1 < polygon.size(); i++) {
        if (length(cross(polygon[i] - polygon[0], polygon[i + 1] - polygon[0])) > DEME_TINY_FLOAT) {
            output.push_back({polygon[0], polygon[i], polygon[i + 1], component_data.owner, component_data.family,
                              component_data.type, component_data.normal_sign});
        }
    }
}

std::array<float3, 8> boxCorners(const float3& lower, const float3& upper) {
    return {make_float3(lower.x, lower.y, lower.z), make_float3(upper.x, lower.y, lower.z),
            make_float3(lower.x, upper.y, lower.z), make_float3(upper.x, upper.y, lower.z),
            make_float3(lower.x, lower.y, upper.z), make_float3(upper.x, lower.y, upper.z),
            make_float3(lower.x, upper.y, upper.z), make_float3(upper.x, upper.y, upper.z)};
}

void orthogonalBasis(const float3& axis, float3& first, float3& second) {
    const float3 reference = std::abs(axis.z) < 0.9f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
    first = normalize(cross(axis, reference));
    second = cross(axis, first);
}

// Intersect an infinite plane with all box edges, sort the resulting convex polygon in-plane, then triangulate it.
void appendPlane(std::vector<OutputTriangle>& output,
                 const AnalyticalOutputComponent& plane,
                 const float3& domain_min,
                 const float3& domain_max) {
    static constexpr unsigned int edges[12][2] = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {4, 5}, {4, 6},
                                                  {5, 7}, {6, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    const auto corners = boxCorners(domain_min, domain_max);
    const float3 normal = normalize(plane.axis);
    std::vector<float3> polygon;
    for (const auto& edge : edges) {
        const float da = dot(corners[edge[0]] - plane.position, normal);
        const float db = dot(corners[edge[1]] - plane.position, normal);
        if ((da > 0) == (db > 0) && std::abs(da) > DEME_TINY_FLOAT && std::abs(db) > DEME_TINY_FLOAT) {
            continue;
        }
        const float denominator = da - db;
        if (std::abs(denominator) <= DEME_TINY_FLOAT) {
            continue;
        }
        const float fraction = da / denominator;
        if (fraction >= 0 && fraction <= 1) {
            const float3 point = corners[edge[0]] + (corners[edge[1]] - corners[edge[0]]) * fraction;
            const bool duplicate = std::any_of(polygon.begin(), polygon.end(), [&point](const float3& existing) {
                return length(point - existing) <= DEME_TINY_FLOAT;
            });
            if (!duplicate) {
                polygon.push_back(point);
            }
        }
    }
    if (polygon.size() < 3) {
        return;
    }
    float3 center = make_float3(0);
    for (const auto& point : polygon) {
        center += point;
    }
    center /= static_cast<float>(polygon.size());
    float3 basis_u, basis_v;
    orthogonalBasis(normal, basis_u, basis_v);
    std::sort(polygon.begin(), polygon.end(), [&center, &basis_u, &basis_v](const float3& a, const float3& b) {
        const float3 da = a - center;
        const float3 db = b - center;
        return std::atan2(dot(da, basis_v), dot(da, basis_u)) < std::atan2(dot(db, basis_v), dot(db, basis_u));
    });
    for (size_t i = 1; i + 1 < polygon.size(); i++) {
        output.push_back(
            {polygon[0], polygon[i], polygon[i + 1], plane.owner, plane.family, plane.type, plane.normal_sign});
    }
}

void appendRoundSurface(std::vector<OutputTriangle>& output,
                        const AnalyticalOutputComponent& surface,
                        const float3& domain_min,
                        const float3& domain_max,
                        unsigned int resolution) {
    const float3 axis = normalize(surface.axis);
    float projection_min = DEME_HUGE_FLOAT;
    float projection_max = -DEME_HUGE_FLOAT;
    for (const auto& corner : boxCorners(domain_min, domain_max)) {
        const float projection = dot(corner - surface.position, axis);
        projection_min = std::min(projection_min, projection);
        projection_max = std::max(projection_max, projection);
    }

    const bool is_cylinder = surface.type == ANAL_OBJ_TYPE_CYL_INF;
    float axial_min = projection_min;
    float axial_max = projection_max;
    if (!is_cylinder) {
        axial_min = std::max(axial_min, surface.size_2);
        axial_max = std::min(axial_max, surface.size_3);
    }
    if (axial_max <= axial_min) {
        return;
    }

    float3 basis_u, basis_v;
    orthogonalBasis(axis, basis_u, basis_v);
    for (unsigned int i = 0; i < resolution; i++) {
        const float angle_0 = 2.f * static_cast<float>(M_PI) * i / resolution;
        const float angle_1 = 2.f * static_cast<float>(M_PI) * (i + 1) / resolution;
        const float3 radial_0 = basis_u * std::cos(angle_0) + basis_v * std::sin(angle_0);
        const float3 radial_1 = basis_u * std::cos(angle_1) + basis_v * std::sin(angle_1);
        const float radius_min = is_cylinder ? surface.size_1 : surface.size_1 * axial_min;
        const float radius_max = is_cylinder ? surface.size_1 : surface.size_1 * axial_max;
        const float3 p00 = surface.position + axis * axial_min + radial_0 * radius_min;
        const float3 p01 = surface.position + axis * axial_min + radial_1 * radius_min;
        const float3 p10 = surface.position + axis * axial_max + radial_0 * radius_max;
        const float3 p11 = surface.position + axis * axial_max + radial_1 * radius_max;
        appendClippedTriangle(output, p00, p10, p11, surface, domain_min, domain_max);
        appendClippedTriangle(output, p00, p11, p01, surface, domain_min, domain_max);
    }
}

}  // namespace

void writeAnalyticalAsVtk(std::ofstream& file,
                          const std::vector<AnalyticalOutputComponent>& components,
                          const float3 domain_min,
                          const float3 domain_max,
                          unsigned int circumferential_resolution) {
    std::vector<OutputTriangle> triangles;
    for (const auto& component : components) {
        switch (component.type) {
            case ANAL_OBJ_TYPE_PLANE:
                appendPlane(triangles, component, domain_min, domain_max);
                break;
            case ANAL_OBJ_TYPE_CYL_INF:
            case ANAL_OBJ_TYPE_CONE_INF:
            case ANAL_OBJ_TYPE_CONE:
                appendRoundSurface(triangles, component, domain_min, domain_max, circumferential_resolution);
                break;
            default:
                break;
        }
    }

    std::ostringstream out;
    out << "# vtk DataFile Version 2.0\nDEME analytical boundaries\nASCII\nDATASET POLYDATA\n";
    out << "POINTS " << 3 * triangles.size() << " float\n";
    for (const auto& triangle : triangles) {
        out << triangle.a.x << " " << triangle.a.y << " " << triangle.a.z << "\n";
        out << triangle.b.x << " " << triangle.b.y << " " << triangle.b.z << "\n";
        out << triangle.c.x << " " << triangle.c.y << " " << triangle.c.z << "\n";
    }
    out << "POLYGONS " << triangles.size() << " " << 4 * triangles.size() << "\n";
    for (size_t i = 0; i < triangles.size(); i++) {
        out << "3 " << 3 * i << " " << 3 * i + 1 << " " << 3 * i + 2 << "\n";
    }
    out << "CELL_DATA " << triangles.size() << "\n";
    auto write_scalar = [&out, &triangles](const std::string& name, const auto& value) {
        out << "SCALARS " << name << " int 1\nLOOKUP_TABLE default\n";
        for (const auto& triangle : triangles) {
            out << value(triangle) << "\n";
        }
    };
    write_scalar("owner", [](const OutputTriangle& triangle) { return triangle.owner; });
    write_scalar("family", [](const OutputTriangle& triangle) { return +triangle.family; });
    write_scalar("component_type", [](const OutputTriangle& triangle) { return +triangle.type; });
    write_scalar("normal_sign", [](const OutputTriangle& triangle) { return static_cast<int>(triangle.normal_sign); });
    file << out.str();
}

}  // namespace deme
