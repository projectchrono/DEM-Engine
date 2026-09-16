// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "DEM/Defines.h"
#include "DEM/utils/VisualizationData.h"
#include <raylib.h>

namespace deme {
namespace visualization {

// Persistent OpenGL resources. The window/context must outlive this object; Close is also safe on empty resources.
// Local geometry is uploaded by SetScene; SetFrame only changes a texture buffer containing three vec4s per owner.
class NativeRenderer {
  public:
    void Initialize();
    void Close();
    void SetScene(const DEMVisualizationScene& scene, int sphere_resolution);
    void SetFrame(const DEMVisualizationFrame& frame, const std::vector<std::array<float, 4>>& colors);
    void Draw(const Camera3D& camera,
              int width,
              int height,
              bool spheres,
              bool triangles,
              bool wireframe,
              bodyID_t selected,
              bool picking = false);
    // Returns zero for background; otherwise one plus the sphere index or sphere_count plus triangle index.
    std::uint32_t Pick(const Camera3D& camera, int width, int height, int x, int y, bool spheres, bool triangles);

  private:
    Shader shader{};
    unsigned int sphere_vao = 0, sphere_vertices = 0, sphere_instances = 0;
    unsigned int mesh_vao = 0, mesh_vertices = 0, mesh_indices = 0;
    unsigned int owner_buffer = 0, owner_texture = 0;
    size_t owner_capacity = 0;
    int sphere_vertex_count = 0, sphere_count = 0, mesh_index_count = 0;
    RenderTexture2D pick_target{};
    std::vector<std::array<float, 4>> owner_data;
};

}  // namespace visualization
}  // namespace deme
