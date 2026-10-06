// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DEME_MESH_DEFORMATION_HPP
#define DEME_MESH_DEFORMATION_HPP

#include <memory>

#include "core/utils/DataMigrationHelper.hpp"
#include "DEM/BdrsAndObjs.h"

namespace deme {
// Persistent, fixed-topology deformation buffers. CPU readers explicitly synchronize the dirty cache;
// device updates never download the vertex array. Patch membership is CSR in local triangle indices.
struct MeshDeformationState {
    MeshDeformationState(size_t* host_bytes, size_t* device_bytes)
        : vertices(host_bytes, device_bytes),
          input(host_bytes, device_bytes),
          faces(host_bytes, device_bytes),
          patch_offsets(host_bytes, device_bytes),
          patch_triangles(host_bytes, device_bytes),
          invalid(host_bytes, device_bytes) {}
    std::shared_ptr<DEMMesh> mesh;
    size_t tri_start = 0;
    size_t patch_start = 0;
    // Rigid meshes need no extra GPU allocations. Materialize the prepared topology only on first deformation
    // or explicit center update, then reuse every allocation for subsequent calls.
    void prepareDevice() {
        if (device_ready)
            return;
        vertices.toDevice();
        faces.toDevice();
        patch_offsets.toDevice();
        patch_triangles.toDevice();
        invalid.toDevice();
        input.resizeDevice(input.size());
        device_ready = true;
    }
    bool device_ready = false;
    bool vertices_dirty = false;
    bool centers_dirty = false;
    DualArray<float3> vertices;
    DualArray<float3> input;
    DualArray<int3> faces;
    DualArray<size_t> patch_offsets;
    DualArray<size_t> patch_triangles;
    DualArray<unsigned int> invalid;
};
}  // namespace deme
#endif
