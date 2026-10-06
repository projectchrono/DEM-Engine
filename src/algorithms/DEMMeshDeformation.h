// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DEME_MESH_DEFORMATION_H
#define DEME_MESH_DEFORMATION_H
#include <cuda_runtime.h>
#include <cstddef>
namespace deme {
// Validate input (and the result of an incremental update) before mutating persistent geometry.
void ValidateMeshVectors(const float3* input,
                         const float3* current,
                         size_t count,
                         unsigned int* invalid,
                         cudaStream_t stream);
// Apply local vertex updates, then scatter through fixed connectivity to the solver's triangle soup.
void DeformMeshVertices(const float3* input,
                        float3* vertices,
                        size_t count,
                        bool increment,
                        const int3* faces,
                        size_t triangles,
                        float3* node1,
                        float3* node2,
                        float3* node3,
                        cudaStream_t stream);
// Preserve the historical local-origin convention for a single patch; otherwise average triangle centroids.
void RefreshMeshPatchCenters(const size_t* offsets,
                             const size_t* triangles,
                             size_t patches,
                             const float3* node1,
                             const float3* node2,
                             const float3* node3,
                             float3* centers,
                             cudaStream_t stream);
}  // namespace deme
#endif
