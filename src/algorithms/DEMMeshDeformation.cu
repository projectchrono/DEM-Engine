// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#include "algorithms/DEMMeshDeformation.h"
#include "core/utils/Logger.hpp"

namespace deme {
namespace {
constexpr unsigned int MESH_DEFORMATION_BLOCK = 128;

// Validate every coordinate before any geometry is changed, including incremental overflow.
__global__ void validateVectors(const float3* input, const float3* current, size_t count, unsigned int* invalid) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    float3 v = input[i];
    if (current) {
        v.x += current[i].x;
        v.y += current[i].y;
        v.z += current[i].z;
    }
    if (!isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z))
        atomicExch(invalid, 1u);
}

// One thread per shared vertex prevents duplicate triangle corners from accumulating increments twice.
__global__ void updateVertices(const float3* input, float3* vertices, size_t count, bool increment) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    float3 v = input[i];
    if (increment) {
        v.x += vertices[i].x;
        v.y += vertices[i].y;
        v.z += vertices[i].z;
    }
    vertices[i] = v;
}

// The connectivity already includes initialization's winding corrections.
__global__ void scatterVertices(const float3* vertices,
                                const int3* faces,
                                size_t count,
                                float3* node1,
                                float3* node2,
                                float3* node3) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const int3 f = faces[i];
    node1[i] = vertices[f.x];
    node2[i] = vertices[f.y];
    node3[i] = vertices[f.z];
}

// A single strided reduction computes whole-mesh bounds once, rather than rescanning the mesh per patch.
// The persistent output scalar and same-stream ordering avoid allocations and host synchronization per refresh.
__global__ void meshDiagonal(const float3* vertices, size_t count, double* diagonal) {
    __shared__ double bounds[6][MESH_DEFORMATION_BLOCK];
    const unsigned int lane = threadIdx.x;
    double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t i = lane; i < count; i += blockDim.x) {
        const double xyz[3] = {vertices[i].x, vertices[i].y, vertices[i].z};
        for (int k = 0; k < 3; ++k) {
            lo[k] = fmin(lo[k], xyz[k]);
            hi[k] = fmax(hi[k], xyz[k]);
        }
    }
    for (int k = 0; k < 3; ++k) {
        bounds[k][lane] = lo[k];
        bounds[k + 3][lane] = hi[k];
    }
    __syncthreads();
    for (unsigned int stride = MESH_DEFORMATION_BLOCK / 2; stride; stride /= 2) {
        if (lane < stride)
            for (int k = 0; k < 3; ++k) {
                bounds[k][lane] = fmin(bounds[k][lane], bounds[k][lane + stride]);
                bounds[k + 3][lane] = fmax(bounds[k + 3][lane], bounds[k + 3][lane + stride]);
            }
        __syncthreads();
    }
    if (lane == 0) {
        double squared = 0;
        if (count)
            for (int k = 0; k < 3; ++k) {
                const double extent = bounds[k + 3][0] - bounds[k][0];
                squared += extent * extent;
            }
        *diagonal = sqrt(squared);
    }
}

// One block reduces each patch's fixed triangle list. The fixed tree is deterministic and avoids atomics.
// The shared helper documents the geometry assumptions and is also used for CPU initialization.
__global__ void patchCenters(const size_t* offsets,
                             const size_t* triangles,
                             size_t patches,
                             const float3* node1,
                             const float3* node2,
                             const float3* node3,
                             float3* centers,
                             const double* diagonal,
                             MeshPatchCenterOptions options) {
    const size_t p = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    const size_t count = offsets[p + 1] - offsets[p];
    if (patches == 1 || count == 0) {
        if (lane == 0)
            centers[p] = make_float3(0, 0, 0);
        return;
    }
    __shared__ double sums[7][MESH_DEFORMATION_BLOCK];
    double local[7] = {};
    for (size_t j = offsets[p] + lane; j < offsets[p + 1]; j += blockDim.x) {
        const size_t t = triangles[j];
        AccumulatePatchTriangle(local, node1[t], node2[t], node3[t]);
    }
    for (int k = 0; k < 7; ++k)
        sums[k][lane] = local[k];
    __syncthreads();
    for (unsigned int stride = MESH_DEFORMATION_BLOCK / 2; stride; stride /= 2) {
        if (lane < stride)
            for (int k = 0; k < 7; ++k)
                sums[k][lane] += sums[k][lane + stride];
        __syncthreads();
    }
    if (lane == 0) {
        for (int k = 0; k < 7; ++k)
            local[k] = sums[k][0];
        centers[p] = FinishPatchCenter(local, count, *diagonal, options);
    }
}
}  // namespace

void ValidateMeshVectors(const float3* input,
                         const float3* current,
                         size_t count,
                         unsigned int* invalid,
                         cudaStream_t stream) {
    if (count) {
        validateVectors<<<(count + MESH_DEFORMATION_BLOCK - 1) / MESH_DEFORMATION_BLOCK, MESH_DEFORMATION_BLOCK, 0,
                          stream>>>(input, current, count, invalid);
        DEME_GPU_CALL(cudaGetLastError());
    }
}

void DeformMeshVertices(const float3* input,
                        float3* vertices,
                        size_t count,
                        bool increment,
                        const int3* faces,
                        size_t triangles,
                        float3* node1,
                        float3* node2,
                        float3* node3,
                        cudaStream_t stream) {
    if (count) {
        updateVertices<<<(count + MESH_DEFORMATION_BLOCK - 1) / MESH_DEFORMATION_BLOCK, MESH_DEFORMATION_BLOCK, 0,
                         stream>>>(input, vertices, count, increment);
        DEME_GPU_CALL(cudaGetLastError());
    }
    if (triangles) {
        scatterVertices<<<(triangles + MESH_DEFORMATION_BLOCK - 1) / MESH_DEFORMATION_BLOCK, MESH_DEFORMATION_BLOCK, 0,
                          stream>>>(vertices, faces, triangles, node1, node2, node3);
        DEME_GPU_CALL(cudaGetLastError());
    }
}

void RefreshMeshPatchCenters(const size_t* offsets,
                             const size_t* triangles,
                             size_t patches,
                             const float3* node1,
                             const float3* node2,
                             const float3* node3,
                             float3* centers,
                             const float3* vertices,
                             size_t vertex_count,
                             double* diagonal,
                             MeshPatchCenterOptions options,
                             cudaStream_t stream) {
    if (patches > 1) {
        meshDiagonal<<<1, MESH_DEFORMATION_BLOCK, 0, stream>>>(vertices, vertex_count, diagonal);
        DEME_GPU_CALL(cudaGetLastError());
    }
    if (patches) {
        patchCenters<<<patches, MESH_DEFORMATION_BLOCK, 0, stream>>>(offsets, triangles, patches, node1, node2, node3,
                                                                     centers, diagonal, options);
        DEME_GPU_CALL(cudaGetLastError());
    }
}
}  // namespace deme
