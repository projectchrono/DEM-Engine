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

// One block reduces each patch's fixed triangle list. Strided loads keep large patches parallel without atomics;
// the fixed reduction tree makes repeated updates deterministic.
__global__ void patchCenters(const size_t* offsets,
                             const size_t* triangles,
                             size_t patches,
                             const float3* node1,
                             const float3* node2,
                             const float3* node3,
                             float3* centers) {
    const size_t p = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    const size_t count = offsets[p + 1] - offsets[p];
    if (patches == 1 || count == 0) {
        if (lane == 0)
            centers[p] = make_float3(0, 0, 0);
        return;
    }
    __shared__ double sums[3][MESH_DEFORMATION_BLOCK];
    double x = 0, y = 0, z = 0;
    for (size_t j = offsets[p] + lane; j < offsets[p + 1]; j += blockDim.x) {
        const size_t t = triangles[j];
        x += ((double)node1[t].x + node2[t].x + node3[t].x) / 3.;
        y += ((double)node1[t].y + node2[t].y + node3[t].y) / 3.;
        z += ((double)node1[t].z + node2[t].z + node3[t].z) / 3.;
    }
    sums[0][lane] = x;
    sums[1][lane] = y;
    sums[2][lane] = z;
    __syncthreads();
    for (unsigned int stride = MESH_DEFORMATION_BLOCK / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            sums[0][lane] += sums[0][lane + stride];
            sums[1][lane] += sums[1][lane + stride];
            sums[2][lane] += sums[2][lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        centers[p] = make_float3((float)(sums[0][0] / count), (float)(sums[1][0] / count), (float)(sums[2][0] / count));
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
                             cudaStream_t stream) {
    if (patches) {
        patchCenters<<<patches, MESH_DEFORMATION_BLOCK, 0, stream>>>(offsets, triangles, patches, node1, node2, node3,
                                                                     centers);
        DEME_GPU_CALL(cudaGetLastError());
    }
}
}  // namespace deme
