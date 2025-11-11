//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// This file contains modifications of the code by Alessandro Tasora and Radu Serban
// Below is the BSD license

// Copyright (c) 2016, Project Chrono Development Team
// All Rghts Reserved.

// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
// following conditions are met:

//  - Redistributions of source code must retain the above copyright notice, this list of conditions and the following
//  disclaimer.
//  - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
//  following disclaimer in the documentation and/or other materials provided with the distribution.
//  - Neither the name of the nor the names of its contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <unordered_map>
#include <cmath>
#include <queue>
#include <vector>
#include <sstream>

#include "../kernel/DEMHelperKernels.cuh"
#include "BdrsAndObjs.h"
#include "../core/utils/WavefrontMeshLoader.hpp"

namespace deme {

using namespace WAVEFRONT;

std::vector<std::vector<float>> DEMMesh::GetCoordsVerticesAsVectorOfVectors() {
    auto vec = GetCoordsVertices();
    std::vector<std::vector<float>> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        res[i] = {vec[i].x, vec[i].y, vec[i].z};
    }
    return res;
}

std::vector<std::vector<int>> DEMMesh::GetIndicesVertexesAsVectorOfVectors() {
    auto vec = GetIndicesVertexes();
    std::vector<std::vector<int>> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        res[i] = {vec[i].x, vec[i].y, vec[i].z};
    }
    return res;
}

bool DEMMesh::LoadWavefrontMesh(std::string input_file, bool load_normals, bool load_uv) {
    this->m_vertices.clear();
    this->m_normals.clear();
    this->m_UV.clear();
    this->m_face_v_indices.clear();
    this->m_face_n_indices.clear();
    this->m_face_uv_indices.clear();

    GeometryInterface emptybm;  // BuildMesh bm;

    filename = input_file;

    OBJ obj;

    int ret = obj.LoadMesh(filename.c_str(), &emptybm, true);
    if (ret == -1) {
        std::cerr << "Error loading OBJ file " << filename << std::endl;
        return false;
    }

    float3 tmp_f3;
    int3 tmp_i3;
    for (unsigned int iv = 0; iv < obj.mVerts.size(); iv += 3) {
        tmp_f3.x = obj.mVerts[iv];
        tmp_f3.y = obj.mVerts[iv + 1];
        tmp_f3.z = obj.mVerts[iv + 2];
        this->m_vertices.push_back(tmp_f3);
    }
    for (unsigned int in = 0; in < obj.mNormals.size(); in += 3) {
        tmp_f3.x = obj.mNormals[in];
        tmp_f3.y = obj.mNormals[in + 1];
        tmp_f3.z = obj.mNormals[in + 2];
        this->m_normals.push_back(tmp_f3);
    }
    for (unsigned int it = 0; it < obj.mTexels.size(); it += 2)  // +2 because only u,v each texel
    {
        tmp_f3.x = obj.mTexels[it];
        tmp_f3.y = obj.mTexels[it + 1];
        tmp_f3.z = 0;
        this->m_UV.push_back(tmp_f3);
    }
    for (unsigned int iiv = 0; iiv < obj.mIndexesVerts.size(); iiv += 3) {
        tmp_i3.x = obj.mIndexesVerts[iiv];
        tmp_i3.y = obj.mIndexesVerts[iiv + 1];
        tmp_i3.z = obj.mIndexesVerts[iiv + 2];
        this->m_face_v_indices.push_back(tmp_i3);
    }
    for (unsigned int iin = 0; iin < obj.mIndexesNormals.size(); iin += 3) {
        tmp_i3.x = obj.mIndexesNormals[iin];
        tmp_i3.y = obj.mIndexesNormals[iin + 1];
        tmp_i3.z = obj.mIndexesNormals[iin + 2];
        this->m_face_n_indices.push_back(tmp_i3);
    }
    for (unsigned int iit = 0; iit < obj.mIndexesTexels.size(); iit += 3) {
        tmp_i3.x = obj.mIndexesTexels[iit];
        tmp_i3.y = obj.mIndexesTexels[iit + 1];
        tmp_i3.z = obj.mIndexesTexels[iit + 2];
        this->m_face_uv_indices.push_back(tmp_i3);
    }

    if (!load_normals) {
        this->m_normals.clear();
        this->m_face_n_indices.clear();
    }
    if (!load_uv) {
        this->m_UV.clear();
        this->m_face_uv_indices.clear();
    }

    this->nTri = m_face_v_indices.size();

    // Initialize default patch info: all triangles in patch 0 (assuming convex mesh)
    this->m_patch_ids.clear();
    this->m_patch_ids.resize(this->nTri, 0);
    this->num_patches = 1;
    this->patches_explicitly_set = false;

    return true;
}

// Write the specified meshes in a Wavefront .obj file
void DEMMesh::WriteWavefront(const std::string& filename, std::vector<DEMMesh>& meshes) {
    std::ofstream mf(filename);

    //// TODO: include normal information if available

    // Create a single object mesh
    std::vector<int> v_offsets;
    int v_off = 1;
    for (auto& m : meshes) {
        for (auto& v : m.m_vertices) {
            mf << "v " << v.x << " " << v.y << " " << v.z << std::endl;
        }
        v_offsets.push_back(v_off);
        v_off += static_cast<int>(m.m_vertices.size());
    }

    std::vector<bool> has_normals;
    std::vector<int> vn_offsets;
    int vn_off = 1;
    for (auto& m : meshes) {
        has_normals.push_back(m.m_normals.size() > 0);
        for (auto& v : m.m_normals) {
            mf << "vn " << v.x << " " << v.y << " " << v.z << std::endl;
        }
        vn_offsets.push_back(vn_off);
        vn_off += static_cast<int>(m.m_normals.size());
    }

    for (size_t i = 0; i < meshes.size(); i++) {
        v_off = v_offsets[i];
        if (has_normals[i]) {
            auto& idxV = meshes[i].m_face_v_indices;
            auto& idxN = meshes[i].m_face_n_indices;
            assert(idxV.size() == idxN.size());
            vn_off = vn_offsets[i];
            for (int j = 0; j < idxV.size(); j++) {
                mf << "f " << idxV[j].x + v_off << "//" << idxN[j].x + vn_off << " " << idxV[j].y + v_off << "//"
                   << idxN[j].y + vn_off << " " << idxV[j].z + v_off << "//" << idxN[j].z + vn_off << std::endl;
            }
        } else {
            for (auto& f : meshes[i].m_face_v_indices) {
                mf << "f " << f.x + v_off << " " << f.y + v_off << " " << f.z + v_off << std::endl;
            }
        }
    }

    mf.close();
}

// Helper function to compute face normal for a triangle
static float3 computeFaceNormal(const float3& v0, const float3& v1, const float3& v2) {
    float3 edge1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 edge2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

    // Cross product
    float3 normal = make_float3(edge1.y * edge2.z - edge1.z * edge2.y, edge1.z * edge2.x - edge1.x * edge2.z,
                                edge1.x * edge2.y - edge1.y * edge2.x);

    // Normalize
    float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (length > DEME_TINY_FLOAT) {
        normal.x /= length;
        normal.y /= length;
        normal.z /= length;
    }

    return normal;
}

// Helper function to compute angle between two normals (in degrees)
static float computeAngleBetweenNormals(const float3& n1, const float3& n2) {
    // Compute dot product
    float dot_product = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;

    // Clamp to [-1, 1] to avoid numerical issues
    dot_product = std::max(-1.0f, std::min(1.0f, dot_product));

    // Compute angle in radians and convert to degrees
    float angle_rad = std::acos(dot_product);
    return angle_rad * 180.0f / deme::PI;
}

// Helper to build adjacency map for triangles (shared edges)
static std::vector<std::vector<size_t>> buildAdjacencyMap(const std::vector<int3>& face_v_indices) {
    size_t num_faces = face_v_indices.size();
    std::vector<std::vector<size_t>> adjacency(num_faces);

    // Map from edge (as pair of vertex indices) to faces that share it
    std::map<std::pair<int, int>, std::vector<size_t>> edge_to_faces;

    for (size_t i = 0; i < num_faces; ++i) {
        const int3& face = face_v_indices[i];

        // Three edges of the triangle (store with smaller index first for consistency)
        std::pair<int, int> edges[3] = {{std::min(face.x, face.y), std::max(face.x, face.y)},
                                        {std::min(face.y, face.z), std::max(face.y, face.z)},
                                        {std::min(face.z, face.x), std::max(face.z, face.x)}};

        for (int e = 0; e < 3; ++e) {
            edge_to_faces[edges[e]].push_back(i);
        }
    }

    // Build adjacency list
    for (const auto& entry : edge_to_faces) {
        const std::vector<size_t>& faces = entry.second;
        // If two faces share an edge, they are adjacent
        if (faces.size() == 2) {
            adjacency[faces[0]].push_back(faces[1]);
            adjacency[faces[1]].push_back(faces[0]);
        }
    }

    return adjacency;
}

// Split mesh into convex patches using region-growing algorithm.
// The algorithm groups adjacent triangles (sharing an edge) if the angle between their
// face normals is below the threshold. Each patch represents a locally convex region.
size_t DEMMesh::SplitIntoConvexPatches(float angle_threshold_deg) {
    if (nTri == 0) {
        patches_explicitly_set = false;
        num_patches = 1;
        return 0;
    }

    // Initialize patch IDs (all -1 means unassigned)
    m_patch_ids.clear();
    m_patch_ids.resize(nTri, -1);

    // Compute face normals for all triangles
    std::vector<float3> face_normals(nTri);
    for (size_t i = 0; i < nTri; ++i) {
        const int3& face = m_face_v_indices[i];
        const float3& v0 = m_vertices[face.x];
        const float3& v1 = m_vertices[face.y];
        const float3& v2 = m_vertices[face.z];
        face_normals[i] = computeFaceNormal(v0, v1, v2);
    }

    // Build adjacency map (which triangles share edges)
    std::vector<std::vector<size_t>> adjacency = buildAdjacencyMap(m_face_v_indices);

    // Region growing algorithm to assign patches
    int current_patch_id = 0;
    std::vector<size_t> queue;

    for (size_t seed = 0; seed < nTri; ++seed) {
        // Skip if already assigned to a patch
        if (m_patch_ids[seed] != -1) {
            continue;
        }

        // Start a new patch from this seed triangle
        queue.clear();
        queue.push_back(seed);
        m_patch_ids[seed] = current_patch_id;

        // Grow the region
        size_t queue_idx = 0;
        while (queue_idx < queue.size()) {
            size_t current = queue[queue_idx++];

            // Check all adjacent triangles
            for (size_t neighbor : adjacency[current]) {
                // Skip if already assigned
                if (m_patch_ids[neighbor] != -1) {
                    continue;
                }

                // Check angle between normals
                float angle = computeAngleBetweenNormals(face_normals[current], face_normals[neighbor]);

                // If angle is below threshold, add to same patch
                if (angle <= angle_threshold_deg) {
                    m_patch_ids[neighbor] = current_patch_id;
                    queue.push_back(neighbor);
                }
            }
        }

        // Move to next patch
        current_patch_id++;
    }

    num_patches = current_patch_id;
    patches_explicitly_set = true;

    return num_patches;
}

// Manually set patch IDs for each triangle
void DEMMesh::SetPatchIDs(const std::vector<int>& patch_ids) {
    if (patch_ids.size() != nTri) {
        std::stringstream ss;
        ss << "SetPatchIDs: Input vector size (" << patch_ids.size() << ") must match the number of triangles (" << nTri
           << ") in the mesh." << std::endl;
        throw std::runtime_error(ss.str());
    }

    // Validate that all patch IDs are non-negative
    for (size_t i = 0; i < patch_ids.size(); ++i) {
        if (patch_ids[i] < 0) {
            std::stringstream ss;
            ss << "SetPatchIDs: Patch ID at index " << i << " is negative (" << patch_ids[i]
               << "). All patch IDs must be non-negative integers." << std::endl;
            throw std::runtime_error(ss.str());
        }
    }

    // Copy the patch IDs
    m_patch_ids = patch_ids;

    // Calculate the number of patches (maximum patch ID + 1)
    if (!patch_ids.empty()) {
        int max_patch_id = *std::max_element(patch_ids.begin(), patch_ids.end());
        num_patches = max_patch_id + 1;
    } else {
        num_patches = 1;
    }

    patches_explicitly_set = true;
}

}  // end namespace deme
