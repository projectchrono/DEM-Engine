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
#include <cstdint>
#include <map>
#include <unordered_map>
#include <cmath>
#include <queue>
#include <vector>
#include <sstream>
#include <cstring>

#include "../kernel/DEMHelperKernels.cuh"
#include "BdrsAndObjs.h"
#include "../core/utils/WavefrontMeshLoader.hpp"
#include "../core/utils/Logger.hpp"
#include "utils/HostSideHelpers.hpp"

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

bool DEMMesh::LoadSTLMesh(std::string input_file, bool load_normals) {
    Clear();
    filename = input_file;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        DEME_ERROR_NOTHROW("Error loading STL file %s", filename.c_str());
        return false;
    }
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (buffer.size() < 84) {
        DEME_ERROR_NOTHROW("STL file %s is too small to contain any triangles.", filename.c_str());
        return false;
    }

    auto set_default_patch_info = [this]() {
        this->nTri = m_face_v_indices.size();
        this->m_patch_ids.clear();
        this->m_patch_ids.resize(this->nTri, 0);
        this->nPatches = 1;
        this->patches_explicitly_set = false;
    };

    auto load_binary = [&](uint32_t tri_count) -> bool {
        size_t expected_size = 84 + static_cast<size_t>(tri_count) * 50;
        if (buffer.size() < expected_size) {
            DEME_ERROR_NOTHROW("Binary STL file %s ended unexpectedly.", filename.c_str());
            return false;
        }
        const unsigned char* data = reinterpret_cast<const unsigned char*>(buffer.data());
        size_t offset = 84;
        for (uint32_t i = 0; i < tri_count; i++) {
            float floats[12];
            std::memcpy(floats, data + offset, sizeof(float) * 12);
            float3 v0 = make_float3(floats[3], floats[4], floats[5]);
            float3 v1 = make_float3(floats[6], floats[7], floats[8]);
            float3 v2 = make_float3(floats[9], floats[10], floats[11]);
            size_t base = m_vertices.size();
            m_vertices.push_back(v0);
            m_vertices.push_back(v1);
            m_vertices.push_back(v2);
            m_face_v_indices.push_back(make_int3((int)base, (int)base + 1, (int)base + 2));
            offset += 50;
        }
        set_default_patch_info();
        return true;
    };

    // Heuristics to decide if STL is binary
    uint32_t tri_count = 0;
    std::memcpy(&tri_count, buffer.data() + 80, sizeof(uint32_t));
    size_t expected_size = 84 + static_cast<size_t>(tri_count) * 50;
    bool looks_binary = expected_size == buffer.size();
    bool looks_ascii = false;
    if (!looks_binary) {
        std::string header(buffer.data(), buffer.data() + std::min<size_t>(buffer.size(), 5));
        if (header == "solid") {
            looks_ascii = true;
        }
    }

    if (looks_binary && load_binary(tri_count)) {
        return true;
    }

    // Fallback to ASCII parsing
    std::istringstream iss(std::string(buffer.begin(), buffer.end()));
    std::string line;
    std::vector<float3> facet_vertices;
    facet_vertices.reserve(3);
    while (std::getline(iss, line)) {
        std::istringstream ls(line);
        std::string token;
        ls >> token;
        if (token == "vertex") {
            float3 v{};
            ls >> v.x >> v.y >> v.z;
            facet_vertices.push_back(v);
            if (facet_vertices.size() == 3) {
                size_t base = m_vertices.size();
                m_vertices.push_back(facet_vertices[0]);
                m_vertices.push_back(facet_vertices[1]);
                m_vertices.push_back(facet_vertices[2]);
                m_face_v_indices.push_back(make_int3((int)base, (int)base + 1, (int)base + 2));
                facet_vertices.clear();
            }
        }
    }

    if (m_face_v_indices.empty()) {
        DEME_ERROR_NOTHROW("Failed to parse STL file %s.", filename.c_str());
        return false;
    }

    // Compute simple per-facet normals (one normal per triangle) so downstream code can rely on normal data.
    if (load_normals) {
        m_normals.clear();
        m_face_n_indices.clear();
        m_normals.reserve(m_face_v_indices.size());
        m_face_n_indices.reserve(m_face_v_indices.size());
        for (size_t i = 0; i < m_face_v_indices.size(); ++i) {
            const int3& f = m_face_v_indices[i];
            const float3& v0 = m_vertices[f.x];
            const float3& v1 = m_vertices[f.y];
            const float3& v2 = m_vertices[f.z];
            float3 n = face_normal(v0, v1, v2);
            m_normals.push_back(n);
            m_face_n_indices.push_back(make_int3((int)i, (int)i, (int)i));
        }
    } else {
        m_normals.clear();
        m_face_n_indices.clear();
    }
    // STL has no UV by design; clear to mirror OBJ loader when UVs are absent.
    m_UV.clear();
    m_face_uv_indices.clear();

    set_default_patch_info();
    {
        size_t boundary_edges = 0;
        size_t nonmanifold_edges = 0;
        if (!IsWatertight(&boundary_edges, &nonmanifold_edges)) {
            DEME_WARNING(
                "Mesh %s is not watertight (boundary edges: %zu, non-manifold edges: %zu). Auto Volume/MOI may be inaccurate.",
                filename.c_str(), boundary_edges, nonmanifold_edges);
        }
    }
    return true;
}

bool DEMMesh::LoadPLYMesh(std::string input_file, bool load_normals) {
    Clear();
    filename = input_file;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        DEME_ERROR_NOTHROW("Error loading PLY file %s", filename.c_str());
        return false;
    }

    std::string line;
    if (!std::getline(file, line) || line != "ply") {
        DEME_ERROR_NOTHROW("PLY file %s is missing magic header.", filename.c_str());
        return false;
    }

    enum class PLYFormat { ASCII, BINARY_LE, BINARY_BE };
    PLYFormat format = PLYFormat::ASCII;
    size_t num_vertices = 0;
    size_t num_faces = 0;
    // Track vertex property order to find position/normal fields
    std::vector<std::string> vertex_props;
    std::vector<std::string> vertex_prop_types;
    bool in_vertex = false;
    // Face list types
    std::string face_count_type;
    std::string face_index_type;

    while (std::getline(file, line)) {
        if (line == "end_header") {
            break;
        }
        std::istringstream ls(line);
        std::string token;
        ls >> token;
        if (token == "format") {
            std::string fmt;
            ls >> fmt;
            if (fmt.find("ascii") == 0) {
                format = PLYFormat::ASCII;
            } else if (fmt.find("binary_little_endian") == 0) {
                format = PLYFormat::BINARY_LE;
            } else if (fmt.find("binary_big_endian") == 0) {
                format = PLYFormat::BINARY_BE;
            }
        } else if (token == "element") {
            std::string elem;
            ls >> elem;
            if (elem == "vertex") {
                ls >> num_vertices;
                in_vertex = true;
            } else if (elem == "face") {
                ls >> num_faces;
                in_vertex = false;
            } else {
                in_vertex = false;
            }
        } else if (token == "property" && in_vertex) {
            std::string type, name;
            ls >> type >> name;
            if (!name.empty()) {
                vertex_props.push_back(name);
                vertex_prop_types.push_back(type);
            }
        } else if (token == "property" && !in_vertex) {
            std::string maybe_list;
            ls >> maybe_list;
            if (maybe_list == "list") {
                ls >> face_count_type >> face_index_type;
                // ignore name
            }
        }
    }

    if (format == PLYFormat::BINARY_BE) {
        DEME_ERROR_NOTHROW("PLY file %s uses big-endian binary, which is not supported.", filename.c_str());
        return false;
    }
    if (num_vertices == 0 || num_faces == 0) {
        DEME_ERROR_NOTHROW("PLY file %s does not contain vertices or faces.", filename.c_str());
        return false;
    }

    auto find_prop = [&](const std::string& name) -> int {
        for (int i = 0; i < static_cast<int>(vertex_props.size()); ++i) {
            if (vertex_props[i] == name)
                return i;
        }
        return -1;
    };
    const int idx_x = find_prop("x");
    const int idx_y = find_prop("y");
    const int idx_z = find_prop("z");
    const int idx_nx = find_prop("nx");
    const int idx_ny = find_prop("ny");
    const int idx_nz = find_prop("nz");
    const bool has_vertex_normals = idx_nx >= 0 && idx_ny >= 0 && idx_nz >= 0;

    m_vertices.reserve(num_vertices);
    m_face_v_indices.reserve(num_faces);

    auto read_scalar_le = [&](std::istream& is, const std::string& type, double& out) -> bool {
        if (type == "float" || type == "float32") {
            float v;
            if (!is.read(reinterpret_cast<char*>(&v), sizeof(float)))
                return false;
            out = static_cast<double>(v);
            return true;
        }
        if (type == "double" || type == "float64") {
            double v;
            if (!is.read(reinterpret_cast<char*>(&v), sizeof(double)))
                return false;
            out = v;
            return true;
        }
        if (type == "uchar" || type == "uint8") {
            std::uint8_t v;
            if (!is.read(reinterpret_cast<char*>(&v), sizeof(std::uint8_t)))
                return false;
            out = static_cast<double>(v);
            return true;
        }
        if (type == "char" || type == "int8") {
            std::int8_t v;
            if (!is.read(reinterpret_cast<char*>(&v), sizeof(std::int8_t)))
                return false;
            out = static_cast<double>(v);
            return true;
        }
        if (type == "int" || type == "int32") {
            std::int32_t v;
            if (!is.read(reinterpret_cast<char*>(&v), sizeof(std::int32_t)))
                return false;
            out = static_cast<double>(v);
            return true;
        }
        if (type == "uint" || type == "uint32") {
            std::uint32_t v;
            if (!is.read(reinterpret_cast<char*>(&v), sizeof(std::uint32_t)))
                return false;
            out = static_cast<double>(v);
            return true;
        }
        return false;
    };

    // Read vertices
    for (size_t i = 0; i < num_vertices; ++i) {
        if (format == PLYFormat::ASCII) {
            if (!std::getline(file, line)) {
                DEME_ERROR_NOTHROW("Unexpected EOF while reading vertices in %s.", filename.c_str());
                return false;
            }
            std::istringstream ls(line);
            std::vector<double> vals;
            double v;
            while (ls >> v) {
                vals.push_back(v);
            }
            if (idx_x < 0 || idx_y < 0 || idx_z < 0 || vals.size() <= std::max({idx_x, idx_y, idx_z})) {
                DEME_ERROR_NOTHROW("Vertex position data missing in %s.", filename.c_str());
                return false;
            }
            float3 p = make_float3(static_cast<float>(vals[idx_x]), static_cast<float>(vals[idx_y]),
                                   static_cast<float>(vals[idx_z]));
            m_vertices.push_back(p);
            if (has_vertex_normals && vals.size() > static_cast<size_t>(std::max({idx_nx, idx_ny, idx_nz}))) {
                float3 n = make_float3(static_cast<float>(vals[idx_nx]), static_cast<float>(vals[idx_ny]),
                                       static_cast<float>(vals[idx_nz]));
                m_normals.push_back(n);
            }
        } else {
            // Binary little-endian
            std::vector<double> vals(vertex_props.size(), 0.0);
            for (size_t p = 0; p < vertex_props.size(); ++p) {
                if (!read_scalar_le(file, vertex_prop_types[p], vals[p])) {
                    DEME_ERROR_NOTHROW("Failed to read vertex data in binary PLY %s.", filename.c_str());
                    return false;
                }
            }
            if (idx_x < 0 || idx_y < 0 || idx_z < 0) {
                DEME_ERROR_NOTHROW("Vertex position data missing in %s.", filename.c_str());
                return false;
            }
            float3 p = make_float3(static_cast<float>(vals[idx_x]), static_cast<float>(vals[idx_y]),
                                   static_cast<float>(vals[idx_z]));
            m_vertices.push_back(p);
            if (has_vertex_normals && vals.size() > static_cast<size_t>(std::max({idx_nx, idx_ny, idx_nz}))) {
                float3 n = make_float3(static_cast<float>(vals[idx_nx]), static_cast<float>(vals[idx_ny]),
                                       static_cast<float>(vals[idx_nz]));
                m_normals.push_back(n);
            }
        }
    }

    // Read faces
    std::vector<int3> faces;
    faces.reserve(num_faces);
    for (size_t i = 0; i < num_faces; ++i) {
        if (format == PLYFormat::ASCII) {
            if (!std::getline(file, line)) {
                DEME_ERROR_NOTHROW("Unexpected EOF while reading faces in %s.", filename.c_str());
                return false;
            }
            std::istringstream ls(line);
            int verts_in_face = 0;
            ls >> verts_in_face;
            if (verts_in_face < 3) {
                continue;  // ignore degenerate
            }
            std::vector<int> idx(verts_in_face);
            for (int j = 0; j < verts_in_face; ++j) {
                ls >> idx[j];
            }
            for (int t = 1; t < verts_in_face - 1; ++t) {
                faces.push_back(make_int3(idx[0], idx[t], idx[t + 1]));
            }
        } else {
            // Binary little-endian faces: expect list uchar count, int indices
            double count_d = 0.0;
            if (!read_scalar_le(file, face_count_type.empty() ? "uchar" : face_count_type, count_d)) {
                DEME_ERROR_NOTHROW("Failed to read face count in binary PLY %s.", filename.c_str());
                return false;
            }
            int verts_in_face = static_cast<int>(count_d);
            if (verts_in_face < 3) {
                // Skip indices
                for (int j = 0; j < verts_in_face; ++j) {
                    double throwaway;
                    if (!read_scalar_le(file, face_index_type.empty() ? "int" : face_index_type, throwaway)) {
                        DEME_ERROR_NOTHROW("Failed to skip face indices in binary PLY %s.", filename.c_str());
                        return false;
                    }
                }
                continue;
            }
            std::vector<int> idx(verts_in_face);
            for (int j = 0; j < verts_in_face; ++j) {
                double v = 0.0;
                if (!read_scalar_le(file, face_index_type.empty() ? "int" : face_index_type, v)) {
                    DEME_ERROR_NOTHROW("Failed to read face indices in binary PLY %s.", filename.c_str());
                    return false;
                }
                idx[j] = static_cast<int>(v);
            }
            for (int t = 1; t < verts_in_face - 1; ++t) {
                faces.push_back(make_int3(idx[0], idx[t], idx[t + 1]));
            }
        }
    }

    if (faces.empty()) {
        DEME_ERROR_NOTHROW("No faces parsed from PLY file %s.", filename.c_str());
        return false;
    }

    m_face_v_indices = std::move(faces);
    nTri = m_face_v_indices.size();

    if (load_normals) {
        m_normals.clear();
        m_face_n_indices.clear();
        m_normals.reserve(nTri);
        m_face_n_indices.reserve(nTri);
        for (size_t i = 0; i < nTri; ++i) {
            const int3& f = m_face_v_indices[i];
            const float3& v0 = m_vertices[f.x];
            const float3& v1 = m_vertices[f.y];
            const float3& v2 = m_vertices[f.z];
            float3 n = face_normal(v0, v1, v2);
            m_normals.push_back(n);
            m_face_n_indices.push_back(make_int3((int)i, (int)i, (int)i));
        }
    } else {
        m_normals.clear();
        m_face_n_indices.clear();
    }
    m_UV.clear();
    m_face_uv_indices.clear();

    // Default patch info: one patch
    m_patch_ids.assign(nTri, 0);
    nPatches = 1;
    patches_explicitly_set = false;
    return true;
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
        DEME_ERROR_NOTHROW("Error loading OBJ file %s", filename.c_str());
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
    this->nPatches = 1;
    this->patches_explicitly_set = false;

    {
        size_t boundary_edges = 0;
        size_t nonmanifold_edges = 0;
        if (!IsWatertight(&boundary_edges, &nonmanifold_edges)) {
            DEME_WARNING(
                "Mesh %s is not watertight (boundary edges: %zu, non-manifold edges: %zu). Volume/MOI may be inaccurate.",
                filename.c_str(), boundary_edges, nonmanifold_edges);
        }
    }

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

// ------------------------------------------------------------
// Helpers for advanced patching
// ------------------------------------------------------------
struct EdgeAdjInfo {
    size_t nbr = 0;
    int va = -1;              // oriented edge vertex A (as appears in the current triangle)
    int vb = -1;              // oriented edge vertex B (as appears in the current triangle)
    bool oriented_ok = false; // true if the neighbor sees the shared edge reversed (good sign for oriented manifold)
};

static inline float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline float3 cross3(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float norm3(const float3& v) {
    return std::sqrt(dot3(v, v));
}
static inline float3 normalize3(const float3& v) {
    float n = norm3(v);
    if (n > DEME_TINY_FLOAT)
        return make_float3(v.x / n, v.y / n, v.z / n);
    return make_float3(0, 0, 0);
}
static inline float3 add3(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static inline float3 mul3(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}
static inline float clamp11(float x) {
    return std::max(-1.0f, std::min(1.0f, x));
}
static inline float deg2rad(float deg) {
    return deg * (deme::PI / 180.0f);
}
static inline float rad2deg(float rad) {
    return rad * (180.0f / deme::PI);
}

static float computeTriangleArea(const float3& v0, const float3& v1, const float3& v2) {
    float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 c = cross3(e1, e2);
    return 0.5f * norm3(c);
}

// Signed dihedral angle (deg) around oriented edge va->vb of the current triangle.
// Sign is meaningful only when edge orientation is reliable (oriented_ok == true).
static float signedDihedralDeg(const float3& n_cur, const float3& n_nbr, const float3& vA, const float3& vB) {
    float3 e = normalize3(make_float3(vB.x - vA.x, vB.y - vA.y, vB.z - vA.z));
    float s = dot3(e, cross3(n_cur, n_nbr));
    float c = clamp11(dot3(n_cur, n_nbr));
    float theta = std::atan2(s, c);  // [-pi, pi]
    return rad2deg(theta);
}

// Build triangle adjacency WITH oriented shared-edge info.
// Non-manifold edges (shared by != 2 faces) are treated as boundaries.
static std::vector<std::vector<EdgeAdjInfo>> buildAdjacencyWithEdgeInfo(const std::vector<int3>& face_v_indices) {
    struct EdgeRec {
        size_t f;
        int a;
        int b;
    };

    const size_t num_faces = face_v_indices.size();
    std::vector<std::vector<EdgeAdjInfo>> adj(num_faces);

    std::map<std::pair<int, int>, std::vector<EdgeRec>> edge_map;

    auto add_edge = [&](size_t f, int a, int b) {
        int lo = std::min(a, b);
        int hi = std::max(a, b);
        edge_map[{lo, hi}].push_back(EdgeRec{f, a, b});
    };

    for (size_t i = 0; i < num_faces; ++i) {
        const int3& tri = face_v_indices[i];
        add_edge(i, tri.x, tri.y);
        add_edge(i, tri.y, tri.z);
        add_edge(i, tri.z, tri.x);
    }

    for (const auto& kv : edge_map) {
        const auto& recs = kv.second;
        if (recs.size() != 2) {
            continue;  // boundary or non-manifold
        }
        const EdgeRec& r0 = recs[0];
        const EdgeRec& r1 = recs[1];

        bool oriented_ok_0 = (r0.a == r1.b && r0.b == r1.a);
        bool oriented_ok_1 = oriented_ok_0;

        adj[r0.f].push_back(EdgeAdjInfo{r1.f, r0.a, r0.b, oriented_ok_0});
        adj[r1.f].push_back(EdgeAdjInfo{r0.f, r1.a, r1.b, oriented_ok_1});
    }

    return adj;
}

// ------------------------------------------------------------
// Smart patch splitter
// ------------------------------------------------------------
unsigned int DEMMesh::SplitIntoConvexPatches(float hard_angle_deg,
                                             const PatchSplitOptions& opt_in,
                                             PatchQualityReport* out_report,
                                             const PatchQualityOptions& qopt) {
    if (nTri == 0) {
        patches_explicitly_set = false;
        nPatches = 1;
        if (out_report) {
            out_report->overall = PatchQualityLevel::SAFE;
            out_report->constraint_status = PatchConstraintStatus::SATISFIED;
            out_report->achieved_patches = 1;
            out_report->requested_min = 1;
            out_report->requested_max = 1;
            out_report->per_patch.clear();
        }
        return 0;
    }

    if (hard_angle_deg <= 0.0f) {
        DEME_ERROR("SplitIntoConvexPatches: hard_angle_deg must be > 0.");
    }
    if (opt_in.patch_min == 0) {
        DEME_ERROR("SplitIntoConvexPatches: patch_min must be >= 1.");
    }
    if (opt_in.patch_min > opt_in.patch_max) {
        DEME_ERROR("SplitIntoConvexPatches: patch_min cannot be > patch_max.");
    }

    // Copy options (we may adjust defaults in a controlled way)
    PatchSplitOptions opt = opt_in;

    hard_angle_deg = std::min(180.0f, std::max(0.0f, hard_angle_deg));

    // Resolve hysteresis
    float soft_angle_deg = (opt.soft_angle_deg >= 0.0f) ? opt.soft_angle_deg : hard_angle_deg;
    soft_angle_deg = std::min(hard_angle_deg, std::max(0.0f, soft_angle_deg));

    // If user activates hysteresis (soft < hard) but didn't enable patch-normal gating, set a sensible default:
    // otherwise the mid-band has no extra decision signal.
    bool patch_gate_enabled = (opt.patch_normal_max_deg >= 0.0f);
    if (!patch_gate_enabled && soft_angle_deg < hard_angle_deg) {
        opt.patch_normal_max_deg = soft_angle_deg;
        patch_gate_enabled = true;
    }

    float patch_normal_max_deg = opt.patch_normal_max_deg;  // may be <0 => disabled
    if (patch_gate_enabled) {
        patch_normal_max_deg = std::min(180.0f, std::max(0.0f, patch_normal_max_deg));
    }

    const float cos_hard = std::cos(deg2rad(hard_angle_deg));
    const float cos_soft = std::cos(deg2rad(soft_angle_deg));
    float cos_patch = -1.0f;
    if (patch_gate_enabled) {
        cos_patch = std::cos(deg2rad(patch_normal_max_deg));
    }

    // Precompute face normals and areas
    std::vector<float3> face_normals(nTri);
    std::vector<float> face_areas(nTri, 0.0f);
    for (size_t i = 0; i < nTri; ++i) {
        const int3& f = m_face_v_indices[i];
        const float3& v0 = m_vertices[f.x];
        const float3& v1 = m_vertices[f.y];
        const float3& v2 = m_vertices[f.z];
        face_normals[i] = computeFaceNormal(v0, v1, v2);
        face_areas[i] = computeTriangleArea(v0, v1, v2);
        if (face_areas[i] <= DEME_TINY_FLOAT)
            face_areas[i] = 0.0f;
    }

    // Adjacency with edge info
    auto adjacency = buildAdjacencyWithEdgeInfo(m_face_v_indices);

    // Seed order
    std::vector<size_t> seeds(nTri);
    for (size_t i = 0; i < nTri; ++i)
        seeds[i] = i;
    if (opt.seed_largest_first) {
        std::stable_sort(seeds.begin(), seeds.end(), [&](size_t a, size_t b) { return face_areas[a] > face_areas[b]; });
    }

    // Core segmentation routine (no post-merge/split)
    auto segment_once = [&](const PatchSplitOptions& o,
                            float soft_deg,
                            bool patch_gate,
                            float cosPatch,
                            std::vector<patchID_t>& out_ids,
                            unsigned int& out_nP) {
        out_ids.assign(nTri, (patchID_t)-1);

        int current_patch_id = 0;
        std::vector<size_t> queue;
        queue.reserve(256);

        for (size_t si = 0; si < nTri; ++si) {
            size_t seed = seeds[si];
            if (out_ids[seed] != (patchID_t)-1)
                continue;

            if (current_patch_id > std::numeric_limits<patchID_t>::max()) {
                DEME_ERROR("SplitIntoPatches: too many patches for patchID_t.");
            }

            float3 sumN = mul3(face_normals[seed], face_areas[seed]);
            float sumA = face_areas[seed];
            float3 patchN = normalize3(sumN);

            queue.clear();
            queue.push_back(seed);
            out_ids[seed] = (patchID_t)current_patch_id;

            size_t qi = 0;
            while (qi < queue.size()) {
                size_t cur = queue[qi++];

                for (const auto& e : adjacency[cur]) {
                    size_t nb = e.nbr;
                    if (out_ids[nb] != (patchID_t)-1)
                        continue;

                    const float3& n_cur = face_normals[cur];
                    const float3& n_nb = face_normals[nb];

                    // Hard barrier (mandatory)
                    float d_cn = clamp11(dot3(n_cur, n_nb));
                    if (d_cn < cos_hard)
                        continue;

                    // Optional concavity barrier
                    if (o.block_concave_edges && e.oriented_ok) {
                        const float3& vA = m_vertices[e.va];
                        const float3& vB = m_vertices[e.vb];
                        float dih = signedDihedralDeg(n_cur, n_nb, vA, vB);
                        if (dih < -o.concave_allow_deg)
                            continue;
                    }

                    // Hysteresis band:
                    // - if below soft: we still require patch gate if enabled (otherwise accept)
                    // - if between soft and hard: require patch gate if enabled; otherwise accept (legacy-like)
                    bool in_soft = (d_cn >= cos_soft);

                    if (patch_gate) {
                        float d_pn = clamp11(dot3(patchN, n_nb));
                        if (d_pn < cosPatch)
                            continue;
                        // pass patch gate => accept
                    } else {
                        // no patch gate => legacy-like behavior (soft only matters if patch gate is active)
                        (void)in_soft;
                    }

                    out_ids[nb] = (patchID_t)current_patch_id;
                    queue.push_back(nb);

                    if (face_areas[nb] > 0.0f) {
                        sumN = add3(sumN, mul3(n_nb, face_areas[nb]));
                        sumA += face_areas[nb];
                        patchN = normalize3(sumN);
                    }
                }
            }

            current_patch_id++;
        }

        out_nP = (unsigned int)current_patch_id;
    };

    // A small helper to compress patch IDs to [0..nP-1]
    auto compress_ids = [&](std::vector<patchID_t>& ids, unsigned int& out_nP) {
        auto res = rank_transform<patchID_t>(ids);
        ids = std::move(res.first);
        // recompute nP
        patchID_t mx = 0;
        for (auto v : ids)
            if (v > mx) mx = v;
        out_nP = (unsigned int)(mx + 1);
    };

    // Enforce patch_max by merging adjacent patches where allowed (hard/concave respected)
    auto enforce_patch_max = [&](std::vector<patchID_t>& ids, unsigned int& pcount, PatchConstraintStatus& cstat) {
        if (pcount <= opt.patch_max)
            return;

        // Build patch mean normals (area-weighted)
        std::vector<float3> pSumN(pcount, make_float3(0, 0, 0));
        std::vector<float> pSumA(pcount, 0.0f);

        for (size_t t = 0; t < nTri; ++t) {
            int p = (int)ids[t];
            if (face_areas[t] > 0.0f) {
                pSumN[p] = add3(pSumN[p], mul3(face_normals[t], face_areas[t]));
                pSumA[p] += face_areas[t];
            }
        }

        struct DSU {
            std::vector<int> parent, rnk;
            std::vector<float3>* sumN;
            std::vector<float>* sumA;

            DSU(int n, std::vector<float3>& sN, std::vector<float>& sA) : parent(n), rnk(n, 0), sumN(&sN), sumA(&sA) {
                for (int i = 0; i < n; ++i) parent[i] = i;
            }
            int find(int x) {
                while (parent[x] != x) {
                    parent[x] = parent[parent[x]];
                    x = parent[x];
                }
                return x;
            }
            bool unite(int a, int b) {
                a = find(a); b = find(b);
                if (a == b) return false;
                if (rnk[a] < rnk[b]) std::swap(a, b);
                parent[b] = a;
                if (rnk[a] == rnk[b]) rnk[a]++;
                (*sumN)[a] = add3((*sumN)[a], (*sumN)[b]);
                (*sumA)[a] += (*sumA)[b];
                return true;
            }
            float3 patchN(int x) {
                x = find(x);
                return normalize3((*sumN)[x]);
            }
        };

        DSU dsu((int)pcount, pSumN, pSumA);

        struct Cand { float cost; int a; int b; };
        struct Cmp { bool operator()(const Cand& x, const Cand& y) const { return x.cost > y.cost; } };

        auto cost_between = [&](int a, int b) {
            float3 na = dsu.patchN(a);
            float3 nb = dsu.patchN(b);
            float d = clamp11(dot3(na, nb));
            return 1.0f - d;  // smaller is better (more parallel)
        };

        // Candidate patch adjacency across mergeable edges (hard + optional concavity)
        std::map<std::pair<int, int>, float> best_cost;

        for (size_t t = 0; t < nTri; ++t) {
            int pt = (int)ids[t];
            for (const auto& e : adjacency[t]) {
                size_t nb = e.nbr;
                int pn = (int)ids[nb];
                if (pt == pn)
                    continue;

                float d = clamp11(dot3(face_normals[t], face_normals[nb]));
                if (d < cos_hard)
                    continue;

                if (opt.block_concave_edges && e.oriented_ok) {
                    const float3& vA = m_vertices[e.va];
                    const float3& vB = m_vertices[e.vb];
                    float dih = signedDihedralDeg(face_normals[t], face_normals[nb], vA, vB);
                    if (dih < -opt.concave_allow_deg)
                        continue;
                }

                int a = std::min(pt, pn);
                int b = std::max(pt, pn);
                float c = cost_between(a, b);

                auto key = std::make_pair(a, b);
                auto it = best_cost.find(key);
                if (it == best_cost.end() || c < it->second)
                    best_cost[key] = c;
            }
        }

        std::priority_queue<Cand, std::vector<Cand>, Cmp> pq;
        for (const auto& kv : best_cost)
            pq.push(Cand{kv.second, kv.first.first, kv.first.second});

        unsigned int cur = pcount;
        while (cur > opt.patch_max && !pq.empty()) {
            auto c = pq.top(); pq.pop();
            int ra = dsu.find(c.a);
            int rb = dsu.find(c.b);
            if (ra == rb)
                continue;
            if (dsu.unite(ra, rb))
                cur--;
        }

        // If we couldn't merge enough, mark as unmergeable
        if (cur > opt.patch_max)
            cstat = PatchConstraintStatus::TOO_MANY_UNMERGEABLE;

        // Write back merged ids and compress
        std::unordered_map<int, patchID_t> rep2new;
        rep2new.reserve(pcount * 2);

        patchID_t next = 0;
        for (size_t i = 0; i < nTri; ++i) {
            int p = (int)ids[i];
            int r = dsu.find(p);
            auto it = rep2new.find(r);
            if (it == rep2new.end()) {
                rep2new.emplace(r, next);
                ids[i] = next;
                next++;
            } else {
                ids[i] = it->second;
            }
        }
        pcount = (unsigned int)next;
    };

    // Enforce patch_min by splitting worst-spread patches (count-only)
    auto enforce_patch_min = [&](std::vector<patchID_t>& ids, unsigned int& pcount, PatchConstraintStatus& cstat) {
        if (pcount >= opt.patch_min)
            return;

        auto rebuild_patch_lists = [&](std::vector<std::vector<size_t>>& pTris) {
            pTris.assign(pcount, {});
            for (size_t i = 0; i < nTri; ++i) {
                int p = (int)ids[i];
                pTris[p].push_back(i);
            }
        };

        std::vector<std::vector<size_t>> pTris;
        rebuild_patch_lists(pTris);

        auto patch_mean_normal = [&](int p) {
            float3 sumN = make_float3(0, 0, 0);
            float sumA = 0.0f;
            for (size_t t : pTris[p]) {
                if (face_areas[t] > 0.0f) {
                    sumN = add3(sumN, mul3(face_normals[t], face_areas[t]));
                    sumA += face_areas[t];
                }
            }
            (void)sumA;
            return normalize3(sumN);
        };

        auto pick_patch_to_split = [&]() -> int {
            float worst = 1.0f;
            int worst_p = -1;
            for (int p = 0; p < (int)pcount; ++p) {
                if (pTris[p].size() < 2)
                    continue;
                float3 pn = patch_mean_normal(p);
                float minDot = 1.0f;
                for (size_t t : pTris[p]) {
                    float d = clamp11(dot3(pn, face_normals[t]));
                    minDot = std::min(minDot, d);
                }
                if (minDot < worst) {
                    worst = minDot;
                    worst_p = p;
                }
            }
            return worst_p;
        };

        struct Node { float cost; size_t tri; int label; };
        struct NodeCmp { bool operator()(const Node& a, const Node& b) const { return a.cost > b.cost; } };

        std::vector<int8_t> label(nTri, -2);
        std::vector<size_t> touched; touched.reserve(2048);

        while (pcount < opt.patch_min) {
            int p = pick_patch_to_split();
            if (p < 0) {
                cstat = PatchConstraintStatus::TOO_FEW_UNSPLITTABLE;
                break;
            }
            const auto& tris = pTris[p];
            if (tris.size() < 2) {
                cstat = PatchConstraintStatus::TOO_FEW_UNSPLITTABLE;
                break;
            }

            // choose 2 seeds with farthest normals (2-sweep)
            size_t t0 = tris[0];
            size_t sA = t0;
            float best = 1.0f;
            for (size_t t : tris) {
                float d = clamp11(dot3(face_normals[t0], face_normals[t]));
                if (d < best) { best = d; sA = t; }
            }
            size_t sB = sA;
            best = 1.0f;
            for (size_t t : tris) {
                float d = clamp11(dot3(face_normals[sA], face_normals[t]));
                if (d < best) { best = d; sB = t; }
            }
            if (sA == sB) {
                cstat = PatchConstraintStatus::TOO_FEW_UNSPLITTABLE;
                break;
            }

            touched.clear();
            for (size_t t : tris) {
                label[t] = -1;
                touched.push_back(t);
            }

            std::priority_queue<Node, std::vector<Node>, NodeCmp> pq;
            label[sA] = 0; label[sB] = 1;
            pq.push(Node{0.0f, sA, 0});
            pq.push(Node{0.0f, sB, 1});

            const float3 seedN[2] = {face_normals[sA], face_normals[sB]};

            while (!pq.empty()) {
                Node cur = pq.top(); pq.pop();
                size_t t = cur.tri;
                int lbl = cur.label;
                if (label[t] != lbl)
                    continue;

                for (const auto& e : adjacency[t]) {
                    size_t nb = e.nbr;
                    if (label[nb] != -1)
                        continue;

                    float d = clamp11(dot3(face_normals[t], face_normals[nb]));
                    if (d < cos_hard)
                        continue;

                    float dn = clamp11(dot3(face_normals[nb], seedN[lbl]));
                    float cost = 1.0f - dn;

                    label[nb] = (int8_t)lbl;
                    pq.push(Node{cost, nb, lbl});
                }
            }

            size_t c0 = 0, c1 = 0;
            for (size_t t : tris) {
                if (label[t] == 0) c0++;
                else if (label[t] == 1) c1++;
            }
            if (c0 == 0 || c1 == 0) {
                for (size_t t : touched) label[t] = -2;
                cstat = PatchConstraintStatus::TOO_FEW_UNSPLITTABLE;
                break;
            }

            patchID_t newP = (patchID_t)pcount;
            pcount++;

            for (size_t t : tris) {
                ids[t] = (label[t] == 1) ? newP : (patchID_t)p;
            }

            for (size_t t : touched) label[t] = -2;

            // compress & rebuild
            compress_ids(ids, pcount);
            rebuild_patch_lists(pTris);
        }
    };

    // Quality report computation
    auto compute_report = [&](const std::vector<patchID_t>& ids,
                              unsigned int pcount,
                              PatchConstraintStatus cstat,
                              PatchQualityReport& rep) {
        rep.per_patch.assign(pcount, PatchQualityPatch{});
        rep.overall = PatchQualityLevel::SAFE;
        rep.constraint_status = cstat;
        rep.achieved_patches = pcount;
        rep.requested_min = opt.patch_min;
        rep.requested_max = opt.patch_max;

        std::vector<std::vector<size_t>> pTris(pcount);
        for (size_t i = 0; i < nTri; ++i) {
            int p = (int)ids[i];
            pTris[p].push_back(i);
        }

        std::vector<float3> pSumN(pcount, make_float3(0, 0, 0));
        std::vector<float>  pSumA(pcount, 0.0f);

        for (int p = 0; p < (int)pcount; ++p) {
            for (size_t t : pTris[p]) {
                if (face_areas[t] > 0.0f) {
                    pSumN[p] = add3(pSumN[p], mul3(face_normals[t], face_areas[t]));
                    pSumA[p] += face_areas[t];
                }
            }
        }

        // reference angle for classification
        float ref_angle_deg = patch_gate_enabled ? patch_normal_max_deg : hard_angle_deg;

        for (int p = 0; p < (int)pcount; ++p) {
            PatchQualityPatch pq;
            pq.n_tris = (unsigned int)pTris[p].size();

            float3 meanN = normalize3(pSumN[p]);
            float sumA = pSumA[p];
            float r = (sumA > DEME_TINY_FLOAT) ? (norm3(pSumN[p]) / sumA) : 0.0f;
            pq.coherence_r = std::min(1.0f, std::max(0.0f, r));

            float minDot = 1.0f;
            for (size_t t : pTris[p]) {
                float d = clamp11(dot3(meanN, face_normals[t]));
                minDot = std::min(minDot, d);
            }
            pq.worst_angle_deg = rad2deg(std::acos(clamp11(minDot)));

            unsigned int hard_cross = 0;
            unsigned int conc_cross = 0;
            unsigned int unoriented = 0;

            for (size_t t : pTris[p]) {
                for (const auto& e : adjacency[t]) {
                    size_t nb = e.nbr;
                    if ((int)ids[nb] != p)
                        continue;

                    float d = clamp11(dot3(face_normals[t], face_normals[nb]));
                    if (d < cos_hard)
                        hard_cross++;

                    if (opt.block_concave_edges) {
                        if (!e.oriented_ok) {
                            unoriented++;
                        } else {
                            const float3& vA = m_vertices[e.va];
                            const float3& vB = m_vertices[e.vb];
                            float dih = signedDihedralDeg(face_normals[t], face_normals[nb], vA, vB);
                            if (dih < -opt.concave_allow_deg)
                                conc_cross++;
                        }
                    }
                }
            }

            pq.hard_crossings = hard_cross / 2;
            pq.concave_crossings = conc_cross / 2;
            pq.unoriented_edges = unoriented / 2;

            PatchQualityLevel lvl = PatchQualityLevel::SAFE;

            if (qopt.hard_crossings_are_critical && pq.hard_crossings > 0) {
                lvl = PatchQualityLevel::CRITICAL;
            }

            if (lvl != PatchQualityLevel::CRITICAL) {
                bool angle_ok = (pq.worst_angle_deg <= ref_angle_deg);
                bool angle_warn = (pq.worst_angle_deg <= ref_angle_deg + qopt.warn_worst_angle_margin_deg);

                if (pq.coherence_r < qopt.warn_r || !angle_warn) {
                    lvl = PatchQualityLevel::CRITICAL;
                } else if (pq.coherence_r < qopt.safe_r || !angle_ok) {
                    lvl = PatchQualityLevel::WARN;
                }
            }

            if (opt.block_concave_edges && pq.concave_crossings > 0) {
                if (qopt.concave_crossings_are_critical)
                    lvl = PatchQualityLevel::CRITICAL;
                else if (lvl == PatchQualityLevel::SAFE)
                    lvl = PatchQualityLevel::WARN;
            }

            if (opt.block_concave_edges && pq.unoriented_edges >= qopt.unoriented_warn_threshold && lvl == PatchQualityLevel::SAFE) {
                lvl = PatchQualityLevel::WARN;
            }

            pq.level = lvl;
            rep.per_patch[p] = pq;

            if ((int)lvl > (int)rep.overall)
                rep.overall = lvl;
        }
    };

    // ------------------------------------------------------------
    // Optional auto tuning (OFF unless opt.auto_tune.enabled == true)
    // ------------------------------------------------------------
    auto run_full = [&](PatchSplitOptions run_opt,
                        std::vector<patchID_t>& ids_out,
                        unsigned int& pcount_out,
                        PatchConstraintStatus& cstat_out,
                        PatchQualityReport* rep_out) {
        cstat_out = PatchConstraintStatus::SATISFIED;

        float run_soft = (run_opt.soft_angle_deg >= 0.0f) ? run_opt.soft_angle_deg : hard_angle_deg;
        run_soft = std::min(hard_angle_deg, std::max(0.0f, run_soft));

        bool run_patch_gate = (run_opt.patch_normal_max_deg >= 0.0f);
        if (!run_patch_gate && run_soft < hard_angle_deg) {
            run_opt.patch_normal_max_deg = run_soft;
            run_patch_gate = true;
        }

        float run_cos_patch = -1.0f;
        if (run_patch_gate) {
            float run_patch_deg = std::min(180.0f, std::max(0.0f, run_opt.patch_normal_max_deg));
            run_cos_patch = std::cos(deg2rad(run_patch_deg));
        }

        // segment
        segment_once(run_opt, run_soft, run_patch_gate, run_cos_patch, ids_out, pcount_out);
        compress_ids(ids_out, pcount_out);

        // enforce max, then min (count-only)
        enforce_patch_max(ids_out, pcount_out, cstat_out);
        enforce_patch_min(ids_out, pcount_out, cstat_out);

        // final compress
        compress_ids(ids_out, pcount_out);

        if (rep_out) {
            PatchQualityReport tmp;
            // Update globals for report reference (patch_gate_enabled etc.) are based on outer opt;
            // for report classification, we reuse "current" (outer) patch_gate_enabled and patch_normal_max_deg.
            // For best accuracy you can compute ref_angle from run_opt as well; keep simple here.
            compute_report(ids_out, pcount_out, cstat_out, tmp);
            *rep_out = std::move(tmp);
        }
    };

    std::vector<patchID_t> best_ids;
    unsigned int best_pcount = 0;
    PatchConstraintStatus best_cstat = PatchConstraintStatus::SATISFIED;
    PatchQualityReport best_rep;

    if (!opt.auto_tune.enabled) {
        run_full(opt, best_ids, best_pcount, best_cstat, out_report ? &best_rep : nullptr);
    } else {
        // Auto-tuning is conservative: it will not run if you hard-fix the count (patch_min == patch_max),
        // because then your intention is explicit ("keep the cube a cube").
        if (opt.patch_min == opt.patch_max) {
            run_full(opt, best_ids, best_pcount, best_cstat, out_report ? &best_rep : nullptr);
        } else {
            // Start from user options; search by tightening/loosening patch_normal_max_deg (and soft if present)
            PatchSplitOptions cur = opt;

            auto severity_score = [&](PatchQualityLevel lvl) { return (int)lvl; };

            bool have_best = false;

            for (unsigned int it = 0; it < opt.auto_tune.max_iters; ++it) {
                std::vector<patchID_t> ids;
                unsigned int pc = 0;
                PatchConstraintStatus cs = PatchConstraintStatus::SATISFIED;
                PatchQualityReport rep;

                run_full(cur, ids, pc, cs, &rep);

                // candidate score: prioritize meeting constraints, then quality, then fewer patches
                bool constraints_ok = (cs == PatchConstraintStatus::SATISFIED);
                int sev = severity_score(rep.overall);

                auto better_than = [&](bool ok, int s, unsigned int p) {
                    if (!have_best) return true;
                    bool best_ok = (best_cstat == PatchConstraintStatus::SATISFIED);
                    int best_sev = severity_score(best_rep.overall);
                    if (ok != best_ok) return ok;          // prefer satisfied
                    if (s != best_sev) return s < best_sev; // prefer safer
                    return p < best_pcount;                // prefer fewer patches
                };

                if (better_than(constraints_ok, sev, pc)) {
                    best_ids = std::move(ids);
                    best_pcount = pc;
                    best_cstat = cs;
                    best_rep = std::move(rep);
                    have_best = true;
                }

                // stop if good enough
                if (constraints_ok && (int)best_rep.overall <= (int)opt.auto_tune.target_level)
                    break;

                // Adjust rules:
                // - If CRITICAL and we can afford more patches => tighten (smaller patch_normal_max, smaller soft)
                // - If too many unmergeable patches => loosen (bigger patch_normal_max, bigger soft, disable concavity if needed)
                // - If too few patches => tighten
                if (cs == PatchConstraintStatus::TOO_MANY_UNMERGEABLE) {
                    // loosen
                    if (cur.patch_normal_max_deg >= 0.0f)
                        cur.patch_normal_max_deg = std::min(180.0f, cur.patch_normal_max_deg + opt.auto_tune.step_deg);
                    if (cur.soft_angle_deg >= 0.0f)
                        cur.soft_angle_deg = std::min(hard_angle_deg, cur.soft_angle_deg + opt.auto_tune.step_deg);
                    if (cur.block_concave_edges && opt.auto_tune.allow_enable_concavity) {
                        // concavity block can prevent merging; relax it
                        cur.block_concave_edges = false;
                    }
                } else if (pc < opt.patch_min || rep.overall == PatchQualityLevel::CRITICAL) {
                    // tighten if possible
                    if (cur.patch_normal_max_deg < 0.0f)
                        cur.patch_normal_max_deg = std::min(hard_angle_deg, 45.0f);  // enable with a sane default
                    else
                        cur.patch_normal_max_deg = std::max(0.0f, cur.patch_normal_max_deg - opt.auto_tune.step_deg);

                    if (cur.soft_angle_deg >= 0.0f)
                        cur.soft_angle_deg = std::max(0.0f, cur.soft_angle_deg - opt.auto_tune.step_deg);

                    if (!cur.block_concave_edges && opt.auto_tune.allow_enable_concavity) {
                        cur.block_concave_edges = true;
                        cur.concave_allow_deg = std::max(0.0f, cur.concave_allow_deg);
                    }
                } else if (pc > opt.patch_max) {
                    // loosen (but note: enforce_patch_max already tries)
                    if (cur.patch_normal_max_deg >= 0.0f)
                        cur.patch_normal_max_deg = std::min(180.0f, cur.patch_normal_max_deg + opt.auto_tune.step_deg);
                    if (cur.soft_angle_deg >= 0.0f)
                        cur.soft_angle_deg = std::min(hard_angle_deg, cur.soft_angle_deg + opt.auto_tune.step_deg);
                } else {
                    // stable but not good enough; slightly tighten coherence if we have headroom under patch_max
                    if (pc < opt.patch_max) {
                        if (cur.patch_normal_max_deg < 0.0f)
                            cur.patch_normal_max_deg = std::min(hard_angle_deg, 45.0f);
                        else
                            cur.patch_normal_max_deg = std::max(0.0f, cur.patch_normal_max_deg - opt.auto_tune.step_deg);
                    } else {
                        break;
                    }
                }
            }

            // If never found, fall back
            if (!have_best) {
                run_full(opt, best_ids, best_pcount, best_cstat, out_report ? &best_rep : nullptr);
            }
        }
    }

    // Commit to mesh state
    m_patch_ids = std::move(best_ids);
    nPatches = best_pcount;
    patches_explicitly_set = true;

    // Feedback output
    if (out_report) {
        *out_report = std::move(best_rep);
    }

    // Material broadcasting (same as existing behavior)
    if (isMaterialSet && materials.size() == 1) {
        materials = std::vector<std::shared_ptr<DEMMaterial>>(nPatches, materials[0]);
    }
    if (isMaterialSet && materials.size() != nPatches) {
        DEME_ERROR(
            "The number of materials set (%zu) does not match the number of patches (%u). Please set the "
            "material for each patch or use a single material for all patches.",
            materials.size(), nPatches);
    }

    return nPatches;
}

// Manually set patch IDs for each triangle
void DEMMesh::SetPatchIDs(const std::vector<patchID_t>& patch_ids) {
    assertTriLength(patch_ids.size(), "SetPatchIDs");

    // Use rank-transformed patch IDs to ensure they are contiguous and start from 0
    auto [compressed_ids, changed] = rank_transform<patchID_t>(patch_ids);

    if (changed) {
        DEME_WARNING(
            std::string("Patch IDs you supplied for a mesh were not contiguous or did not start from 0.\nThey have "
                        "been transformed to be contiguous and start from 0."));
    }

    // Copy the patch IDs
    m_patch_ids = compressed_ids;

    // Calculate the number of patches (maximum patch ID + 1)
    if (!compressed_ids.empty()) {
        int max_patch_id = *std::max_element(compressed_ids.begin(), compressed_ids.end());
        nPatches = max_patch_id + 1;
    } else {
        nPatches = 1;
    }

    patches_explicitly_set = true;

    // If material is set and we can broadcast it to all patches, we do so
    if (isMaterialSet && materials.size() == 1) {
        materials = std::vector<std::shared_ptr<DEMMaterial>>(nPatches, materials[0]);
    }
    // If material is set and we cannot broadcast it to all patches, we raise error
    if (isMaterialSet && materials.size() != nPatches) {
        DEME_ERROR(
            "The number of materials set (%zu) does not match the number of patches (%u). Please set the "
            "material for each patch or use a single material for all patches.",
            materials.size(), nPatches);
    }
}

// Compute patch locations (relative to CoM, which is implicitly at 0,0,0)
// If not explicitly set, calculates as:
// - Single patch: (0,0,0)
// - Multiple patches: average of triangle centroids per patch
std::vector<float3> DEMMesh::ComputePatchLocations() const {
    std::vector<float3> patch_locations(nPatches, make_float3(0, 0, 0));

    if (nPatches == 1) {
        // Single patch: location is just CoM (0,0,0)
        return patch_locations;
    }

    // Multiple patches: compute average of triangle centroids per patch
    std::vector<int> patch_triangle_counts(nPatches, 0);

    for (size_t i = 0; i < nTri; ++i) {
        const int3& face = m_face_v_indices[i];
        const float3& v0 = m_vertices[face.x];
        const float3& v1 = m_vertices[face.y];
        const float3& v2 = m_vertices[face.z];

        float3 centroid = triangleCentroid<float3>(v0, v1, v2);
        patchID_t patch_id = (i < m_patch_ids.size()) ? m_patch_ids[i] : 0;

        // Validate patch_id is within bounds
        if (patch_id >= 0 && patch_id < static_cast<patchID_t>(nPatches)) {
            patch_locations[patch_id].x += centroid.x;
            patch_locations[patch_id].y += centroid.y;
            patch_locations[patch_id].z += centroid.z;
            patch_triangle_counts[patch_id]++;
        }
    }

    // Average the accumulated centroids
    for (unsigned int p = 0; p < nPatches; ++p) {
        if (patch_triangle_counts[p] > 0) {
            patch_locations[p].x /= patch_triangle_counts[p];
            patch_locations[p].y /= patch_triangle_counts[p];
            patch_locations[p].z /= patch_triangle_counts[p];
        }
    }

    return patch_locations;
}

// Compute volume, centroid and MOI in CoM frame (unit density).
// ATTENTION: Only correct for "watertight" meshes with fine and non-degenerated triangles.
void DEMMesh::ComputeMassProperties(double& volume, float3& center, float3& inertia) const {
    double vol = 0.0;
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    double ix2 = 0.0;
    double iy2 = 0.0;
    double iz2 = 0.0;
    double ixy = 0.0;
    double iyz = 0.0;
    double izx = 0.0;

    for (const auto& face : m_face_v_indices) {
        const float3& a = m_vertices[face.x];
        const float3& b = m_vertices[face.y];
        const float3& c = m_vertices[face.z];

        const float3 bcross = cross(b, c);
        const double v = static_cast<double>(dot(a, bcross)) / 6.0;

        vol += v;
        mx += v * (static_cast<double>(a.x) + b.x + c.x) / 4.0;
        my += v * (static_cast<double>(a.y) + b.y + c.y) / 4.0;
        mz += v * (static_cast<double>(a.z) + b.z + c.z) / 4.0;

        const double ax = a.x, ay = a.y, az = a.z;
        const double bx = b.x, by = b.y, bz = b.z;
        const double cx = c.x, cy = c.y, cz = c.z;

        const double f1x = ax * ax + bx * bx + cx * cx + ax * bx + bx * cx + cx * ax;
        const double f1y = ay * ay + by * by + cy * cy + ay * by + by * cy + cy * ay;
        const double f1z = az * az + bz * bz + cz * cz + az * bz + bz * cz + cz * az;

        ix2 += v * f1x / 10.0;
        iy2 += v * f1y / 10.0;
        iz2 += v * f1z / 10.0;

        const double fxy = 2.0 * (ax * ay + bx * by + cx * cy) +
                           (ax * by + ay * bx + bx * cy + by * cx + cx * ay + cy * ax);
        const double fyz = 2.0 * (ay * az + by * bz + cy * cz) +
                           (ay * bz + az * by + by * cz + bz * cy + cy * az + cz * ay);
        const double fzx = 2.0 * (az * ax + bz * bx + cz * cx) +
                           (az * bx + ax * bz + bz * cx + bx * cz + cz * ax + cx * az);

        ixy += v * fxy / 20.0;
        iyz += v * fyz / 20.0;
        izx += v * fzx / 20.0;
    }

    if (vol == 0.0) {
        volume = 0.0;
        center = make_float3(0, 0, 0);
        inertia = make_float3(0, 0, 0);
        return;
    }

    if (vol < 0.0) {
        vol = -vol;
        mx = -mx;
        my = -my;
        mz = -mz;
        ix2 = -ix2;
        iy2 = -iy2;
        iz2 = -iz2;
        ixy = -ixy;
        iyz = -iyz;
        izx = -izx;
    }

    const double cx = mx / vol;
    const double cy = my / vol;
    const double cz = mz / vol;

    double Ixx = iy2 + iz2;
    double Iyy = ix2 + iz2;
    double Izz = ix2 + iy2;
    double Ixy = -ixy;
    double Iyz = -iyz;
    double Izx = -izx;

    // Shift to center of mass.
    Ixx -= vol * (cy * cy + cz * cz);
    Iyy -= vol * (cx * cx + cz * cz);
    Izz -= vol * (cx * cx + cy * cy);
    Ixy += vol * cx * cy;
    Iyz += vol * cy * cz;
    Izx += vol * cz * cx;

    volume = vol;
    center = make_float3(static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(cz));
    inertia = make_float3(static_cast<float>(Ixx), static_cast<float>(Iyy), static_cast<float>(Izz));
}

// Section for Watertight test, false if not

struct QuantKey3 {
    int64_t x, y, z;
    bool operator==(const QuantKey3& o) const noexcept { return x==o.x && y==o.y && z==o.z; }
};
struct QuantKey3Hash {
    size_t operator()(const QuantKey3& k) const noexcept {
        size_t h1 = std::hash<int64_t>{}(k.x);
        size_t h2 = std::hash<int64_t>{}(k.y);
        size_t h3 = std::hash<int64_t>{}(k.z);
        size_t h = h1;
        h ^= h2 + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        h ^= h3 + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        return h;
    }
};

static inline int64_t q(double v, double eps) {
    return (int64_t)std::llround(v / eps);
}

bool DEMMesh::IsWatertight(size_t* boundary_edges, size_t* nonmanifold_edges) const {
    if (boundary_edges) *boundary_edges = 0;
    if (nonmanifold_edges) *nonmanifold_edges = 0;
    if (m_face_v_indices.empty()) return true;

    auto count_edges_by_index = [&](size_t& boundary, size_t& nonmanifold) {
        std::map<std::pair<size_t, size_t>, size_t> edge_counts;

        for (const auto& face : m_face_v_indices) {
            const int fx = face.x, fy = face.y, fz = face.z;
            if (fx < 0 || fy < 0 || fz < 0) continue;

            const size_t a = (size_t)fx, b = (size_t)fy, c = (size_t)fz;
            if (a == b || b == c || c == a) continue;

            std::pair<size_t, size_t> edges[3] = {
                {std::min(a,b), std::max(a,b)},
                {std::min(b,c), std::max(b,c)},
                {std::min(c,a), std::max(c,a)}
            };
            edge_counts[edges[0]]++;
            edge_counts[edges[1]]++;
            edge_counts[edges[2]]++;
        }

        boundary = 0; nonmanifold = 0;
        for (const auto& kv : edge_counts) {
            if (kv.second == 1) boundary++;
            else if (kv.second > 2) nonmanifold++;
        }
    };

    size_t boundary1 = 0, nonmanifold1 = 0;
    count_edges_by_index(boundary1, nonmanifold1);

    if (boundary1 == 0 && nonmanifold1 == 0) {
        if (boundary_edges) *boundary_edges = 0;
        if (nonmanifold_edges) *nonmanifold_edges = 0;
        return true;
    }

    if (m_vertices.empty()) {
        if (boundary_edges) *boundary_edges = boundary1;
        if (nonmanifold_edges) *nonmanifold_edges = nonmanifold1;
        return false;
    }

    double minx = m_vertices[0].x, miny = m_vertices[0].y, minz = m_vertices[0].z;
    double maxx = minx, maxy = miny, maxz = minz;
    for (const auto& v : m_vertices) {
        minx = std::min(minx, (double)v.x); miny = std::min(miny, (double)v.y);
        minz = std::min(minz, (double)v.z);
        maxx = std::max(maxx, (double)v.x); maxy = std::max(maxy, (double)v.y);
        maxz = std::max(maxz, (double)v.z);
    }
    const double dx = maxx - minx, dy = maxy - miny, dz = maxz - minz;
    const double diag = std::sqrt(dx*dx + dy*dy + dz*dz);
    const double eps = std::max(diag * 1e-9, 1e-12);

    std::unordered_map<QuantKey3, size_t, QuantKey3Hash> rep;
    rep.reserve(m_vertices.size());

    std::vector<size_t> canon(m_vertices.size(), (size_t)-1);
    size_t next_id = 0;

    for (size_t i = 0; i < m_vertices.size(); ++i) {
        const auto& v = m_vertices[i];
        QuantKey3 key{ q(v.x, eps), q(v.y, eps), q(v.z, eps) };

        auto it = rep.find(key);
        if (it == rep.end()) {
            rep.emplace(key, next_id);
            canon[i] = next_id;
            next_id++;
        } else {
            canon[i] = it->second;
        }
    }

    std::map<std::pair<size_t, size_t>, size_t> edge_counts2;
    for (const auto& face : m_face_v_indices) {
        const int fx = face.x, fy = face.y, fz = face.z;
        if (fx < 0 || fy < 0 || fz < 0) continue;

        const size_t a0 = (size_t)fx, b0 = (size_t)fy, c0 = (size_t)fz;
        if (a0 >= canon.size() || b0 >= canon.size() || c0 >= canon.size()) continue;

        const size_t a = canon[a0], b = canon[b0], c = canon[c0];
        if (a == b || b == c || c == a) continue;

        std::pair<size_t, size_t> edges[3] = {
            {std::min(a,b), std::max(a,b)},
            {std::min(b,c), std::max(b,c)},
            {std::min(c,a), std::max(c,a)}
        };
        edge_counts2[edges[0]]++;
        edge_counts2[edges[1]]++;
        edge_counts2[edges[2]]++;
    }

    size_t boundary2 = 0, nonmanifold2 = 0;
    for (const auto& kv : edge_counts2) {
        if (kv.second == 1) boundary2++;
        else if (kv.second > 2) nonmanifold2++;
    }

    if (boundary_edges) *boundary_edges = boundary2;
    if (nonmanifold_edges) *nonmanifold_edges = nonmanifold2;
    return boundary2 == 0 && nonmanifold2 == 0;
}

}  // end namespace deme
