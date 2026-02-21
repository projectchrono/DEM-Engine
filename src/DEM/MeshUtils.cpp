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

    bool parsed = false;
    if (looks_binary) {
        parsed = load_binary(tri_count);
    }
    if (!parsed) {
        // Fallback to ASCII parsing
        std::istringstream iss(std::string(buffer.begin(), buffer.end()));
        std::string line;
        std::vector<float3> facet_vertices;
        facet_vertices.reserve(3);
        while (std::getline(iss, line)) {
            std::istringstream ls(line);
            std::string token;
            ls >> token;
            if (token == "facet") {
                continue;
            }
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
    }

    if (m_face_v_indices.empty()) {
        DEME_ERROR_NOTHROW("Failed to parse STL file %s.", filename.c_str());
        return false;
    }

    set_default_patch_info();

    // Compute one geometric normal per facet from triangle winding.
    // This restores the previous DEM contact behavior and avoids relying on
    // inconsistent/inverted normals embedded in STL files.
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
            const float3 n = face_normal(v0, v1, v2);
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

    {
        size_t boundary_edges = 0;
        size_t nonmanifold_edges = 0;
        if (!IsWatertight(&boundary_edges, &nonmanifold_edges)) {
            DEME_WARNING(
                "Mesh %s is not watertight (boundary edges: %zu, non-manifold edges: %zu). Auto Volume/MOI may be "
                "inaccurate.",
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
                "Mesh %s is not watertight (boundary edges: %zu, non-manifold edges: %zu). Volume/MOI may be "
                "inaccurate.",
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
    int va = -1;               // oriented edge vertex A (as appears in the current triangle)
    int vb = -1;               // oriented edge vertex B (as appears in the current triangle)
    bool oriented_ok = false;  // true if the neighbor sees the shared edge reversed (good sign for oriented manifold)
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
    float3 inertia_products = make_float3(0, 0, 0);
    ComputeMassProperties(volume, center, inertia, inertia_products);
}

// Compute volume, centroid and full inertia tensor in CoM frame (unit density).
// ATTENTION: Only correct for "watertight" meshes with fine and non-degenerated triangles.
void DEMMesh::ComputeMassProperties(double& volume, float3& center, float3& inertia, float3& inertia_products) const {
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

        const double fxy =
            2.0 * (ax * ay + bx * by + cx * cy) + (ax * by + ay * bx + bx * cy + by * cx + cx * ay + cy * ax);
        const double fyz =
            2.0 * (ay * az + by * bz + cy * cz) + (ay * bz + az * by + by * cz + bz * cy + cy * az + cz * ay);
        const double fzx =
            2.0 * (az * ax + bz * bx + cz * cx) + (az * bx + ax * bz + bz * cx + bx * cz + cz * ax + cx * az);

        ixy += v * fxy / 20.0;
        iyz += v * fyz / 20.0;
        izx += v * fzx / 20.0;
    }

    if (vol == 0.0) {
        volume = 0.0;
        center = make_float3(0, 0, 0);
        inertia = make_float3(0, 0, 0);
        inertia_products = make_float3(0, 0, 0);
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
    inertia_products = make_float3(static_cast<float>(Ixy), static_cast<float>(Iyz), static_cast<float>(Izx));
}

// Section for Watertight test, false if not

bool DEMMesh::IsWatertight(size_t* boundary_edges, size_t* nonmanifold_edges) const {
    if (boundary_edges)
        *boundary_edges = 0;
    if (nonmanifold_edges)
        *nonmanifold_edges = 0;
    if (m_face_v_indices.empty())
        return true;

    auto count_edges_by_index = [&](size_t& boundary, size_t& nonmanifold) {
        std::map<std::pair<size_t, size_t>, size_t> edge_counts;

        for (const auto& face : m_face_v_indices) {
            const int fx = face.x, fy = face.y, fz = face.z;
            if (fx < 0 || fy < 0 || fz < 0)
                continue;

            const size_t a = (size_t)fx, b = (size_t)fy, c = (size_t)fz;
            if (a == b || b == c || c == a)
                continue;

            std::pair<size_t, size_t> edges[3] = {
                {std::min(a, b), std::max(a, b)}, {std::min(b, c), std::max(b, c)}, {std::min(c, a), std::max(c, a)}};
            edge_counts[edges[0]]++;
            edge_counts[edges[1]]++;
            edge_counts[edges[2]]++;
        }

        boundary = 0;
        nonmanifold = 0;
        for (const auto& kv : edge_counts) {
            if (kv.second == 1)
                boundary++;
            else if (kv.second > 2)
                nonmanifold++;
        }
    };

    size_t boundary1 = 0, nonmanifold1 = 0;
    count_edges_by_index(boundary1, nonmanifold1);

    if (boundary1 == 0 && nonmanifold1 == 0) {
        if (boundary_edges)
            *boundary_edges = 0;
        if (nonmanifold_edges)
            *nonmanifold_edges = 0;
        return true;
    }

    if (m_vertices.empty()) {
        if (boundary_edges)
            *boundary_edges = boundary1;
        if (nonmanifold_edges)
            *nonmanifold_edges = nonmanifold1;
        return false;
    }

    const double eps = computeVertexQuantEps(m_vertices);
    const auto canon = buildCanonicalVertexMap(m_vertices, eps);

    std::map<std::pair<size_t, size_t>, size_t> edge_counts2;
    for (const auto& face : m_face_v_indices) {
        const int fx = face.x, fy = face.y, fz = face.z;
        if (fx < 0 || fy < 0 || fz < 0)
            continue;

        const size_t a0 = (size_t)fx, b0 = (size_t)fy, c0 = (size_t)fz;
        if (a0 >= canon.size() || b0 >= canon.size() || c0 >= canon.size())
            continue;

        const size_t a = canon[a0], b = canon[b0], c = canon[c0];
        if (a == b || b == c || c == a)
            continue;

        std::pair<size_t, size_t> edges[3] = {
            {std::min(a, b), std::max(a, b)}, {std::min(b, c), std::max(b, c)}, {std::min(c, a), std::max(c, a)}};
        edge_counts2[edges[0]]++;
        edge_counts2[edges[1]]++;
        edge_counts2[edges[2]]++;
    }

    size_t boundary2 = 0, nonmanifold2 = 0;
    for (const auto& kv : edge_counts2) {
        if (kv.second == 1)
            boundary2++;
        else if (kv.second > 2)
            nonmanifold2++;
    }

    if (boundary_edges)
        *boundary_edges = boundary2;
    if (nonmanifold_edges)
        *nonmanifold_edges = nonmanifold2;
    return boundary2 == 0 && nonmanifold2 == 0;
}

}  // end namespace deme
